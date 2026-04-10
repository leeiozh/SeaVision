import numpy as np
from src.io.structs import Output, Wave, WaveOutput


class Averager:
    """
    Accumulates and averages processing outputs.
    """

    def __init__(self, mean: int, n_freq: int, n_dirs: int, n_shots: int, cut_num: int):
        self.mean = mean

        self.outputs = [WaveOutput(1, Wave(0, 0, 0, 0, 0, 0, 0, False),
                                   Wave(0, 0, 0, 0, 0, 0, 0, False),
                                   Wave(0, 0, 0, 0, 0, 0, 0, False),
                                   Wave(0, 0, 0, 0, 0, 0, 0, False),
                                   np.zeros(n_freq), np.zeros((n_dirs, n_freq)))] * mean
        self.spec_1d = np.zeros((mean, n_freq))
        self.spec_2d = np.zeros((mean, n_dirs, n_freq))
        self.port = np.zeros((mean, n_shots, cut_num))
        self.index = 0

    def push(
            self,
            wave_out: WaveOutput,
            spec_1d: np.ndarray,
            spec_2d: np.ndarray,
            port: np.ndarray,
    ):
        i = self.index % self.mean
        self.outputs[i] = wave_out
        self.spec_1d[i] = spec_1d
        self.spec_2d[i] = spec_2d
        self.port[i] = port
        self.index += 1

    def get_mean(
            self,
            pulse: int,
            step: float,
            rpm: float,
            n_shots: int,
            asp: int,
            adp: int,
    ):
        size = min(self.index, self.mean)

        res_out = Output(
            pulse=pulse,
            step=step,
            rps=rpm,
            n_in_win=n_shots - 1,
            n_wins=size,
            step_area=step,
            n_area=asp * 2,
            n_start=adp,
            cog_proc=0,
            sog_proc=0,
            max_sys=3,
            ide_sys=1,
            wave_sum=Wave(0, 0, 0, 0, 0, 0, 0, False),
            wave_win=Wave(0, 0, 0, 0, 0, 0, 0, False),
            wave_sw1=Wave(0, 0, 0, 0, 0, 0, 0, False),
            wave_sw2=Wave(0, 0, 0, 0, 0, 0, 0, False),
            n_dis=32,
            spec_1d=np.zeros_like(self.spec_1d[0]),
            spec_2d=np.zeros_like(self.spec_2d[0])
        )

        if size == 0:
            return res_out, None, None, None

        for j in range(size):
            res_out.spec_1d += self.outputs[j].spec_1d / size
            res_out.spec_2d += self.outputs[j].spec_2d / size
            res_out.wave_sum = res_out.wave_sum.sum(self.outputs[j].wave_sum, size)

        spec_1d = np.mean(self.spec_1d[:size], axis=0)
        spec_2d = np.mean(self.spec_2d[:size], axis=0)
        port = np.mean(self.port[:size], axis=0)

        # normalization
        if np.nanmax(res_out.spec_1d) > 0:
            res_out.spec_1d = res_out.spec_1d / np.nanmax(res_out.spec_1d) * 255
        if np.nanmax(spec_1d) > 0:
            spec_1d = spec_1d / np.nanmax(spec_1d) * 255

        # normalization
        if np.nanmax(res_out.spec_2d) > 0:
            res_out.spec_2d = res_out.spec_2d / np.nanmax(res_out.spec_2d) * 255
        if np.nanmax(spec_2d) > 0:
            spec_2d = spec_2d / np.nanmax(spec_2d) * 255

        if np.nanmax(port) > 0:
            port = port / np.nanmax(port) * 255

        res_out.spec_1d[np.isnan(res_out.spec_1d)] = 0
        res_out.spec_1d = res_out.spec_1d.astype(int)

        res_out.spec_2d[np.isnan(res_out.spec_2d)] = 0
        res_out.spec_2d = res_out.spec_2d.astype(int)

        spec_1d = spec_1d.astype(int)
        spec_2d = spec_2d.astype(int)
        port = port.astype(int)

        return res_out, spec_1d, spec_2d, port
