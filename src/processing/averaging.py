import numpy as np
from src.io.structs import Output, Wave, WaveOutput

_ZERO_WAVE = Wave()


class Averager:
    """Accumulates and averages processing outputs over MEAN frames."""

    def __init__(self, mean: int, n_freq: int, n_dirs: int, n_shots: int, cut_num: int):
        self.n_dirs = n_dirs
        self.mean = mean

        self.outputs = [WaveOutput(1, Wave(), Wave(), Wave(), Wave(),
                                   np.zeros(n_freq), np.zeros((n_dirs, n_freq)))] * mean
        self.spec_1d = np.zeros((mean, n_freq))
        self.spec_2d = np.zeros((mean, n_dirs, n_freq))
        self.port = np.zeros((mean, n_shots, cut_num))
        self.index = 0

    def push(self, wave_out: WaveOutput,
             spec_1d: np.ndarray, spec_2d: np.ndarray, port: np.ndarray):
        i = self.index % self.mean
        self.outputs[i] = wave_out
        self.spec_1d[i] = spec_1d
        self.spec_2d[i] = spec_2d
        self.port[i] = port
        self.index += 1

    def get_mean(self, pulse: int, step: float, rpm: float,
                 n_shots: int, asp: int, adp: int):
        size = min(self.index, self.mean)

        res_out = Output(
            pulse=pulse, step=step, rps=rpm,
            n_in_win=n_shots - 1, n_wins=size,
            step_area=step, n_area=asp * 2, n_start=adp,
            cog_proc=0, sog_proc=0, max_sys=3, ide_sys=1,
            wave_sum=Wave(), wave_win=Wave(), wave_sw1=Wave(), wave_sw2=Wave(),
            n_dis=32,
            spec_1d=np.zeros_like(self.spec_1d[0]),
            spec_2d=np.zeros_like(self.spec_2d[0]),
        )

        if size == 0:
            return res_out, None, None, None

        for j in range(size):
            res_out.spec_1d += self.outputs[j].spec_1d / size
            res_out.spec_2d += self.outputs[j].spec_2d / size
            res_out.wave_sum = res_out.wave_sum.sum(self.outputs[j].wave_sum, size)
            res_out.wave_win = res_out.wave_win.sum(self.outputs[j].wave_win, size)
            res_out.wave_sw1 = res_out.wave_sw1.sum(self.outputs[j].wave_sw1, size)
            res_out.wave_sw2 = res_out.wave_sw2.sum(self.outputs[j].wave_sw2, size)
        # Use most recent n_sys (not the hardcoded initial value of 1)
        res_out.ide_sys = self.outputs[(self.index - 1) % self.mean].ide_sys

        spec_1d = np.mean(self.spec_1d[:size], axis=0)
        spec_2d = np.mean(self.spec_2d[:size], axis=0)
        port = np.mean(self.port[:size], axis=0)

        # Normalise to [0, 255] for UDP packet
        for arr in (res_out.spec_1d, res_out.spec_2d):
            mx = np.nanmax(arr)
            if mx > 0:
                arr[:] = arr / mx * 255

        for arr in (spec_1d, spec_2d):
            mx = np.nanmax(arr)
            if mx > 0:
                arr[:] = arr / mx * 255

        mx = np.nanmax(port)
        if mx > 0:
            port = port / mx * 255

        res_out.spec_1d[np.isnan(res_out.spec_1d)] = 0
        res_out.spec_1d = res_out.spec_1d.astype(int)
        res_out.spec_2d[np.isnan(res_out.spec_2d)] = 0
        res_out.spec_2d = res_out.spec_2d.astype(int)

        return res_out, spec_1d.astype(int), spec_2d.astype(int), port.astype(int)
