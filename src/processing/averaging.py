import numpy as np
from src.io.structs import Output, Wave, WaveOutput


def _circ_mean_deg(sin_acc: float, cos_acc: float) -> float:
    """Circular mean [0, 360) from pre-accumulated mean_sin / mean_cos."""
    return float(np.degrees(np.arctan2(sin_acc, cos_acc)) % 360)


class Averager:
    """Accumulates processing outputs over MEAN frames and returns a blended result.

    Averaging policy
    ────────────────
    Averaged (arithmetic):    spec_1d, spec_2d, wave_sum.swh, wave_sum.t_m,
                              wspd, curr_speed.
    Averaged (circular mean): wave_sum.d_m, wind_dir, curr_dir.
    Last frame only:          wave_sum.t_p, wave_sum.d_p, wave_sum.snr,
                              wave_win, wave_sw1, wave_sw2, ide_sys.

    Rationale: peak parameters and per-system partitions vary discretely and have
    no physically meaningful arithmetic mean.  Summary scalars (SWH, Tm, Dm,
    wind, current) change slowly and benefit from smoothing.  Spectral arrays are
    averaged for display continuity.
    """

    def __init__(self, mean: int, n_freq: int, n_freq_2d: int, n_dirs: int,
                 n_shots: int, cut_num: int):
        self.n_dirs = n_dirs
        self.mean   = mean
        self.outputs = [
            WaveOutput(ide_sys=1,
                       wave_sum=Wave(), wave_win=Wave(), wave_sw1=Wave(), wave_sw2=Wave(),
                       spec_1d=np.zeros(n_freq), spec_2d=np.zeros((n_dirs, n_freq_2d)))
            for _ in range(mean)
        ]
        self.port  = np.zeros((mean, n_shots, cut_num))
        self.index = 0

    def push(self, wave_out: WaveOutput, port: np.ndarray):
        i = self.index % self.mean
        self.outputs[i] = wave_out
        self.port[i]    = port
        self.index += 1

    def get_mean(self, pulse: int, step: float, rpm: float,
                 n_shots: int, asp: int, adp: int):
        size = min(self.index, self.mean)
        last = self.outputs[(self.index - 1) % self.mean]

        # ── Peak and per-system params: always from the most recent computation ─
        ws = last.wave_sum
        res_out = Output(
            pulse=pulse, step=step, rps=rpm,
            n_in_win=n_shots - 1, n_wins=size,
            step_area=step, n_area=asp * 2, n_start=adp,
            cog_proc=0, sog_proc=0, max_sys=3, ide_sys=last.ide_sys,
            wave_sum=Wave(snr=ws.snr, swh=ws.swh,
                          t_p=ws.t_p,  t_m=ws.t_m,
                          d_p=ws.d_p,  d_m=ws.d_m),
            wave_win=Wave(swh=last.wave_win.swh,
                          t_p=last.wave_win.t_p, d_p=last.wave_win.d_p),
            wave_sw1=Wave(swh=last.wave_sw1.swh,
                          t_p=last.wave_sw1.t_p, d_p=last.wave_sw1.d_p),
            wave_sw2=Wave(swh=last.wave_sw2.swh,
                          t_p=last.wave_sw2.t_p, d_p=last.wave_sw2.d_p),
            n_dis=32,
            spec_1d=np.zeros_like(last.spec_1d, dtype=float),
            spec_2d=np.zeros_like(last.spec_2d, dtype=float),
        )

        if size == 0:
            return res_out, None

        # ── Accumulate averaged fields ─────────────────────────────────────────
        sum_swh  = 0.0
        sum_tm   = 0.0
        sin_dm   = 0.0;  cos_dm   = 0.0
        sum_wspd = 0.0
        sin_wdir = 0.0;  cos_wdir = 0.0
        sum_curr = 0.0
        sin_cdir = 0.0;  cos_cdir = 0.0

        for j in range(size):
            o = self.outputs[j]
            w = 1.0 / size

            res_out.spec_1d += o.spec_1d * w
            res_out.spec_2d += o.spec_2d * w

            sum_swh  += (o.wave_sum.swh if np.isfinite(o.wave_sum.swh) else 0.0) * w
            sum_tm   += (o.wave_sum.t_m if np.isfinite(o.wave_sum.t_m) else 0.0) * w
            sin_dm   += np.sin(np.radians(o.wave_sum.d_m)) * w
            cos_dm   += np.cos(np.radians(o.wave_sum.d_m)) * w

            sum_wspd += (o.wspd       if np.isfinite(o.wspd)       else 0.0) * w
            sin_wdir += np.sin(np.radians(o.wind_dir)) * w
            cos_wdir += np.cos(np.radians(o.wind_dir)) * w

            sum_curr += (o.curr_speed if np.isfinite(o.curr_speed) else 0.0) * w
            sin_cdir += np.sin(np.radians(o.curr_dir)) * w
            cos_cdir += np.cos(np.radians(o.curr_dir)) * w

        # ── Write averaged scalars ─────────────────────────────────────────────
        res_out.wave_sum.swh = sum_swh
        res_out.wave_sum.t_m = sum_tm
        res_out.wave_sum.d_m = _circ_mean_deg(sin_dm,   cos_dm)
        res_out.wspd         = sum_wspd
        res_out.wind_dir     = _circ_mean_deg(sin_wdir,  cos_wdir)
        res_out.curr_speed   = sum_curr
        res_out.curr_dir     = _circ_mean_deg(sin_cdir,  cos_cdir)

        # ── Normalise spectra to [0, 255] ──────────────────────────────────────
        for arr in (res_out.spec_1d, res_out.spec_2d):
            mx = np.nanmax(arr)
            if mx > 0:
                arr[:] = arr / mx * 255
        res_out.spec_1d[np.isnan(res_out.spec_1d)] = 0
        res_out.spec_1d = res_out.spec_1d.astype(int)
        res_out.spec_2d[np.isnan(res_out.spec_2d)] = 0
        res_out.spec_2d = res_out.spec_2d.astype(int)

        port = np.mean(self.port[:size], axis=0)
        mx   = np.nanmax(port)
        if mx > 0:
            port = port / mx * 255

        return res_out, port.astype(int)
