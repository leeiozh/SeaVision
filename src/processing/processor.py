import numpy as np
from scipy.interpolate import interp1d
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import logging as _logging
_log = _logging.getLogger(__name__)

from src.processing.state import ProcessorState
from src.processing.averaging import Averager
from src.algorithms.area import Area
from src.algorithms.dispersion import calc_current_vector
from src.algorithms.spectrum2d import (
    calc_spec3d, calc_port,
    apply_doppler_3d_vec,
    separate_signal_noise, apply_mtf,
    compute_snr, compute_frequency_spectrum,
    calc_spec2d,
)

_SIGNAL_BAND  = 10    # ±bins around ω=√(gk) for signal extraction (shared by all steps)
_MAX_CURRENT  = 2.55  # physical clip for ocean current [m/s]

# Quality thresholds (hardcoded, not in config)
_SNR_QUALITY_MIN = 1.5    # min snr_tot
_WIND_SIG_MIN    = 5.0   # min std of backscatter in ADP±ASP ring
_T_PEAK_MIN      = 5.5    # min peak period [s]

from src.algorithms.partition import calc_wspd, calc_partitions
from src.io.structs import Wave, WaveOutput


# ── debug helpers ─────────────────────────────────────────────────────────────

def _save_debug_segments(cbck, raw_ports, raw_spec3ds, azimuths, k_max, om_max, path):
    """
    Grid: NUM_AREA rows × 3 cols:
      col 1 — mean backscatter (Cartesian, interpolated)
      col 2 — raw ω-k portrait + dispersion curve
      col 3 — kx-ky slice summed over ω > 0.1·om_max
    Each row is labelled with its azimuth.
    """
    import os
    out_path = (path if path.endswith('.png')
                else os.path.join(path, 'debug_segments.png'))

    num_area = len(raw_ports)
    k_num = raw_ports[0].shape[1]
    k_arr = np.linspace(0, k_max, k_num)
    disp_om = np.sqrt(9.81 * k_arr)

    # Unified scale for all portraits
    port_vmax = max(float(p.max()) for p in raw_ports)
    port_vmax = max(port_vmax, 1e-9)

    from matplotlib import cm as mpl_cm
    from matplotlib.colors import Normalize
    norm_port = Normalize(vmin=0, vmax=port_vmax)
    cmap_port = mpl_cm.gnuplot2

    fig = Figure(figsize=(18, num_area * 1.8))
    FigureCanvasAgg(fig)

    portrait_axes = []

    for i in range(num_area):
        az = float(azimuths[i])

        # ── col 1: mean backscatter ──────────────────────────────────────────
        ax = fig.add_subplot(num_area, 3, 3 * i + 1)
        bck_mean = cbck[i, -1]
        ax.imshow(bck_mean, cmap='gray', aspect='equal', origin='lower',
                  vmin=100, vmax=150)
        ax.set_title(f'{az:.0f}°', fontsize=15, pad=3, fontweight='bold')
        ax.axis('off')

        # ── col 2: ω-k portrait (unified colorbar) ───────────────────────────
        ax = fig.add_subplot(num_area, 3, 3 * i + 2)
        port = raw_ports[i]
        ax.imshow(port, aspect='auto', origin='lower', cmap=cmap_port, norm=norm_port,
                  extent=[0, k_max, 0, om_max])
        ax.plot(k_arr, disp_om, 'w--', lw=0.8)
        ax.set_xlim(0, k_max)
        ax.set_ylim(0, om_max)
        ax.text(0.97, 0.03, f'{port.max():e}', transform=ax.transAxes,
                color='white', fontsize=15, ha='right', va='bottom')
        ax.axis('off')
        portrait_axes.append(ax)

        # ── col 3: kx-ky slice above 0.1·om_max, zoomed to ±k_max/2 ─────────
        ax = fig.add_subplot(num_area, 3, 3 * i + 3)
        spec3d = raw_spec3ds[i]  # (half, 2*k_num, 2*k_num)
        om_min_idx = max(1, round(0.1 * spec3d.shape[0]))
        kxy = spec3d[om_min_idx:].sum(axis=0)  # (2*k_num, 2*k_num)
        ax.imshow(kxy, aspect='equal', origin='lower', cmap='gnuplot2',
                  extent=[-k_max, k_max, -k_max, k_max],
                  vmin=0, vmax=max(float(kxy.max()), 1e-9))
        ax.axhline(0, color='w', lw=0.4, alpha=0.5)
        ax.axvline(0, color='w', lw=0.4, alpha=0.5)
        ax.set_xlim(-k_max / 2, k_max / 2)
        ax.set_ylim(-k_max / 2, k_max / 2)
        ax.axis('off')

    fig.subplots_adjust(wspace=0.03, hspace=0.05, left=0.01, right=0.97, top=0.99, bottom=0.01)
    try:
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
    except Exception:
        pass


def _save_debug_portrait(port_fixed, step, asp, k_num, om_max,
                         sys_dict, Ux, Uy, path):
    """Save Doppler-corrected ω-k portrait with per-system residual dispersion curves."""
    import os
    out_path = (path if path.endswith('.png')
                else os.path.join(path, 'debug_portrait.png'))

    k_max = np.pi / asp / step * k_num
    k_arr = np.linspace(0, k_max, k_num)

    fig = Figure(figsize=(9, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    vmax = max(float(port_fixed.max()), 1e-9)
    ax.imshow(port_fixed, aspect='auto', origin='lower', cmap='gnuplot2',
              extent=[0, k_max, 0, om_max], vmin=0, vmax=vmax)
    ax.plot(k_arr, np.sqrt(9.81 * k_arr), 'w--', lw=1.2, label='ω = √(gk)', alpha=0.6)

    sys_styles = [
        ('w_s', 'cyan', 'wind sea'),
        ('sw_1', 'lime', 'swell 1'),
        ('sw_2', 'orange', 'swell 2'),
    ]
    for key, clr, lbl in sys_styles:
        s = sys_dict.get(key)
        if s is None or s['t_p'] <= 0:
            continue
        d_p_rad = np.deg2rad(s['d_p'])
        vco_sys = float(Ux * np.cos(d_p_rad) + Uy * np.sin(d_p_rad))
        om_curve = np.sqrt(9.81 * k_arr) + k_arr * vco_sys
        mask = (om_curve >= 0) & (om_curve <= om_max)
        ax.plot(k_arr[mask], om_curve[mask], color=clr, lw=1.5,
                label=f'{lbl}  residual vco={vco_sys:+.2f} m/s')

    ax.set_xlabel('k  [rad/m]')
    ax.set_ylabel('ω  [rad/s]')
    ax.set_title(f'Doppler-corrected ω–k portrait  (Ux={Ux:+.2f} Uy={Uy:+.2f} m/s)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, k_max)
    ax.set_ylim(0, om_max)

    try:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
    except Exception:
        pass


def _save_debug_spec2d(s_om_th, omega_vals, wave_sum, sys_dict, num_area, path):
    """Save directional spectrum image with detected system markers."""
    import os
    out_path = (path if path.endswith('.png')
                else os.path.join(path, 'debug_spec2d.png'))

    fig = Figure(figsize=(9, 4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    vmax = max(float(s_om_th.max()), 1e-9)
    ax.imshow(s_om_th, aspect='auto', origin='lower', cmap='gnuplot2',
              extent=[0, omega_vals[-1], 0, num_area], vmin=0, vmax=vmax)

    dir_bin = wave_sum.d_p / 360 * num_area
    ax.axhline(dir_bin, color='red', lw=2, label=f'Sum {wave_sum.d_p:.0f}°')
    if wave_sum.t_p > 0:
        ax.axvline(2 * np.pi / wave_sum.t_p, color='red', lw=1, ls='--')

    colors = {'w_s': 'cyan', 'sw_1': 'lime', 'sw_2': 'orange'}
    for key, clr in colors.items():
        sys = sys_dict.get(key)
        if sys is None:
            continue
        ax.axhline(sys['d_p'] / 360 * num_area, color=clr, lw=1.5,
                   label=f"{key} {sys['d_p']:.0f}°")
        if sys['t_p'] > 0:
            ax.axvline(2 * np.pi / sys['t_p'], color=clr, lw=1, ls='--')

    ax.set_xlabel('ω  [rad/s]')
    ax.set_ylabel('Direction bin')
    ax.set_title('Directional spectrum  (ω–dir, all segments averaged)')
    ax.legend(fontsize=8, loc='upper right')

    try:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
    except Exception:
        pass


# ── processor ─────────────────────────────────────────────────────────────────

class Processor:

    def __init__(self, config, pics=False):
        self.pics = pics
        self.cfg = config
        self.cst = config.const
        # Nyquist angular frequency: ω_max = π * RPM / 60
        self.om_max = np.pi / (60 / self.cfg.const.RPM)

        self.state = ProcessorState()
        self.state.init_arrays(
            n_shots=self.cst.N_SHOTS,
            num_area=self.cst.NUM_AREA,
            mean=self.cst.MEAN,
            asp=self.cst.ASP,
        )

        self.averager = Averager(
            mean=self.cst.MEAN,
            n_freq=self.cst.N_FREQ,
            n_dirs=self.cst.N_DIRS,
            n_shots=self.cst.N_SHOTS // 2,  # portrait spans positive-ω half only
            cut_num=self.cst.K_NUM,
        )

        # All NUM_AREA segments positioned at equal azimuth steps, non-rotated
        self.seg_azimuths = np.linspace(0, 360, self.cst.NUM_AREA, endpoint=False)
        self.msh = [
            Area(self.cst.ASP * 2, self.cst.ADP, np.deg2rad(ang), 0, self.cst.AAP).calc_mask()
            for ang in self.seg_azimuths
        ]

        # Geographic direction for each direction bin [deg]
        self.dir_array = np.linspace(0, 360, self.cst.N_DIRS, endpoint=False)

    def stop(self):
        return 0

    def update(self, back, navi):
        s = self.state
        cst = self.cst

        s.curr_step = back.step
        s.curr_pulse = back.pulse
        s.speed[s.index % cst.MEAN] = navi.sog
        s.heading[s.index % cst.MEAN] = navi.hdg
        s.cog[s.index % cst.MEAN] = navi.cog

        # Fill rolling buffer for all NUM_AREA segments
        bck = back.bck
        t = s.index % cst.N_SHOTS
        for i in range(cst.NUM_AREA):
            (x, y), (wx, wy) = self.msh[i]
            row0 = bck[y, x] * (1.0 - wx) + bck[y, x + 1] * wx
            row1 = bck[y + 1, x] * (1.0 - wx) + bck[y + 1, x + 1] * wx
            s.cbck[i, t] = row0 * (1.0 - wy) + row1 * wy

        s.indices = np.roll(s.indices, -1)

        result = None
        port_out = None

        if s.index >= cst.N_SHOTS and s.index % int(self.cfg.output["out_times"]) == 0:

            sig, wdir = calc_wspd(bck)
            ring_sig = float(np.std(bck[:, max(0, cst.ADP - cst.ASP): cst.ADP + cst.ASP]))

            k_max = np.pi / cst.ASP / s.curr_step * cst.K_NUM
            half = cst.N_SHOTS // 2
            omega_vals = np.linspace(0, self.om_max, half)
            k_vals = np.linspace(0, k_max, cst.K_NUM)

            spec_3d_corr = np.zeros((half, 2 * cst.K_NUM, 2 * cst.K_NUM), dtype=np.float32)

            collect_ports = self.pics not in (False, "false")
            raw_ports = [] if collect_ports else None
            raw_spec3ds = [] if collect_ports else None

            for i in range(cst.NUM_AREA):
                spec_3d_i = calc_spec3d(s.cbck[i], cst.K_NUM)
                if collect_ports:
                    port_i, _ = calc_port(spec_3d_i)
                    raw_ports.append(port_i)
                    raw_spec3ds.append(spec_3d_i)
                spec_3d_corr += spec_3d_i

            spec_3d_corr /= cst.NUM_AREA

            sog_mean = float(np.median(s.speed))
            cog_mean = float(np.median(s.cog))
            Ux, Uy = calc_current_vector(spec_3d_corr, k_max, self.om_max, band=_SIGNAL_BAND,
                                         sog=sog_mean, cog_deg=cog_mean)
            spec_3d_fixed = apply_doppler_3d_vec(spec_3d_corr, k_max, Ux, Uy, self.om_max)

            port_fixed, _ = calc_port(spec_3d_fixed)
            signal, noise = separate_signal_noise(port_fixed, k_vals, self.om_max, band=_SIGNAL_BAND)
            signal_mtf = apply_mtf(signal, k_vals, exp=1.2)

            snr_tot = compute_snr(signal_mtf, noise)
            s_omega, m0, T_peak, T_mean = compute_frequency_spectrum(signal_mtf, k_vals, omega_vals)

            s_om_th, peak_dir, mean_dir = calc_spec2d(spec_3d_fixed, omega_vals, k_max, cst.N_DIRS, band=_SIGNAL_BAND)

            swh = 0.01 * (cst.SNR_A + cst.SNR_B * np.sqrt(snr_tot))

            wave_sum = Wave(swh=swh, snr=m0,
                            t_p=T_peak, t_m=T_mean,
                            d_p=peak_dir, d_m=mean_dir)

            sys = calc_partitions(s_om_th, omega_vals, self.dir_array, wdir, swh)

            def _sys_wave(d):
                if d is None:
                    return Wave()
                return Wave(swh=d["h_s"], snr=0.0,
                            t_p=d["t_p"], t_m=d["t_m"],
                            d_p=d["d_p"], d_m=d["d_m"])

            if sys["w_s"] is None:
                wave_win = Wave()
            else:
                wave_win = Wave(swh=sys["w_s"]["h_s"], snr=0.0,
                                t_p=sys["w_s"]["t_p"], t_m=sys["w_s"]["t_m"],
                                d_p=sys["w_s"]["d_p"], d_m=sys["w_s"]["d_m"])
            wave_sw1 = _sys_wave(sys["sw_1"])
            wave_sw2 = _sys_wave(sys["sw_2"])

            quality = int(
                snr_tot >= _SNR_QUALITY_MIN
                and ring_sig >= _WIND_SIG_MIN
                and T_peak >= _T_PEAK_MIN
                and sys["n_sys"] >= 1
            )
            if not quality:
                reasons = []
                if snr_tot    < _SNR_QUALITY_MIN: reasons.append(f'snr={snr_tot:.2f}<{_SNR_QUALITY_MIN}')
                if ring_sig   < _WIND_SIG_MIN:    reasons.append(f'ring_sig={ring_sig:.2f}<{_WIND_SIG_MIN}')
                if T_peak     < _T_PEAK_MIN:      reasons.append(f'T_peak={T_peak:.2f}<{_T_PEAK_MIN}')
                if sys["n_sys"] < 1:              reasons.append('n_sys=0')
                msg = f'quality=BAD: {", ".join(reasons)}'
                _log.info(msg)
                print(msg)

            # (Ux, Uy) = apparent velocity in radar frame = v_current + v_ship.
            # True ocean current: subtract ship velocity.
            cog_rad = np.deg2rad(cog_mean)
            u_curr_x = float(Ux) - sog_mean * np.sin(cog_rad)   # East [m/s]
            u_curr_y = float(Uy) - sog_mean * np.cos(cog_rad)   # North [m/s]
            curr_speed = float(np.hypot(u_curr_x, u_curr_y))
            if curr_speed > _MAX_CURRENT:
                _f = _MAX_CURRENT / curr_speed
                u_curr_x *= _f; u_curr_y *= _f
                curr_speed = _MAX_CURRENT
            # Compass bearing: arctan2(East, North) = bearing from North, clockwise
            curr_dir = float(np.degrees(np.arctan2(u_curr_x, u_curr_y)) % 360)

            wspd = 0.01 * float(cst.WSPD_A + cst.WSPD_B * ring_sig)

            # Interpolate spectra onto output frequency grid
            spec_1d = np.interp(np.linspace(0, self.om_max, cst.N_FREQ), omega_vals, s_omega)

            f_interp = interp1d(omega_vals, s_om_th, axis=1, kind='linear',
                                fill_value=0.0, bounds_error=False)
            spec_2d = f_interp(np.linspace(0, self.om_max, cst.N_FREQ))

            wave_out = WaveOutput(
                ide_sys=sys["n_sys"],
                wave_sum=wave_sum, wave_win=wave_win,
                wave_sw1=wave_sw1, wave_sw2=wave_sw2,
                spec_1d=spec_1d, spec_2d=spec_2d,
            )

            if self.pics not in (False, "false"):
                _pics = self.pics if isinstance(self.pics, str) else "./"
                _save_debug_portrait(port_fixed, s.curr_step, cst.ASP, cst.K_NUM,
                                     self.om_max, sys, Ux, Uy, _pics)
                _save_debug_spec2d(s_om_th, omega_vals, wave_sum, sys,
                                   cst.N_DIRS, _pics)
                _save_debug_segments(s.cbck, raw_ports, raw_spec3ds,
                                     self.seg_azimuths, k_max, self.om_max, _pics)

            self.averager.push(wave_out, spec_1d, spec_2d, port_fixed)

            result, spec_1d, spec_2d, port_out = self.averager.get_mean(
                pulse=s.curr_pulse, step=s.curr_step, rpm=cst.RPM,
                n_shots=cst.N_SHOTS, asp=cst.ASP, adp=cst.ADP,
            )

            if result is not None:
                result.sog_proc = navi.sog
                result.cog_proc = navi.cog
                result.n_start = round(navi.hdg)
                result.curr_speed = curr_speed  # vco slot (un[26])
                result.curr_dir  = curr_dir    # repurposed rps slot (un[3])
                result.wind_dir  = wdir        # repurposed n_area slot (un[7])
                result.wspd      = wspd        # repurposed step_area slot (un[6])
                result.n_dis = quality    # quality flag (un[25])

        s.index += 1
        return {
            "out": result,
            "pulse": s.curr_pulse,
            "step": s.curr_step,
            "navi": navi,
            "port": port_out,
        }
