import numpy as np
from scipy.interpolate import interp1d
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import logging as _logging
_log = _logging.getLogger(__name__)

from src.processing.state import ProcessorState
from src.processing.averaging import Averager
from src.algorithms.area import Area
from src.algorithms.dispersion import calc_current_vector, calc_current_multiwave
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
_SNR_QUALITY_MIN = 5.0
_WIND_SIG_MIN    = 5.5
_T_PEAK_MIN      = 5.5

from src.algorithms.partition import calc_wspd, calc_partitions, find_freq_peaks, find_system_dirs
from src.io.structs import Wave, WaveOutput

# Debug colour palette — index matches system position in systems_draft
_SYS_COLORS = ['cyan', 'lime', 'orange']


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
    k_num    = raw_ports[0].shape[1]
    k_arr    = np.linspace(0, k_max, k_num)
    disp_om  = np.sqrt(9.81 * k_arr)

    port_vmax = max(float(p.max()) for p in raw_ports)
    port_vmax = max(port_vmax, 1e-9)

    from matplotlib import cm as mpl_cm
    from matplotlib.colors import Normalize
    norm_port = Normalize(vmin=0, vmax=port_vmax)
    cmap_port = mpl_cm.gnuplot2

    fig = Figure(figsize=(18, num_area * 1.8))
    FigureCanvasAgg(fig)

    for i in range(num_area):
        az = float(azimuths[i])

        ax = fig.add_subplot(num_area, 3, 3 * i + 1)
        bck_mean = cbck[i, -1]
        ax.imshow(bck_mean, cmap='gray', aspect='equal', origin='lower',
                  vmin=100, vmax=150)
        ax.set_title(f'{az:.0f}°', fontsize=15, pad=3, fontweight='bold')
        ax.axis('off')

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

        ax = fig.add_subplot(num_area, 3, 3 * i + 3)
        spec3d   = raw_spec3ds[i]
        om_min_i = max(1, round(0.1 * spec3d.shape[0]))
        kxy      = spec3d[om_min_i:].sum(axis=0)
        ax.imshow(kxy, aspect='equal', origin='lower', cmap='gnuplot2',
                  extent=[-k_max, k_max, -k_max, k_max],
                  vmin=0, vmax=max(float(kxy.max()), 1e-9))
        ax.axhline(0, color='w', lw=0.4, alpha=0.5)
        ax.axvline(0, color='w', lw=0.4, alpha=0.5)
        ax.set_xlim(-k_max / 2, k_max / 2)
        ax.set_ylim(-k_max / 2, k_max / 2)
        ax.axis('off')

    fig.subplots_adjust(wspace=0.03, hspace=0.05,
                        left=0.01, right=0.97, top=0.99, bottom=0.01)
    try:
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
    except Exception:
        pass


def _save_debug_portrait(port_fixed, step, asp, k_num, om_max,
                         sys_dict, Ux, Uy,
                         systems_draft, sys_scatter, path,
                         sog=0.0, cog_deg=0.0):
    """
    Doppler-corrected ω-k portrait.

    Background: imshow of port_fixed.
    White dashed: undisturbed ω = √(gk).
    Coloured curves: per-final-system residual dispersion.
    Coloured scatter: centroid points per systems_draft member — colour = system,
                      size ∝ log intensity.  Only size encodes intensity; colour
                      encodes system identity.
    """
    import os
    out_path = (path if path.endswith('.png')
                else os.path.join(path, 'debug_portrait.png'))

    k_max = np.pi / asp / step * k_num
    k_arr = np.linspace(0, k_max, k_num)

    fig = Figure(figsize=(9, 5))
    FigureCanvasAgg(fig)
    ax  = fig.add_subplot(111)

    vmax = max(float(port_fixed.max()), 1e-9)
    ax.imshow(port_fixed, aspect='auto', origin='lower', cmap='gnuplot2',
              extent=[0, k_max, 0, om_max], vmin=0, vmax=vmax)
    ax.plot(k_arr, np.sqrt(9.81 * k_arr), 'w--', lw=1.2,
            label='ω = √(gk)', alpha=0.6)

    # ── per-final-system residual curves ─────────────────────────────────────
    # True ocean current (for residual curve labels)
    cog_rad = np.deg2rad(cog_deg)
    u_curr_x = float(Ux) - sog * np.sin(cog_rad)
    u_curr_y = float(Uy) - sog * np.cos(cog_rad)

    sys_styles = [
        ('w_s',  'cyan',   'wind sea'),
        ('sw_1', 'lime',   'swell 1'),
        ('sw_2', 'orange', 'swell 2'),
    ]
    for key, clr, lbl in sys_styles:
        s = sys_dict.get(key)
        if s is None or s['t_p'] <= 0:
            continue
        d_p_rad = np.deg2rad(s['d_p'])
        # Residual = projection of true current onto wave direction.
        # After full Doppler correction energy should sit at ω=√(gk);
        # a non-zero residual indicates correction error or mis-identified direction.
        u_res = float(u_curr_x * np.cos(d_p_rad) + u_curr_y * np.sin(d_p_rad))
        om_curve = np.sqrt(9.81 * k_arr) + k_arr * u_res
        mask = (om_curve >= 0) & (om_curve <= om_max)
        ax.plot(k_arr[mask], om_curve[mask], color=clr, lw=1.5,
                label=f'{lbl}  u_curr_proj={u_res:+.2f} m/s')

    # ── scatter: centroid points per pre-analysis system ─────────────────────
    MAX_PTS = 200   # subsample cap per system
    rng = np.random.default_rng(0)

    for s_idx, (k_pts, om_pts, w_pts) in enumerate(sys_scatter):
        if len(k_pts) == 0:
            continue
        clr = _SYS_COLORS[s_idx % len(_SYS_COLORS)]

        # Subsample if needed
        n = len(k_pts)
        if n > MAX_PTS:
            idx = rng.choice(n, MAX_PTS, replace=False)
            k_pts  = k_pts[idx]
            om_pts = om_pts[idx]
            w_pts  = w_pts[idx]

        # Size: relative log-intensity, clipped to [10, 80] pt²
        if w_pts.max() > 0:
            sz = np.log1p(w_pts / w_pts.max() * 1e3)
            sz = 10.0 + 70.0 * sz / np.log1p(1e3)
        else:
            sz = np.full(len(k_pts), 20.0)

        label_s = (f'sys {s_idx}  T={2*np.pi/systems_draft[s_idx]["om"]:.1f}s'
                   f'  {systems_draft[s_idx]["dir_deg"]:.0f}°'
                   if s_idx < len(systems_draft) else f'sys {s_idx}')
        ax.scatter(k_pts, om_pts, c=clr, s=sz, alpha=0.55,
                   linewidths=0, label=label_s, zorder=5)

    ax.set_xlabel('k  [rad/m]')
    ax.set_ylabel('ω  [rad/s]')
    ax.set_title(f'Doppler-corrected ω–k portrait  '
                 f'(Ux={Ux:+.2f} Uy={Uy:+.2f} m/s)')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(0, k_max)
    ax.set_ylim(0, om_max)

    try:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
    except Exception:
        pass


def _save_debug_freq_spec(s_omega_pre, omega_vals, freq_peaks, systems_draft, path):
    """
    1-D frequency spectrum from the ship-corrected pre-analysis pass.
    Vertical coloured lines mark each identified frequency peak; shaded regions
    show the attributed band.  Period annotation next to each line.
    """
    import os
    out_path = (path if path.endswith('.png')
                else os.path.join(path, 'debug_freq_spec.png'))

    fig = Figure(figsize=(8, 3.5))
    FigureCanvasAgg(fig)
    ax  = fig.add_subplot(111)

    ax.plot(omega_vals, s_omega_pre, color='white', lw=1.2, zorder=3)
    ax.fill_between(omega_vals, s_omega_pre, alpha=0.25, color='steelblue')
    ax.set_facecolor('#111111')
    fig.patch.set_facecolor('#111111')

    s_max = float(s_omega_pre.max()) if s_omega_pre.max() > 0 else 1.0

    for s_idx, sys in enumerate(systems_draft):
        clr = _SYS_COLORS[s_idx % len(_SYS_COLORS)]
        om  = sys['om']
        T   = 2 * np.pi / om if om > 0 else 0.0
        # Shaded frequency band
        ax.axvspan(sys['om_lo'], sys['om_hi'], alpha=0.15, color=clr, zorder=1)
        # Peak line
        ax.axvline(om, color=clr, lw=1.5, zorder=4,
                   label=f'sys {s_idx}: T={T:.1f}s  {sys["dir_deg"]:.0f}°')
        # Period annotation
        ax.text(om, s_max * 0.92, f'T={T:.1f}s', color=clr,
                fontsize=8, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.15', fc='#222222', alpha=0.7))

    ax.set_xlabel('ω  [rad/s]', color='white')
    ax.set_ylabel('S(ω)  [a.u.]', color='white')
    ax.set_title('Frequency spectrum (ship-corrected, pre-analysis)',
                 color='white', fontsize=9)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')
    if systems_draft:
        ax.legend(fontsize=7, loc='upper right',
                  facecolor='#222222', labelcolor='white')
    ax.set_xlim(0, omega_vals[-1])
    ax.set_ylim(0)

    try:
        fig.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    except Exception:
        pass


def _save_debug_spec2d(s_om_th, omega_vals, wave_sum, sys_dict,
                       num_area, freq_peaks, systems_draft, path):
    """
    Directional-frequency spectrum s_om_th (dir bin × ω).

    Background: heatmap.
    Red lines / dashes: summary wave_sum.
    Coloured horizontal lines: per-final-system d_p (solid) and ω_p (dashed).
    Coloured crosses: pre-analysis systems_draft location (dir × ω).
    """
    import os
    out_path = (path if path.endswith('.png')
                else os.path.join(path, 'debug_spec2d.png'))

    fig = Figure(figsize=(9, 4))
    FigureCanvasAgg(fig)
    ax  = fig.add_subplot(111)

    vmax = max(float(s_om_th.max()), 1e-9)
    ax.imshow(s_om_th, aspect='auto', origin='lower', cmap='gnuplot2',
              extent=[0, omega_vals[-1], 0, num_area], vmin=0, vmax=vmax)

    # Summary wave
    dir_bin = wave_sum.d_p / 360 * num_area
    ax.axhline(dir_bin, color='red', lw=2, label=f'Sum {wave_sum.d_p:.0f}°')
    if wave_sum.t_p > 0:
        ax.axvline(2 * np.pi / wave_sum.t_p, color='red', lw=1, ls='--')

    # Final classified systems (solid = direction, dashed = period)
    final_styles = {'w_s': 'cyan', 'sw_1': 'lime', 'sw_2': 'orange'}
    for key, clr in final_styles.items():
        s = sys_dict.get(key)
        if s is None:
            continue
        ax.axhline(s['d_p'] / 360 * num_area, color=clr, lw=1.5,
                   label=f"{key} {s['d_p']:.0f}°")
        if s['t_p'] > 0:
            ax.axvline(2 * np.pi / s['t_p'], color=clr, lw=1, ls='--')

    # Pre-analysis systems_draft: cross marker at (ω_peak, dir_bin)
    for s_idx, sys in enumerate(systems_draft):
        clr = _SYS_COLORS[s_idx % len(_SYS_COLORS)]
        om  = sys['om']
        db  = sys['dir_idx']
        ax.plot(om, db, marker='+', color=clr, ms=14, mew=2, zorder=6,
                label=f"draft {s_idx}: {sys['dir_deg']:.0f}° T={2*np.pi/om:.1f}s")
        # Frequency band: vertical shaded region
        ax.axvspan(sys['om_lo'], sys['om_hi'], alpha=0.08, color=clr)

    ax.set_xlabel('ω  [rad/s]')
    ax.set_ylabel('Direction bin  (0=East, 90°/bin = North)')
    ax.set_title('Directional spectrum  (dir × ω, averaged over segments)')
    ax.legend(fontsize=7, loc='upper right')

    try:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
    except Exception:
        pass


# ── processor ─────────────────────────────────────────────────────────────────

class Processor:

    def __init__(self, config, pics=False):
        self.pics = pics
        self.cfg  = config
        self.cst  = config.const
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
            n_shots=self.cst.N_SHOTS // 2,
            cut_num=self.cst.K_NUM,
        )

        self.seg_azimuths = np.linspace(0, 360, self.cst.NUM_AREA, endpoint=False)
        self.msh = [
            Area(self.cst.ASP * 2, self.cst.ADP, np.deg2rad(ang), 0, self.cst.AAP).calc_mask()
            for ang in self.seg_azimuths
        ]

        # Geographic direction per direction bin [deg], math convention (0=East)
        self.dir_array = np.linspace(0, 360, self.cst.N_DIRS, endpoint=False)

    def stop(self):
        return 0

    def update(self, back, navi):
        s   = self.state
        cst = self.cst

        s.curr_step  = back.step
        s.curr_pulse = back.pulse
        s.speed[s.index % cst.MEAN]   = navi.sog
        s.heading[s.index % cst.MEAN] = navi.hdg
        s.cog[s.index % cst.MEAN]     = navi.cog

        bck = back.bck
        t   = s.index % cst.N_SHOTS
        for i in range(cst.NUM_AREA):
            (x, y), (wx, wy) = self.msh[i]
            row0 = bck[y, x] * (1.0 - wx) + bck[y, x + 1] * wx
            row1 = bck[y + 1, x] * (1.0 - wx) + bck[y + 1, x + 1] * wx
            s.cbck[i, t] = row0 * (1.0 - wy) + row1 * wy

        s.indices = np.roll(s.indices, -1)

        result   = None
        port_out = None

        if s.index >= cst.N_SHOTS and s.index % int(self.cfg.output["out_times"]) == 0:

            sig, wdir = calc_wspd(bck)
            ring_sig  = float(np.std(
                bck[:, max(0, cst.ADP - cst.ASP): cst.ADP + cst.ASP]))

            k_max     = np.pi / cst.ASP / s.curr_step * cst.K_NUM
            half      = cst.N_SHOTS // 2
            omega_vals = np.linspace(0, self.om_max, half)
            k_vals    = np.linspace(0, k_max, cst.K_NUM)

            spec_3d_corr = np.zeros((half, 2 * cst.K_NUM, 2 * cst.K_NUM),
                                    dtype=np.float32)

            collect_ports = self.pics not in (False, "false")
            raw_ports   = [] if collect_ports else None
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
            cog_rad  = np.deg2rad(cog_mean)

            # ── Phase 1: ship-speed correction for pre-analysis ───────────────
            Ux_ship = sog_mean * np.sin(cog_rad)
            Uy_ship = sog_mean * np.cos(cog_rad)

            spec_3d_ship   = apply_doppler_3d_vec(
                spec_3d_corr, k_max, Ux_ship, Uy_ship, self.om_max)
            port_pre, _    = calc_port(spec_3d_ship)
            signal_pre, _  = separate_signal_noise(
                port_pre, k_vals, self.om_max, band=_SIGNAL_BAND)
            sig_mtf_pre    = apply_mtf(signal_pre, k_vals, exp=1.2)
            s_omega_pre, _, _, _ = compute_frequency_spectrum(
                sig_mtf_pre, k_vals, omega_vals)

            # ── Phase 2: system identification from frequency + direction ─────
            freq_peaks = find_freq_peaks(s_omega_pre, omega_vals)

            s_om_th_pre, _, _ = calc_spec2d(
                spec_3d_ship, omega_vals, k_max, cst.N_DIRS, band=_SIGNAL_BAND)
            systems_draft = find_system_dirs(
                s_om_th_pre, freq_peaks, omega_vals, self.dir_array)

            # ── Phase 3: current estimation ───────────────────────────────────
            Ux, Uy       = None, None
            sys_scatter  = []

            if len(systems_draft) >= 2:
                Ucx, Ucy, sys_scatter = calc_current_multiwave(
                    spec_3d_ship, k_max, self.om_max, systems_draft, _SIGNAL_BAND)
                if Ucx is not None:
                    Ux = Ucx + Ux_ship
                    Uy = Ucy + Uy_ship
                    _log.info(
                        f'multi-wave current: Ux={Ux:+.2f} Uy={Uy:+.2f} m/s '
                        f'({len(systems_draft)} systems: '
                        + ', '.join(f'T={2*np.pi/d["om"]:.1f}s {d["dir_deg"]:.0f}°'
                                    for d in systems_draft) + ')')

            if Ux is None:
                Ux, Uy = calc_current_vector(
                    spec_3d_corr, k_max, self.om_max, band=_SIGNAL_BAND,
                    sog=sog_mean, cog_deg=cog_mean)
                # Build empty scatter so debug functions have consistent input
                sys_scatter = [(np.array([]), np.array([]), np.array([]))
                               for _ in systems_draft]

            # ── Phase 4: final Doppler correction & spectral quantities ───────
            spec_3d_fixed = apply_doppler_3d_vec(
                spec_3d_corr, k_max, Ux, Uy, self.om_max)
            port_fixed, _ = calc_port(spec_3d_fixed)
            signal, noise = separate_signal_noise(
                port_fixed, k_vals, self.om_max, band=_SIGNAL_BAND)
            signal_mtf    = apply_mtf(signal, k_vals, exp=1.2)

            snr_tot = compute_snr(signal_mtf, noise)
            s_omega, m0, T_peak, T_mean = compute_frequency_spectrum(
                signal_mtf, k_vals, omega_vals)

            s_om_th, peak_dir, mean_dir = calc_spec2d(
                spec_3d_fixed, omega_vals, k_max, cst.N_DIRS, band=_SIGNAL_BAND)

            swh = 0.01 * (cst.SNR_A + cst.SNR_B * np.sqrt(snr_tot))

            wave_sum = Wave(swh=swh, snr=m0,
                            t_p=T_peak, t_m=T_mean,
                            d_p=peak_dir, d_m=mean_dir)

            # ── Phase 5: partitioning ─────────────────────────────────────────
            sys = calc_partitions(s_om_th, omega_vals, self.dir_array, wdir, swh)

            def _sys_wave(d):
                if d is None:
                    return Wave()
                return Wave(swh=d["h_s"], snr=0.0,
                            t_p=d["t_p"], t_m=d["t_m"],
                            d_p=d["d_p"], d_m=d["d_m"])

            wave_win = _sys_wave(sys["w_s"])
            wave_sw1 = _sys_wave(sys["sw_1"])
            wave_sw2 = _sys_wave(sys["sw_2"])

            # ── Quality flag ──────────────────────────────────────────────────
            quality = int(
                snr_tot  >= _SNR_QUALITY_MIN
                and ring_sig >= _WIND_SIG_MIN
                and T_peak   >= _T_PEAK_MIN
                and sys["n_sys"] >= 1
            )
            if not quality:
                reasons = []
                if snr_tot  < _SNR_QUALITY_MIN: reasons.append(f'snr={snr_tot:.2f}<{_SNR_QUALITY_MIN}')
                if ring_sig < _WIND_SIG_MIN:    reasons.append(f'ring_sig={ring_sig:.2f}<{_WIND_SIG_MIN}')
                if T_peak   < _T_PEAK_MIN:      reasons.append(f'T_peak={T_peak:.2f}<{_T_PEAK_MIN}')
                if sys["n_sys"] < 1:            reasons.append('n_sys=0')
                msg = f'quality=BAD: {", ".join(reasons)}'
                _log.info(msg)
                print(msg)

            # ── True ocean current (subtract ship velocity) ───────────────────
            u_curr_x = float(Ux) - sog_mean * np.sin(cog_rad)  # East [m/s]
            u_curr_y = float(Uy) - sog_mean * np.cos(cog_rad)  # North [m/s]
            curr_speed = float(np.hypot(u_curr_x, u_curr_y))
            if curr_speed > _MAX_CURRENT:
                _f = _MAX_CURRENT / curr_speed
                u_curr_x *= _f;  u_curr_y *= _f
                curr_speed = _MAX_CURRENT
            curr_dir = float(np.degrees(np.arctan2(u_curr_x, u_curr_y)) % 360)

            wspd = 0.01 * float(cst.WSPD_A + cst.WSPD_B * ring_sig)

            spec_1d = np.interp(
                np.linspace(0, self.om_max, cst.N_FREQ), omega_vals, s_omega)
            f_interp = interp1d(omega_vals, s_om_th, axis=1, kind='linear',
                                fill_value=0.0, bounds_error=False)
            spec_2d = f_interp(np.linspace(0, self.om_max, cst.N_FREQ))

            wave_out = WaveOutput(
                ide_sys=sys["n_sys"],
                wave_sum=wave_sum, wave_win=wave_win,
                wave_sw1=wave_sw1, wave_sw2=wave_sw2,
                spec_1d=spec_1d, spec_2d=spec_2d,
            )

            # ── Debug plots ───────────────────────────────────────────────────
            if self.pics not in (False, "false"):
                _pics = self.pics if isinstance(self.pics, str) else "./"
                _save_debug_portrait(
                    port_fixed, s.curr_step, cst.ASP, cst.K_NUM,
                    self.om_max, sys, Ux, Uy,
                    systems_draft, sys_scatter, _pics,
                    sog=sog_mean, cog_deg=cog_mean)
                _save_debug_freq_spec(
                    s_omega_pre, omega_vals, freq_peaks, systems_draft, _pics)
                _save_debug_spec2d(
                    s_om_th, omega_vals, wave_sum, sys,
                    cst.N_DIRS, freq_peaks, systems_draft, _pics)
                _save_debug_segments(
                    s.cbck, raw_ports, raw_spec3ds,
                    self.seg_azimuths, k_max, self.om_max, _pics)

            self.averager.push(wave_out, spec_1d, spec_2d, port_fixed)

            result, spec_1d, spec_2d, port_out = self.averager.get_mean(
                pulse=s.curr_pulse, step=s.curr_step, rpm=cst.RPM,
                n_shots=cst.N_SHOTS, asp=cst.ASP, adp=cst.ADP,
            )

            if result is not None:
                result.sog_proc  = navi.sog
                result.cog_proc  = navi.cog
                result.n_start   = round(navi.hdg)
                result.curr_speed = curr_speed
                result.curr_dir   = curr_dir
                result.wind_dir   = wdir
                result.wspd       = wspd
                result.n_dis      = quality

        s.index += 1
        return {
            "out":   result,
            "pulse": s.curr_pulse,
            "step":  s.curr_step,
            "navi":  navi,
            "port":  port_out,
        }
