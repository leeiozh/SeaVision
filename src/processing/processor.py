import numpy as np
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

_SIGNAL_BAND     = 10     # dispersion band half-width [k-bins]; recalibrate SNR_A/B if changed
_MAX_CURRENT     = 3.0    # physical clip for apparent current [m/s]
_SNR_QUALITY_MIN = 5.0    # min snr_tot for quality=GOOD
_T_PEAK_MIN      = 5.5    # min peak period [s] for quality=GOOD

_FALLBACK_RPM    = 25.0   # default antenna rate until a live estimate is available
_RPM_MIN_DT      = 0.3    # plausible rotation period band [s] (≈200 RPM upper)
_RPM_MAX_DT      = 12.0   #                                    (≈5 RPM lower)

from src.algorithms.partition import calc_wspd, calc_partitions, find_freq_peaks, find_system_dirs
from src.io.structs import Wave, WaveOutput

# Debug colour palette — index matches system position in systems_draft
_SYS_COLORS = ['cyan', 'lime', 'orange']

_F_DISPLAY = 0.20   # Hz — radial limit on polar spectrum display
_PULSE_STR = {1: 'SP', 2: 'MP', 3: 'LP'}


# ── single combined debug figure (replaces 4 separate PNGs) ──────────────────

def _save_debug_pic(
    s_omega_pre, omega_vals, systems_draft,
    s_om_th, sys_dict, n_dirs,
    port_corr, k_vals, k_max, Ux, Uy, sys_scatter, sog=0.0, cog_deg=0.0,
    swh=0.0, T_peak=0.0, T_mean=0.0, peak_dir=0.0,
    quality=0, snr_tot=0.0, wdir=0.0, wspd=0.0, wind_sig=0.0,
    pulse=0, n_sys=0,
    raw_bck=None, rpm=0.0,
    path='.',
):
    """Single diagnostic figure combining 1D spectrum, polar directional spectrum,
    ω-k portrait and parameter table.  Mirrors batch_process._save_pic."""
    import os
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.gridspec import GridSpec

    out_path = path if path.endswith('.png') else os.path.join(path, 'debug_combined.png')

    cog_rad     = np.deg2rad(cog_deg)
    Ux_ship     = sog * np.sin(cog_rad)
    Uy_ship     = sog * np.cos(cog_rad)
    peak_rad    = np.deg2rad(peak_dir)
    u_proj      = float(Ux * np.cos(peak_rad) + Uy * np.sin(peak_rad))
    u_ship_proj = float(Ux_ship * np.cos(peak_rad) + Uy_ship * np.sin(peak_rad))
    u_curr_x    = float(Ux) - Ux_ship
    u_curr_y    = float(Uy) - Uy_ship
    curr_speed  = float(np.hypot(u_curr_x, u_curr_y))
    curr_dir    = float(np.degrees(np.arctan2(u_curr_x, u_curr_y)) % 360)

    fig = Figure(figsize=(21, 9))
    FigureCanvasAgg(fig)
    gs = GridSpec(2, 4, figure=fig, height_ratios=[3, 1.2],
                  hspace=0.22, wspace=0.22,
                  left=0.05, right=0.98, top=0.95, bottom=0.04)

    f_axis = omega_vals / (2 * np.pi)
    f_max  = float(omega_vals[-1]) / (2 * np.pi)

    # ── [0,0] 1D frequency spectrum (ship-corrected, pre-analysis) ───────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor('#111111'); fig.patch.set_facecolor('#111111')
    ax0.plot(f_axis, s_omega_pre, color='white', lw=1.2)
    ax0.fill_between(f_axis, s_omega_pre, alpha=0.22, color='steelblue')
    s_max = float(s_omega_pre.max()) if s_omega_pre.max() > 0 else 1.0
    if T_peak > 0:
        ax0.axvline(1.0 / T_peak, color='white', ls='-',  lw=1.3)
    if T_mean > 0:
        ax0.axvline(1.0 / T_mean, color='white', ls='--', lw=1.0, alpha=0.7)
    for s_idx, sys in enumerate(systems_draft or []):
        clr = _SYS_COLORS[s_idx % len(_SYS_COLORS)]
        f_lo = sys['om_lo'] / (2 * np.pi)
        f_hi = sys['om_hi'] / (2 * np.pi)
        f_pk = sys['om']    / (2 * np.pi)
        T_s  = 1.0 / f_pk if f_pk > 0 else 0.0
        ax0.axvspan(f_lo, f_hi, alpha=0.14, color=clr)
        ax0.axvline(f_pk, color=clr, lw=1.5, zorder=4,
                    label=f'sys{s_idx} T={T_s:.1f}s {sys["dir_deg"]:.0f}°')
        ax0.text(f_pk, s_max * 0.93, f'T={T_s:.1f}s', color=clr,
                 fontsize=7, ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.1', fc='#222222', alpha=0.7))
    ax0.set_xlim(0, f_max); ax0.set_ylim(0)
    ax0.set_xlabel('f [Hz]', color='white'); ax0.set_ylabel('S(f)  [a.u.]', color='white')
    ax0.set_title('Spectrum (ship-corrected)', color='white', fontsize=9)
    ax0.tick_params(colors='white')
    for sp in ax0.spines.values(): sp.set_edgecolor('#555555')
    if systems_draft:
        ax0.legend(fontsize=7, loc='upper right', facecolor='#222222', labelcolor='white')

    # ── [0,1] polar directional spectrum (final, after full correction) ──────
    ax1 = fig.add_subplot(gs[0, 1], projection='polar')
    ax1.set_theta_zero_location('N'); ax1.set_theta_direction(-1)
    n_om_s   = s_om_th.shape[1]
    f_vals_s = np.linspace(0, f_max, n_om_s)
    f_mask   = f_vals_s <= _F_DISPLAY
    f_disp   = f_vals_s[f_mask]
    s_disp   = s_om_th[:, f_mask]                           # (n_dirs, n_f_disp)
    n_f_disp = s_disp.shape[1]
    # theta edges: n_dirs+1 to close ring; r edges: n_f_disp+1
    theta_e  = np.linspace(0, 2 * np.pi, n_dirs + 1)       # (37,)
    r_e      = np.linspace(0, _F_DISPLAY, n_f_disp + 1)    # (n_f_disp+1,)
    T2D, R2D = np.meshgrid(theta_e, r_e)                   # (n_f_disp+1, 37) each
    vmax     = float(s_disp.max()) or 1.0
    # C must be (n_f_disp, n_dirs) for shading='flat'
    ax1.pcolormesh(T2D, R2D, s_disp.T, cmap='gnuplot2', vmin=0, vmax=vmax, shading='flat')
    ax1.set_ylim(0, _F_DISPLAY); ax1.set_rlabel_position(45)
    ax1.tick_params(labelsize=7)
    # Final system direction lines (only for detected systems with Tp > 0)
    sys_styles = [('w_s', 'cyan', 'W sea'), ('sw_1', 'lime', 'Swell-1'), ('sw_2', 'orange', 'Swell-2')]
    for key, clr, lbl in sys_styles:
        s = sys_dict.get(key)
        if s and s.get('t_p', 0) > 0:
            ax1.plot([np.deg2rad(s['d_p'])] * 2, [0, _F_DISPLAY * 0.88],
                     color=clr, lw=2.2, alpha=0.9, label=lbl)
    ax1.plot([np.deg2rad(peak_dir)] * 2, [0, _F_DISPLAY * 0.82],
             color='red', lw=2.0, alpha=0.85, label='Sum')
    ax1.plot([np.deg2rad(cog_deg)] * 2, [0, _F_DISPLAY * 0.45],
             color='deepskyblue', lw=1.5, ls='--', alpha=0.75, label='COG')
    # NOTE: systems_draft '+' markers are intentionally NOT shown on the polar plot
    # (they are already shown in the 1D spectrum panel) to avoid visual confusion.
    ax1.legend(fontsize=6, loc='lower right', bbox_to_anchor=(1.35, -0.05))
    ax1.set_title('Dir. spectrum (final)', color='white', fontsize=9, pad=12)
    ax1.set_facecolor('#111111')

    # ── [0,2] ω-k portrait (pre-correction) ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#111111')
    if port_corr is not None:
        k_arr   = k_vals
        om_disp = np.sqrt(9.81 * k_arr)
        vmax_p  = max(float(port_corr.max()), 1e-9)
        ax2.imshow(port_corr, aspect='auto', origin='lower', cmap='gnuplot2',
                   extent=[0, k_max, 0, float(omega_vals[-1])],
                   vmin=0, vmax=vmax_p, interpolation='none')
        # Undisturbed dispersion
        ax2.plot(k_arr, om_disp, 'w--', lw=1.0, label='ω=√(gk)')
        # Full apparent velocity projection
        om_full = om_disp + k_arr * u_proj
        m_full  = (om_full >= 0) & (om_full <= float(omega_vals[-1]))
        ax2.plot(k_arr[m_full], om_full[m_full], color='red', lw=1.5,
                 label=f'full {u_proj:+.2f}')
        # Ship-only projection
        om_ship = om_disp + k_arr * u_ship_proj
        m_ship  = (om_ship >= 0) & (om_ship <= float(omega_vals[-1]))
        ax2.plot(k_arr[m_ship], om_ship[m_ship], color='red', lw=1.5, ls='--',
                 label=f'ship {u_ship_proj:+.2f}')
        # Scatter from multiwave regression
        rng = np.random.default_rng(0)
        for s_idx, (k_pts, om_pts, w_pts) in enumerate(sys_scatter or []):
            if len(k_pts) == 0:
                continue
            clr = _SYS_COLORS[s_idx % len(_SYS_COLORS)]
            n = len(k_pts)
            if n > 150:
                idx = rng.choice(n, 150, replace=False)
                k_pts, om_pts, w_pts = k_pts[idx], om_pts[idx], w_pts[idx]
            sz = (10 + 60 * np.log1p(w_pts / (w_pts.max() + 1e-30) * 1e3)
                  / np.log1p(1e3)) if w_pts.max() > 0 else np.full(len(k_pts), 15.0)
            ax2.scatter(k_pts, om_pts, c=clr, s=sz, alpha=0.6, linewidths=0, zorder=4)
        ax2.set_xlim(0, k_max); ax2.set_ylim(0, float(omega_vals[-1]))
        ax2.legend(fontsize=6, loc='upper right')
        ax2.text(0.02, 0.97,
                 f'Ux={Ux:+.2f} Uy={Uy:+.2f} m/s\ncurr={curr_speed:.2f}→{curr_dir:.0f}°\n'
                 f'SOG={sog:.2f} COG={cog_deg:.0f}°\n'
                 f'RPM={rpm:.2f}  fmax={float(omega_vals[-1])/(2*np.pi):.3f}Hz',
                 transform=ax2.transAxes, fontsize=6, va='top', color='white',
                 bbox=dict(facecolor='black', alpha=0.55, pad=2))
    ax2.set_xlabel('k [rad/m]', color='white'); ax2.set_ylabel('ω [rad/s]', color='white')
    ax2.set_title('ω-k portrait (pre-correction)', color='white', fontsize=9)
    ax2.tick_params(colors='white')
    for sp in ax2.spines.values(): sp.set_edgecolor('#555555')

    # ── [0,3] raw backscatter (last frame, unprocessed) ──────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_facecolor('#111111')
    if raw_bck is not None:
        raw = np.asarray(raw_bck)
        # decimate for display — full frame is (AAP, ARDP) ≈ 4096×2048
        step_y = max(1, raw.shape[0] // 1024)
        step_x = max(1, raw.shape[1] // 1024)
        raw_disp = raw[::step_y, ::step_x]
        vmax_r = float(np.percentile(raw_disp, 99)) or 1.0
        ax4.imshow(raw_disp, aspect='auto', origin='lower', cmap='gray',
                   vmin=0, vmax=vmax_r, interpolation='none')
        ax4.set_xlabel('range [px]', color='white')
        ax4.set_ylabel('azimuth [px]', color='white')
    else:
        ax4.text(0.5, 0.5, 'no raw frame', color='#888888',
                 ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Raw backscatter (last frame)', color='white', fontsize=9)
    ax4.tick_params(colors='white')
    for sp in ax4.spines.values(): sp.set_edgecolor('#555555')

    # ── [1,:] parameter table ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off'); ax3.set_facecolor('#111111')
    ws  = sys_dict.get('w_s')
    sw1 = sys_dict.get('sw_1')
    sw2 = sys_dict.get('sw_2')
    def _fv(d, k, fmt):
        v = d.get(k) if d else None
        return (fmt % v) if v else '—'
    rows = [
        ['Quality',     'GOOD' if quality else 'BAD',
         'Pulse',       _PULSE_STR.get(pulse, str(pulse)),
         'Hs [m]',      f'{swh:.3f}',
         'Tp [s]',      f'{T_peak:.2f}',
         'Tm [s]',      f'{T_mean:.2f}',
         'Dp [°]',      f'{peak_dir:.1f}'],
        ['N sys',       str(n_sys),
         'Wind from[°]', f'{wdir:.1f}',
         'Ring std',    f'{wind_sig:.2f}',
         'SNR tot',     f'{snr_tot:.2f}',
         'WSPD [m/s]',  f'{wspd:.2f}',
         'Dm [°]',      '—'],
        ['Wind Hs',     _fv(ws,  'h_s', '%.3f') if ws else '—',
         'Wind Tp',     _fv(ws,  't_p', '%.2f') if ws else '—',
         'Wind Dp',     _fv(ws,  'd_p', '%.1f') if ws else '—',
         'Wind Tm',     _fv(ws,  't_m', '%.2f') if ws else '—',
         'Ux [m/s]',    f'{Ux:.3f}',
         'Uy [m/s]',    f'{Uy:.3f}'],
        ['Sw1 Hs',      _fv(sw1, 'h_s', '%.3f') if sw1 else '—',
         'Sw1 Tp',      _fv(sw1, 't_p', '%.2f') if sw1 else '—',
         'Sw1 Dp',      _fv(sw1, 'd_p', '%.1f') if sw1 else '—',
         'Sw1 Tm',      _fv(sw1, 't_m', '%.2f') if sw1 else '—',
         'Curr [m/s]',  f'{curr_speed:.3f}',
         'Curr dir[°]', f'{curr_dir:.1f}'],
    ]
    col_labels = ['Par', 'Val'] * 6
    tbl = ax3.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.4)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#404040')
        tbl[0, j].get_text().set_color('white')
    tbl[1, 1].set_facecolor('#204020' if quality else '#402020')
    tbl[1, 1].get_text().set_color('white')

    try:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    except Exception:
        pass


# ── processor ─────────────────────────────────────────────────────────────────

class Processor:
    """Stateful wave parameter estimator for a single radar installation.

    Maintains a ring buffer of N_SHOTS frames (ProcessorState) and computes
    spectral estimates every out_times frames once the buffer is full.

    The algorithm runs in five phases on each output cycle:
      1. 3-D Welch FFT over NUM_AREA spatial segments → spec_3d_corr.
      2. Ship-speed Doppler correction → identify frequency peaks and wave
         system directions (systems_draft).
      3. Current estimation via calc_current_multiwave (≥2 systems) or
         calc_current_vector (fallback).
      4. Full Doppler correction → ω-k portrait, MTF, SNR, 1-D spectrum,
         directional spectrum, SWH.
      5. Partitioning into wind-sea / swell systems.

    Results are held in an Averager ring buffer (depth MEAN); get_mean()
    returns the averaged Output ready for UDP/CSV transmission.
    """

    def __init__(self, config, pics=False):
        """
        config — AppConfig from load_config().
        pics   — debug figure path (str) or False to disable.  When a path is
                 given, debug_combined.png is written after each output cycle.
                 matplotlib is imported lazily and only when pics is active.
        """
        self.pics = pics
        self.cfg  = config
        self.cst  = config.const

        # ── antenna rate ──────────────────────────────────────────────────────
        # Fixed from config, or estimated live from inter-frame timing when
        # RPM=false (cst.RPM is None). om_max = π·RPM/60 is the master quantity:
        # omega_vals and every frequency/period/dispersion calc derive from it.
        self._dyn_rpm = self.cst.RPM is None
        self.rpm = _FALLBACK_RPM if self.cst.RPM is None else float(self.cst.RPM)
        self.om_max = np.pi * self.rpm / 60.0
        # ring buffer of last N_SHOTS inter-frame intervals [s]
        self._dt_buf = np.zeros(self.cst.N_SHOTS, dtype=np.float64)
        self._dt_count = 0
        self._t_prev = 0.0

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
            n_freq_2d=self.cst.N_FREQ_2D,
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

    def _update_rpm(self, recv_time: float):
        """Estimate antenna RPM from inter-frame arrival times.

        One frame == one rotation, so the interval between consecutive frames is
        the rotation period.  We keep a ring buffer of the last N_SHOTS intervals
        and take the median (robust to dropped rotations / packet loss), updating
        self.rpm and self.om_max on every frame.

        No-op unless RPM=false (self._dyn_rpm) and a valid recv_time is present
        (live UDP only — file sources leave recv_time=0.0).
        """
        if not self._dyn_rpm or not recv_time:
            return
        if self._t_prev > 0.0:
            dt = recv_time - self._t_prev
            if _RPM_MIN_DT < dt < _RPM_MAX_DT:          # reject implausible gaps
                self._dt_buf[self._dt_count % self.cst.N_SHOTS] = dt
                self._dt_count += 1
        self._t_prev = recv_time

        n = min(self._dt_count, self.cst.N_SHOTS)
        if n > 0:
            med = float(np.median(self._dt_buf[:n]))
            if med > 0.0:
                self.rpm = 60.0 / med
                self.om_max = np.pi * self.rpm / 60.0

    def update(self, back: "BackData", navi: "Navi") -> dict:
        """Process one radar frame and return a result dict.

        Called once per antenna rotation (~2.4 s at RPM=25).  Accumulates
        frames into the ring buffer; triggers a full spectral computation every
        out_times frames once N_SHOTS frames have been buffered.

        Returns a dict with keys:
          "out"   — Output (averaged, ready for sinks) or None if not yet due.
          "pulse" — current pulse code.
          "step"  — current range resolution [m/px].
          "navi"  — the Navi passed in.
          "port"  — averaged ω-k portrait as int array [0–255], or None.
        """
        s   = self.state
        cst = self.cst

        # Refresh live RPM/om_max estimate before any frequency computation.
        self._update_rpm(back.recv_time)

        s.curr_step  = back.step
        s.curr_pulse = back.pulse
        s.speed[s.index % cst.N_SHOTS]   = navi.sog   # N_SHOTS ring buf — same window as cbck
        s.heading[s.index % cst.MEAN]    = navi.hdg   # MEAN buf — HDG output only
        s.cog[s.index % cst.N_SHOTS]     = navi.cog   # N_SHOTS ring buf — same window as cbck

        bck = back.bck
        t   = s.index % cst.N_SHOTS
        for i in range(cst.NUM_AREA):
            (x, y), (wx, wy) = self.msh[i]
            row0 = bck[y, x] * (1.0 - wx) + bck[y, x + 1] * wx
            row1 = bck[y + 1, x] * (1.0 - wx) + bck[y + 1, x + 1] * wx
            s.cbck[i, t] = row0 * (1.0 - wy) + row1 * wy

        result   = None
        port_out = None

        if s.index >= cst.N_SHOTS and s.index % int(self.cfg.output.get("out_times", 32)) == 0:

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

            # oldest frame sits at (t+1) % N_SHOTS in the ring buffer
            offset = (t + 1) % cst.N_SHOTS
            for i in range(cst.NUM_AREA):
                spec_3d_corr += calc_spec3d(s.cbck[i], cst.K_NUM, offset=offset)

            spec_3d_corr /= cst.NUM_AREA

            port_corr = calc_port(spec_3d_corr)[0] if collect_ports else None

            # Navi averaged over the same N_SHOTS window as cbck (same offset → same frames).
            # Eliminates HF spectral artifacts from SOG/COG mismatch at high ship speeds.
            _spd_win = np.roll(s.speed, -offset)
            _cog_win = np.roll(s.cog,   -offset)
            sog_mean = float(np.median(_spd_win))
            _cr = np.deg2rad(_cog_win)
            cog_mean = float(
                np.degrees(np.arctan2(np.mean(np.sin(_cr)), np.mean(np.cos(_cr)))) % 360
            )
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
                return Wave(swh=d["h_s"], t_p=d["t_p"], d_p=d["d_p"])

            wave_win = _sys_wave(sys["w_s"])
            wave_sw1 = _sys_wave(sys["sw_1"])
            wave_sw2 = _sys_wave(sys["sw_2"])

            # ── Quality flag ──────────────────────────────────────────────────
            quality = int(
                snr_tot  >= _SNR_QUALITY_MIN
                and ring_sig >= cst.WIND_SIG_MIN
                and T_peak   >= _T_PEAK_MIN
                and sys["n_sys"] >= 1
            )
            if not quality:
                reasons = []
                if snr_tot  < _SNR_QUALITY_MIN:    reasons.append(f'snr={snr_tot:.2f}<{_SNR_QUALITY_MIN}')
                if ring_sig < cst.WIND_SIG_MIN:    reasons.append(f'ring_sig={ring_sig:.2f}<{cst.WIND_SIG_MIN}')
                if T_peak   < _T_PEAK_MIN:         reasons.append(f'T_peak={T_peak:.2f}<{_T_PEAK_MIN}')
                if sys["n_sys"] < 1:            reasons.append('n_sys=0')
                _log.info(f'quality=BAD: {", ".join(reasons)}')

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
            _om_out = np.linspace(0, self.om_max, cst.N_FREQ_2D)
            spec_2d = np.vstack([
                np.interp(_om_out, omega_vals, row, left=0.0, right=0.0)
                for row in s_om_th
            ])

            wave_out = WaveOutput(
                ide_sys=sys["n_sys"],
                wave_sum=wave_sum, wave_win=wave_win,
                wave_sw1=wave_sw1, wave_sw2=wave_sw2,
                spec_1d=spec_1d, spec_2d=spec_2d,
                wspd=wspd, wind_dir=wdir,
                curr_speed=curr_speed, curr_dir=curr_dir,
            )

            # ── Debug plots ───────────────────────────────────────────────────
            if collect_ports:
                _pics = self.pics if isinstance(self.pics, str) else "./"
                _save_debug_pic(
                    s_omega_pre, omega_vals, systems_draft, s_om_th, sys, cst.N_DIRS,
                    port_corr, k_vals, k_max, Ux, Uy, sys_scatter, sog=sog_mean, cog_deg=cog_mean,
                    swh=swh, T_peak=T_peak, T_mean=T_mean, peak_dir=peak_dir,
                    quality=quality, snr_tot=snr_tot, wdir=wdir,
                    wspd=wspd, wind_sig=ring_sig, pulse=s.curr_pulse, n_sys=sys["n_sys"],
                    raw_bck=bck, rpm=self.rpm, path=_pics,
                )

            self.averager.push(wave_out, port_fixed)

            result, port_out = self.averager.get_mean(
                pulse=s.curr_pulse, step=s.curr_step, rpm=self.rpm,
                n_shots=cst.N_SHOTS, asp=cst.ASP, adp=cst.ADP,
            )

            if result is not None:
                result.sog_proc = navi.sog
                result.cog_proc = navi.cog
                result.n_start  = round(navi.hdg)
                result.n_dis    = quality   # quality: always from current computation

        s.index += 1
        return {
            "out":   result,
            "pulse": s.curr_pulse,
            "step":  s.curr_step,
            "navi":  navi,
            "port":  port_out,
        }
