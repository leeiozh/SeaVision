#!/usr/bin/env python3
"""
Batch processing of NetCDF radar files — one result per file from first N_SHOTS frames.
No averaging, no processing/step parameters.

Usage:
    python batch_process.py [--csv META_upd.csv] [--base-path /storage/thalassa/DATA/RADAR/]
                            [--out batch_out] [--config config.ini]

Input CSV: any CSV with a 'name' column of relative NC paths; --base-path is prepended.
Output:
    {out}/params.csv                  — DataFrame with computed columns
    {out}/spec/{name}_freqspec.npy    — shape (N_FREQ,),        int [0..255]
    {out}/spec/{name}_dirspec.npy     — shape (N_DIRS, N_FREQ), int [0..255]
    {out}/pics/{name}.png             — diagnostic figure
"""

import argparse
import gc
import os
import traceback
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.config import load_config
from src.io.input import NCInputSource
from src.algorithms.area import Area
from src.algorithms.spectrum2d import (
    calc_spec3d, calc_port,
    apply_doppler_3d_vec,
    separate_signal_noise, apply_mtf,
    compute_snr, compute_frequency_spectrum,
    calc_spec2d,
)
from src.algorithms.dispersion import calc_current_vector
from src.algorithms.partition import calc_wspd, calc_partitions
from src.runtime.logger import setup_logger

_SIGNAL_BAND = 10    # must match processor.py

# Quality thresholds — hardcoded, same as processor.py
_SNR_QUALITY_MIN = 1.5
_WIND_SIG_MIN    = 10.0
_T_PEAK_MIN      = 6.0

_F_DISPLAY   = 0.20   # Hz — radial limit on polar spectrum display
_BUOY_SKIP_SEC = 420  # skip first 7 min of buoy data (deployment)


_PARAMS_FIELDS = [
    "name", "pulse", "quality",
    "swh", "t_p", "t_m", "d_p", "d_m",
    "wswh", "wt_p", "wt_m", "wd_p",
    "sw1_swh", "sw1_t_p", "sw1_d_p",
    "sw2_swh", "sw2_t_p", "sw2_d_p",
    "ide_sys", "u_proj",
    "u_x", "u_y",
    "wind_sig", "wind_dir",
    "sog_proc", "cog_proc", "hdg_proc",
]


def _norm255(arr):
    mx = np.nanmax(arr)
    if mx > 0:
        arr = arr / mx * 255
    return arr.astype(int)


def _sys_fields(prefix, d):
    if d is None:
        return {f"{prefix}_swh": 0.0, f"{prefix}_t_p": 0.0, f"{prefix}_d_p": 0.0}
    return {
        f"{prefix}_swh": float(d['h_s']),
        f"{prefix}_t_p": float(d['t_p']),
        f"{prefix}_d_p": float(d['d_p']),
    }


# ── buoy data (batch only, not production) ────────────────────────────────────

def _load_buoy_data(nc_path):
    """
    Load buoy displacement data from NC file.
    Skips first _BUOY_SKIP_SEC seconds (deployment).
    Prints diagnostic info to stdout.
    Returns dict {time, x, y, z} or None.
    """
    try:
        from netCDF4 import Dataset as NCDataset
        with NCDataset(nc_path, 'r') as ds:
            available = list(ds.variables.keys())
            if 'time_buoy' not in ds.variables:
                print(f'[buoy] No buoy variables in file. Available keys: {available[:15]}')
                return None
            time_b = np.ma.filled(ds.variables['time_buoy'][:], np.nan).astype(np.float64)
            x_b    = np.ma.filled(ds.variables['x_buoy'][:],    np.nan).astype(np.float64)
            y_b    = np.ma.filled(ds.variables['y_buoy'][:],    np.nan).astype(np.float64)
            z_b    = np.ma.filled(ds.variables['z_buoy'][:],    np.nan).astype(np.float64)
    except Exception as exc:
        print(f'[buoy] Load error: {exc}')
        traceback.print_exc()
        return None

    valid = np.isfinite(time_b) & np.isfinite(x_b) & np.isfinite(y_b) & np.isfinite(z_b)
    time_b, x_b, y_b, z_b = time_b[valid], x_b[valid], y_b[valid], z_b[valid]
    if len(time_b) < 100:
        print(f'[buoy] Only {len(time_b)} valid samples — skipping')
        return None

    mask = (time_b - time_b[0]) >= _BUOY_SKIP_SEC
    n_kept = int(mask.sum())
    if n_kept < 100:
        print(f'[buoy] Only {n_kept} samples after {_BUOY_SKIP_SEC}s skip — skipping')
        return None

    dt = float(np.median(np.diff(time_b[mask])))
    print(f'[buoy] Loaded {n_kept} samples  dt={dt:.3f}s  fs={1/dt:.2f}Hz  '
          f'z: mean={x_b[mask].mean():.3f} std={z_b[mask].std():.3f}m')
    return {'time': time_b[mask], 'x': x_b[mask], 'y': y_b[mask], 'z': z_b[mask]}


def _compute_buoy_spectra(buoy_raw, n_freq, om_max):
    """
    Compute 1D Welch spectrum and (optionally) EWDM directional spectrum.
    All errors are printed to stdout.
    Returns dict {freq_hz, s_freq_255, ewdm} or None on Welch failure.
    """
    from scipy.signal import welch as scipy_welch

    z = buoy_raw['z']
    t = buoy_raw['time']
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt

    # --- 1D Welch frequency spectrum ---
    nperseg = min(512, len(z) // 4)
    try:
        f_w, Szz = scipy_welch(z - z.mean(), fs=fs, nperseg=nperseg, scaling='density')
        print(f'[buoy] Welch: {len(f_w)} bins, '
              f'f=[{f_w[1]:.3f}..{f_w[-1]:.3f}]Hz, '
              f'peak_f={f_w[np.argmax(Szz)]:.3f}Hz')
    except Exception as exc:
        print(f'[buoy] Welch failed: {exc}')
        traceback.print_exc()
        return None

    out_om   = np.linspace(0, om_max, n_freq)
    # Convert PSD from m²/Hz to m²/(rad/s) then interpolate onto radar om-grid
    s_freq_raw = np.interp(out_om, f_w * 2 * np.pi, Szz / (2 * np.pi),
                           left=0.0, right=0.0)
    # Normalise to [0..255] for overlay on the radar plot
    mx = s_freq_raw.max()
    s_freq_255 = (s_freq_raw / mx * 255) if mx > 0 else s_freq_raw

    # --- EWDM directional spectrum ---
    ewdm_out = None
    try:
        # np.trapz removed in NumPy 2.0; patch before ewdm imports numpy
        import numpy as _np
        if not hasattr(_np, 'trapz'):
            _np.trapz = _np.trapezoid
        import ewdm
        import xarray as xr

        x = buoy_raw['x']
        y = buoy_raw['y']

        t_ns = ((t - t[0]) * 1e9).astype('timedelta64[ns]')
        t_dt = np.datetime64('2024-01-01', 'ns') + t_ns

        ds = xr.Dataset(
            {
                'surface_elevation':     ('time', z.astype(np.float32)),
                'eastward_displacement':  ('time', x.astype(np.float32)),
                'northward_displacement': ('time', y.astype(np.float32)),
            },
            coords={'time': t_dt},
            attrs={'sampling_rate': float(fs)},
        )

        spec_obj = ewdm.Triplets(ds)
        ewdm_out = spec_obj.compute(
            omin=-5, omax=None, nvoice=16, dd=5, kappa=36,
            use='displacements', block_size='60min',
        )
        freq_e = ewdm_out.frequency.values
        dirs_e = ewdm_out.direction.values
        print(f'[buoy] EWDM ok: f=[{freq_e[0]:.3f}..{freq_e[-1]:.3f}]Hz  '
              f'dir=[{dirs_e[0]:.0f}..{dirs_e[-1]:.0f}]°')
    except ImportError:
        print('[buoy] ewdm not installed — directional spectrum skipped')
    except Exception as exc:
        print(f'[buoy] EWDM failed: {exc}')
        traceback.print_exc()

    return {
        'freq_hz':    out_om / (2 * np.pi),
        's_freq_255': s_freq_255,
        'ewdm':       ewdm_out,
    }


# ── figure ────────────────────────────────────────────────────────────────────

def _save_pic(name, spec_1d, spec_2d, freq_out, ring, sys_dict,
              swh, T_peak, T_mean, peak_dir, wdir, wind_sig,
              Ux, Uy, u_curr_x, u_curr_y, u_proj, quality,
              last_navi, adp, asp, n_dirs, pics_dir,
              buoy_proc=None):
    """
    Diagnostic figure.

    spec_1d  : shape (N_FREQ,),        int [0..255]  — same as transmitted
    spec_2d  : shape (N_DIRS, N_FREQ), int [0..255]  — same as transmitted
    freq_out : shape (N_FREQ,), Hz

    Layout (no buoy EWDM): 3×2
      [0,0] 1-D frequency spectrum (0..255)
      [0,1] Directional spectrum (polar, North-up clockwise)
      [1,0] Backscatter ring (range × azimuth, transposed)
      [1,1] Backscatter histogram (x clipped 50..200)
      [2,:] Parameter table

    Layout (with buoy EWDM): 3×3 — adds [0,2] EWDM polar spectrum.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.gridspec import GridSpec

    has_ewdm = (buoy_proc is not None and buoy_proc.get('ewdm') is not None)
    ncols = 3 if has_ewdm else 2
    figw  = 16 if has_ewdm else 8

    fig = Figure(figsize=(figw, 8))
    FigureCanvasAgg(fig)
    fig.suptitle(name, fontsize=11, y=0.998)

    gs = GridSpec(3, ncols, figure=fig, height_ratios=[3, 1, 2],
                  hspace=0.45, wspace=0.30,
                  left=0.06, right=0.97, top=0.96, bottom=0.03)

    # ── Panel 0,0: 1D frequency spectrum (0..255, same as UDP) ────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(freq_out, spec_1d, lw=1.4, color='steelblue', label='Radar')
    ax0.fill_between(freq_out, spec_1d, alpha=0.20, color='steelblue')

    if T_peak > 0:
        ax0.axvline(1.0 / T_peak, color='r', ls='--', lw=1, label=f'Tp={T_peak:.1f}s')
    if T_mean > 0:
        ax0.axvline(1.0 / T_mean, color='orange', ls=':', lw=1, label=f'Tm={T_mean:.1f}s')

    if buoy_proc is not None and buoy_proc.get('s_freq_255') is not None:
        ax0.plot(buoy_proc['freq_hz'], buoy_proc['s_freq_255'],
                 lw=1.2, color='tomato', alpha=0.85, label='Buoy (Welch)')

    ax0.set_xlim(0, freq_out[-1])
    ax0.set_ylim(0, 255)
    ax0.set_xlabel('f [Hz]')
    ax0.set_ylabel('S(f)  [0–255]')
    ax0.legend(fontsize=8)

    # ── helper: polar directional spectrum ────────────────────────────────────
    def _polar_dir_spec(ax, spec_nd, freq_hz, sys_d, extra_lines=None, title=''):
        """spec_nd: (N_DIRS, N_FREQ), values arbitrary."""
        n_d, _ = spec_nd.shape
        f_mask    = freq_hz <= _F_DISPLAY
        freq_disp = freq_hz[f_mask]
        s_disp    = spec_nd[:, f_mask]

        theta_vals   = np.linspace(0, 2 * np.pi, n_d, endpoint=False)
        theta_closed = np.append(theta_vals, 2 * np.pi)
        s_closed     = np.vstack([s_disp, s_disp[:1, :]])
        theta_g, r_g = np.meshgrid(theta_closed, freq_disp)

        vmax = float(s_disp.max()) or 1.0
        ax.pcolormesh(theta_g, r_g, s_closed.T,
                      cmap='gnuplot2', vmin=0, vmax=vmax, shading='auto')

        r_max = _F_DISPLAY

        def _peak_r(d_deg):
            if len(freq_disp) == 0:
                return r_max
            d_idx = int(round(d_deg / 360.0 * n_d)) % n_d
            sl = s_disp[d_idx]
            return float(freq_disp[int(np.argmax(sl))]) if sl.max() > 0 else r_max

        styles = [
            ('sum',  peak_dir, 'red',    f'Sum {peak_dir:.0f}°'),
            ('w_s',  None,     'cyan',   'Wind sea'),
            ('sw_1', None,     'lime',   'Swell 1'),
            ('sw_2', None,     'orange', 'Swell 2'),
        ]
        for key, fixed_dir, clr, lbl in styles:
            d_deg = fixed_dir if key == 'sum' else (sys_d[key]['d_p'] if sys_d.get(key) else None)
            if d_deg is None:
                continue
            t_r = np.deg2rad(d_deg)
            ax.plot([t_r, t_r], [0, _peak_r(d_deg)], color=clr, lw=1.8, label=lbl)
            ax.plot(t_r, _peak_r(d_deg), 'o', color=clr, ms=5)

        # Wind direction — white solid line
        wind_t = np.deg2rad(wdir)
        ax.plot([wind_t, wind_t], [0, _peak_r(wdir)],
                color='white', lw=1.5, ls='-', label=f'Wind {wdir:.0f}°')

        if extra_lines:
            for t_rad, r_val, clr, lbl in extra_lines:
                ax.plot([t_rad, t_rad], [0, r_val], color=clr, lw=2, label=lbl)
                ax.plot(t_rad, r_val, '^', color=clr, ms=7)

        ax.set_rlim(0, r_max)
        ax.grid(False)
        ax.legend(fontsize=7, loc='lower right', bbox_to_anchor=(1.35, -0.05))

    # Navigation overlays
    curr_speed   = float(np.hypot(u_curr_x, u_curr_y))
    curr_dir_deg = float(np.degrees(np.arctan2(u_curr_y, u_curr_x)) % 360)
    extra = [
        (np.deg2rad(last_navi.cog), _F_DISPLAY * 0.4,
         'deepskyblue', f'Ship {last_navi.sog:.1f}m/s'),
    ]
    if curr_speed > 0:
        extra.append((np.deg2rad(curr_dir_deg), _F_DISPLAY * 0.3,
                      'magenta', f'Curr {curr_speed:.2f}m/s'))

    # ── Panel 0,1: Radar directional spectrum ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1], projection='polar')
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    _polar_dir_spec(ax1, spec_2d.astype(float), freq_out, sys_dict,
                    extra_lines=extra)

    # ── Panel 0,2: EWDM directional spectrum ──────────────────────────────────
    if has_ewdm:
        ax_ewdm = fig.add_subplot(gs[0, 2], projection='polar')
        ax_ewdm.set_theta_zero_location('N')
        ax_ewdm.set_theta_direction(-1)
        try:
            ew      = buoy_proc['ewdm']
            freq_ew = np.asarray(ew.frequency.values, dtype=float)
            dirs_ew = np.asarray(ew.direction.values, dtype=float)
            S_ft    = np.asarray(ew.directional_spectrum.values, dtype=float)
            if S_ft.ndim == 3:
                S_ft = S_ft.mean(axis=0)   # average over time blocks → (n_freq, n_dir)

            dirs_360  = dirs_ew % 360
            sort_idx  = np.argsort(dirs_360)
            dirs_360  = dirs_360[sort_idx]
            S_ft      = S_ft[:, sort_idx]  # (n_freq, n_dir_sorted)

            f_mask_ew = freq_ew <= _F_DISPLAY
            freq_d_ew = freq_ew[f_mask_ew]
            S_disp    = S_ft[f_mask_ew, :]  # (n_f_disp, n_dir)

            theta_ew        = np.deg2rad(dirs_360)
            theta_closed_ew = np.append(theta_ew, theta_ew[0] + 2 * np.pi)
            S_closed_ew     = np.hstack([S_disp, S_disp[:, :1]])
            theta_g_ew, r_g_ew = np.meshgrid(theta_closed_ew, freq_d_ew)

            vmax_ew = float(S_disp.max()) or 1.0
            ax_ewdm.pcolormesh(theta_g_ew, r_g_ew, S_closed_ew,
                               cmap='gnuplot2', vmin=0, vmax=vmax_ew, shading='auto')
            ax_ewdm.set_rlim(0, _F_DISPLAY)
            ax_ewdm.grid(False)
        except Exception as exc:
            ax_ewdm.set_title(f'EWDM plot error:\n{exc}', pad=12)
            print(f'[buoy] EWDM plot failed: {exc}')
            traceback.print_exc()

    # ── Panel 1,0: Backscatter ring — transposed, range×azimuth ───────────────
    ax2 = fig.add_subplot(gs[1, 0])
    r_lo = max(0, adp - asp)
    r_hi = r_lo + ring.shape[1]
    # ring: (AAP, 2*ASP)  →  ring.T: (2*ASP, AAP) = (range, azimuth)
    ax2.imshow(ring.T, vmin=80, vmax=140, cmap='binary', aspect='auto',
               origin='upper', extent=[0, 360, r_hi, r_lo])
    ax2.set_xlabel('Azimuth [°]')
    ax2.set_ylabel('Range [px]')

    # ── Panel 1,1: Histogram, clipped 50–200 ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    vals   = ring.ravel().astype(float)
    mean_v = float(np.mean(vals))
    std_v  = float(np.std(vals))
    ax3.hist(vals, bins=np.arange(49.5, 200.5, 1),
             color='steelblue', edgecolor='none', alpha=0.85)
    ax3.axvline(mean_v,           color='red',    lw=2.5, label=f'mean={mean_v:.1f}')
    ax3.axvline(mean_v - std_v,   color='orange', lw=2.2, ls='--', label=f'±1σ ({std_v:.1f})')
    ax3.axvline(mean_v + std_v,   color='orange', lw=2.2, ls='--')
    ax3.axvline(mean_v - 3*std_v, color='lime',   lw=2.0, ls=':',  label='±3σ')
    ax3.axvline(mean_v + 3*std_v, color='lime',   lw=2.0, ls=':')
    ax3.set_xlim(50, 200)
    ax3.set_xlabel('Backscatter intensity')
    ax3.set_ylabel('Count')
    ax3.set_title(f'std={std_v:.2f}')
    ax3.legend(fontsize=8)

    # ── Panel 2,: Parameter table ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    ws  = sys_dict.get('w_s')
    sw1 = sys_dict.get('sw_1')
    sw2 = sys_dict.get('sw_2')

    rows = [
        ['Quality',      'GOOD' if quality else 'BAD',
         'Hs [m]',       f'{swh:.3f}',
         'Tp [s]',       f'{T_peak:.2f}',
         'Tm [s]',       f'{T_mean:.2f}',
         'Dp [°]',       f'{peak_dir:.1f}'],
        ['N sys',        str(sys_dict['n_sys']),
         'Wind from [°]', f'{wdir:.1f}',
         'Ring std',     f'{wind_sig:.2f}',
         'SNR tot',      '—',
         'u_proj [m/s]', f'{u_proj:.3f}'],
        ['Wind Hs [m]',  f"{ws['h_s']:.3f}"  if ws  else '—',
         'Wind Tp [s]',  f"{ws['t_p']:.2f}"  if ws  else '—',
         'Wind Dp [°]',  f"{ws['d_p']:.1f}"  if ws  else '—',
         'Ux [m/s]',     f'{Ux:.3f}',
         'Uy [m/s]',     f'{Uy:.3f}'],
        ['Sw1 Hs [m]',   f"{sw1['h_s']:.3f}" if sw1 else '—',
         'Sw1 Tp [s]',   f"{sw1['t_p']:.2f}" if sw1 else '—',
         'Sw1 Dp [°]',   f"{sw1['d_p']:.1f}" if sw1 else '—',
         'Curr [m/s]',   f'{curr_speed:.3f}',
         'Curr dir [°]', f'{curr_dir_deg:.1f}'],
        ['Sw2 Hs [m]',   f"{sw2['h_s']:.3f}" if sw2 else '—',
         'Sw2 Tp [s]',   f"{sw2['t_p']:.2f}" if sw2 else '—',
         'Sw2 Dp [°]',   f"{sw2['d_p']:.1f}" if sw2 else '—',
         'SOG [m/s]',    f'{last_navi.sog:.2f}',
         'COG [°]',      f'{last_navi.cog:.1f}'],
    ]

    col_labels = ['Par', 'Val', 'Par', 'Val', 'Par', 'Val', 'Par', 'Val', 'Par', 'Val']
    tbl = ax4.table(cellText=rows, colLabels=col_labels,
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#c0c0c0')
    tbl[1, 1].set_facecolor('#90ee90' if quality else '#ffb0b0')

    out_path = os.path.join(pics_dir, f'{name}.png')
    try:
        fig.savefig(out_path, dpi=100, bbox_inches='tight')
    except Exception as exc:
        print(f'[pic] save failed: {exc}')


# ── processing ────────────────────────────────────────────────────────────────

def _process_file(name, nc_path, cfg, spec_dir, pics_dir, log):
    """
    Read first N_SHOTS frames, run full algorithm pipeline, return params dict.
    Returns (row_dict, spec_1d, spec_2d) or None on failure.
    """
    cst    = cfg.const
    om_max = np.pi / (60.0 / cst.RPM)
    step   = 1.875
    k_max  = np.pi / cst.ASP / step * cst.K_NUM
    half   = cst.N_SHOTS // 2
    omega_vals = np.linspace(0, om_max, half)
    k_vals     = np.linspace(0, k_max, cst.K_NUM)
    dir_array  = np.linspace(0, 360, cst.N_DIRS, endpoint=False)

    seg_azimuths = np.linspace(0, 360, cst.NUM_AREA, endpoint=False)
    msh = [Area(cst.ASP * 2, cst.ADP, az, 0, cst.AAP).calc_mask()
           for az in seg_azimuths]

    try:
        source = NCInputSource(nc_path)
    except Exception as exc:
        log.error(f'{name}: cannot open {nc_path!r}: {exc}')
        return None

    cbck    = np.zeros((cst.NUM_AREA, cst.N_SHOTS, 2 * cst.ASP, 2 * cst.ASP), dtype=np.float32)
    pulse   = 1
    last_bck  = None
    last_navi = None

    try:
        for t in range(cst.N_SHOTS):
            back = source.get_bck()
            if back.step == 0.0:
                log.warning(f'{name}: only {t} frames (need {cst.N_SHOTS}), skipping')
                return None
            navi      = source.get_navi()
            pulse     = back.pulse
            last_bck  = back.bck
            last_navi = navi
            for i in range(cst.NUM_AREA):
                (x, y), (wx, wy) = msh[i]
                row0 = last_bck[y, x] * (1.0 - wx) + last_bck[y, x + 1] * wx
                row1 = last_bck[y + 1, x] * (1.0 - wx) + last_bck[y + 1, x + 1] * wx
                cbck[i, t] = row0 * (1.0 - wy) + row1 * wy
    except Exception as exc:
        log.error(f'{name}: read error: {exc}', exc_info=True)
        return None
    finally:
        try:
            source.close()
        except Exception:
            pass

    try:
        _, wdir = calc_wspd(last_bck)

        ring     = last_bck[:, max(0, cst.ADP - cst.ASP): cst.ADP + cst.ASP]
        wind_sig = float(np.std(ring))

        spec_3d_corr = np.zeros((half, 2 * cst.K_NUM, 2 * cst.K_NUM), dtype=np.float32)
        for i in range(cst.NUM_AREA):
            spec_3d_corr += calc_spec3d(cbck[i], cst.K_NUM)
        spec_3d_corr /= cst.NUM_AREA

        Ux, Uy        = calc_current_vector(spec_3d_corr, k_max, om_max)
        spec_3d_fixed = apply_doppler_3d_vec(spec_3d_corr, k_max, Ux, Uy, om_max)

        port_fixed, _ = calc_port(spec_3d_fixed)
        signal, noise = separate_signal_noise(port_fixed, k_vals, om_max, band=_SIGNAL_BAND)
        signal_mtf    = apply_mtf(signal, k_vals, exp=1.2)

        snr_tot                         = compute_snr(signal_mtf, noise)
        s_omega, m0, T_peak, T_mean     = compute_frequency_spectrum(signal_mtf, k_vals, omega_vals)
        s_om_th, peak_dir, mean_dir     = calc_spec2d(
            spec_3d_fixed, omega_vals, k_max, cst.N_DIRS, band=_SIGNAL_BAND)

        swh = 0.01 * (cst.SNR_A + cst.SNR_B * np.sqrt(snr_tot))
        sys = calc_partitions(s_om_th, omega_vals, dir_array, wdir, swh)

        cog_rad  = np.deg2rad(float(last_navi.cog))
        u_curr_x = float(Ux) + float(last_navi.sog) * np.cos(cog_rad)
        u_curr_y = float(Uy) + float(last_navi.sog) * np.sin(cog_rad)

        peak_dir_rad = np.deg2rad(peak_dir)
        u_proj = float(u_curr_x * np.cos(peak_dir_rad) + u_curr_y * np.sin(peak_dir_rad))

        quality = int(
            snr_tot    >= _SNR_QUALITY_MIN
            and wind_sig   >= _WIND_SIG_MIN
            and T_peak     >= _T_PEAK_MIN
            and sys["n_sys"] >= 1
        )
        if not quality:
            reasons = []
            if snr_tot       < _SNR_QUALITY_MIN: reasons.append(f'snr={snr_tot:.2f}<{_SNR_QUALITY_MIN}')
            if wind_sig      < _WIND_SIG_MIN:    reasons.append(f'ring_sig={wind_sig:.2f}<{_WIND_SIG_MIN}')
            if T_peak        < _T_PEAK_MIN:      reasons.append(f'T_peak={T_peak:.2f}<{_T_PEAK_MIN}')
            if sys["n_sys"]  < 1:                reasons.append('n_sys=0')
            msg = f'[quality] {name}: BAD  [{", ".join(reasons)}]'
            log.info(msg)
            print(msg)

        # Output grid — same discretisation as UDP packet (N_FREQ bins)
        out_om      = np.linspace(0, om_max, cst.N_FREQ)
        freq_out    = out_om / (2 * np.pi)
        s_omega_out = np.interp(out_om, omega_vals, s_omega)
        s_om_th_out = interp1d(omega_vals, s_om_th, axis=1, kind='linear',
                                fill_value=0.0, bounds_error=False)(out_om)

        spec_1d = _norm255(s_omega_out.copy())   # [0..255] int — transmitted
        spec_2d = _norm255(s_om_th_out.copy())   # [0..255] int — transmitted

    except Exception as exc:
        log.error(f'{name}: algorithm error: {exc}', exc_info=True)
        return None

    # Buoy processing (only for "0606" files, batch only)
    buoy_proc = None
    if '0606' in name:
        raw_buoy = _load_buoy_data(nc_path)
        if raw_buoy is not None:
            buoy_proc = _compute_buoy_spectra(raw_buoy, cst.N_FREQ, om_max)

    if pics_dir is not None:
        try:
            _save_pic(
                name, spec_1d, spec_2d, freq_out, ring, sys,
                swh, T_peak, T_mean, peak_dir, wdir, wind_sig,
                Ux, Uy, u_curr_x, u_curr_y, u_proj, quality,
                last_navi, cst.ADP, cst.ASP, cst.N_DIRS, pics_dir,
                buoy_proc=buoy_proc,
            )
        except Exception as exc:
            log.warning(f'{name}: pic save failed: {exc}', exc_info=True)

    ws = sys.get('w_s')
    row = {
        'name':     name,
        'pulse':    int(pulse),
        'quality':  int(quality),
        'swh':      float(swh),
        't_p':      float(T_peak),
        't_m':      float(T_mean),
        'd_p':      float(peak_dir),
        'd_m':      float(mean_dir),
        'wswh':     float(ws['h_s']) if ws else 0.0,
        'wt_p':     float(ws['t_p']) if ws else 0.0,
        'wt_m':     float(ws['t_m']) if ws else 0.0,
        'wd_p':     float(ws['d_p']) if ws else 0.0,
        **_sys_fields('sw1', sys.get('sw_1')),
        **_sys_fields('sw2', sys.get('sw_2')),
        'ide_sys':  int(sys['n_sys']),
        'u_proj':   float(u_proj),
        'u_x':      float(u_curr_x),
        'u_y':      float(u_curr_y),
        'wind_sig': float(wind_sig),
        'wind_dir': float(wdir),
        'sog_proc': float(last_navi.sog),
        'cog_proc': float(last_navi.cog),
        'hdg_proc': float(last_navi.hdg),
    }
    return row, spec_1d, spec_2d


def main():
    parser = argparse.ArgumentParser(description='Batch NC radar file processing')
    parser.add_argument('--csv',       default='META_upd2.csv')
    parser.add_argument('--base-path', default='') # /storage/thalassa/DATA/RADAR/
    parser.add_argument('--out',       default='batch_out')
    parser.add_argument('--config',    default='config.ini')
    args = parser.parse_args()

    log = setup_logger('batch')
    cfg = load_config(args.config)

    spec_dir    = os.path.join(args.out, 'spec')
    pics_dir    = os.path.join(args.out, 'pics')
    params_path = os.path.join(args.out, 'params.csv')
    os.makedirs(args.out,   exist_ok=True)
    os.makedirs(spec_dir,   exist_ok=True)
    os.makedirs(pics_dir,   exist_ok=True)

    df = pd.read_csv(args.csv)
    for field in _PARAMS_FIELDS:
        if field not in df.columns:
            df[field] = np.nan

    log.info(f"Batch: {len(df)} rows → '{args.out}'  (N_SHOTS={cfg.const.N_SHOTS})")

    for i, row in df.iterrows():
        name = row['name'].split('/')[-1][:-3]
        path = args.base_path + row['name']

        try:
            result = _process_file(name, path, cfg, spec_dir, pics_dir, log)
            if result is None:
                log.warning(f'{name}: processing failed')
                continue

            params, spec_1d, spec_2d = result
            for key, value in params.items():
                df.loc[i, key] = value

            np.save(os.path.join(spec_dir, f'{name}_freqspec.npy'), spec_1d)
            np.save(os.path.join(spec_dir, f'{name}_dirspec.npy'),  spec_2d)

            df.to_csv(params_path, index=False)
            log.info(f'{name}: done  quality={params["quality"]}')

        except Exception as exc:
            log.error(f'{name}: fatal error: {exc}', exc_info=True)
            df.to_csv(params_path, index=False)
        finally:
            gc.collect()  # flush HDF5 file descriptors before next file

    log.info(f"Done. Params → '{params_path}'")


if __name__ == '__main__':
    main()
