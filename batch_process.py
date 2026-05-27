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

_SIGNAL_BAND  = 10    # must match processor.py
_MAX_CURRENT  = 2.55  # physical clip for ocean current [m/s]

# Quality thresholds — hardcoded, same as processor.py
_SNR_QUALITY_MIN = 1.5
_WIND_SIG_MIN    = 5.0
_T_PEAK_MIN      = 5.5

_F_DISPLAY   = 0.20   # Hz — radial limit on polar spectrum display
_BUOY_SKIP_SEC = 420  # skip first 7 min of buoy data (deployment)


_PARAMS_FIELDS = [
    "name", "pulse", "quality",
    "swh", "t_p", "t_m", "d_p", "d_m",
    "wswh", "wt_p", "wt_m", "wd_p",
    "sw1_swh", "sw1_t_p", "sw1_d_p",
    "sw2_swh", "sw2_t_p", "sw2_d_p",
    "ide_sys", "curr_speed", "curr_dir", "curr_x", "curr_y", "wspd_proc",
    "u_x", "u_y",
    "wind_sig", "wind_dir",
    "sog_proc", "cog_proc", "hdg_proc",
]


_PULSE_TO_STR = {1: 'SP', 2: 'MP', 3: 'LP'}
_PULSE_TO_INT = {'SP': 1, 'MP': 2, 'LP': 3}

def _pulse_str(p):
    if isinstance(p, str):
        return p.strip().upper()
    return _PULSE_TO_STR.get(int(p), str(p))

def _pulse_int(p):
    if isinstance(p, str):
        return _PULSE_TO_INT.get(p.strip().upper(), 0)
    return int(p)


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


def _dispersion_centroids(spec_3d, k_max, om_max, sog=0.0, cog_deg=0.0):
    """
    For each valid (kx, ky) cell, find the argmax of energy within a wide window
    centred on the ship-velocity-shifted dispersion curve (mirrors Pass-1 of
    calc_current_vector). Returns (k_abs, om_peak) scatter arrays for debug overlay.
    """
    n_om, n2, _ = spec_3d.shape
    k_num = n2 // 2
    kx_arr = np.arange(-k_num, k_num, dtype=float) / k_num * k_max
    ky_arr = np.arange(-k_num, k_num, dtype=float) / k_num * k_max
    KX, KY = np.meshgrid(kx_arr, ky_arr, indexing='ij')
    K_abs  = np.sqrt(KX ** 2 + KY ** 2)
    om_arr = np.linspace(0, om_max, n_om)
    omega_ref = np.sqrt(9.81 * K_abs)

    k_lo = k_max * 0.08
    k_hi = k_max * 0.65
    i_vals, j_vals = np.where((K_abs > k_lo) & (K_abs < k_hi))

    cog_rad = np.deg2rad(cog_deg)
    Ux_prior = sog * np.sin(cog_rad)
    Uy_prior = sog * np.cos(cog_rad)

    win = max(10, n_om // 16)
    k_list, om_list, w_list = [], [], []
    for i, j in zip(i_vals, j_vals):
        om_ctr = float(omega_ref[i, j] + KX[i, j] * Ux_prior + KY[i, j] * Uy_prior)
        ci = int(round(om_ctr / om_max * (n_om - 1)))
        ci = max(0, min(n_om - 1, ci))
        lo = max(1, ci - win)
        hi = min(n_om, ci + win + 1)
        if hi <= lo:
            continue
        sl = spec_3d[lo:hi, i, j].astype(np.float64)
        pk = int(np.argmax(sl))
        if sl[pk] <= 0:
            continue
        k_list.append(float(K_abs[i, j]))
        om_list.append(float(om_arr[lo + pk]))
        w_list.append(float(sl[pk]))

    return np.array(k_list), np.array(om_list), np.array(w_list)


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
    except ImportError:
        print('[buoy] ewdm not installed — directional spectrum skipped')
    except Exception as exc:
        print(f'[buoy] EWDM failed: {exc}')
        traceback.print_exc()

    # Buoy Welch peak and mean frequency
    f_peak_hz = float(f_w[np.argmax(Szz)]) if len(Szz) > 0 else None
    denom = float(np.sum(Szz))
    f_mean_hz = float(np.dot(f_w, Szz) / denom) if denom > 0 else None

    # EWDM directional peak and mean
    dp_buoy = dm_buoy = None
    if ewdm_out is not None:
        try:
            S_all = np.asarray(ewdm_out.directional_spectrum.values, dtype=float)
            if S_all.ndim == 3:
                S_all = S_all.mean(axis=0)          # (n_freq, n_dir)
            dirs_all = np.asarray(ewdm_out.direction.values, dtype=float) % 360
            s_dir_b  = S_all.sum(axis=0)
            dp_buoy  = float(dirs_all[np.argmax(s_dir_b)])
            cx = float(np.dot(s_dir_b, np.cos(np.deg2rad(dirs_all))))
            cy = float(np.dot(s_dir_b, np.sin(np.deg2rad(dirs_all))))
            dm_buoy  = float(np.degrees(np.arctan2(cy, cx)) % 360)
        except Exception:
            pass

    return {
        'freq_hz':    out_om / (2 * np.pi),
        's_freq_255': s_freq_255,
        'ewdm':       ewdm_out,
        'f_peak_hz':  f_peak_hz,
        'f_mean_hz':  f_mean_hz,
        'dp_buoy':    dp_buoy,
        'dm_buoy':    dm_buoy,
    }


# ── figure ────────────────────────────────────────────────────────────────────

def _save_pic(name, spec_1d, spec_2d, freq_out, ring, sys_dict,
              swh, T_peak, T_mean, peak_dir, mean_dir, wdir, wind_sig, wspd, snr_tot,
              Ux, Uy, u_proj, u_curr_x, u_curr_y, curr_speed, curr_dir_in, quality,
              pulse, last_navi, adp, asp, n_dirs, pics_dir,
              port_corr=None, k_vals=None, omega_vals=None,
              wdir_meta=None, buoy_proc=None,
              cent_k=None, cent_om=None, cent_w=None, u_ship_proj=0.0,
              sog_mean=0.0, cog_mean=0.0):
    """
    Diagnostic figure.

    spec_1d  : shape (N_FREQ,),        int [0..255]  — same as transmitted
    spec_2d  : shape (N_DIRS, N_FREQ), int [0..255]  — same as transmitted
    freq_out : shape (N_FREQ,), Hz

    Layout (3 columns):
      [0,0] 1-D frequency spectrum (0..255)
      [0,1] Directional spectrum (polar, North-up clockwise)
      [0,2] ω-k dispersion portrait (pre-Doppler-correction)
      [1,:] Backscatter ring full-width (azimuth × range)
      [2,:] Parameter table
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.gridspec import GridSpec

    pulse_str = _pulse_str(pulse)

    fig = Figure(figsize=(15, 10))
    FigureCanvasAgg(fig)
    fig.suptitle(f'{name}   [{pulse_str}]', fontsize=11, y=0.998)

    gs = GridSpec(3, 3, figure=fig, height_ratios=[2, 1, 1],
                  hspace=0.45, wspace=0.30,
                  left=0.06, right=0.97, top=0.96, bottom=0.03)

    # ── Panel 0,0: 1D frequency spectrum (0..255, same as UDP) ────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(freq_out, spec_1d, lw=1.4, color='steelblue', label='Radar')
    ax0.fill_between(freq_out, spec_1d, alpha=0.20, color='steelblue')

    if T_peak > 0:
        ax0.axvline(1.0 / T_peak, color='steelblue', ls='-', lw=1.5, label=f'Radar Tp={T_peak:.1f}s')
    if T_mean > 0:
        ax0.axvline(1.0 / T_mean, color='steelblue', ls='--', lw=1.2, label=f'Radar Tm={T_mean:.1f}s')

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

        # Wind direction from backscatter fit — white solid line
        wind_t = np.deg2rad(wdir)
        ax.plot([wind_t, wind_t], [0, _peak_r(wdir)],
                color='white', lw=1.5, ls='-', label=f'Wind bck {wdir:.0f}°')

        # Wind direction from ERA5 u_10/v_10 — white dashed line
        if wdir_meta is not None:
            wm_t = np.deg2rad(wdir_meta)
            ax.plot([wm_t, wm_t], [0, r_max * 0.95],
                    color='white', lw=1.5, ls='--', label=f'Wind ERA5 {wdir_meta:.0f}°')

        if extra_lines:
            for t_rad, r_val, clr, lbl in extra_lines:
                ax.plot([t_rad, t_rad], [0, r_val], color=clr, lw=2, label=lbl)
                ax.plot(t_rad, r_val, '^', color=clr, ms=7)

        ax.set_rlim(0, r_max)
        ax.grid(False)
        ax.legend(fontsize=7, loc='lower right', bbox_to_anchor=(1.35, -0.05))

    # Navigation overlays
    curr_speed   = float(np.hypot(u_curr_x, u_curr_y))
    curr_dir_deg = float(curr_dir_in)
    extra = [
        (np.deg2rad(cog_mean), _F_DISPLAY * 0.4,
         'deepskyblue', f'Ship {sog_mean:.1f}m/s'),
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

    # ── Panel 0,2: ω-k dispersion portrait (pre-Doppler-correction) ─────────────
    ax_disp = fig.add_subplot(gs[0, 2])
    if port_corr is not None and k_vals is not None and omega_vals is not None:
        _om_max = float(omega_vals[-1])
        _k_max  = float(k_vals[-1])
        _k_arr  = np.linspace(0, _k_max, port_corr.shape[1])
        _om_undist = np.sqrt(9.81 * _k_arr)
        _om_shift  = _om_undist + _k_arr * u_proj

        ax_disp.imshow(port_corr, aspect='auto', origin='lower', cmap='gnuplot2',
                       extent=[0, _k_max, 0, _om_max],
                       vmin=0, vmax=max(float(port_corr.max()), 1e-9),
                       interpolation='none')

        # Centroid scatter — where energy actually sits (Pass-1 wide window)
        if cent_k is not None and len(cent_k) > 0:
            if cent_w is not None and len(cent_w) == len(cent_k) and cent_w.max() > 0:
                # log-scale within [0,1] so faint cells are visible
                w_log = np.log1p(cent_w)
                w_norm = (w_log - w_log.min()) / (w_log.max() - w_log.min() + 1e-12)
                ax_disp.scatter(cent_k, cent_om, s=10 * w_norm, c=w_norm,
                                cmap='rainbow', vmin=0, vmax=1, alpha=0.8,
                                linewidths=0, zorder=3, label='centroids')
            else:
                ax_disp.scatter(cent_k, cent_om, s=3, c='red', alpha=0.35,
                                linewidths=0, zorder=3, label='centroids')

        ax_disp.plot(_k_arr, _om_undist, 'w--', lw=1.0, label='ω=√(gk)')

        # Cyan: full apparent velocity (algorithm result)
        _mask = (_om_shift >= 0) & (_om_shift <= _om_max)
        ax_disp.plot(_k_arr[_mask], _om_shift[_mask], color='red', lw=1.5,
                     label=f'u_proj={u_proj:+.2f}')

        # Yellow: ship-only curve (if water current = 0)
        _om_ship = _om_undist + _k_arr * u_ship_proj
        _mask_ship = (_om_ship >= 0) & (_om_ship <= _om_max)
        ax_disp.plot(_k_arr[_mask_ship], _om_ship[_mask_ship], color='red',
                     lw=1.5, ls='--', label=f'ship_only={u_ship_proj:+.2f}')

        ax_disp.set_xlim(0, _k_max)
        ax_disp.set_ylim(0, _om_max)
        ax_disp.legend(fontsize=6, loc='upper right')
        _info = (f'Ux={Ux:+.2f} Uy={Uy:+.2f} m/s\n'
                 f'u_proj={u_proj:+.2f}  ship={u_ship_proj:+.2f} m/s\n'
                 f'curr={curr_speed:.2f}→{curr_dir_deg:.0f}°\n'
                 f'SOG={sog_mean:.2f} COG={cog_mean:.0f}°')
        ax_disp.text(0.02, 0.97, _info, transform=ax_disp.transAxes,
                     fontsize=6, va='top', color='white',
                     bbox=dict(facecolor='black', alpha=0.55, pad=2))
    else:
        ax_disp.axis('off')
    ax_disp.set_title('ω-k (pre-corr)', fontsize=8)
    ax_disp.set_xlabel('k [rad/m]', fontsize=7)
    ax_disp.set_ylabel('ω [rad/s]', fontsize=7)

    # ── Panel 1,:: Backscatter ring — full width, range×azimuth ──────────────
    ax2 = fig.add_subplot(gs[1, :])
    r_lo = max(0, adp - asp)
    r_hi = r_lo + ring.shape[1]
    ax2.imshow(ring.T, vmin=80, vmax=140, cmap='binary_r', aspect='auto',
               origin='upper', extent=[0, 360, r_hi, r_lo])
    _dir_lines = [
        (peak_dir, 'red',   f'Sum {peak_dir:.0f}°'),
        (wdir,     'white', f'Wind {wdir:.0f}°'),
    ]
    for key, clr in [('w_s', 'cyan'), ('sw_1', 'lime'), ('sw_2', 'orange')]:
        s = sys_dict.get(key)
        if s:
            _dir_lines.append((s['d_p'], clr, f'{key} {s["d_p"]:.0f}°'))
    for d_deg, clr, lbl in _dir_lines:
        ax2.axvline(d_deg % 360, color=clr, lw=1.4, ls='--', alpha=0.85, label=lbl)
    ax2.legend(fontsize=7, loc='upper right', framealpha=0.5)
    ax2.set_xlabel('Azimuth [°]')
    ax2.set_ylabel('Range [px]')

    # ── Panel 2,: Parameter table ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    ws  = sys_dict.get('w_s')
    sw1 = sys_dict.get('sw_1')
    sw2 = sys_dict.get('sw_2')

    rows = [
        ['Quality',      'GOOD' if quality else 'BAD',
         'Pulse',        pulse_str,
         'Hs [m]',       f'{swh:.3f}',
         'Tp [s]',       f'{T_peak:.2f}',
         'Tm [s]',       f'{T_mean:.2f}',
         'Dp [°]',       f'{peak_dir:.1f}'],
        ['N sys',        str(sys_dict['n_sys']),
         'Wind from [°]', f'{wdir:.1f}',
         'Ring std',     f'{wind_sig:.2f}',
         'SNR tot',      f'{snr_tot:.2f}',
         'WSPD [m/s]',   f'{wspd:.2f}',
         'Dm [°]',       f'{mean_dir:.1f}'],
        ['Wind Hs [m]',  f"{ws['h_s']:.3f}"  if ws  else '—',
         'Wind Tp [s]',  f"{ws['t_p']:.2f}"  if ws  else '—',
         'Wind Dp [°]',  f"{ws['d_p']:.1f}"  if ws  else '—',
         'Wind Tm [s]',  f"{ws['t_m']:.2f}"  if ws  else '—',
         'Ux [m/s]',     f'{Ux:.3f}',
         'Uy [m/s]',     f'{Uy:.3f}'],
        ['Sw1 Hs [m]',   f"{sw1['h_s']:.3f}" if sw1 else '—',
         'Sw1 Tp [s]',   f"{sw1['t_p']:.2f}" if sw1 else '—',
         'Sw1 Dp [°]',   f"{sw1['d_p']:.1f}" if sw1 else '—',
         'Sw1 Tm [s]',   f"{sw1['t_m']:.2f}" if sw1 else '—',
         'Curr [m/s]',   f'{curr_speed:.3f}',
         'Curr dir [°]', f'{curr_dir_deg:.1f}'],
        ['Sw2 Hs [m]',   f"{sw2['h_s']:.3f}" if sw2 else '—',
         'Sw2 Tp [s]',   f"{sw2['t_p']:.2f}" if sw2 else '—',
         'Sw2 Dp [°]',   f"{sw2['d_p']:.1f}" if sw2 else '—',
         'Sw2 Tm [s]',   f"{sw2['t_m']:.2f}" if sw2 else '—',
         'SOG [m/s]',    f'{sog_mean:.2f}',
         'COG [°]',      f'{cog_mean:.1f}'],
    ]

    col_labels = ['Par', 'Val'] * 6
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

def _load_frames(name, nc_path, pulse, cfg, log):
    """
    I/O phase: read N_SHOTS frames from NC file + buoy data if applicable.
    Returns dict with raw arrays or None on failure.
    Designed to run in a background thread while the previous file is computed.
    """
    cst = cfg.const
    seg_azimuths = np.linspace(0, 360, cst.NUM_AREA, endpoint=False)
    msh = [Area(cst.ASP * 2, cst.ADP, np.deg2rad(az), 0, cst.AAP).calc_mask()
           for az in seg_azimuths]

    try:
        source = NCInputSource(nc_path)
    except Exception as exc:
        log.error(f'{name}: cannot open {nc_path!r}: {exc}')
        return None

    cbck = np.zeros((cst.NUM_AREA, cst.N_SHOTS, 2 * cst.ASP, 2 * cst.ASP), dtype=np.float32)
    last_bck = None
    last_navi = None
    sog_acc = 0.0
    cog_sin_acc = cog_cos_acc = 0.0
    hdg_sin_acc = hdg_cos_acc = 0.0
    n_navi = 0

    try:
        for t in range(cst.N_SHOTS):
            back = source.get_bck()
            if back.step == 0.0:
                log.warning(f'{name}: only {t} frames (need {cst.N_SHOTS}), skipping')
                return None
            navi = source.get_navi()
            last_bck = back.bck
            last_navi = navi
            sog_acc += float(navi.sog)
            _cr = np.deg2rad(float(navi.cog)); cog_sin_acc += np.sin(_cr); cog_cos_acc += np.cos(_cr)
            _hr = np.deg2rad(float(navi.hdg)); hdg_sin_acc += np.sin(_hr); hdg_cos_acc += np.cos(_hr)
            n_navi += 1
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
    n_navi = max(n_navi, 1)
    om_max = np.pi / (60.0 / cst.RPM)

    buoy_proc = None
    if '0606' in name:
        raw_buoy = _load_buoy_data(nc_path)
        if raw_buoy is not None:
            buoy_proc = _compute_buoy_spectra(raw_buoy, cst.N_FREQ, om_max)

    return {
        'cbck':      cbck,
        'pulse':     pulse,
        'last_bck':  last_bck,
        'last_navi': last_navi,
        'nc_path':   nc_path,
        'sog_mean':  sog_acc / n_navi,
        'cog_mean':  float(np.degrees(np.arctan2(cog_sin_acc, cog_cos_acc)) % 360),
        'hdg_mean':  float(np.degrees(np.arctan2(hdg_sin_acc, hdg_cos_acc)) % 360),
        'buoy_proc': buoy_proc,
    }


def _compute_from_frames(name, frames, cfg, spec_dir, pics_dir, log, wind_meta=None):
    """
    Compute phase: run full algorithm on pre-loaded frames dict.
    Returns (row_dict, spec_1d, spec_2d) or None on failure.
    """
    cst       = cfg.const
    om_max    = np.pi / (60.0 / cst.RPM)
    step      = 1.875
    k_max     = np.pi / cst.ASP / step * cst.K_NUM
    half      = cst.N_SHOTS // 2
    omega_vals = np.linspace(0, om_max, half)
    k_vals     = np.linspace(0, k_max, cst.K_NUM)
    dir_array  = np.linspace(0, 360, cst.N_DIRS, endpoint=False)

    cbck      = frames['cbck']
    pulse     = frames['pulse']
    last_bck  = frames['last_bck']
    last_navi = frames['last_navi']
    nc_path   = frames['nc_path']
    sog_mean  = frames['sog_mean']
    cog_mean  = frames['cog_mean']
    hdg_mean  = frames['hdg_mean']
    buoy_proc = frames['buoy_proc']

    try:
        _, wdir = calc_wspd(last_bck)

        ring     = last_bck[:, cst.ADP - cst.ASP: cst.ADP + cst.ASP]
        wind_sig = float(np.std(ring))

        spec_3d_corr = np.zeros((half, 2 * cst.K_NUM, 2 * cst.K_NUM), dtype=np.float32)
        for i in range(cst.NUM_AREA):
            spec_3d_corr += calc_spec3d(cbck[i], cst.K_NUM)
        spec_3d_corr /= cst.NUM_AREA

        port_corr, _ = calc_port(spec_3d_corr)   # pre-correction, for debug portrait

        Ux, Uy        = calc_current_vector(spec_3d_corr, k_max, om_max, band=_SIGNAL_BAND,
                                            sog=sog_mean, cog_deg=cog_mean)

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

        cog_rad  = np.deg2rad(cog_mean)
        u_curr_x = float(Ux) - sog_mean * np.sin(cog_rad)   # East [m/s]
        u_curr_y = float(Uy) - sog_mean * np.cos(cog_rad)   # North [m/s]
        curr_speed = float(np.hypot(u_curr_x, u_curr_y))
        if curr_speed > _MAX_CURRENT:
            _f = _MAX_CURRENT / curr_speed
            u_curr_x *= _f; u_curr_y *= _f
            curr_speed = _MAX_CURRENT
        curr_dir   = float(np.degrees(np.arctan2(u_curr_x, u_curr_y)) % 360)  # compass

        # Apparent current projected onto dominant wave direction (for dispersion portrait).
        # peak_dir is in math convention (0°=East, ccw): projection = Ux·cos + Uy·sin.
        peak_dir_rad = np.deg2rad(peak_dir)
        u_proj = float(Ux * np.cos(peak_dir_rad) + Uy * np.sin(peak_dir_rad))

        # Ship-only projection: apparent speed if water current = 0
        Ux_ship_vis = sog_mean * np.sin(cog_rad)   # East component of ship velocity
        Uy_ship_vis = sog_mean * np.cos(cog_rad)   # North component
        u_ship_proj = float(Ux_ship_vis * np.cos(peak_dir_rad) + Uy_ship_vis * np.sin(peak_dir_rad))

        # Per-cell ω peak positions for scatter overlay on dispersion portrait
        cent_k, cent_om, cent_w = _dispersion_centroids(spec_3d_corr, k_max, om_max,
                                                       sog=sog_mean, cog_deg=cog_mean)

        wspd = 0.01 * float(cst.WSPD_A + cst.WSPD_B * wind_sig)

        # Wind direction from ERA5 u_10/v_10 (FROM direction, compass)
        wdir_meta = None
        if wind_meta:
            u10 = float(wind_meta.get('u_10', 0.0))
            v10 = float(wind_meta.get('v_10', 0.0))
            if u10 != 0.0 or v10 != 0.0:
                # u10/v10 are East/North components of wind blowing toward that direction
                # FROM direction = opposite: arctan2(u10, v10) + 180°
                wdir_meta = float((np.degrees(np.arctan2(u10, v10)) + 180) % 360)

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

    if pics_dir is not None:
        try:
            _save_pic(
                name, spec_1d, spec_2d, freq_out, ring, sys,
                swh, T_peak, T_mean, peak_dir, mean_dir, wdir, wind_sig, wspd, snr_tot,
                Ux, Uy, u_proj, u_curr_x, u_curr_y, curr_speed, curr_dir, quality,
                pulse, last_navi, cst.ADP, cst.ASP, cst.N_DIRS, pics_dir,
                port_corr=port_corr, k_vals=k_vals, omega_vals=omega_vals,
                wdir_meta=wdir_meta, buoy_proc=buoy_proc,
                cent_k=cent_k, cent_om=cent_om, cent_w=cent_w, u_ship_proj=u_ship_proj,
                sog_mean=sog_mean, cog_mean=cog_mean,
            )
        except Exception as exc:
            log.warning(f'{name}: pic save failed: {exc}', exc_info=True)

    ws = sys.get('w_s')
    row = {
        'name':     name,
        'pulse':    _pulse_str(pulse),
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
        'ide_sys':    int(sys['n_sys']),
        'curr_speed': float(curr_speed),
        'curr_dir':   float(curr_dir),
        'curr_x':     float(u_curr_x),
        'curr_y':     float(u_curr_y),
        'wspd_proc':       float(wspd),
        'u_x':        float(Ux),
        'u_y':        float(Uy),
        'wind_sig': float(wind_sig),
        'wind_dir': float(wdir),
        'sog_proc': sog_mean,
        'cog_proc': cog_mean,
        'hdg_proc': hdg_mean,
    }
    return row, spec_1d, spec_2d


def _process_file(name, nc_path, pulse, cfg, spec_dir, pics_dir, log, wind_meta=None):
    """Thin wrapper: load frames then compute. Used by single-process batch and legacy callers."""
    frames = _load_frames(name, nc_path, pulse, cfg, log)
    if frames is None:
        return None
    return _compute_from_frames(name, frames, cfg, spec_dir, pics_dir, log, wind_meta)


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
    if 'pulse' in df.columns:
        df['pulse'] = df['pulse'].apply(_pulse_str)
    for field in _PARAMS_FIELDS:
        if field not in df.columns:
            df[field] = np.nan

    log.info(f"Batch: {len(df)} rows → '{args.out}'  (N_SHOTS={cfg.const.N_SHOTS})")

    for i, row in df.iterrows():
        name = row['name'].split('/')[-1][:-3]
        path = args.base_path + row['name']
        pulse = row["pulse"]
        try:
            wind_meta = None
            for col in ('u_10', 'v_10'):
                if col not in df.columns:
                    break
            else:
                u10, v10 = row.get('u_10'), row.get('v_10')
                if pd.notna(u10) and pd.notna(v10):
                    wind_meta = {'u_10': float(u10), 'v_10': float(v10)}

            result = _process_file(name, path, pulse, cfg, spec_dir, pics_dir, log,
                                   wind_meta=wind_meta)
            if result is None:
                log.warning(f'{name}: processing failed')
                continue

            params, spec_1d, spec_2d = result
            for key, value in params.items():
                df.loc[i, key] = value

            # np.save(os.path.join(spec_dir, f'{name}_freqspec.npy'), spec_1d)
            # np.save(os.path.join(spec_dir, f'{name}_dirspec.npy'),  spec_2d)

            df.to_csv(params_path, index=False, float_format='%.2f')
            log.info(f'{name}: done  quality={params["quality"]}')

        except Exception as exc:
            log.error(f'{name}: fatal error: {exc}', exc_info=True)
            df.to_csv(params_path, index=False)
        finally:
            gc.collect()  # flush HDF5 file descriptors before next file

    log.info(f"Done. Params → '{params_path}'")


if __name__ == '__main__':
    main()
