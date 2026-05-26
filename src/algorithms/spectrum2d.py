import numpy as np
from scipy.fft import fft2, fftshift
from scipy.signal import welch
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter, gaussian_filter1d


# ── 3-D spectrum ─────────────────────────────────────────────────────────────

def calc_spec3d(cbck_i, k_num):
    """
    3-D Welch power spectrum (ω, kx, ky) for one segment.
    Input: cbck_i shape (N_SHOTS, 2*ASP, 2*ASP).
    Returns positive-frequency half: (N_SHOTS//2, 2*K_NUM, 2*K_NUM), float32.
    """
    asp = cbck_i.shape[1] // 2
    _, back_f3 = welch(
        fftshift(fft2(cbck_i, axes=(1, 2)), axes=(1, 2))
        [:, asp - k_num:asp + k_num, asp - k_num:asp + k_num],
        axis=0, return_onesided=False,
    )
    half = back_f3.shape[0] // 2
    return back_f3[:half].astype(np.float32)


# ── omega-k portrait ─────────────────────────────────────────────────────────

def calc_port(spec_3d):
    """
    Collapse 3-D spectrum (ω, kx, ky) → (ω, |k|) portrait using all quadrants.
    Returns (port, k_norm) where k_norm[i,j] = sqrt(kx²+ky²) / k_num ∈ [0, √2].
    """
    n_om, n2, _ = spec_3d.shape
    k_num = n2 // 2

    kx = np.arange(-k_num, k_num, dtype=float)
    ky = np.arange(-k_num, k_num, dtype=float)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K_abs = np.sqrt(KX ** 2 + KY ** 2)
    k_norm = K_abs / k_num

    k_div = np.floor(K_abs).astype(int)
    k_mod = (K_abs - k_div).astype(np.float32)
    mask0 = (k_div < k_num).ravel()
    mask1 = (k_div < k_num - 1).ravel()
    kd0 = k_div.ravel()[mask0]
    km0 = k_mod.ravel()[mask0]
    kd1 = k_div.ravel()[mask1]
    km1 = k_mod.ravel()[mask1]

    sf = spec_3d.reshape(n_om, -1)  # (n_om, n2*n2)
    port = np.zeros((n_om, k_num), dtype=np.float32)
    np.add.at(port, (slice(None), kd0), sf[:, mask0] * (1.0 - km0))
    np.add.at(port, (slice(None), kd1 + 1), sf[:, mask1] * km1)
    port[:, 0] = 0.0
    return port, k_norm


# ── Doppler correction ────────────────────────────────────────────────────────

def apply_doppler_2d(port, k_vals, u_proj, omega_vals):
    """
    Shift each k-column of the omega-k portrait by the Doppler amount.
    port: (n_om, n_k);  k_vals: (n_k,);  omega_vals: (n_om,).
    """
    n_om = port.shape[0]
    dom = omega_vals[1] - omega_vals[0]
    shift = (-np.rint(k_vals * u_proj / dom)).astype(int)   # (n_k,)
    idx = (np.arange(n_om)[:, None] - shift[None, :]) % n_om
    return np.take_along_axis(port, idx, axis=0)


def apply_doppler_3d(spec_3d, k_max, u_proj, om_max):
    """Scalar Doppler correction (isotropic approximation). Prefer apply_doppler_3d_vec."""
    n_om, n2, _ = spec_3d.shape
    k_num = n2 // 2
    kx = np.arange(-k_num, k_num, dtype=float)
    ky = np.arange(-k_num, k_num, dtype=float)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k_abs = np.sqrt(KX ** 2 + KY ** 2) / k_num * k_max
    shift = (-np.rint(k_abs * u_proj / om_max * n_om)).astype(int)
    idx = (np.arange(n_om)[:, None, None] - shift[None]) % n_om
    return np.take_along_axis(spec_3d, idx, axis=0)


def apply_doppler_3d_vec(spec_3d, k_max, Ux, Uy, om_max):
    """
    Vectorial Doppler correction: each (kx, ky) cell is shifted by
    Δω = kx·Ux + ky·Uy, which is exact for any number of wave systems
    simultaneously. Requires current vector (Ux, Uy) in radar image coords [m/s].
    """
    n_om, n2, _ = spec_3d.shape
    k_num = n2 // 2
    kx = np.arange(-k_num, k_num, dtype=float) / k_num * k_max
    ky = np.arange(-k_num, k_num, dtype=float) / k_num * k_max
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    delta_om = KX * Ux + KY * Uy                                      # (n2, n2)
    shift = (-np.rint(delta_om / om_max * n_om)).astype(int)
    idx = (np.arange(n_om)[:, None, None] - shift[None]) % n_om
    return np.take_along_axis(spec_3d, idx, axis=0)


# ── signal / noise separation ─────────────────────────────────────────────────

def separate_signal_noise(port_corr, k_vals, om_max, band=15):
    """
    Separate signal (within ±band frequency bins of ω=√(gk)) from noise.
    Returns (signal, noise), both shape (n_om, n_k).
    """
    n_om, n_k = port_corr.shape
    omega_ref = np.sqrt(9.81 * k_vals)
    om_idx = np.rint(omega_ref / om_max * n_om).astype(int).clip(0, n_om - 1)

    rows = np.arange(n_om)[:, None]
    signal_mask = np.abs(rows - om_idx[None, :]) <= band

    signal = np.where(signal_mask, port_corr, 0.0)
    noise = np.where(~signal_mask, port_corr, 0.0)
    return signal.astype(np.float32), noise.astype(np.float32)


def apply_mtf(signal, k_vals, exp=1.2):
    """Apply MTF weight k^{-exp} to the omega-k signal."""
    weight = np.ones_like(k_vals)
    nz = k_vals > 0
    weight[nz] = k_vals[nz] ** (-exp)
    return (signal * weight[None, :]).astype(np.float32)


def compute_snr(signal_mtf, noise):
    """Integrated signal-to-noise ratio."""
    noi = float(trapezoid(trapezoid(noise)))
    sig = float(trapezoid(trapezoid(signal_mtf)))
    return sig / noi if noi > 0 else 0.0


# ── 1-D frequency spectrum ────────────────────────────────────────────────────

def compute_frequency_spectrum(signal_mtf, k_vals, omega_vals):
    """
    Integrate MTF-weighted signal over k for each ω to get 1-D omega spectrum.
    Returns (s_omega, m0, T_peak, T_mean).
    T_peak and T_mean in seconds.
    """
    s_omega = gaussian_filter1d(
        np.maximum(trapezoid(signal_mtf, k_vals, axis=1), 0.0), sigma=2)

    m0 = float(trapezoid(s_omega, omega_vals)) if len(omega_vals) > 1 else 0.0

    peak_idx = int(np.nanargmax(s_omega))
    om_peak = omega_vals[peak_idx]
    T_peak = float(2 * np.pi / om_peak) if om_peak > 0 else 0.0

    denom = float(np.sum(s_omega))
    om_mean = float(np.sum(s_omega * omega_vals)) / denom if denom > 0 else 0.0
    T_mean = float(2 * np.pi / om_mean) if om_mean > 0 else T_peak

    return s_omega, m0, T_peak, T_mean


# ── directional spectrum ──────────────────────────────────────────────────────

def calc_spec2d(spec_3d_corr, omega_vals, k_max, num_area, band=7):
    """
    Build directional spectrum s_om_th(dir_bin, ω) by integrating the Doppler-
    corrected 3-D spectrum along a band around the dispersion curve for each
    (kx, ky) direction bin.

    Returns (s_om_th, peak_dir_deg, mean_dir_deg).
    Direction bin 0 corresponds to arctan2(KY=0, KX>0) = 0 rad (radar +x axis).
    """
    n_om, n2, _ = spec_3d_corr.shape
    k_num = n2 // 2

    kx = np.arange(-k_num, k_num, dtype=float)
    ky = np.arange(-k_num, k_num, dtype=float)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K_abs = np.sqrt(KX ** 2 + KY ** 2)
    k_abs = K_abs / k_num * k_max
    valid = (K_abs > 0) & (K_abs < k_num * 0.95)

    phi = np.arctan2(KY, KX)                                      # (-π, π]
    dir_idx = (np.rint(phi / (2 * np.pi) * num_area)).astype(int) % num_area

    omega_ref = np.sqrt(9.81 * k_abs)
    om_idx = np.rint(omega_ref / omega_vals[-1] * n_om).astype(int).clip(0, n_om - 1)

    s_om_th = np.zeros((num_area, n_om), dtype=np.float32)

    i_vals, j_vals = np.where(valid)
    for n in range(len(i_vals)):
        i, j = i_vals[n], j_vals[n]
        d = dir_idx[i, j]
        or_ij = om_idx[i, j]
        lo = max(0, or_ij - band)
        hi = min(n_om, or_ij + band + 1)
        s_om_th[d, lo:hi] += spec_3d_corr[lo:hi, i, j]

    s_om_th = gaussian_filter(s_om_th, sigma=[1.0, 1.0])

    # Peak direction
    s_dir = s_om_th.sum(axis=1)
    peak_idx = int(np.argmax(s_dir))
    peak_dir = float(peak_idx / num_area * 360)

    # Mean direction (circular)
    theta = np.linspace(0, 2 * np.pi, num_area, endpoint=False)
    cx = float(np.dot(s_dir, np.cos(theta)))
    cy = float(np.dot(s_dir, np.sin(theta)))
    mean_dir = float(np.degrees(np.arctan2(cy, cx)) % 360)

    return s_om_th, peak_dir, mean_dir
