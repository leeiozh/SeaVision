import numpy as np


def calc_vco(port, om_max, k_max):
    """
    Estimate radial Doppler velocity projection via weighted least-squares fit
    of the dispersion relation ω = √(gk) + k·vco (Senet et al. 2001,
    Young et al. 1985, Carrasco et al. 2012).

    For each wavenumber bin k_i (within the valid k range), the energy-weighted
    centroid of ω is computed within a ±band window around the theoretical
    dispersion curve.  Then vco is solved via:

        vco = Σ_k [w_k · k · (ω̄(k) − √(gk))] / Σ_k [w_k · k²]

    where w_k = Σ_ω P(ω, k_i) is the integrated power at that wavenumber.

    port: (n_om, k_num), positive-frequency ω-k portrait.
    Returns vco [m/s].
    """
    n_om, k_num = port.shape
    om_arr = np.linspace(0, om_max, n_om)
    k_arr = np.linspace(0, k_max, k_num)

    k_lo = max(1, round(0.08 * k_num))
    k_hi = round(0.65 * k_num)
    om_lo = max(1, round(0.04 * n_om))
    band = max(7, round(0.12 * n_om))   # ±band bins around ω=√(gk)

    weights = np.zeros(k_num)
    om_centroid = np.zeros(k_num)

    for ki in range(k_lo, k_hi):
        k = k_arr[ki]
        om_ref = np.sqrt(9.81 * k)
        ref_idx = int(round(om_ref / om_max * (n_om - 1)))
        lo = max(om_lo, ref_idx - band)
        hi = min(n_om, ref_idx + band + 1)
        if hi <= lo:
            continue
        sl = port[lo:hi, ki].astype(float)
        w = sl.sum()
        if w <= 0:
            continue
        weights[ki] = w
        om_centroid[ki] = float(np.dot(sl, om_arr[lo:hi])) / w

    valid = (weights > 0) & (k_arr > 0)
    if not np.any(valid):
        return 0.0

    k_v = k_arr[valid]
    w_v = weights[valid]
    residual = om_centroid[valid] - np.sqrt(9.81 * k_v)

    num = float(np.dot(w_v * k_v, residual))
    den = float(np.dot(w_v, k_v ** 2))
    return float(num / den) if den > 0 else 0.0


def calc_current_vector(spec_3d, k_max, om_max):
    """
    Estimate current velocity vector (Ux, Uy) [m/s] from the averaged 3D spectrum
    via weighted least-squares fit of the dispersion centroid displacement:

        ω̄(kx, ky) − √(g·|k|) = kx·Ux + ky·Uy

    Each valid (kx, ky) cell contributes one equation weighted by its integrated
    spectral power. Returns (Ux, Uy) in radar image coordinates [m/s].
    """
    n_om, n2, _ = spec_3d.shape
    k_num = n2 // 2
    band = max(8, round(0.15 * n_om))

    kx_arr = np.arange(-k_num, k_num, dtype=float) / k_num * k_max
    ky_arr = np.arange(-k_num, k_num, dtype=float) / k_num * k_max
    KX, KY = np.meshgrid(kx_arr, ky_arr, indexing='ij')
    K_abs = np.sqrt(KX ** 2 + KY ** 2)

    om_arr = np.linspace(0, om_max, n_om)
    omega_ref = np.sqrt(9.81 * K_abs)
    ref_idx = np.clip(np.rint(omega_ref / om_max * (n_om - 1)).astype(int), 0, n_om - 1)
    om_lo = np.maximum(1, ref_idx - band)
    om_hi = np.minimum(n_om, ref_idx + band + 1)

    k_lo = k_max * 0.08
    k_hi = k_max * 0.65
    i_vals, j_vals = np.where((K_abs > k_lo) & (K_abs < k_hi))

    eq_A = []
    eq_b = []
    eq_w = []

    for i, j in zip(i_vals, j_vals):
        lo, hi = int(om_lo[i, j]), int(om_hi[i, j])
        if hi <= lo:
            continue
        sl = spec_3d[lo:hi, i, j].astype(np.float64)
        w = sl.sum()
        if w <= 0:
            continue
        om_centroid = float(np.dot(sl, om_arr[lo:hi])) / w
        eq_A.append([float(KX[i, j]), float(KY[i, j])])
        eq_b.append(om_centroid - float(omega_ref[i, j]))
        eq_w.append(w)

    if len(eq_A) < 10:
        return 0.0, 0.0

    A = np.array(eq_A)
    b = np.array(eq_b)
    w = np.sqrt(np.array(eq_w))
    result, _, _, _ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    return float(result[0]), float(result[1])


def dispersion_curve(k, vco, depth=None):
    """ω = √(gk·tanh(kd)) + k·vco  (deep water: depth=None)."""
    if depth is None:
        return np.sqrt(9.81 * k) + k * vco
    return np.sqrt(9.81 * k * np.tanh(k * depth)) + k * vco
