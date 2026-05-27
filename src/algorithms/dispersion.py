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


def calc_current_vector(spec_3d, k_max, om_max, band, sog=0.0, cog_deg=0.0):
    """
    Two-pass estimation of (Ux, Uy) [m/s] from the 3-D spectrum.
    Equation: ω_peak(kx,ky) − √(g|k|) = kx·Ux + ky·Uy, solved by weighted least-squares.

    Pass 1 — argmax in wide window centred on ship-velocity-shifted dispersion curve.
              The ship-velocity prior ensures the window contains the wave energy even
              at high SOG; argmax finds the peak without centroid bias at window edges.
    Pass 2 — energy centroid in narrow window centred on Pass-1 result.
              Gives sub-bin precision around the already-well-centred peak.

    Returns (Ux, Uy) [m/s] in geographic (East, North) frame.
    (Ux, Uy) = apparent velocity = water_current − ship_velocity.
    """
    n_om, n2, _ = spec_3d.shape
    k_num = n2 // 2

    kx_arr = np.arange(-k_num, k_num, dtype=float) / k_num * k_max
    ky_arr = np.arange(-k_num, k_num, dtype=float) / k_num * k_max
    KX, KY = np.meshgrid(kx_arr, ky_arr, indexing='ij')
    K_abs = np.sqrt(KX ** 2 + KY ** 2)

    om_arr = np.linspace(0, om_max, n_om)
    omega_ref = np.sqrt(9.81 * K_abs)

    k_lo = k_max * 0.08
    k_hi = k_max * 0.65
    i_vals, j_vals = np.where((K_abs > k_lo) & (K_abs < k_hi))

    # Ship moves at (sog, cog_deg) → apparent velocity ≈ +ship_velocity in radar frame
    cog_rad = np.deg2rad(cog_deg)
    Ux_ship = sog * np.sin(cog_rad)
    Uy_ship = sog * np.cos(cog_rad)

    def _argmax_pass(Ux_p, Uy_p, win):
        """Peak position (argmax) within window — no centroid bias at window edges."""
        eq_A, eq_b, eq_w = [], [], []
        for i, j in zip(i_vals, j_vals):
            om_ctr = float(omega_ref[i, j] + KX[i, j] * Ux_p + KY[i, j] * Uy_p)
            ci = int(round(om_ctr / om_max * (n_om - 1)))
            ci = max(0, min(n_om - 1, ci))
            lo = max(1, ci - win)
            hi = min(n_om, ci + win + 1)
            if hi <= lo:
                continue
            sl = spec_3d[lo:hi, i, j].astype(np.float64)
            pk = int(np.argmax(sl))
            peak_val = float(sl[pk])
            if peak_val <= 0:
                continue
            eq_A.append([float(KX[i, j]), float(KY[i, j])])
            eq_b.append(float(om_arr[lo + pk]) - float(omega_ref[i, j]))
            eq_w.append(peak_val)

        if len(eq_A) < 10:
            return Ux_p, Uy_p

        A = np.array(eq_A)
        b = np.array(eq_b)
        wsq = np.sqrt(np.array(eq_w))
        res, _, _, _ = np.linalg.lstsq(A * wsq[:, None], b * wsq, rcond=None)
        return float(res[0]), float(res[1])

    def _centroid_pass(Ux_p, Uy_p, win):
        """Energy centroid in narrow window — sub-bin precision when well-centred."""
        eq_A, eq_b, eq_w = [], [], []
        for i, j in zip(i_vals, j_vals):
            om_ctr = float(omega_ref[i, j] + KX[i, j] * Ux_p + KY[i, j] * Uy_p)
            ci = int(round(om_ctr / om_max * (n_om - 1)))
            ci = max(0, min(n_om - 1, ci))
            lo = max(1, ci - win)
            hi = min(n_om, ci + win + 1)
            if hi <= lo:
                continue
            sl = spec_3d[lo:hi, i, j].astype(np.float64)
            w = sl.sum()
            if w <= 0:
                continue
            om_c = float(np.dot(sl, om_arr[lo:hi])) / w
            eq_A.append([float(KX[i, j]), float(KY[i, j])])
            eq_b.append(om_c - float(omega_ref[i, j]))
            eq_w.append(w)

        if len(eq_A) < 10:
            return Ux_p, Uy_p

        A = np.array(eq_A)
        b = np.array(eq_b)
        wsq = np.sqrt(np.array(eq_w))
        res, _, _, _ = np.linalg.lstsq(A * wsq[:, None], b * wsq, rcond=None)
        return float(res[0]), float(res[1])

    # Pass 1: argmax, wide window, centred on ship-shifted dispersion curve
    wide = max(band, n_om // 4)
    Ux, Uy = _argmax_pass(Ux_ship, Uy_ship, wide)

    # Pass 2: centroid, narrow window, centred on Pass-1 result (sub-bin refinement)
    Ux, Uy = _centroid_pass(Ux, Uy, band)

    return Ux, Uy


def dispersion_curve(k, vco, depth=None):
    """ω = √(gk·tanh(kd)) + k·vco  (deep water: depth=None)."""
    if depth is None:
        return np.sqrt(9.81 * k) + k * vco
    return np.sqrt(9.81 * k * np.tanh(k * depth)) + k * vco
