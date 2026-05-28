import numpy as np

# ── Thresholds for calc_current_multiwave ────────────────────────────────────
_MULTI_DIR_HALF_DEG = 45.0   # half-width of directional cone per system [deg]
_MULTI_MIN_CELLS    = 5      # min spectral cells per system to enter regression
_MULTI_MIN_SV_RATIO = 0.05   # min σ_min/σ_max for 2×2 solve (ill-cond. guard)
_MULTI_MAX_CURRENT  = 2.55   # physical clip for residual current [m/s]
_MULTI_K_MIN_REL    = 0.08   # min k / k_max for a cell to be used in regression;
                              # very long swell (k → 0) gives Δω = k·U → 0 and
                              # contributes only noise to the regression


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


def calc_current_vector(spec_3d, k_max, om_max, band, sog=0.0, cog_deg=0.0, max_current=2.55):
    """
    Two-pass estimation of (Ux, Uy) [m/s] from the 3-D spectrum.
    Equation: ω_peak(kx,ky) − √(g|k|) = kx·Ux + ky·Uy, solved by weighted least-squares.

    Pass 1 — argmax in wide window centred on ship-velocity-shifted dispersion curve.
              The ship-velocity prior ensures the window contains the wave energy even
              at high SOG; argmax finds the peak without centroid bias at window edges.
    Pass 2 — energy centroid in narrow window centred on Pass-1 result.
              Gives sub-bin precision around the already-well-centred peak.

    After each pass the true ocean current (Ux−Ux_ship, Uy−Uy_ship) is clipped to
    max_current [m/s] so physically implausible outliers cannot accumulate.

    Returns (Ux, Uy) [m/s] in geographic (East, North) frame.
    (Ux, Uy) = apparent velocity = water_current + ship_velocity.
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
    k_hi = k_max * 0.45
    # Exclude cells where the unshifted dispersion curve exceeds the Nyquist ω.
    # Those cells always produce a negative ω-residual (ω_ref is clipped to n_om-1)
    # and systematically bias the velocity regression.
    i_vals, j_vals = np.where((K_abs > k_lo) & (K_abs < k_hi) & (omega_ref < om_max))

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
            # Divide by K_abs: outer rings have more cells (∝k), normalise to equal
            # ring contribution so high-k cells don't dominate the regression.
            eq_w.append(peak_val / float(K_abs[i, j]))

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
            eq_w.append(w / float(K_abs[i, j]))

        if len(eq_A) < 10:
            return Ux_p, Uy_p

        A = np.array(eq_A)
        b = np.array(eq_b)
        wsq = np.sqrt(np.array(eq_w))
        res, _, _, _ = np.linalg.lstsq(A * wsq[:, None], b * wsq, rcond=None)
        return float(res[0]), float(res[1])

    def _clip(Ux_r, Uy_r):
        dx, dy = Ux_r - Ux_ship, Uy_r - Uy_ship
        mag = float(np.hypot(dx, dy))
        if mag > max_current:
            f = max_current / mag
            return float(Ux_ship + dx * f), float(Uy_ship + dy * f)
        return Ux_r, Uy_r

    # Pass 1: argmax, wide window, centred on ship-shifted dispersion curve
    wide = max(band, n_om // 4)
    Ux, Uy = _argmax_pass(Ux_ship, Uy_ship, wide)
    Ux, Uy = _clip(Ux, Uy)

    # Pass 2: centroid, narrow window, centred on Pass-1 result (sub-bin refinement)
    Ux, Uy = _centroid_pass(Ux, Uy, band)
    Ux, Uy = _clip(Ux, Uy)

    return Ux, Uy


def calc_current_multiwave(spec_3d_ship, k_max, om_max, systems_draft, band):
    """
    Per-system Doppler estimation from ship-speed-corrected 3-D spectrum.

    spec_3d_ship  — spec_3d after Doppler-correcting for ship velocity only.
                    Residual Doppler = true ocean current.
    systems_draft — list of dicts from find_system_dirs:
                    {"om", "om_lo", "om_hi", "dir_deg", ...}
    band          — ±band bins used throughout the pipeline.

    For each system a directional cone [dir_deg ± _MULTI_DIR_HALF_DEG] and
    frequency band [om_lo, om_hi] define a spectral mask.  Within that mask,
    the energy-centroid ω gives  kx·Ucx + ky·Ucy = Δω  per cell.

    Systems contribute equally to the joint regression (per-system weight
    normalisation), so a high-energy swell does not drown a low-energy wind-sea.

    Returns (Ucx, Ucy) [m/s] — true ocean current, East/North frame.
    Returns (None, None) if the problem is ill-conditioned (too few cells or
    singular-value ratio too small); caller falls back to calc_current_vector.

    Also returns sys_scatter: list of (k_arr, om_arr, w_arr) per system —
    centroid positions used for debug visualisation.
    """
    n_om, n2, _ = spec_3d_ship.shape
    k_num = n2 // 2

    kx = np.arange(-k_num, k_num, dtype=float) / k_num * k_max   # [rad/m]
    ky = np.arange(-k_num, k_num, dtype=float) / k_num * k_max
    KX, KY   = np.meshgrid(kx, ky, indexing='ij')
    K_abs    = np.sqrt(KX ** 2 + KY ** 2)       # already physical [rad/m]
    k_phys   = K_abs                             # alias for clarity
    phi      = np.arctan2(KY, KX)               # direction, math convention

    om_arr    = np.linspace(0, om_max, n_om)
    omega_ref = np.sqrt(9.81 * k_phys)

    dir_half = np.deg2rad(_MULTI_DIR_HALF_DEG)

    all_A, all_b, all_w = [], [], []
    sys_scatter = []                             # [(k_pts, om_pts, w_pts), ...]

    for sys in systems_draft:
        k_lo_s  = sys["om_lo"] ** 2 / 9.81
        k_hi_s  = sys["om_hi"] ** 2 / 9.81
        theta_s = np.deg2rad(sys["dir_deg"])

        dangle = np.abs(phi - theta_s)
        dangle = np.minimum(dangle, 2 * np.pi - dangle)

        k_min_useful = k_max * _MULTI_K_MIN_REL
        mask = (
            (k_phys    > max(k_lo_s, k_min_useful)) & (k_phys < k_hi_s) &
            (dangle    < dir_half) &
            (omega_ref < om_max * 0.92)          # avoid Nyquist aliasing
        )
        i_v, j_v = np.where(mask)
        if len(i_v) < _MULTI_MIN_CELLS:
            sys_scatter.append((np.array([]), np.array([]), np.array([])))
            continue

        sys_A, sys_b, sys_w = [], [], []
        k_pts, om_pts, w_pts = [], [], []

        for i, j in zip(i_v, j_v):
            om_ref_ij = float(omega_ref[i, j])
            ci = int(round(om_ref_ij / om_max * (n_om - 1)))
            ci = max(1, min(n_om - 1, ci))
            lo = max(1,    ci - band)
            hi = min(n_om, ci + band + 1)
            if hi <= lo:
                continue
            sl = spec_3d_ship[lo:hi, i, j].astype(np.float64)
            w  = sl.sum()
            if w <= 0:
                continue
            om_c = float(np.dot(sl, om_arr[lo:hi])) / w
            sys_A.append([float(KX[i, j]), float(KY[i, j])])
            sys_b.append(om_c - om_ref_ij)
            sys_w.append(w / max(float(K_abs[i, j]), 1e-6))
            k_pts.append(float(k_phys[i, j]))
            om_pts.append(om_c)
            w_pts.append(w)

        if len(sys_A) < _MULTI_MIN_CELLS:
            sys_scatter.append((np.array([]), np.array([]), np.array([])))
            continue

        # Normalise per-system: each system contributes equally regardless of energy
        w_arr = np.array(sys_w);  w_arr /= w_arr.sum()

        all_A.extend(sys_A);  all_b.extend(sys_b);  all_w.extend(w_arr.tolist())
        sys_scatter.append((np.array(k_pts), np.array(om_pts), np.array(w_pts)))

    if len(all_A) < 4:
        return None, None, sys_scatter

    A   = np.array(all_A)
    b   = np.array(all_b)
    wsq = np.sqrt(np.array(all_w))

    _, sv, _ = np.linalg.svd(A * wsq[:, None], full_matrices=False)
    if sv[-1] < _MULTI_MIN_SV_RATIO * sv[0]:
        return None, None, sys_scatter

    res, _, _, _ = np.linalg.lstsq(A * wsq[:, None], b * wsq, rcond=None)
    Ucx, Ucy = float(res[0]), float(res[1])

    mag = float(np.hypot(Ucx, Ucy))
    if mag > _MULTI_MAX_CURRENT:
        f = _MULTI_MAX_CURRENT / mag
        Ucx, Ucy = Ucx * f, Ucy * f

    return Ucx, Ucy, sys_scatter


def dispersion_curve(k, vco, depth=None):
    """ω = √(gk·tanh(kd)) + k·vco  (deep water: depth=None)."""
    if depth is None:
        return np.sqrt(9.81 * k) + k * vco
    return np.sqrt(9.81 * k * np.tanh(k * depth)) + k * vco
