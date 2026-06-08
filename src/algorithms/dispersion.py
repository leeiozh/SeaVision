import numpy as np

# ── Thresholds for calc_current_multiwave ────────────────────────────────────
_MULTI_DIR_HALF_DEG = 45.0   # half-width of directional cone per system [deg]
_MULTI_MIN_CELLS    = 5      # min spectral cells per system to enter regression
_MULTI_MIN_SV_RATIO = 0.05   # min σ_min/σ_max for 2×2 solve (ill-cond. guard)
_MULTI_MAX_CURRENT  = 3.0    # physical clip for residual current [m/s]
_MULTI_K_MIN_REL    = 0.08   # min k / k_max for a cell to be used in regression;
                              # very long swell (k → 0) gives Δω = k·U → 0 and
                              # contributes only noise to the regression


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

    Two-pass estimation (mirrors calc_current_vector):
      Pass 1 — argmax in wide window centred on ω_ref = √(gk).
               Captures large currents that would fall outside the narrow band.
               Wide window = max(band, n_om//4) bins.
      Pass 2 — energy centroid in narrow window (±band bins) centred on
               ω_ref + KX·Ucx0 + KY·Ucy0 from pass 1.  Sub-bin precision.

    Systems contribute equally to the joint regression (per-system weight
    normalisation), so a high-energy swell does not drown a low-energy wind-sea.

    Returns (Ucx, Ucy) [m/s] — true ocean current, East/North frame.
    Returns (None, None) if the problem is ill-conditioned (too few cells or
    singular-value ratio too small); caller falls back to calc_current_vector.

    Also returns sys_scatter: list of (k_arr, om_arr, w_arr) per system —
    centroid positions from pass 2, used for debug visualisation.
    """
    n_om, n2, _ = spec_3d_ship.shape
    k_num = n2 // 2

    kx = np.arange(-k_num, k_num, dtype=float) / k_num * k_max   # [rad/m]
    ky = np.arange(-k_num, k_num, dtype=float) / k_num * k_max
    KX, KY   = np.meshgrid(kx, ky, indexing='ij')
    K_abs    = np.sqrt(KX ** 2 + KY ** 2)
    k_phys   = K_abs
    phi      = np.arctan2(KY, KX)

    om_arr    = np.linspace(0, om_max, n_om)
    omega_ref = np.sqrt(9.81 * k_phys)

    dir_half = np.deg2rad(_MULTI_DIR_HALF_DEG)
    wide     = max(band, n_om // 4)   # same wide window as calc_current_vector pass 1

    # Precompute per-system spatial masks — reused by both passes
    _empty = (np.array([]), np.array([]), np.array([]))
    sys_masks = []
    for sys in systems_draft:
        k_lo_s       = sys["om_lo"] ** 2 / 9.81
        k_hi_s       = sys["om_hi"] ** 2 / 9.81
        theta_s      = np.deg2rad(sys["dir_deg"])
        dangle       = np.abs(phi - theta_s)
        dangle       = np.minimum(dangle, 2 * np.pi - dangle)
        k_min_useful = k_max * _MULTI_K_MIN_REL
        mask = (
            (k_phys > max(k_lo_s, k_min_useful)) & (k_phys < k_hi_s) &
            (dangle < dir_half) &
            (omega_ref < om_max * 0.92)
        )
        sys_masks.append(np.where(mask))

    def _run_pass(Ucx_p, Ucy_p, win, use_argmax):
        """Collect WLS equations for all systems.

        Ucx_p, Ucy_p — current prior used to centre each cell's window.
        win           — half-width of search window in ω bins.
        use_argmax    — True: take argmax (unbiased at window edges, for pass 1).
                        False: energy centroid (sub-bin precision, for pass 2).
        Returns (all_A, all_b, all_w, scatter_list).
        scatter_list always has len == len(systems_draft).
        """
        all_A, all_b, all_w = [], [], []
        scatter = []
        for s_idx, _ in enumerate(systems_draft):
            i_v, j_v = sys_masks[s_idx]
            if len(i_v) < _MULTI_MIN_CELLS:
                scatter.append(_empty)
                continue
            sA, sb, sw = [], [], []
            k_pts, om_pts, w_pts = [], [], []
            for i, j in zip(i_v, j_v):
                om_ref_ij = float(omega_ref[i, j])
                om_ctr    = om_ref_ij + float(KX[i, j]) * Ucx_p + float(KY[i, j]) * Ucy_p
                ci = int(round(om_ctr / om_max * (n_om - 1)))
                ci = max(0, min(n_om - 1, ci))
                lo = max(1,    ci - win)
                hi = min(n_om, ci + win + 1)
                if hi <= lo:
                    continue
                sl = spec_3d_ship[lo:hi, i, j].astype(np.float64)
                if use_argmax:
                    pk       = int(np.argmax(sl))
                    peak_val = float(sl[pk])
                    if peak_val <= 0:
                        continue
                    om_c = float(om_arr[lo + pk])
                    wt   = peak_val / max(float(K_abs[i, j]), 1e-6)
                    w_sc = peak_val
                else:
                    w = sl.sum()
                    if w <= 0:
                        continue
                    om_c = float(np.dot(sl, om_arr[lo:hi])) / w
                    wt   = w / max(float(K_abs[i, j]), 1e-6)
                    w_sc = w
                sA.append([float(KX[i, j]), float(KY[i, j])])
                sb.append(om_c - om_ref_ij)
                sw.append(wt)
                k_pts.append(float(k_phys[i, j]))
                om_pts.append(om_c)
                w_pts.append(w_sc)
            if len(sA) < _MULTI_MIN_CELLS:
                scatter.append(_empty)
                continue
            w_arr = np.array(sw);  w_arr /= w_arr.sum()
            all_A.extend(sA);  all_b.extend(sb);  all_w.extend(w_arr.tolist())
            scatter.append((np.array(k_pts), np.array(om_pts), np.array(w_pts)))
        return all_A, all_b, all_w, scatter

    def _solve(all_A, all_b, all_w):
        if len(all_A) < 4:
            return None, None
        A   = np.array(all_A)
        b   = np.array(all_b)
        wsq = np.sqrt(np.array(all_w))
        _, sv, _ = np.linalg.svd(A * wsq[:, None], full_matrices=False)
        if sv[-1] < _MULTI_MIN_SV_RATIO * sv[0]:
            return None, None
        res, _, _, _ = np.linalg.lstsq(A * wsq[:, None], b * wsq, rcond=None)
        return float(res[0]), float(res[1])

    def _clip(Ucx, Ucy):
        mag = float(np.hypot(Ucx, Ucy))
        if mag > _MULTI_MAX_CURRENT:
            f = _MULTI_MAX_CURRENT / mag
            return Ucx * f, Ucy * f
        return Ucx, Ucy

    empty_scatter = [_empty for _ in systems_draft]

    # Pass 1: argmax in wide window, no current prior
    A1, b1, w1, _ = _run_pass(0.0, 0.0, win=wide, use_argmax=True)
    Ucx0, Ucy0 = _solve(A1, b1, w1)
    if Ucx0 is None:
        return None, None, empty_scatter
    Ucx0, Ucy0 = _clip(Ucx0, Ucy0)

    # Pass 2: centroid in narrow window centred on pass-1 estimate
    A2, b2, w2, sys_scatter = _run_pass(Ucx0, Ucy0, win=band, use_argmax=False)
    Ucx, Ucy = _solve(A2, b2, w2)
    if Ucx is None:
        # Pass 2 degenerate (very rare); keep pass-1 result
        return Ucx0, Ucy0, sys_scatter
    Ucx, Ucy = _clip(Ucx, Ucy)

    return Ucx, Ucy, sys_scatter
