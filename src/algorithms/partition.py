import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter


def calc_wspd(bck):
    """
    Estimate wind direction and mean backscatter intensity from the full image.
    Fits  I(θ) = a + b·cos²(0.5·(θ − c))  to azimuthal mean intensity.
    Returns (sig, wdir_deg).
    """

    sig = float(np.mean(bck))
    intensity = bck.mean(axis=1).astype(float)
    n = len(intensity)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    def _model(x, a, b, c):
        return a + b * np.cos(0.5 * (x - c)) ** 2

    try:
        p0 = [float(np.mean(intensity)),
              max(float(np.std(intensity)), 0.1),
              0.0]
        popt, _ = curve_fit(
            _model, theta, intensity, p0=p0,
            bounds=([0, 0, -np.pi], [np.inf, np.inf, 2 * np.pi]),
            maxfev=2000,
        )
        wdir = float(np.degrees(popt[2]) % 360)
    except Exception:
        wdir = 0.0

    return sig, wdir


def calc_partitions(s_om_th, omega_vals, dir_array, wdir, swh):
    """
    Partition the directional spectrum s_om_th (num_area × n_om) into up to 3
    wave systems by iterative peak extraction.

    dir_array: geographic directions for each direction bin [deg], length num_area.
    wdir: wind direction [deg] — used to help label wind sea.
    swh: total significant wave height [m].

    Returns dict:
      { "n_sys": int,
        "w_s":  {"h_s", "t_p", "t_m", "d_p", "d_m"} or None,
        "sw_1": { … } or None,
        "sw_2": { … } or None }
    """
    # Two systems are considered the same if they are closer than these thresholds
    # in BOTH direction AND period simultaneously.
    MIN_DIR_SEP = 30.0    # degrees
    MIN_PER_RATIO = 1.15  # T_large / T_small must exceed this to be distinct

    # Minimum fraction of total energy a peak must carry to be kept as a system.
    # The blanking window captures only ~50% of a system's energy, so set lower than
    # the naïve 1/N_sys estimate (was 0.30 — too restrictive for 3-system cases).
    MIN_ENERGY_FRAC = 0.10

    num_area, n_om = s_om_th.shape
    dir_array = np.asarray(dir_array, dtype=float)
    omega_vals = np.asarray(omega_vals, dtype=float)

    spec = gaussian_filter(s_om_th.astype(float), sigma=[1, 2])
    total_energy = float(spec.sum())

    if total_energy <= 0 or swh <= 0:
        return {"n_sys": 0, "w_s": None, "sw_1": None, "sw_2": None}

    # Blanking radius: 15 % of each axis
    om_r = max(2, round(n_om * 0.15))
    dir_r = max(1, round(num_area * 0.15))

    def _dir_diff_local(a, b):
        d = abs(a - b) % 360
        return min(d, 360 - d)

    def _too_similar(d_p_new, t_p_new):
        for s in systems:
            dir_sep = _dir_diff_local(d_p_new, s['d_p'])
            # Direction alone is sufficient — enforce minimum separation regardless of period
            if dir_sep < MIN_DIR_SEP:
                return True
            # Also merge if periods are too similar even when directions differ
            t_lo = max(min(t_p_new, s['t_p']), 0.1)
            t_hi = max(t_p_new, s['t_p'])
            if t_hi / t_lo < MIN_PER_RATIO:
                return True
        return False

    systems = []

    for _ in range(3):
        if spec.max() <= 0:
            break

        # Noise floor stop: peak must be > 5× median background
        pos = spec[spec > 0]
        if pos.size == 0:
            break
        if spec.max() < np.median(pos) * 5.0:
            break

        # Remaining energy fraction (before blanking this system)
        rem_energy = float(spec.sum())
        if rem_energy < total_energy * MIN_ENERGY_FRAC:
            break

        p = int(np.argmax(spec))
        d_idx, om_idx = divmod(p, n_om)

        om_peak = omega_vals[om_idx] if om_idx < len(omega_vals) else 0.0
        t_p = float(2 * np.pi / om_peak) if om_peak > 0 else 0.0
        d_p = float(dir_array[d_idx]) if d_idx < len(dir_array) else 0.0

        # Snapshot pre-blank energy and directional slice — BEFORE zeroing
        sys_energy = float(
            spec[
                max(0, d_idx - dir_r): min(num_area, d_idx + dir_r + 1),
                max(0, om_idx - om_r): min(n_om, om_idx + om_r + 1),
            ].sum()
        )
        s1d_preblank = spec[d_idx, :].copy()

        # Blank peak region (always, even if we end up skipping this system)
        for di in range(-dir_r, dir_r + 1):
            di_mod = (d_idx + di) % num_area
            lo = max(0, om_idx - om_r)
            hi = min(n_om, om_idx + om_r + 1)
            spec[di_mod, lo:hi] = 0.0

        # Skip if too similar to an already-found system
        if _too_similar(d_p, t_p):
            continue

        # Skip if this system carries too little energy (phantom peak)
        frac = sys_energy / total_energy
        if frac < MIN_ENERGY_FRAC:
            continue

        # Mean period from pre-blank directional slice
        s1d_sum = float(s1d_preblank.sum())
        if s1d_sum > 0 and np.any(omega_vals > 0):
            om_mean = float(np.dot(s1d_preblank, omega_vals[:n_om])) / s1d_sum
            t_m = float(2 * np.pi / om_mean) if om_mean > 0 else t_p
        else:
            t_m = t_p

        systems.append({
            "frac": frac,
            "t_p": t_p,
            "t_m": t_m,
            "d_p": d_p,
            "d_m": d_p,
        })

    if not systems:
        return {"n_sys": 0, "w_s": None, "sw_1": None, "sw_2": None}

    # Normalize fracs so sum(h_s_i²) = swh² (energy conservation).
    # Raw sys_energy values cover only the blanking window, so their sum < total_energy.
    # Normalising redistributes the background energy proportionally.
    total_frac = sum(s["frac"] for s in systems)
    for s in systems:
        s["h_s"] = float(swh * np.sqrt(s["frac"] / total_frac))

    # Classify wind sea vs swell using wdir.
    # Wind sea: direction within 45° of wdir, prefer shortest T among candidates.
    # No fallback — if no system matches wdir, all systems are swell (w_s = None).
    WIND_DIR_THRESH = 45.0

    def _dir_diff(a, b):
        d = abs(a - b) % 360
        return min(d, 360 - d)

    wind_candidates = [
        (i, s) for i, s in enumerate(systems)
        if _dir_diff(s["d_p"], wdir) <= WIND_DIR_THRESH
    ]
    if wind_candidates:
        wind_idx = min(wind_candidates, key=lambda x: x[1]["t_p"])[0]
        wind_sys = systems[wind_idx]
        swell_sys = sorted(
            [s for i, s in enumerate(systems) if i != wind_idx],
            key=lambda s: s["t_p"],
        )
    else:
        wind_sys = None
        swell_sys = sorted(systems, key=lambda s: s["t_p"])

    return {
        "n_sys": len(systems),
        "w_s":   wind_sys,
        "sw_1":  swell_sys[0] if len(swell_sys) > 0 else None,
        "sw_2":  swell_sys[1] if len(swell_sys) > 1 else None,
    }
