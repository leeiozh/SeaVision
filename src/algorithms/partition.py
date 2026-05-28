import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter, gaussian_filter1d

# ── Thresholds for calc_partitions ───────────────────────────────────────────
_PART_MIN_DIR_SEP     = 20.0   # min angular separation [deg] for two distinct systems
                                # (was 30°; with N_DIRS=36 → 10°/bin, two bins = 20°
                                # is already the grid resolution, so 20° is the minimum
                                # meaningful separation)
_PART_MIN_PER_RATIO   = 1.1    # T_large / T_small to consider systems distinct
                                # (was 1.2; real multi-system spectra often have
                                # adjacent peaks with T_ratio ≈ 1.1–1.15, e.g. 13s/14.9s)
_PART_MIN_ENERGY_FRAC = 0.05   # min window-energy / total_energy for a valid system
                                # (was 0.30 / 0.08; background inflates window energy
                                # so each of 3 equal systems gets only ~10-12% of total;
                                # use _PART_NOISE_SNR as the primary stop criterion)
_PART_NOISE_SNR       = 3.0    # peak must exceed median background × this factor
_PART_BLANK_FRAC      = 0.08   # blanking radius [fraction of each axis]
                                # (was 0.15 = 19 bins at N_SHOTS//2=128 → ±0.19 rad/s;
                                # that spans from T=16s all the way down to T=10s in one
                                # blanking step; 0.08 = 10 bins = ±0.10 rad/s)
_PART_WIND_DIR_THRESH = 45.0   # max angle from wdir for wind-sea classification [deg]

# ── Thresholds for find_freq_peaks ───────────────────────────────────────────
_FPEAK_MIN_PROM_REL  = 0.15    # min prominence [relative to global max]
_FPEAK_BAND_HALF_FRAC = 0.10   # half-width of each peak's band [fraction of n_om]
_FPEAK_MIN_SEP_FRAC  = 0.12    # min separation between peaks [fraction of n_om]
_FPEAK_SMOOTH_SIGMA  = 2.0     # Gaussian pre-smoothing [bins]
_FPEAK_MAX           = 3       # max number of peaks

# ── Thresholds for find_system_dirs ──────────────────────────────────────────
_SDIR_SMOOTH_SIGMA   = 1.0     # Gaussian smoothing of directional profile [bins]


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


def find_freq_peaks(s_omega, omega_vals):
    """
    Find up to _FPEAK_MAX peaks in 1-D frequency spectrum.

    Returns list of dicts sorted by amplitude (highest first):
      {"idx": int, "om": float, "om_lo": float, "om_hi": float, "amp": float}
    om_lo/om_hi define the band attributed to this peak (used for direction search
    and per-system current estimation).
    """
    s = gaussian_filter1d(np.maximum(s_omega, 0.0).astype(float),
                          sigma=_FPEAK_SMOOTH_SIGMA)
    s[0] = 0.0  # ignore DC / zero-freq bin
    if s.max() <= 0:
        return []

    n         = len(s)
    min_sep   = max(2, int(round(_FPEAK_MIN_SEP_FRAC   * n)))
    band_half = max(1, int(round(_FPEAK_BAND_HALF_FRAC * n)))
    threshold = _FPEAK_MIN_PROM_REL * float(s.max())

    peaks  = []
    s_work = s.copy()

    for _ in range(_FPEAK_MAX):
        if float(s_work.max()) <= threshold:
            break
        idx    = int(np.argmax(s_work))
        amp    = float(s_work[idx])
        lo_idx = max(0,     idx - band_half)
        hi_idx = min(n - 1, idx + band_half)
        peaks.append({
            "idx":   idx,
            "om":    float(omega_vals[idx]),
            "om_lo": float(omega_vals[lo_idx]),
            "om_hi": float(omega_vals[hi_idx]),
            "amp":   amp,
        })
        s_work[max(0, idx - min_sep): min(n, idx + min_sep + 1)] = 0.0

    peaks.sort(key=lambda p: -p["amp"])
    return peaks


def find_system_dirs(s_om_th, freq_peaks, omega_vals, dir_array):
    """
    For each frequency peak, find the dominant propagation direction by summing
    s_om_th over the peak's omega band and finding the angular argmax.

    s_om_th   — directional spectrum (num_dirs × n_om).
    freq_peaks — output of find_freq_peaks.
    dir_array  — geographic direction per bin [deg], math convention (0=East).

    Returns list of dicts (same fields as freq_peaks plus 'dir_deg', 'dir_idx').
    Order matches freq_peaks (amplitude descending).
    """
    if not freq_peaks:
        return []

    n_om  = s_om_th.shape[1]
    om_max = float(omega_vals[-1])
    result = []

    for peak in freq_peaks:
        lo_bin = max(0,      int(round(peak["om_lo"] / om_max * (n_om - 1))))
        hi_bin = min(n_om-1, int(round(peak["om_hi"] / om_max * (n_om - 1))))
        if hi_bin <= lo_bin:
            hi_bin = min(n_om - 1, lo_bin + 1)

        dir_prof = s_om_th[:, lo_bin:hi_bin + 1].sum(axis=1).astype(float)
        dir_prof = gaussian_filter1d(dir_prof, sigma=_SDIR_SMOOTH_SIGMA)

        if dir_prof.max() <= 0:
            continue

        dir_idx = int(np.argmax(dir_prof))
        result.append({
            **peak,
            "dir_deg": float(dir_array[dir_idx]),
            "dir_idx": dir_idx,
        })

    return result


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
    num_area, n_om = s_om_th.shape
    dir_array  = np.asarray(dir_array,  dtype=float)
    omega_vals = np.asarray(omega_vals, dtype=float)

    spec         = gaussian_filter(s_om_th.astype(float), sigma=[1, 2])
    total_energy = float(spec.sum())

    if total_energy <= 0 or swh <= 0:
        return {"n_sys": 0, "w_s": None, "sw_1": None, "sw_2": None}

    om_r  = max(2, round(n_om    * _PART_BLANK_FRAC))
    dir_r = max(1, round(num_area * _PART_BLANK_FRAC))

    def _dir_diff(a, b):
        d = abs(a - b) % 360
        return min(d, 360 - d)

    def _too_similar(d_p_new, t_p_new):
        for s in systems:
            if _dir_diff(d_p_new, s['d_p']) < _PART_MIN_DIR_SEP:
                return True
            t_lo = max(min(t_p_new, s['t_p']), 0.1)
            t_hi = max(t_p_new, s['t_p'])
            if t_hi / t_lo < _PART_MIN_PER_RATIO:
                return True
        return False

    systems = []

    for _ in range(5):   # up to 3 accepted; extras consumed by near-duplicate blanking
        if spec.max() <= 0:
            break
        pos = spec[spec > 0]
        if pos.size == 0:
            break
        if spec.max() < np.median(pos) * _PART_NOISE_SNR:
            break
        rem_energy = float(spec.sum())
        if rem_energy < total_energy * _PART_MIN_ENERGY_FRAC:
            break

        p = int(np.argmax(spec))
        d_idx, om_idx = divmod(p, n_om)

        om_peak = omega_vals[om_idx] if om_idx < len(omega_vals) else 0.0
        t_p     = float(2 * np.pi / om_peak) if om_peak > 0 else 0.0
        d_p     = float(dir_array[d_idx])    if d_idx < len(dir_array) else 0.0

        sys_energy = float(
            spec[
                max(0, d_idx - dir_r): min(num_area, d_idx + dir_r + 1),
                max(0, om_idx - om_r): min(n_om,     om_idx + om_r + 1),
            ].sum()
        )
        s1d_preblank = spec[d_idx, :].copy()

        for di in range(-dir_r, dir_r + 1):
            di_mod = (d_idx + di) % num_area
            lo = max(0,    om_idx - om_r)
            hi = min(n_om, om_idx + om_r + 1)
            spec[di_mod, lo:hi] = 0.0

        if _too_similar(d_p, t_p):
            continue

        frac = sys_energy / total_energy
        if frac < _PART_MIN_ENERGY_FRAC:
            continue

        s1d_sum = float(s1d_preblank.sum())
        if s1d_sum > 0 and np.any(omega_vals > 0):
            om_mean = float(np.dot(s1d_preblank, omega_vals[:n_om])) / s1d_sum
            t_m = float(2 * np.pi / om_mean) if om_mean > 0 else t_p
        else:
            t_m = t_p

        systems.append({"frac": frac, "t_p": t_p, "t_m": t_m, "d_p": d_p, "d_m": d_p})

    if not systems:
        return {"n_sys": 0, "w_s": None, "sw_1": None, "sw_2": None}

    total_frac = sum(s["frac"] for s in systems)
    for s in systems:
        s["h_s"] = float(swh * np.sqrt(s["frac"] / total_frac))

    wind_candidates = [
        (i, s) for i, s in enumerate(systems)
        if _dir_diff(s["d_p"], wdir) <= _PART_WIND_DIR_THRESH
    ]
    if wind_candidates:
        wind_idx = min(wind_candidates, key=lambda x: x[1]["t_p"])[0]
        wind_sys = systems[wind_idx]
        swell_sys = sorted(
            [s for i, s in enumerate(systems) if i != wind_idx],
            key=lambda s: s["t_p"],
        )
    else:
        wind_sys  = None
        swell_sys = sorted(systems, key=lambda s: s["t_p"])

    return {
        "n_sys": len(systems),
        "w_s":   wind_sys,
        "sw_1":  swell_sys[0] if len(swell_sys) > 0 else None,
        "sw_2":  swell_sys[1] if len(swell_sys) > 1 else None,
    }
