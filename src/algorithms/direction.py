import numpy as np
from scipy.optimize import curve_fit


def wind_direction(bck: np.ndarray, adp: int, asp: int) -> float:
    """
    Estimate wind direction from azimuthal backscatter intensity.
    Fits I(θ) = a + b·cos²(0.5·(θ − c)) and returns c in degrees [0, 360).
    """
    intensity = bck[:, adp - asp: adp + asp].mean(axis=1).astype(float)
    aap = len(intensity)
    theta = np.linspace(0, 2 * np.pi, aap, endpoint=False)

    def _model(x, a, b, c):
        return a + b * np.cos(0.5 * (x - c)) ** 2

    try:
        a0 = float(intensity.mean())
        b0 = float(intensity.max() - intensity.min())
        c0 = float(theta[int(np.argmax(intensity))])
        popt, _ = curve_fit(
            _model, theta, intensity,
            p0=[a0, b0, c0],
            bounds=([0.0, 0.0, -np.pi], [255.0, 255.0, 3 * np.pi]),
            maxfev=1000,
        )
        return float(np.degrees(popt[2]) % 360)
    except Exception:
        return 0.0


def trim_segment(arr, index, num_area, AAP, ADP, ASP):
    """
    trimming segment from polar backscatter by index
    :param arr: input polar backscatter (theta, range)
    :param index: index of segment angle
    :return:
    """
    win = int(1 / num_area * AAP)
    lb_t = int(index * win - ASP)
    rb_t = int(index * win + ASP)

    lb_r = ADP - ASP
    rb_r = ADP + ASP

    if lb_t < 0:
        return np.concatenate((arr[lb_t:, lb_r:rb_r], arr[:int(2 * ASP + lb_t), lb_r:rb_r]), axis=0)
    elif rb_t > AAP:
        return np.concatenate((arr[AAP - rb_t:, lb_r:rb_r], arr[:int(2 * ASP - (rb_t - AAP)), lb_r:rb_r]), axis=0)
    else:
        return arr[lb_t:rb_t, lb_r:rb_r]


def calc_dir_ind(arr, num, AAP, ADP, ASP):
    """
    calculate main wave direction
    """
    rose = np.array([
        np.std(np.sum(trim_segment(arr, n, num, AAP, ADP, ASP), axis=0), axis=0)
        for n in range(num)
    ])
    return int(np.argmax(rose))
