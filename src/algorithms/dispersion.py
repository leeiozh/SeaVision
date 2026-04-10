import numpy as np


def dispersion_curve(k, vco, depth=None):
    """
    dispersion relation for waves on deep water
    :param k: wave number [rad / m]
    :param vco: multiplier in doppler term (speed * cosine on angle between speed and wave vector) [m / s]
    :param depth: mean depth in trimmed segment
    :return: angular velocity [rad / s]
    """
    if depth is None:  # for waves on deep water
        return np.sqrt(9.81 * k) + k * vco
    else:
        return np.sqrt(9.81 * k * np.tanh(k * depth)) + k * vco


def find_curve(portrait, speed, step, inv, asp, k_num, om_max, n_shots):
    """
    approximating dispersion curve through zero and max points
    :param portrait: omega-k spectrum
    :param speed: ship speed over ground [m/s]
    :param step: range resolution [m]
    :param inv: inverse spectra or not
    :return: inversion flag, multiplier in doppler term, wave number of maxima
    """
    k_max = np.pi / asp / step * k_num  # max wave number in rad / m

    port_sm = np.apply_along_axis(lambda row: np.convolve(row, np.ones(4) / 4, mode="same"), axis=0, arr=portrait)

    f_b = round(0.04 * portrait.shape[0])
    port_sm[:f_b, :] = 0
    port_sm[n_shots - f_b:, :] = 0
    port_sm[:, :round(0.1 * portrait.shape[1])] = 0
    port_sm[:, round(0.6 * portrait.shape[1]):] = 0

    max_p = np.argmax(port_sm)
    om_m = (max_p // k_num) / n_shots * om_max  # frequency of point of maxima
    k_m = (max_p % k_num) / k_num * k_max  # wave number of point if maxima

    if inv is not None:
        f = inv
    else:
        f = False
        if om_m > 0.5 * om_max:
            om_m = om_max - om_m
            f = True

    if k_m != 0:
        vco = (om_m - np.sqrt(9.81 * k_m)) / k_m
    else:
        vco = 0

    if abs(vco) > speed:
        vco = np.sign(vco) * speed

    return f, vco, k_m
