import numpy as np
from scipy.ndimage import gaussian_filter1d
from src.algorithms.dispersion import dispersion_curve, find_curve


def process_portrait(port, inv, speed, freq_win, step, asp, k_num, om_max, n_shots, ax_2d=None, ax_1d=None):
    """
    processing omega-k spectrum
    :param port: omega-k spectrum
    :param inv: flag of inversion
    :param speed: ship speed over ground [m / s]
    :param freq_win: frequency window for trimming dispersion curve [rad / s]
    :param step: range resolution [m]
    :param ax: ax if plotting needed
    :return: port, tp, m0, spec_1d, ks, vco
    """
    if inv:
        port = np.flip(port, axis=0)
    port_pr = np.copy(port)
    _, vco, ks = find_curve(port, speed, step, inv, asp, k_num, om_max, n_shots)
    k_max = np.pi / asp / step * k_num
    k_arr = np.linspace(0, k_max, k_num)

    shift = -np.round(k_arr * vco / om_max * n_shots).astype(int)
    row_idx = (np.arange(n_shots)[:, None] - shift[None, :]) % n_shots
    port_pr = np.take_along_axis(port_pr, row_idx, axis=0)

    # for i, k in enumerate(k_arr):
    #     port_pr[:, i] = np.roll(port_pr[:, i], shift=-round(k * vco / OM_MAX * N_SHOTS))

    f = dispersion_curve(k_arr, 0)
    do_arg = ((f - freq_win) / om_max * n_shots).astype(int)
    do_arg = np.clip(do_arg, 0, n_shots - 1)
    up_arg = ((f + freq_win) / om_max * n_shots).astype(int)
    up_arg = np.clip(up_arg, 0, n_shots - 1)
    noise = np.copy(port_pr)

    if ax_2d is not None:
        ax_2d.imshow(port, cmap="gnuplot", origin="lower", aspect="auto", interpolation="None",
                     extent=[0, k_max, 0, om_max])
        ax_2d.set_xlim(0, k_max)
        ax_2d.set_ylim(0, om_max)
        ax_2d.plot(k_arr, dispersion_curve(k_arr, vco), lw=1, c="w", ls=":", label="original disp. curve")
        ax_2d.plot(k_arr, f, lw=1, c="w", label="shifted disp. curve")
        ax_2d.plot(k_arr, do_arg / n_shots * om_max, lw=0.5, c="grey", label=r'bounds of "signal"')
        ax_2d.plot(k_arr, up_arg / n_shots * om_max, lw=0.5, c="grey")
        # ax_2d.legend(loc="lower right")

    rows = np.arange(n_shots)[:, None]
    noise[(rows >= do_arg[None, :]) & (rows < up_arg[None, :])] = 0

    signal = port_pr - noise

    k_idx = np.arange(1, port_pr.shape[1])
    k = k_idx / port_pr.shape[1] * k_max
    signal[:, 1:] *= (k ** (-1.2))

    f_sig = np.trapz(signal, axis=1)
    f_noi = np.trapz(noise, axis=1)
    spec_1d = gaussian_filter1d(f_sig / f_noi, sigma=2)

    f_noi_trap = np.trapz(f_noi)
    spec1d_argmax = np.nanargmax(spec_1d)
    if f_noi_trap > 0:
        m0 = np.trapz(f_sig) / f_noi_trap
    else:
        m0 = 0
    if spec1d_argmax > 0:
        tp = 2 * np.pi / (spec1d_argmax / n_shots * om_max)
    else:
        tp = 0

    if ax_1d is not None:
        ax_1d.plot(np.linspace(0, om_max / 2 / np.pi, spec_1d.shape[0]), spec_1d)
        ax_1d.set_ylim(0, np.max(spec_1d))

    return port_pr, tp, m0, spec_1d, ks, vco
