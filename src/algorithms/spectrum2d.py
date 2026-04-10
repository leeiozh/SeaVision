import numpy as np
from scipy.signal import welch
from scipy.fft import fft2, fftshift
from scipy.interpolate import griddata
from src.algorithms.dispersion import dispersion_curve


def calc_spec2d(back, angle, vco, step, asp, k_num, om_max, n_shots, num_area, ax=None):
    """
    calculate 2D spectrum
    :param back: timeseries of square segments trimmed from input backscatter
    :param angle: angle between center of square [degrees]
    :param vco: multiplier in doppler term (k, V) / |k| [m/s]
    :param step: range resolution [m]
    :return:
    """
    k_max = np.pi / asp / step * k_num

    '''image spectrum calculation'''
    _, back_f3 = welch(fftshift(fft2(back, axes=(1, 2)), axes=(1, 2))
                       [:, asp - k_num:asp + k_num, asp - k_num:asp + k_num], axis=0, return_onesided=False)

    kx = np.arange(-k_num, k_num)
    ky = np.arange(-k_num, k_num)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    KABS = np.rint(np.sqrt(KX ** 2 + KY ** 2)).astype(int)
    k_norm = KABS / k_num * k_max
    valid = KABS < k_num
    nonzero = KABS != 0

    '''doppler term removing'''
    shift = -np.rint(k_norm * vco / om_max * n_shots).astype(int)
    ix, iy = np.where(valid)
    for i, j in zip(ix, iy):
        back_f3[:, i, j] = np.roll(back_f3[:, i, j], shift[i, j])

    '''signal and noise separation'''
    omega = np.rint(dispersion_curve(k_norm, 0) / om_max * n_shots).astype(int)
    omega = np.clip(omega, 0, n_shots - 1)
    bf = back_f3[:, KX + k_num, KY + k_num]  # (time, kx, ky)
    noise_mask = np.abs(np.arange(n_shots)[:, None, None] - omega[None, :, :]) > 15
    noi = bf * noise_mask
    sig = bf - noi

    '''applying MTF k^{-1.2}'''
    weight = np.ones_like(KABS, dtype=float)
    weight[nonzero] = KABS[nonzero] ** (-1.2)
    sig *= weight[None, :, :]

    '''integrating over omega'''
    mean_noi = np.mean(noi, axis=0)
    mean_sig = np.mean(sig, axis=0)
    snr = np.zeros_like(mean_sig)
    valid_snr = valid & (mean_noi != 0)
    snr[valid_snr] = mean_sig[valid_snr] / mean_noi[valid_snr]

    '''integrating '''
    sp2d = np.zeros((num_area, n_shots))
    ang = np.rint((np.deg2rad(angle) - np.arctan2(KY, KX)) / (2 * np.pi) * sp2d.shape[0]).astype(int) % sp2d.shape[0]
    np.add.at(sp2d, (ang[valid_snr], omega[valid_snr]), snr[valid_snr])

    '''interpolation over azimuth to smooth 2d spectrum'''
    sp2d[sp2d == 0] = np.nan
    sp2d[:, 0] = np.nan
    pad = sp2d.shape[0] // 3
    sp2d_ext = np.vstack((sp2d[-pad:, :], sp2d, sp2d[:pad, :]))
    x, y = np.indices(sp2d_ext.shape)
    val = ~np.isnan(sp2d_ext)
    sp2d_int = sp2d_ext.copy()
    sp2d_int[np.isnan(sp2d_ext)] = griddata((x[val], y[val]), sp2d_ext[val],
                                            (x[np.isnan(sp2d_ext)], y[np.isnan(sp2d_ext)]),
                                            method='linear', fill_value=0)
    spec2d = sp2d_int[pad:-pad, :]

    if ax is not None:
        ax.contourf(np.linspace(0, 2 * np.pi, num_area), 2 * np.pi / np.linspace(1e-10, om_max, n_shots), spec2d.T,
                    cmap="gnuplot2")
        ax.set_yticks([5, 7, 10, 12, 15, 25])
        ax.set_ylim(5, 25)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(np.linspace(0, 2 * np.pi, 37)[:-1])
        ax.set_xticklabels([""] * 36)
        ax.grid(True, lw=0.3)

    '''integrating (kx, ky) -> (|k|, ang)'''
    ii = np.arange(k_num)
    jj = np.arange(k_num)
    II, JJ = np.meshgrid(ii, jj, indexing="ij")
    k = np.sqrt(II ** 2 + JJ ** 2)
    k_div = np.floor(k).astype(int)
    k_mod = k - k_div
    mask0 = k_div < k_num
    mask1 = k_div < k_num - 1
    bf = back_f3[:, II + k_num, JJ + k_num]
    port = np.zeros((n_shots, k_num), dtype=bf.dtype)
    np.add.at(port, (slice(None), k_div[mask0]), bf[:, mask0] * (1.0 - k_mod[mask0]))
    np.add.at(port, (slice(None), k_div[mask1] + 1), bf[:, mask1] * k_mod[mask1])
    port[0, 0] = 0.0
    return spec2d, port
