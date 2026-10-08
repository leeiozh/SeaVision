"""Spectral axes sit on the true FFT bins, and peaks are located between bins.

Guards two accuracy fixes (ALGO_VERSION 2):
  1. Axes — bin i of an n-bin half-axis is at i·v_max/n.  linspace(0, v_max, n)
     stretched ω by n/(n−1) (0.8 % at n=128) and k by 3.2 % (K_NUM=32); in the
     current regression that read as a spurious drift along the wave direction.
  2. Peaks — argmax alone quantises T_peak to the ω grid and directions to 10°;
     a parabola through the peak and its neighbours recovers the position.

Run:  python test/test_spectral_axes.py       (no radar / sockets needed)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.spectrum2d import (
    fft_axis, parabolic_offset, compute_frequency_spectrum)
from src.algorithms.partition import calc_partitions


def test_axis_matches_fft_bins():
    n_shots, rpm = 256, 25.0
    dt = 60.0 / rpm
    om_max = np.pi * rpm / 60.0
    true = 2 * np.pi * np.fft.fftfreq(n_shots, dt)[: n_shots // 2]
    assert np.allclose(fft_axis(om_max, n_shots // 2), true), \
        "FAIL: omega axis is off the FFT bin centres"
    print("PASS [axis]: fft_axis equals the FFT bin frequencies")


def test_parabolic_offset():
    x = np.array([-1.0, 0.0, 1.0])
    for true_off in (-0.4, -0.1, 0.0, 0.25, 0.45):
        y = 5.0 - (x - true_off) ** 2
        assert abs(parabolic_offset(*y) - true_off) < 1e-9
    assert parabolic_offset(1.0, 1.0, 1.0) == 0.0     # flat — not a maximum
    assert parabolic_offset(3.0, 1.0, 3.0) == 0.0     # a minimum
    print("PASS [parabola]: exact on a parabola, 0 when there is no maximum")


def test_t_peak_between_bins():
    om = fft_axis(np.pi * 25 / 60, 128)
    d_om = om[1] - om[0]
    k_vals = np.array([0.0, 1.0])
    worst = 0.0
    for frac in (0.0, 0.2, 0.5, 0.8):
        om_true = om[50] + frac * d_om
        s = np.exp(-0.5 * ((om - om_true) / (4 * d_om)) ** 2)
        signal = np.repeat(s[:, None], 2, axis=1)      # ∫dk over [0,1] returns s
        _, _, t_peak, _ = compute_frequency_spectrum(signal, k_vals, om)
        worst = max(worst, abs(2 * np.pi / t_peak - om_true) / d_om)
    assert worst < 0.05, f"FAIL: T_peak off by {worst:.2f} bin (argmax gives up to 0.5)"
    print(f"PASS [T_peak]: worst error {worst:.3f} bin")


def test_partition_peak_between_bins():
    om = fft_axis(np.pi * 25 / 60, 128)
    dirs = np.linspace(0, 360, 36, endpoint=False)
    om_true, dir_true = om[50] + 0.3 * (om[1] - om[0]), 123.0
    dd = (dirs[:, None] - dir_true + 180) % 360 - 180
    spec = np.exp(-0.5 * (dd / 25.0) ** 2) * \
        np.exp(-0.5 * ((om[None, :] - om_true) / 0.06) ** 2)
    res = calc_partitions(spec, om, dirs, wdir=0.0, swh=2.0)
    assert res["n_sys"] == 1, f"FAIL: expected one system, got {res['n_sys']}"
    sys_ = res["sw_1"]
    assert abs(sys_["d_p"] - dir_true) < 1.0, f"FAIL: d_p={sys_['d_p']:.1f}, true {dir_true}"
    assert abs(sys_["t_p"] - 2 * np.pi / om_true) < 0.05, f"FAIL: t_p={sys_['t_p']:.2f}"
    print(f"PASS [partition]: d_p={sys_['d_p']:.1f} (true {dir_true}), "
          f"t_p={sys_['t_p']:.2f}s (true {2 * np.pi / om_true:.2f}s)")


if __name__ == "__main__":
    test_axis_matches_fft_bins()
    test_parabolic_offset()
    test_t_peak_between_bins()
    test_partition_peak_between_bins()
    print("ALL TESTS PASSED")
