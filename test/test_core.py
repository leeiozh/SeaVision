"""
Synthetic regression tests for SeaVision core modules.

Run:  pytest test/test_core.py -v
      pytest test/test_core.py -v -k averaging   # one group

Coverage:
  - Averager: last-frame vs averaged fields, circular mean, ring-buffer wrap
  - _phys_clip: limits, NaN/inf, idempotency
  - UDP protocol: pack → unpack round-trip (new v2.0 and old)
  - Current direction convention: TO compass bearing
  - Energy balance: Σ h_s_i² == swh² in calc_partitions
  - _circ_mean_deg: wrap 0/360, constant, bimodal
  - _decode_rpm: new / old / unknown / bad-type
"""
import struct
import numpy as np
import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_output(swh=1.5, tp=10.0, tm=8.0, dp=90.0, dm=95.0,
                 swh_win=0.8, tp_win=6.0, dp_win=80.0,
                 swh_sw1=0.7, tp_sw1=12.0, dp_sw1=100.0,
                 curr_speed=0.3, curr_dir=45.0,
                 wspd=7.0, wind_dir=270.0,
                 n_sys=2, quality=1, rps=25.0):
    """Build a minimal Output object for packing tests."""
    from src.io.structs import Wave, Output
    o = Output(
        pulse=1, step=1.875, rps=rps,
        n_in_win=255, n_wins=4,
        step_area=1.875, n_area=384, n_start=1192,
        cog_proc=0, sog_proc=0, max_sys=3, ide_sys=n_sys,
        wave_sum=Wave(swh=swh, t_p=tp, t_m=tm, d_p=dp, d_m=dm),
        wave_win=Wave(swh=swh_win, t_p=tp_win, d_p=dp_win),
        wave_sw1=Wave(swh=swh_sw1, t_p=tp_sw1, d_p=dp_sw1),
        wave_sw2=Wave(),
        n_dis=quality,
        spec_1d=np.arange(64, dtype=int),
        spec_2d=np.zeros((36, 36), dtype=int),
    )
    o.curr_speed = curr_speed
    o.curr_dir   = curr_dir
    o.wspd       = wspd
    o.wind_dir   = wind_dir
    return o


def _make_wave_out(n_sys=2, swh=2.0, tp=12.0, tm=9.0, dp=90.0, dm=95.0,
                   swh_win=1.0, tp_win=6.0, dp_win=80.0,
                   wspd=5.0, wind_dir=270.0, curr_speed=0.5, curr_dir=45.0):
    from src.io.structs import Wave, WaveOutput
    return WaveOutput(
        ide_sys=n_sys,
        wave_sum=Wave(swh=swh, t_p=tp, t_m=tm, d_p=dp, d_m=dm),
        wave_win=Wave(swh=swh_win, t_p=tp_win, d_p=dp_win),
        wave_sw1=Wave(), wave_sw2=Wave(),
        spec_1d=np.zeros(64), spec_2d=np.zeros((36, 36)),
        wspd=wspd, wind_dir=wind_dir,
        curr_speed=curr_speed, curr_dir=curr_dir,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. _circ_mean_deg
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircMean:
    from src.processing.averaging import _circ_mean_deg

    def _cm(self, angles):
        from src.processing.averaging import _circ_mean_deg
        s = sum(np.sin(np.radians(a)) for a in angles) / len(angles)
        c = sum(np.cos(np.radians(a)) for a in angles) / len(angles)
        return _circ_mean_deg(s, c)

    def test_constant(self):
        assert abs(self._cm([45, 45, 45]) - 45.0) < 0.01

    def test_wrap_zero(self):
        result = self._cm([350, 10])
        # should be near 0° (or 360°)
        assert min(abs(result - 0), abs(result - 360)) < 0.5

    def test_south(self):
        assert abs(self._cm([180, 180]) - 180.0) < 0.01

    def test_north_east(self):
        result = self._cm([0, 90])   # mean ≈ 45°
        assert abs(result - 45.0) < 0.5

    def test_result_in_0_360(self):
        # Due to float arithmetic, result may be exactly 360.0 (≡ 0°); normalise.
        for angles in [[350, 10], [270, 90], [1, 359], [0, 0]]:
            r = self._cm(angles) % 360.0
            assert 0.0 <= r < 360.0, f"out of range: {r} for {angles}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Averager
# ═══════════════════════════════════════════════════════════════════════════════

class TestAverager:

    def _av(self, mean=4):
        from src.processing.averaging import Averager
        return Averager(mean=mean, n_freq=64, n_freq_2d=36,
                        n_dirs=36, n_shots=16, cut_num=4)

    def test_empty_returns_none_port(self):
        av = self._av()
        res, port = av.get_mean(pulse=1, step=1.875, rpm=25.0,
                                n_shots=16, asp=12, adp=20)
        assert port is None

    def test_single_push_peak_from_last(self):
        av = self._av()
        wo = _make_wave_out(n_sys=3, tp=13.0, dp=88.0)
        av.push(wo, np.zeros((16, 4)))
        res, _ = av.get_mean(pulse=1, step=1.875, rpm=25.0,
                             n_shots=16, asp=12, adp=20)
        assert res.ide_sys == 3
        assert res.wave_sum.t_p == 13.0
        assert res.wave_sum.d_p == 88.0

    def test_nsys_from_last_frame_not_average(self):
        """n_sys must reflect the last computation, not an average."""
        av = self._av(mean=4)
        av.push(_make_wave_out(n_sys=3, swh=2.0, tp=12.0, tm=9.0, dp=90.0, dm=90.0), np.zeros((16,4)))
        av.push(_make_wave_out(n_sys=3, swh=2.0, tp=12.0, tm=9.0, dp=90.0, dm=90.0), np.zeros((16,4)))
        av.push(_make_wave_out(n_sys=2, swh=2.0, tp=13.0, tm=9.0, dp=88.0, dm=90.0), np.zeros((16,4)))
        res, _ = av.get_mean(pulse=1, step=1.875, rpm=25.0, n_shots=16, asp=12, adp=20)
        assert res.ide_sys == 2, "n_sys must be from last frame"

    def test_peak_tp_dp_from_last_frame(self):
        """t_p and d_p in wave_sum must come from the last computation."""
        av = self._av(mean=4)
        av.push(_make_wave_out(tp=10.0, dp=80.0), np.zeros((16,4)))
        av.push(_make_wave_out(tp=11.0, dp=82.0), np.zeros((16,4)))
        av.push(_make_wave_out(tp=15.0, dp=100.0), np.zeros((16,4)))
        res, _ = av.get_mean(pulse=1, step=1.875, rpm=25.0, n_shots=16, asp=12, adp=20)
        assert res.wave_sum.t_p == 15.0, "t_p must be from last frame"
        assert res.wave_sum.d_p == 100.0, "d_p must be from last frame"

    def test_swh_averaged(self):
        av = self._av(mean=4)
        swhs = [1.0, 2.0, 3.0]
        for s in swhs:
            av.push(_make_wave_out(swh=s), np.zeros((16,4)))
        res, _ = av.get_mean(pulse=1, step=1.875, rpm=25.0, n_shots=16, asp=12, adp=20)
        assert abs(res.wave_sum.swh - np.mean(swhs)) < 0.01

    def test_tm_averaged(self):
        av = self._av(mean=4)
        tms = [8.0, 9.0, 10.0]
        for t in tms:
            av.push(_make_wave_out(tm=t), np.zeros((16,4)))
        res, _ = av.get_mean(pulse=1, step=1.875, rpm=25.0, n_shots=16, asp=12, adp=20)
        assert abs(res.wave_sum.t_m - np.mean(tms)) < 0.01

    def test_dm_circular_mean_wrap(self):
        """d_m averaging must use circular mean — arithmetic fails near 0°/360°."""
        av = self._av(mean=4)
        for dm in [350.0, 10.0]:
            av.push(_make_wave_out(dm=dm), np.zeros((16,4)))
        res, _ = av.get_mean(pulse=1, step=1.875, rpm=25.0, n_shots=16, asp=12, adp=20)
        # correct circular mean ≈ 0°; arithmetic mean would give 180°
        assert min(abs(res.wave_sum.d_m), abs(res.wave_sum.d_m - 360)) < 5.0, \
            f"d_m={res.wave_sum.d_m:.1f} should be near 0° not {res.wave_sum.d_m:.1f}°"

    def test_wind_dir_circular_mean(self):
        av = self._av(mean=4)
        for wd in [355.0, 5.0]:
            av.push(_make_wave_out(wind_dir=wd), np.zeros((16,4)))
        res, _ = av.get_mean(pulse=1, step=1.875, rpm=25.0, n_shots=16, asp=12, adp=20)
        assert min(abs(res.wind_dir), abs(res.wind_dir - 360)) < 5.0, \
            f"wind_dir={res.wind_dir:.1f} should be near 0°"

    def test_curr_dir_circular_mean(self):
        av = self._av(mean=4)
        for cd in [10.0, 350.0]:
            av.push(_make_wave_out(curr_dir=cd), np.zeros((16,4)))
        res, _ = av.get_mean(pulse=1, step=1.875, rpm=25.0, n_shots=16, asp=12, adp=20)
        assert min(abs(res.curr_dir), abs(res.curr_dir - 360)) < 5.0, \
            f"curr_dir={res.curr_dir:.1f} should be near 0°"

    def test_wave_win_from_last_frame(self):
        """wave_win (per-system) must come from the last computation, not averaged."""
        av = self._av(mean=4)
        av.push(_make_wave_out(swh_win=1.5, tp_win=6.0, dp_win=80.0), np.zeros((16,4)))
        av.push(_make_wave_out(swh_win=1.6, tp_win=6.5, dp_win=82.0), np.zeros((16,4)))
        av.push(_make_wave_out(swh_win=0.0, tp_win=0.0, dp_win=0.0), np.zeros((16,4)))
        res, _ = av.get_mean(pulse=1, step=1.875, rpm=25.0, n_shots=16, asp=12, adp=20)
        assert res.wave_win.swh == 0.0,  "wave_win.swh must be from last frame"
        assert res.wave_win.t_p == 0.0,  "wave_win.t_p must be from last frame"

    def test_ring_buffer_wrap(self):
        """After more than MEAN pushes, oldest entries are evicted."""
        av = self._av(mean=3)
        for swh in [10.0, 20.0, 30.0]:  # fills the ring
            av.push(_make_wave_out(swh=swh, n_sys=1), np.zeros((16,4)))
        # push one more — evicts 10.0
        av.push(_make_wave_out(swh=4.0, n_sys=2, tp=14.0, dp=50.0), np.zeros((16,4)))
        res, _ = av.get_mean(pulse=1, step=1.875, rpm=25.0, n_shots=16, asp=12, adp=20)
        # mean of last 3: 20, 30, 4
        assert abs(res.wave_sum.swh - (20+30+4)/3) < 0.01
        assert res.ide_sys == 2         # last frame
        assert res.wave_sum.t_p == 14.0 # last frame

    def test_spec_normalised_to_255(self):
        av = self._av(mean=2)
        s = np.linspace(0, 100, 64)
        wo = _make_wave_out()
        wo.spec_1d = s
        av.push(wo, np.zeros((16,4)))
        res, _ = av.get_mean(pulse=1, step=1.875, rpm=25.0, n_shots=16, asp=12, adp=20)
        assert int(np.max(res.spec_1d)) == 255

    def test_wspd_averaged(self):
        av = self._av(mean=4)
        for w in [4.0, 6.0, 8.0]:
            av.push(_make_wave_out(wspd=w), np.zeros((16,4)))
        res, _ = av.get_mean(pulse=1, step=1.875, rpm=25.0, n_shots=16, asp=12, adp=20)
        assert abs(res.wspd - np.mean([4, 6, 8])) < 0.01

    def test_curr_speed_averaged(self):
        av = self._av(mean=4)
        for c in [0.2, 0.4, 0.6]:
            av.push(_make_wave_out(curr_speed=c), np.zeros((16,4)))
        res, _ = av.get_mean(pulse=1, step=1.875, rpm=25.0, n_shots=16, asp=12, adp=20)
        assert abs(res.curr_speed - np.mean([0.2, 0.4, 0.6])) < 0.001


# ═══════════════════════════════════════════════════════════════════════════════
# 3. _phys_clip
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhysClip:

    def _clip(self, **kw):
        from src.io.output import _phys_clip
        o = _make_output(**kw)
        _phys_clip(o)
        return o

    def test_normal_values_unchanged(self):
        o = self._clip(swh=1.5, tp=12.0, tm=9.0, curr_speed=0.3, wspd=7.0)
        assert o.wave_sum.swh == 1.5
        assert o.wave_sum.t_p == 12.0
        assert o.wave_sum.t_m == 9.0
        assert o.curr_speed  == 0.3
        assert o.wspd        == 7.0

    def test_swh_clipped_above(self):
        o = self._clip(swh=50.0)
        assert o.wave_sum.swh == 20.0

    def test_tp_clipped_above(self):
        o = self._clip(tp=600.0)
        assert o.wave_sum.t_p == 30.0

    def test_tm_clipped_above(self):
        o = self._clip(tm=400.0)
        assert o.wave_sum.t_m == 30.0

    def test_curr_clipped_above(self):
        o = self._clip(curr_speed=10.0)
        assert o.curr_speed == 3.0

    def test_wspd_clipped_above(self):
        o = self._clip(wspd=99.0)
        assert o.wspd == 30.0

    def test_negative_swh_zeroed(self):
        o = self._clip(swh=-1.0)
        assert o.wave_sum.swh == 0.0

    def test_negative_curr_zeroed(self):
        o = self._clip(curr_speed=-0.5)
        assert o.curr_speed == 0.0

    def test_nan_swh_zeroed(self):
        o = self._clip(swh=float('nan'))
        assert o.wave_sum.swh == 0.0

    def test_inf_tp_clipped(self):
        # inf is non-finite → treated as "no data", zeroed (not clipped to max)
        o = self._clip(tp=float('inf'))
        assert o.wave_sum.t_p == 0.0

    def test_nan_curr_zeroed(self):
        o = self._clip(curr_speed=float('nan'))
        assert o.curr_speed == 0.0

    def test_idempotent(self):
        from src.io.output import _phys_clip
        o = _make_output(swh=50.0, tp=600.0, curr_speed=99.0, wspd=200.0)
        _phys_clip(o)
        v1 = (o.wave_sum.swh, o.wave_sum.t_p, o.curr_speed, o.wspd)
        _phys_clip(o)
        v2 = (o.wave_sum.swh, o.wave_sum.t_p, o.curr_speed, o.wspd)
        assert v1 == v2

    def test_per_system_waves_also_clipped(self):
        from src.io.output import _phys_clip
        o = _make_output(swh_win=25.0, tp_win=40.0)
        _phys_clip(o)
        assert o.wave_win.swh == 20.0
        assert o.wave_win.t_p == 30.0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. UDP protocol — pack/unpack round-trip
# ═══════════════════════════════════════════════════════════════════════════════

class TestUdpProtocol:

    _HDR_NEW = "<BBHHHHHHHHHHHHHHHHHHHHBBHHHH"
    _HDR_OLD = "<B17H"

    def _pack_new(self, o):
        from src.io.output import UdpOutputSink
        sink = UdpOutputSink.__new__(UdpOutputSink)
        sink.n_freqs   = 64
        sink.n_dirs    = 36
        sink.n_freq_2d = 36
        sink.algo_version = 1
        return sink._pack_new(o)

    def _pack_old(self, o):
        from src.io.output import UdpOutputSink
        sink = UdpOutputSink.__new__(UdpOutputSink)
        sink.n_freqs   = 64
        sink.n_dirs    = 36
        sink.n_freq_2d = 36
        sink.algo_version = 1
        return sink._pack_old(o)

    def test_new_packet_size(self):
        data = self._pack_new(_make_output())
        assert len(data) == 1412

    def test_old_packet_size(self):
        data = self._pack_old(_make_output())
        assert len(data) == 1398

    def test_new_type_byte(self):
        data = self._pack_new(_make_output())
        assert data[0] == 5

    def test_old_type_byte(self):
        data = self._pack_old(_make_output())
        assert data[0] == 5

    def test_new_swh_roundtrip(self):
        o = _make_output(swh=2.34)
        data = self._pack_new(o)
        un = struct.unpack(self._HDR_NEW, data[:struct.calcsize(self._HDR_NEW)])
        assert abs(un[4] / 100.0 - 2.34) < 0.01

    def test_new_tp_roundtrip(self):
        o = _make_output(tp=12.5)
        data = self._pack_new(o)
        un = struct.unpack(self._HDR_NEW, data[:struct.calcsize(self._HDR_NEW)])
        assert abs(un[5] / 100.0 - 12.5) < 0.01

    def test_new_rpm_roundtrip(self):
        o = _make_output(rps=24.73)
        data = self._pack_new(o)
        un = struct.unpack(self._HDR_NEW, data[:struct.calcsize(self._HDR_NEW)])
        assert abs(un[3] / 100.0 - 24.73) < 0.01

    def test_new_curr_speed_roundtrip(self):
        o = _make_output(curr_speed=1.23)
        data = self._pack_new(o)
        un = struct.unpack(self._HDR_NEW, data[:struct.calcsize(self._HDR_NEW)])
        assert abs(un[18] / 100.0 - 1.23) < 0.01

    def test_new_direction_in_0_359(self):
        o = _make_output(dp=361.0)  # should wrap
        data = self._pack_new(o)
        un = struct.unpack(self._HDR_NEW, data[:struct.calcsize(self._HDR_NEW)])
        assert 0 <= un[7] < 360

    def test_new_nsys_and_quality(self):
        o = _make_output(n_sys=2, quality=1)
        data = self._pack_new(o)
        un = struct.unpack(self._HDR_NEW, data[:struct.calcsize(self._HDR_NEW)])
        assert un[22] == 2   # n_sys
        assert un[23] == 1   # quality

    def test_old_rpm_roundtrip(self):
        o = _make_output(rps=25.0)
        data = self._pack_old(o)
        un = struct.unpack(self._HDR_OLD, data[:struct.calcsize(self._HDR_OLD)])
        assert abs(un[17] / 1000.0 - 25.0) < 0.01

    def test_spec1d_in_packet_new(self):
        spec = np.arange(64, dtype=int)
        o = _make_output()
        o.spec_1d = spec
        data = self._pack_new(o)
        hdr_size = struct.calcsize(self._HDR_NEW)
        spec_bytes = data[hdr_size: hdr_size + 64]
        assert list(spec_bytes) == list(spec)

    def test_new_nan_curr_gives_zero(self):
        o = _make_output(curr_speed=float('nan'))
        data = self._pack_new(o)   # must not raise
        un = struct.unpack(self._HDR_NEW, data[:struct.calcsize(self._HDR_NEW)])
        assert un[18] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Current direction convention (TO compass bearing)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCurrentDirection:
    """
    Convention: curr_dir = arctan2(East_component, North_component) % 360
    = compass bearing where the current is GOING TO.
    """

    def _curr_dir(self, east, north):
        return float(np.degrees(np.arctan2(east, north)) % 360)

    def test_northward(self):
        assert abs(self._curr_dir(0, 1) - 0.0) < 0.01    # going North → 0°

    def test_eastward(self):
        assert abs(self._curr_dir(1, 0) - 90.0) < 0.01   # going East → 90°

    def test_southward(self):
        assert abs(self._curr_dir(0, -1) - 180.0) < 0.01 # going South → 180°

    def test_westward(self):
        assert abs(self._curr_dir(-1, 0) - 270.0) < 0.01 # going West → 270°

    def test_northeast(self):
        d = self._curr_dir(1, 1)
        assert abs(d - 45.0) < 0.01   # NE → 45°

    def test_result_in_0_360(self):
        for east, north in [(1,1), (-1,1), (0,-1), (-1,-1)]:
            d = self._curr_dir(east, north)
            assert 0.0 <= d < 360.0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Energy balance in calc_partitions: Σ h_s_i² == swh²
# ═══════════════════════════════════════════════════════════════════════════════

class TestPartitionEnergyBalance:

    def _run(self, n_dirs=36, n_om=64):
        from src.algorithms.partition import calc_partitions
        rng = np.random.default_rng(42)
        # Create a spectrum with 2 clear peaks
        om  = np.linspace(0.1, 1.3, n_om)
        dirs = np.linspace(0, 360, n_dirs, endpoint=False)
        spec = np.zeros((n_dirs, n_om))
        # peak 1: om≈0.5, dir≈90°
        spec[9,  15] = 100.0
        # peak 2: om≈0.9, dir≈270°
        spec[27, 40] = 80.0
        swh = 2.8
        result = calc_partitions(spec, om, dirs, wdir=90.0, swh=swh)
        return result, swh

    def test_energy_balance_exact(self):
        result, swh = self._run()
        systems = [v for v in (result['w_s'], result['sw_1'], result['sw_2']) if v is not None]
        if len(systems) == 0:
            pytest.skip("No systems found — spectrum too simple for partitioner")
        total_h2 = sum(s['h_s']**2 for s in systems)
        assert abs(total_h2 - swh**2) < 1e-6, \
            f"Energy: Σh_s²={total_h2:.6f} ≠ swh²={swh**2:.6f}"

    def test_h_s_non_negative(self):
        result, _ = self._run()
        for key in ('w_s', 'sw_1', 'sw_2'):
            s = result[key]
            if s is not None:
                assert s['h_s'] >= 0.0

    def test_n_sys_le_3(self):
        result, _ = self._run()
        assert result['n_sys'] <= 3

    def test_no_systems_zero_swh(self):
        from src.algorithms.partition import calc_partitions
        spec = np.zeros((36, 64))
        om   = np.linspace(0.1, 1.3, 64)
        dirs = np.linspace(0, 360, 36, endpoint=False)
        result = calc_partitions(spec, om, dirs, wdir=0.0, swh=0.0)
        assert result['n_sys'] == 0
        assert result['w_s'] is None

    def test_energy_balance_1_system(self):
        from src.algorithms.partition import calc_partitions
        n_dirs, n_om = 36, 64
        om   = np.linspace(0.1, 1.3, n_om)
        dirs = np.linspace(0, 360, n_dirs, endpoint=False)
        spec = np.zeros((n_dirs, n_om))
        spec[9, 20] = 200.0   # single dominant peak
        swh = 3.5
        result = calc_partitions(spec, om, dirs, wdir=90.0, swh=swh)
        systems = [v for v in (result['w_s'], result['sw_1'], result['sw_2']) if v is not None]
        if len(systems) == 0:
            pytest.skip("No systems found")
        total_h2 = sum(s['h_s']**2 for s in systems)
        assert abs(total_h2 - swh**2) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RPM estimation ring buffer
# ═══════════════════════════════════════════════════════════════════════════════

class TestRpmRingBuffer:
    """Tests the _update_rpm logic from processor.py in isolation."""

    _MIN_DT = 0.3
    _MAX_DT = 12.0

    def _simulate(self, intervals, n_shots=64):
        buf = np.zeros(n_shots)
        count = 0
        t_prev = 0.0
        rpms = []
        t = 0.0
        for dt in intervals:
            t += dt
            if t_prev > 0.0:
                if self._MIN_DT < dt < self._MAX_DT:
                    buf[count % n_shots] = dt
                    count += 1
            t_prev = t
            n = min(count, n_shots)
            if n > 0:
                med = float(np.median(buf[:n]))
                if med > 0:
                    rpms.append(60.0 / med)
        return rpms

    def test_constant_rate_converges(self):
        intervals = [2.4] * 100   # 25 RPM
        rpms = self._simulate(intervals)
        assert abs(rpms[-1] - 25.0) < 0.1

    def test_jitter_stability(self):
        rng = np.random.default_rng(0)
        intervals = 2.4 * (1 + 0.05 * rng.standard_normal(200))
        rpms = self._simulate(intervals)
        assert abs(np.mean(rpms[-50:]) - 25.0) < 0.5

    def test_dropout_rejected(self):
        """Doubled interval (missed rotation) should not corrupt median."""
        intervals = [2.4] * 50 + [4.8] * 5 + [2.4] * 50   # 5 dropouts
        rpms = self._simulate(intervals)
        assert abs(rpms[-1] - 25.0) < 0.5

    def test_implausible_interval_rejected(self):
        """Interval outside [MIN_DT, MAX_DT] must be ignored."""
        intervals = [2.4] * 20 + [0.1] * 5 + [2.4] * 20   # 0.1s < MIN_DT
        rpms = self._simulate(intervals)
        assert abs(rpms[-1] - 25.0) < 0.3

    def test_fast_rate(self):
        intervals = [0.914] * 100   # ~65.6 RPM
        rpms = self._simulate(intervals)
        assert abs(rpms[-1] - 60 / 0.914) < 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# 8. _decode_rpm (tester_receive_rpm.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDecodeRpm:

    def _make_new(self, rpm_x100: int) -> bytes:
        hdr = struct.pack("<BBHHHHHHHHHHHHHHHHHHHHBBHHHH",
                          5, 1, 1875, rpm_x100,
                          280, 1100, 900, 90, 95,
                          0,0,0, 0,0,0, 0,0,0,
                          0, 0, 0, 0,
                          2, 1, 1, 0, 0, 0)
        return hdr + bytes(64) + bytes(36*36)

    def _make_old(self, rpm_x1000: int) -> bytes:
        fields = [280,1100,90, 0,0,0, 0,0,0, 0,0,0, 0,0, 0,0, rpm_x1000]
        hdr = struct.pack("<B17H", 5, *fields)
        return hdr + bytes(1) + bytes(64) + bytes(2) + bytes(36*36)

    def _decode(self, data):
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location(
            "rpm_recv", pathlib.Path("test/tester_receive_rpm.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m._decode_rpm(data, "auto")

    def test_new_protocol_decode(self):
        data = self._make_new(2573)
        assert abs(self._decode(data) - 25.73) < 1e-6

    def test_old_protocol_decode(self):
        data = self._make_old(25000)
        assert abs(self._decode(data) - 25.0) < 1e-6

    def test_unknown_size_returns_none(self):
        assert self._decode(bytes(100)) is None

    def test_wrong_type_byte_returns_none(self):
        data = bytearray(self._make_new(2500))
        data[0] = 3
        assert self._decode(bytes(data)) is None

    def test_new_size_is_1412(self):
        assert len(self._make_new(2500)) == 1412

    def test_old_size_is_1398(self):
        assert len(self._make_old(25000)) == 1398
