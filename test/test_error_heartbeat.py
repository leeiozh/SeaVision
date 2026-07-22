"""Synthetic test: a processing failure must NOT silence the output and must NOT
blank the receiver's picture.

Drives the real Manager pipeline and checks two things on the wire format:

  1. On a processing exception the module keeps emitting UDP packets carrying the
     ERROR state (algo_state = 4, byte 42) — so an external module without
     console access still sees that something went wrong.

  2. Once a good result has been produced, those heartbeat packets RESEND the
     last good result (spectra + wave parameters) instead of zeros — so the
     receiver keeps showing the previous picture while the state byte changes.

  3. Before any result exists, a heartbeat is a zeroed status packet (nothing to
     show yet).

Run:  python test/test_error_heartbeat.py        (no radar / sockets needed)
"""
import os
import sys
import time
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io.structs import BackData, Navi, Output, Wave
from src.io.output import OutputSink, UdpOutputSink
from src.runtime.manager import Manager, _STATE_ERROR, _STATE_RESET


# ── minimal config doubles ──────────────────────────────────────────────────
class _Const:
    RPM = 25
    N_SHOTS = 8
    MEAN = 2
    N_FREQ = 64
    N_DIRS = 36
    N_FREQ_2D = 36


class _Pipeline:
    queue_size = 8


class _Config:
    const = _Const()
    output = {"out_times": 4}
    pipeline = _Pipeline()


class FakeInput:
    def get_bck(self):
        time.sleep(0.05)
        return BackData(step=1.875, pulse=2,
                        bck=np.zeros((16, 16), dtype=np.uint8), recv_time=0.0)

    def get_navi(self):
        return Navi(0, 0, 0, 0, 0, 0)

    def close(self):
        pass


# Sentinel good result: distinctive swh and a fully non-zero spectrum.
_GOOD_SWH = 3.5
_GOOD_SPEC = 200


def _make_good_output():
    return Output(
        pulse=2, step=1.875, rps=25.0, n_in_win=0, n_wins=1, step_area=1.875,
        n_area=16, n_start=0, cog_proc=0, sog_proc=0, max_sys=3, ide_sys=1,
        wave_sum=Wave(swh=_GOOD_SWH, t_p=8.0, t_m=6.0, d_p=120, d_m=120),
        wave_win=Wave(swh=_GOOD_SWH, t_p=8.0, d_p=120),
        wave_sw1=Wave(), wave_sw2=Wave(), n_dis=1,
        spec_1d=np.full(_Const.N_FREQ, _GOOD_SPEC, dtype=int),
        spec_2d=np.full((_Const.N_DIRS, _Const.N_FREQ_2D), _GOOD_SPEC, dtype=int),
    )


class GoodThenRaising:
    """First update() returns a good result; every later call raises.

    A fresh instance is created on each processor restart, so the pipeline
    oscillates: good result → failure (ERROR + RESET) → restart → good → …
    The Manager caches the last good Output across restarts.
    """
    rpm = 25.0

    def __init__(self):
        self._n = 0

    def update(self, back, navi):
        self._n += 1
        if self._n == 1:
            return {"out": _make_good_output(), "pulse": 2, "step": 1.875,
                    "navi": navi, "port": None}
        raise RuntimeError("synthetic failure after a good result")


class AlwaysRaising:
    rpm = 25.0

    def update(self, back, navi):
        raise RuntimeError("synthetic failure from the start")


class WireCaptureSink(OutputSink):
    """Packs each result exactly as UdpOutputSink would and records wire bytes."""

    def __init__(self):
        self.packer = UdpOutputSink.__new__(UdpOutputSink)
        self.packer.n_freqs = _Const.N_FREQ
        self.packer.n_dirs = _Const.N_DIRS
        self.packer.n_freq_2d = _Const.N_FREQ_2D
        self.packer.algo_version = 1
        self.packer.protocol = "new"
        self.records = []      # dict per packet
        self.lock = threading.Lock()

    def send(self, result):
        data = self.packer._pack_new(result.output)
        rec = {
            "len": len(data),
            "state": data[42],
            "progress": data[46],
            "swh_raw": int.from_bytes(data[6:8], "little"),   # ×100
            "spec1d_sum": sum(data[52:116]),
            "is_status": getattr(result, "is_status", False),
        }
        with self.lock:
            self.records.append(rec)

    def close(self):
        pass


def _run(processor_cls, seconds=2.0):
    sink = WireCaptureSink()
    mgr = Manager(_Config(), lambda: processor_cls(), FakeInput(), [sink])
    mgr.start()
    time.sleep(seconds)
    mgr.stop()
    with sink.lock:
        return list(sink.records)


def test_error_keeps_picture():
    recs = _run(GoodThenRaising)
    assert recs, "FAIL: no packets emitted at all"

    bad_len = [r["len"] for r in recs if r["len"] != 1412]
    assert not bad_len, f"FAIL: wrong packet size(s): {bad_len}"

    err = [r for r in recs if r["state"] == _STATE_ERROR]
    assert err, f"FAIL: no ERROR packet; states seen: {sorted({r['state'] for r in recs})}"

    # The key requirement: ERROR heartbeats resend the last good picture.
    want = round(_GOOD_SWH * 100)
    good_picture = [r for r in err if r["swh_raw"] == want and r["spec1d_sum"] > 0]
    assert good_picture, (
        "FAIL: ERROR packets blanked the picture — "
        f"swh_raw={[r['swh_raw'] for r in err][:5]}, "
        f"spec1d_sum={[r['spec1d_sum'] for r in err][:5]} (expected swh_raw={want}, spec>0)"
    )
    assert all(r["is_status"] for r in err), "FAIL: ERROR packet not flagged is_status"
    n_reset = sum(1 for r in recs if r["state"] == _STATE_RESET)
    print(f"PASS [keeps picture]: {len(err)} ERROR + {n_reset} RESET packets; "
          f"ERROR heartbeats carry last good Hs={_GOOD_SWH}m and non-zero spectra; "
          f"all 1412 bytes ({len(recs)} datagrams).")


def test_error_from_start_is_zeroed():
    recs = _run(AlwaysRaising)
    err = [r for r in recs if r["state"] == _STATE_ERROR]
    assert err, "FAIL: no ERROR packet emitted on immediate failure"
    # No good result ever existed → heartbeats must be zeroed (nothing to show).
    assert all(r["swh_raw"] == 0 and r["spec1d_sum"] == 0 for r in err), (
        "FAIL: expected zeroed status packets before any result existed"
    )
    print(f"PASS [from start]: {len(err)} zeroed ERROR packet(s) before first "
          f"result (no stale picture).")


if __name__ == "__main__":
    test_error_keeps_picture()
    test_error_from_start_is_zeroed()
    print("ALL TESTS PASSED")
