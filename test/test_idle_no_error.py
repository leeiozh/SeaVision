"""Idle path must stay silent: no data is a normal state, not an error.

Guards the Python-version trap that broke the 3.8 (Windows 7-compatible) build:
on Python 3.10+ socket.timeout IS TimeoutError, on 3.8/3.9 it is a separate
OSError subclass.  A bare `except TimeoutError` therefore catches nothing on 3.8,
and a quiet "waiting for data" wait turns into an "Input error" traceback.

Two checks:
  1. Functional — a source raising socket.timeout produces only WAIT heartbeats
     and no error in the log.
  2. Structural — the tuple actually used for catching contains socket.timeout,
     and no bare `except TimeoutError` is left in the socket paths.  This one is
     meaningful on every Python version; check 1 only bites on 3.8/3.9.

Run:  python test/test_idle_no_error.py       (no radar / sockets needed)
"""
import io
import os
import re
import socket
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io.structs import Navi
from src.io.output import OutputSink
from src.io.service import SocketTimeout
from src.runtime.manager import Manager, _STATE_WAIT

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


class SilentRadar:
    """Live radar that sends nothing — raises socket.timeout like UdpInputSource."""

    def get_bck(self):
        time.sleep(0.05)
        raise socket.timeout("timed out")

    def get_navi(self):
        return Navi(0, 0, 0, 0, 0, 0)

    def close(self):
        pass


class StateSink(OutputSink):
    def __init__(self):
        self.states = []

    def send(self, result):
        self.states.append(result.output.algo_state)

    def close(self):
        pass


class NeverCalled:
    rpm = 25.0

    def update(self, back, navi):
        raise AssertionError("processor must not be called when no data arrives")


def test_idle_is_silent():
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logging.getLogger("manager").addHandler(handler)
    try:
        sink = StateSink()
        mgr = Manager(_Config(), lambda: NeverCalled(), SilentRadar(), [sink])
        mgr.start()
        time.sleep(2.5)
        mgr.stop()
    finally:
        logging.getLogger("manager").removeHandler(handler)

    text = buf.getvalue()
    assert "Input error" not in text, f"FAIL: idle path logged an error:\n{text}"
    assert sink.states, "FAIL: no heartbeat emitted while waiting for data"
    bad = {s for s in sink.states if s != _STATE_WAIT}
    assert not bad, f"FAIL: unexpected states while idle: {bad}"
    merged = "same class" if socket.timeout is TimeoutError else "separate classes"
    print(f"PASS [idle silent]: {len(sink.states)} WAIT heartbeat(s), no error logged "
          f"(Python {sys.version_info.major}.{sys.version_info.minor}, "
          f"socket.timeout/TimeoutError — {merged}).")


def test_catches_socket_timeout_on_every_python():
    assert socket.timeout in SocketTimeout, (
        "FAIL: SocketTimeout must list socket.timeout explicitly — on Python 3.8 "
        "it is not a TimeoutError subclass"
    )
    assert TimeoutError in SocketTimeout, "FAIL: SocketTimeout must also list TimeoutError"

    # No bare `except TimeoutError` may remain where a socket read can time out.
    offenders = []
    for rel in ("src/io/input.py", "src/runtime/manager.py"):
        for i, line in enumerate(open(os.path.join(_ROOT, rel), encoding="utf-8"), 1):
            if re.search(r"except\s+TimeoutError\s*[:(]", line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, (
        "FAIL: bare `except TimeoutError` misses socket.timeout on Python 3.8 — "
        "use SocketTimeout:\n  " + "\n  ".join(offenders)
    )
    print("PASS [catch tuple]: SocketTimeout covers both classes; no bare "
          "`except TimeoutError` in the socket paths.")


if __name__ == "__main__":
    test_catches_socket_timeout_on_every_python()
    test_idle_is_silent()
    print("ALL TESTS PASSED")
