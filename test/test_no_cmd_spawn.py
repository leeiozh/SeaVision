"""Startup must not launch cmd.exe (or any other child process).

Guards the "black window" failure on locked-down operator stations: with the
policy DisableCMD=1, cmd.exe prints "disabled by your administrator, press any
key" and waits.  Python 3.8's platform.win32_ver() runs `ver` through cmd.exe,
and numpy.testing (pulled in by scipy.signal / scipy.optimize) triggers it at
import — so the bundle hung before its first line of output and sent nothing.

Two checks, both with subprocess.Popen replaced by a trap:
  1. platform.uname()/machine()/win32_ver() answer without spawning anything.
  2. Importing the whole processing stack spawns nothing.

Only bites on Windows (the CI runs it on the same Python 3.8 as the bundle);
elsewhere the stdlib never takes the `ver` path and the test passes trivially.

Run:  python test/test_no_cmd_spawn.py       (no radar / sockets needed)
"""
import os
import platform
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_spawned = []
_RealPopen = subprocess.Popen


class _TrapPopen(_RealPopen):
    def __init__(self, args, *a, **kw):
        _spawned.append(args)
        raise OSError(f"child process spawned at startup: {args!r}")


subprocess.Popen = _TrapPopen

import src  # noqa: E402 — applies winguard.disable_cmd_probe()


def test_platform_probe_spawns_nothing():
    platform._uname_cache = None          # force a fresh probe
    u = platform.uname()
    platform.machine()
    if sys.platform == "win32":
        release, version, _, _ = platform.win32_ver()
        assert version, "FAIL: win32_ver() lost the OS version without the `ver` probe"
    assert not _spawned, f"FAIL: platform probe spawned a process: {_spawned}"
    print(f"PASS [platform probe]: {u.system} {u.release} resolved, nothing spawned")


def test_import_spawns_nothing():
    from src.processing.processor import Processor   # noqa: F401 — numpy + scipy stack
    from src.runtime.manager import Manager          # noqa: F401
    import numpy.testing                             # noqa: F401 — the original trigger
    assert not _spawned, f"FAIL: import spawned a process: {_spawned}"
    print("PASS [import]: processing stack imported, nothing spawned")


if __name__ == "__main__":
    test_platform_probe_spawns_nothing()
    test_import_spawns_nothing()
    print("ALL TESTS PASSED")
