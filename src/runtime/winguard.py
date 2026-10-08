import sys


def disable_cmd_probe():
    """Stop the stdlib from launching cmd.exe behind our back on Windows.

    Python 3.8's platform.win32_ver() learns the OS build by running the shell
    command `ver` (platform._syscmd_ver → subprocess, shell=True → cmd.exe).
    Nothing in this app asks for it, but numpy.testing calls platform.machine()
    at import, and scipy.signal / scipy.optimize import numpy.testing — so the
    bundle spawns cmd.exe on every start, before the first line of output.

    On a locked-down operator station the policy DisableCMD=1 makes cmd.exe
    print "The command prompt has been disabled by your administrator. Press
    any key to continue" and wait for a key.  The text goes into subprocess's
    pipe (invisible), the wait is on our console: the window stays black, the
    import never returns, no packet is ever sent.

    Replacing the probe with a no-op makes win32_ver() fall back to
    sys.getwindowsversion(), which is what it does anyway when `ver` fails.
    Must run before numpy/scipy are imported — src/__init__.py does that.
    """
    if sys.platform != "win32":
        return
    import platform

    def _no_syscmd_ver(system='', release='', version='', supported_platforms=()):
        return system, release, version

    platform._syscmd_ver = _no_syscmd_ver
