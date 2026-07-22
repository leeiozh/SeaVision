import logging
import sys


def _force_utf8_stdio():
    """Make stdout/stderr able to carry the non-ASCII text this app prints.

    Every log line and progress bar here contains characters outside ASCII
    (—, →, °, █, ░).  On Windows, Python writes Unicode straight to a real
    console, but a *redirected* stdout (`seavision.exe > run.log`, a service
    wrapper, a CI step) falls back to the locale codepage — cp1252 on English
    Windows, cp866 on Russian.  Neither covers the full set, and the resulting
    UnicodeEncodeError is raised inside print(), killing whichever thread was
    logging.  That took down the Process thread on the 3.8 build.

    UTF-8 + errors="replace" makes such a write impossible to fail.  Runs at
    import so every entry point (main.py, tests) is covered before any output.
    """
    for stream in (sys.stdout, sys.stderr):
        # None in windowed builds; reconfigure() exists since Python 3.7
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass   # never let logging setup break startup


_force_utf8_stdio()


def setup_logger(name="app", level=logging.INFO):
    """Return a named logger with a stdout handler (idempotent — safe to call repeatedly)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(threadName)s | %(message)s"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger
