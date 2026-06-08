import logging
import sys


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
