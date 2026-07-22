import sys
from socket import (socket, error, timeout as _socket_timeout,
                    AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_RCVBUF, SO_SNDBUF)

_FRAME_SIZE_BYTES = 4096 * 2 * 1032   # ≈ 8.45 MB

# Catch socket read timeouts on every supported Python.
#
# On Python 3.10+ socket.timeout IS TimeoutError (they were merged), so a bare
# `except TimeoutError` works.  On 3.8/3.9 they are DIFFERENT classes — both
# derive from OSError, but socket.timeout is not a TimeoutError.  The Windows
# bundle is built on 3.8 (Win7 compatibility), where a bare `except TimeoutError`
# silently misses every socket timeout: what should be a quiet "no data yet"
# wait turns into an "Input error" traceback.  Always catch this tuple.
SocketTimeout = (_socket_timeout, TimeoutError)


def create_inp_socket(ip, port, size=0):
    """Create, configure, and bind a UDP receive socket.

    size — receive buffer size in MB (0 = OS default).  Warns if the actual
    buffer is smaller than one full radar frame (~8.45 MB).
    Calls sys.exit(1) on bind failure.
    """
    try:
        sock = socket(AF_INET, SOCK_DGRAM)
        if size != 0:
            requested = 1024 * 1024 * size
            sock.setsockopt(SOL_SOCKET, SO_RCVBUF, requested)
            actual = sock.getsockopt(SOL_SOCKET, SO_RCVBUF)
            if actual < _FRAME_SIZE_BYTES:
                print(
                    f'[service] WARNING: UDP recv buffer {actual//1024} KB < '
                    f'frame size {_FRAME_SIZE_BYTES//1024//1024} MB — packet drops likely!\n'
                    f'  Fix: sudo sysctl -w net.core.rmem_max=33554432'
                )
            else:
                print(f'[service] UDP recv buffer OK: {actual//1024//1024} MB')
        sock.bind((ip, port))
        return sock
    except error as err:
        print(f"Failed to create and bind socket to {ip}:{port}. Error: {err}")
        sys.exit(1)


def create_out_socket(port, size=0):
    """Create an unbound UDP send socket.  size — send buffer in MB (0 = OS default)."""
    try:
        sock = socket(AF_INET, SOCK_DGRAM)
        if size != 0:
            sock.setsockopt(SOL_SOCKET, SO_SNDBUF, 1024 * 1024 * size)
        return sock
    except error as err:
        print(f"Failed to create and bind socket to {port}. Error: {err}")
        sys.exit(1)
