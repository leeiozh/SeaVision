import sys
from socket import socket, error, AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_RCVBUF, SO_BROADCAST

_FRAME_SIZE_BYTES = 4096 * 2 * 1032   # ≈ 8.45 MB


def create_inp_socket(ip, port, size=0):
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
    try:
        sock = socket(AF_INET, SOCK_DGRAM)
        if size != 0:
            sock.setsockopt(SOL_SOCKET, SO_BROADCAST, 1024 * 1024 * size)
        return sock
    except error as err:
        print(f"Failed to create and bind socket to {port}. Error: {err}")
        sys.exit(1)
