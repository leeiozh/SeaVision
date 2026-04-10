import sys
from socket import socket, error, AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_RCVBUF, SO_BROADCAST


def create_inp_socket(ip, port, size=0):
    try:
        sock = socket(AF_INET, SOCK_DGRAM)
        if size != 0:
            sock.setsockopt(SOL_SOCKET, SO_RCVBUF, 1024 * 1024 * size)
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
