import gc
import socket
import struct
import time
import numpy as np
from netCDF4 import Dataset

NETCDF_FILE = "/media/leeiozh/EZH/DATA/old_radar_data/0606_4338.nc"
# SERVER_IP = '192.168.192.201'
SERVER_IP = '127.0.0.1'
PRLI_PORT = 4001
NAVI_PORT = 4002

AAP = 4096   # azimuth lines per rotation
N_PARTS = 2  # 2 × 1024 B = 2048 B = AREA_READ_DIST_PX (parser rejects parts 3 and 4 anyway)
ADP = 1192   # AREA_DISTANCE_PX — center of the processing window
ASP = 192    # AREA_SIZE_PX    — half-width of the processing window

# ── throttle ──────────────────────────────────────────────────────────────────
# Problem: OS default rmem_max ≈ 208 KB, but one frame = 8.45 MB.
# Without LINE_SLEEP, all 8192 packets burst out in <2 ms; the kernel drops
# everything beyond the ~400-packet buffer → ~95% packet loss.
# Fix: sleep LINE_SLEEP seconds after each azimuth line (= after 2 packets).
# At 200 µs/line: send time ≈ 410 ms, well within ROTATION_SLEEP.
# Permanent OS fix: sudo sysctl -w net.core.rmem_max=33554432
LINE_SLEEP = 0.00015   # 100 µs between lines; keeps send rate ~13k pkt/s < kernel buffer limit
                      # (default rmem_max ≈ 413 packets; 100µs → buffer at ~13%, no drops)
                      # For 0 sleep: sudo sysctl -w net.core.rmem_max=33554432 first

# Gap between rotations (added on top of send time).
ROTATION_SLEEP = 0.3

PRLI_SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
PRLI_SOCK.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024 * 16)
NAVI_SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_navi_packet(lat, lon, spd, hdg, cog, sog):
    NAVI_SOCK.sendto(
        struct.pack('<bbbbHhiiHH',
                    1, 0, 0, 0,
                    int(spd * 100),
                    int(hdg * 100),
                    int(lat * 1_000_000),
                    int(lon * 1_000_000),
                    int(cog * 100),
                    int(sog * 100)),
        (SERVER_IP, NAVI_PORT),
    )


def send_prli_packets(bcksctr, step, pulse):
    step_i = int(step * 1000)
    for line in range(AAP):
        for part in range(1, N_PARTS + 1):
            start = (part - 1) * 1024
            header = struct.pack('<BHHBBB', 8, line, step_i, part, N_PARTS, pulse)
            PRLI_SOCK.sendto(header + bcksctr[line][start:start + 1024].tobytes(),
                             (SERVER_IP, PRLI_PORT))
        if LINE_SLEEP > 0:
            time.sleep(LINE_SLEEP)   # throttle: let receiver drain kernel buffer


def stream_data():
    with Dataset(NETCDF_FILE, 'r') as ds:
        n = len(ds['lat_radar'])
        # Load all nav arrays upfront — small, avoids repeated HDF5 scalar reads
        lat_arr = np.asarray(ds['lat_radar'][:])
        lon_arr = np.asarray(ds['lon_radar'][:])
        sog_arr = np.asarray(ds['sog_radar'][:])
        hdg_arr = np.asarray(ds['giro_radar'][:])
        cog_arr = np.asarray(ds['cog_radar'][:])

        for i in range(n):
            # np.asarray detaches bck from the HDF5 object immediately
            bck = np.asarray(ds['bsktr_radar'][i])
            mean_bck = float(bck[:, ADP - ASP:ADP + ASP].mean())
            send_navi_packet(float(lat_arr[i]), float(lon_arr[i]),
                             float(sog_arr[i]), float(hdg_arr[i]),
                             float(cog_arr[i]), float(sog_arr[i]))
            print(f"[{i + 1}/{n}]  lat={lat_arr[i]:.6f}  lon={lon_arr[i]:.6f}  mean_bck={mean_bck:.1f}")
            send_prli_packets(bck, 1.875, 1)
            del bck
            if i % 100 == 99:
                gc.collect()
            time.sleep(ROTATION_SLEEP)
    print("Stream complete")


if __name__ == '__main__':
    stream_data()
