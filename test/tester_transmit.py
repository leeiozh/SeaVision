import argparse
import gc
import socket
import struct
import time
import numpy as np
from netCDF4 import Dataset

NETCDF_FILE = "/media/leeiozh/EZH/DATA/old_radar_data/0606_4392.nc"
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
# At 150 µs/line: send time ≈ 614 ms, well within ROTATION_SLEEP.
# Permanent OS fix: sudo sysctl -w net.core.rmem_max=33554432
LINE_SLEEP = 0.00015   # 150 µs between lines; keeps send rate ~6.7k pkt/s < kernel buffer limit
                       # (default rmem_max ≈ 413 packets; 150µs → buffer at ~6%, no drops)
                       # For 0 sleep: sudo sysctl -w net.core.rmem_max=33554432 first

# Gap between rotations (added on top of send time).
# Send time ≈ AAP * LINE_SLEEP = 4096 * 0.00015 ≈ 0.614 s.
# To mimic real RPM: ROTATION_SLEEP = 60/RPM - AAP*LINE_SLEEP
#   25 RPM → ROTATION_SLEEP ≈ 2.4 - 0.614 = 1.786 s
#   30 RPM → ROTATION_SLEEP ≈ 2.0 - 0.614 = 1.386 s
ROTATION_SLEEP = 0.3   # default fast mode for functional testing (≈ 65 RPM equivalent)

def _parse_args():
    p = argparse.ArgumentParser(description="SeaVision radar test transmitter")
    p.add_argument("--rotation-sleep", type=float, default=None,
                   help="Inter-rotation gap [s] after packet burst (overrides ROTATION_SLEEP). "
                        "To mimic RPM=R: --rotation-sleep $(python3 -c "
                        "'print(round(60/R - 4096*0.00015, 3))')")
    p.add_argument("--line-sleep", type=float, default=None,
                   help="Sleep between azimuth lines [s] (overrides LINE_SLEEP). "
                        "Default 0.00015. Decrease only after raising rmem_max.")
    p.add_argument("--rpm", type=float, default=None,
                   help="Target RPM shortcut: auto-computes --rotation-sleep = 60/RPM - AAP*line_sleep.")
    p.add_argument("--file", type=str, default=NETCDF_FILE,
                   help="NetCDF file to stream (default: %(default)s)")
    return p.parse_args()


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


def send_prli_packets(bcksctr, step, pulse, line_sleep=LINE_SLEEP):
    step_i = int(step * 1000)
    for line in range(AAP):
        for part in range(1, N_PARTS + 1):
            start = (part - 1) * 1024
            header = struct.pack('<BHHBBB', 8, line, step_i, part, N_PARTS, pulse)
            PRLI_SOCK.sendto(header + bcksctr[line][start:start + 1024].tobytes(),
                             (SERVER_IP, PRLI_PORT))
        if line_sleep > 0:
            time.sleep(line_sleep)   # throttle: let receiver drain kernel buffer


def stream_data(netcdf_file=NETCDF_FILE, rotation_sleep=ROTATION_SLEEP, line_sleep=LINE_SLEEP):
    period = AAP * line_sleep + rotation_sleep
    print(f"Streaming {netcdf_file}")
    print(f"  line_sleep={line_sleep*1e6:.0f} µs  rotation_sleep={rotation_sleep:.3f} s")
    print(f"  frame period ≈ {period:.3f} s  →  simulated RPM ≈ {60/period:.2f}")
    with Dataset(netcdf_file, 'r') as ds:
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
            send_prli_packets(bck, 1.875, 1, line_sleep=line_sleep)
            del bck
            if i % 100 == 99:
                gc.collect()
            time.sleep(rotation_sleep)
    print("Stream complete")


if __name__ == '__main__':
    args = _parse_args()
    ls = args.line_sleep if args.line_sleep is not None else LINE_SLEEP
    if args.rpm is not None:
        rs = 60.0 / args.rpm - AAP * ls
        if rs < 0:
            print(f"WARNING: requested RPM={args.rpm} too high for line_sleep={ls*1e6:.0f}µs "
                  f"(send time {AAP*ls:.3f}s > rotation period {60/args.rpm:.3f}s). "
                  f"Setting rotation_sleep=0.")
            rs = 0.0
    elif args.rotation_sleep is not None:
        rs = args.rotation_sleep
    else:
        rs = ROTATION_SLEEP
    stream_data(netcdf_file=args.file, rotation_sleep=rs, line_sleep=ls)
