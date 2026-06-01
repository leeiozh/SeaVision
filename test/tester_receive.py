import socket
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# MY_IP   = '192.168.192.185'
MY_IP = '127.0.0.1'
IN_PORT = 4000

N_FREQS   = 64    # spec_1d frequency bins
N_DIRS    = 36    # direction bins
N_FREQ_2D = 36    # spec_2d frequency bins
RPM       = 25
F_MAX     = RPM / 120.0
F_DISPLAY = 0.20
N_DISPLAY = round(F_DISPLAY / F_MAX * N_FREQ_2D)

# ── Packet format v2.0 ─────────────────────────────────────────────────────────
# Header 52 B | spec_1d 64 B | spec_2d 36×36 = 1296 B → 1412 bytes total
#
# [0]  B  type=5
# [1]  B  pulse
# [2]  H  step_mm
# [3]  H  rpm_x100
# [4]  H  swh_sum ×100
# [5]  H  t_p_sum ×100
# [6]  H  t_m_sum ×100
# [7]  H  dir_p_sum  (integer degrees)
# [8]  H  dir_m_sum  (integer degrees)
# [9]  H  swh_win ×100
# [10] H  t_p_win ×100
# [11] H  dir_p_win
# [12] H  swh_sw1 ×100
# [13] H  t_p_sw1 ×100
# [14] H  dir_p_sw1
# [15] H  swh_sw2 ×100
# [16] H  t_p_sw2 ×100
# [17] H  dir_p_sw2
# [18] H  curr_speed ×100  (uint16, max 655 m/s)
# [19] H  curr_dir
# [20] H  wspd ×10
# [21] H  wind_dir
# [22] B  n_sys
# [23] B  quality  (0=BAD, 1=GOOD)
# [24-27] H×4  reserved
_HDR_FMT  = "<BBHHHHHHHHHHHHHHHHHHHHBBHHHH"
_HDR_SIZE = struct.calcsize(_HDR_FMT)                     # 52 bytes
_PKT_SIZE = _HDR_SIZE + N_FREQS + N_FREQ_2D * N_DIRS      # 52+64+1296 = 1412 bytes

_PULSE = {1: "SP", 2: "MP", 3: "LP"}
_F_AXIS  = np.linspace(0, F_MAX, N_FREQS)

_THETA_E   = np.linspace(0, 2 * np.pi, N_DIRS + 1)
_R_E       = np.linspace(0, F_DISPLAY, N_DISPLAY + 1)
_T2D, _R2D = np.meshgrid(_THETA_E, _R_E)
_YTICKS    = np.array([f for f in np.arange(0, F_MAX + 0.01, 0.05) if f <= F_DISPLAY + 1e-9])
_YTICK_LABELS = [f'{f:.2f}' for f in _YTICKS]

_DIR_CFG = [
    ('dir_sum',  'tomato',      'Total'),
    ('dir_win',  'deepskyblue', 'Wind sea'),
    ('dir_sw1',  'limegreen',   'Swell 1'),
    ('dir_sw2',  'orange',      'Swell 2'),
    ('wind_dir', 'gold',        'Wind backsc.'),
]

_CMAP = LinearSegmentedColormap.from_list('wave_spectrum', [
    (1.00, 1.00, 1.00),
    (0.10, 0.30, 0.85),
    (0.00, 0.75, 0.85),
    (0.15, 0.80, 0.15),
    (0.90, 0.90, 0.00),
    (1.00, 0.40, 0.00),
    (0.75, 0.00, 0.00),
], N=256)

_RENDER_DT   = 0.35
_last_render = 0.0


def _setup():
    plt.ion()
    fig = plt.figure(figsize=(15, 6))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.4, 0.9], wspace=0.38)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlabel("Frequency  [Hz]")
    ax1.set_ylabel("Power  [norm.]")
    ax1.set_xlim(0, F_MAX)
    ax1.set_ylim(0, 260)
    ax1.grid(True, alpha=0.3)
    line1d, = ax1.plot(_F_AXIS, np.zeros(N_FREQS), color='steelblue', lw=1.5)
    fill1d  = ax1.fill_between(_F_AXIS, np.zeros(N_FREQS), alpha=0.25, color='steelblue')

    ax2 = fig.add_subplot(gs[1], projection='polar')
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_ylim(0, F_DISPLAY)
    ax2.set_yticks(_YTICKS)
    ax2.set_yticklabels(_YTICK_LABELS, fontsize=7)
    ax2.set_rlabel_position(45)

    pcm = ax2.pcolormesh(_T2D, _R2D, np.zeros((N_DISPLAY, N_DIRS)),
                         cmap=_CMAP, shading='flat', vmin=0, vmax=1.0)

    dir_lines = []
    for _, clr, lbl in _DIR_CFG:
        ln, = ax2.plot([0, 0], [0, F_DISPLAY * 0.88],
                       color=clr, lw=1.8, alpha=0.85, label=lbl)
        dir_lines.append(ln)
    ax2.legend(loc='lower right', fontsize=7,
               bbox_to_anchor=(1.35, -0.05), framealpha=0.6)

    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    info = ax3.text(0.02, 0.98, "Waiting for data…",
                    transform=ax3.transAxes, va='top', ha='left',
                    fontfamily='monospace', fontsize=9, linespacing=1.75)

    fig.suptitle("Waiting for data…", fontsize=11)
    plt.tight_layout()
    plt.pause(0.05)
    return fig, ax1, line1d, fill1d, ax2, pcm, dir_lines, info


def _decode(data: bytes):
    n_hdr_1d = _HDR_SIZE + N_FREQS
    if len(data) < n_hdr_1d:
        return None

    un = struct.unpack(_HDR_FMT + f"{N_FREQS}B", data[:n_hdr_1d])
    if un[0] != 5:
        print(f"Unexpected packet type: {un[0]}")
        return None

    spec_1d = np.array(un[28:28 + N_FREQS], dtype=np.float32)

    if len(data) >= _PKT_SIZE:
        spec_2d = (np.frombuffer(data[n_hdr_1d:_PKT_SIZE], dtype=np.uint8)
                   .reshape(N_DIRS, N_FREQ_2D).astype(np.float32))
    else:
        spec_2d = np.zeros((N_DIRS, N_FREQ_2D), dtype=np.float32)

    return {
        'pulse':    un[1],
        'step':     un[2] / 1000.0,
        'rpm':      un[3] / 100.0,
        # summary (swh, t_p, t_m, dir_p, dir_m)
        'swh_sum':  un[4]  / 100.0,
        'per_sum':  un[5]  / 100.0,
        'tm_sum':   un[6]  / 100.0,
        'dir_sum':  un[7],
        'dirm_sum': un[8],
        # wind wave (swh, t_p, dir_p)
        'swh_win':  un[9]  / 100.0,
        'per_win':  un[10] / 100.0,
        'dir_win':  un[11],
        # swell 1 (swh, t_p, dir_p)
        'swh_sw1':  un[12] / 100.0,
        'per_sw1':  un[13] / 100.0,
        'dir_sw1':  un[14],
        # swell 2 (swh, t_p, dir_p)
        'swh_sw2':  un[15] / 100.0,
        'per_sw2':  un[16] / 100.0,
        'dir_sw2':  un[17],
        # current
        'curr_spd': un[18] / 100.0,
        'curr_dir': un[19],
        # wind environment
        'wspd':     un[20] / 10.0,
        'wind_dir': un[21],
        # misc
        'ide_sys':  un[22],
        'quality':  un[23],
        # spectra
        'spec_1d':  spec_1d,
        'spec_2d':  spec_2d,
    }


def _make_fill_verts(ydata):
    n = len(ydata)
    return np.column_stack([
        np.concatenate([_F_AXIS, _F_AXIS[::-1]]),
        np.concatenate([ydata, np.zeros(n)]),
    ])


def _update(fig, ax1, line1d, fill1d, ax2, pcm, dir_lines, info, d: dict):
    global _last_render

    line1d.set_ydata(d['spec_1d'])
    fill1d.set_verts([_make_fill_verts(d['spec_1d'])])

    spec_disp = d['spec_2d'][:, :N_DISPLAY]    # (N_DIRS, N_DISPLAY)
    vmax = max(float(spec_disp.max()), 1.0)
    pcm.set_array(spec_disp.T.ravel())           # (N_DISPLAY, N_DIRS)
    pcm.set_clim(0, vmax)

    for ln, (key, _, _) in zip(dir_lines, _DIR_CFG):
        angle = np.radians(d.get(key, 0))
        ln.set_xdata([angle, angle])

    SEP = '─' * 48

    def wrow(name, swh, dr, per, tm=None):
        if per < 0.1:
            return f"  {name:<11}  —"
        tm_str = f"  Tm={tm:.1f}s" if tm is not None and tm > 0.1 else ""
        return f"  {name:<11} {swh:5.2f}m  {dr:5.0f}°  Tp={per:.1f}s{tm_str}"

    txt = (
        f"  {'':11} {'SWH':>5}   {'Dir':>5}   {'T':>5}\n"
        f"  {SEP}\n"
        f"{wrow('Summary',  d['swh_sum'], d['dir_sum'], d['per_sum'], d['tm_sum'])}\n"
        f"{wrow('Wind wave',d['swh_win'], d['dir_win'], d['per_win'])}\n"
        f"{wrow('Swell 1',  d['swh_sw1'], d['dir_sw1'], d['per_sw1'])}\n"
        f"{wrow('Swell 2',  d['swh_sw2'], d['dir_sw2'], d['per_sw2'])}\n"
        f"  {SEP}\n"
        f"  {'Current:':<13} {d['curr_spd']:5.2f} m/s  →  {d['curr_dir']:4}°\n"
        f"  {'Wind:':<13}     {d['wspd']:5.2f} m/s  →  {d['wind_dir']:4}°\n"
        f"  {SEP}\n"
        f"  step: {d['step']:.3f} m   [{_PULSE.get(d['pulse'], '?')}]"
        f"   RPM: {d['rpm']:.0f}   N_sys: {d['ide_sys']}\n"
        f"  quality: {'GOOD ✓' if d['quality'] else 'BAD  ✗'}"
    )
    info.set_text(txt)

    fig.suptitle(
        f"Hs={d['swh_sum']:.2f}m   Dir={d['dir_sum']}°   Tp={d['per_sum']:.1f}s"
        f"   Tm={d['tm_sum']:.1f}s"
        f"   Curr={d['curr_spd']:.2f}m/s→{d['curr_dir']}°"
        f"   Wind={d['wspd']:.1f}m/s→{d['wind_dir']}°"
        f"   N_sys={d['ide_sys']}   [{_PULSE.get(d['pulse'], '?')}]",
        fontsize=10,
    )

    now = time.monotonic()
    if now - _last_render >= _RENDER_DT:
        fig.canvas.draw()
        fig.canvas.flush_events()
        _last_render = now


if __name__ == '__main__':
    IN_SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    IN_SOCK.bind((MY_IP, IN_PORT))

    fig, ax1, line1d, fill1d, ax2, pcm, dir_lines, info = _setup()
    while True:
        data, _ = IN_SOCK.recvfrom(_PKT_SIZE)
        d = _decode(data)
        if d is None:
            continue
        _update(fig, ax1, line1d, fill1d, ax2, pcm, dir_lines, info, d)
        print(
            f"Hs={d['swh_sum']:.2f}m  Tp={d['per_sum']:.1f}s  Tm={d['tm_sum']:.1f}s"
            f"  Dir={d['dir_sum']}°  Curr={d['curr_spd']:.2f}m/s"
            f"  Wind={d['wspd']:.1f}m/s  Nsys={d['ide_sys']}"
            f"  [{_PULSE.get(d['pulse'], '?')}]"
        )
