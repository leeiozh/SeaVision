import socket
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SERVER_IP = '127.0.0.1'
IN_PORT = 4000

N_FREQS   = 64    # N_FREQ  from config.ini
N_AREA    = 36    # N_DIRS  from config.ini (spectrum direction bins)
RPM       = 25    # RPM from config.ini
F_MAX     = RPM / 120.0   # Nyquist frequency [Hz] ≈ 0.208
F_DISPLAY = 0.20          # radial limit for polar display [Hz]
N_DISPLAY = round(F_DISPLAY / F_MAX * N_FREQS)

# ── Packet format ──────────────────────────────────────────────────────────────
# Full format string matches output.py exactly:
# <BBHHBBHHHHHBBHHHHHHHHHHHHHhHH + N_FREQS×B + (N_FREQS×N_AREA)×B
_HDR_FMT  = "<BBHHBBHHHHHBBHHHHHHHHHHHHHhHH"
_HDR_SIZE = struct.calcsize(_HDR_FMT)                # 52 bytes
_PKT_SIZE = _HDR_SIZE + N_FREQS + N_FREQS * N_AREA  # 2420 bytes

_PULSE = {1: "SP", 2: "MP", 3: "LP"}

_F_AXIS  = np.linspace(0, F_MAX, N_FREQS)
_THETA_E = np.linspace(0, 2 * np.pi, N_AREA + 1)
_R_E     = np.linspace(0, F_DISPLAY, N_DISPLAY + 1)
_T2D, _R2D = np.meshgrid(_THETA_E, _R_E)
_YTICKS  = np.array([f for f in np.arange(0, F_MAX + 0.01, 0.05) if f <= F_DISPLAY + 1e-9])
_YTICK_LABELS = [f'{f:.2f}' for f in _YTICKS]

# Direction marker config: (dict-key, colour, legend-label)
_DIR_CFG = [
    ('dir_sum', 'tomato',       'Total'),
    ('dir_win', 'deepskyblue',  'Wind sea'),
    ('dir_sw1', 'limegreen',    'Swell 1'),
    ('dir_sw2', 'orange',       'Swell 2'),
    ('wind_dir','gold',         'Wind backsc.'),
]

# Smooth colormap: white at zero → dark blue → cyan → green → yellow → red
_CMAP = LinearSegmentedColormap.from_list('wave_spectrum', [
    (1.00, 1.00, 1.00),
    (0.10, 0.30, 0.85),
    (0.00, 0.75, 0.85),
    (0.15, 0.80, 0.15),
    (0.90, 0.90, 0.00),
    (1.00, 0.40, 0.00),
    (0.75, 0.00, 0.00),
], N=256)

_RENDER_DT = 0.35
_last_render = 0.0


def _setup():
    plt.ion()
    fig = plt.figure(figsize=(15, 6))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.4, 0.9], wspace=0.38)

    # ── Panel 1: 1D Wave Spectrum ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlabel("Frequency  [Hz]")
    ax1.set_ylabel("Power  [norm.]")
    ax1.set_xlim(0, F_MAX)
    ax1.set_ylim(0, 260)
    ax1.grid(True, alpha=0.3)
    line1d, = ax1.plot(_F_AXIS, np.zeros(N_FREQS), color='steelblue', lw=1.5)
    fill1d  = ax1.fill_between(_F_AXIS, np.zeros(N_FREQS), alpha=0.25, color='steelblue')

    # ── Panel 2: Directional spectrum (polar) ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1], projection='polar')
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_ylim(0, F_DISPLAY)
    ax2.set_yticks(_YTICKS)
    ax2.set_yticklabels(_YTICK_LABELS, fontsize=7)
    ax2.set_rlabel_position(45)

    pcm = ax2.pcolormesh(_T2D, _R2D, np.zeros((N_DISPLAY, N_AREA)),
                         cmap=_CMAP, shading='flat', vmin=0, vmax=1.0)

    dir_lines = []
    for _, clr, lbl in _DIR_CFG:
        ln, = ax2.plot([0, 0], [0, F_DISPLAY * 0.88],
                       color=clr, lw=1.8, alpha=0.85, label=lbl)
        dir_lines.append(ln)
    ax2.legend(loc='lower right', fontsize=7,
               bbox_to_anchor=(1.35, -0.05), framealpha=0.6)

    # ── Panel 3: Parameter table ──────────────────────────────────────────────
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
    n_hdr_spec = _HDR_SIZE + N_FREQS
    if len(data) < n_hdr_spec:
        return None

    un = struct.unpack(_HDR_FMT + f"{N_FREQS}B", data[:n_hdr_spec])
    if un[0] != 5:
        print(f"Unexpected packet type: {un[0]}")
        return None

    spec_1d = np.array(un[29:29 + N_FREQS], dtype=np.float32)

    if len(data) >= _PKT_SIZE:
        spec_2d = (np.frombuffer(data[n_hdr_spec:_PKT_SIZE], dtype=np.uint8)
                   .reshape(N_AREA, N_FREQS).astype(np.float32))
    else:
        spec_2d = np.zeros((N_AREA, N_FREQS), dtype=np.float32)

    return {
        # Radar metadata
        'pulse':    un[1],
        'step':     un[2] / 1000.0,          # range resolution [m]
        'rpm':      un[3] / 100.0,           # RPM (об/мин) — кодируется как RPM × 100
        'n_in_win': un[4],                   # frames in window
        'n_wins':   un[5],                   # averagings
        'hdg':      un[8],                   # HDG [°], integer
        'cog':      round(un[9]  / 100.0),   # COG [°]
        'sog':      un[10] / 100.0,          # SOG [m/s]
        'ide_sys':  un[12],                  # number of wave systems
        # Summary wave
        'swh_sum':  un[13] / 100.0,
        'dir_sum':  round(un[14] / 100.0),   # math convention [°]
        'per_sum':  un[15] / 100.0,
        # Wind wave
        'swh_win':  un[16] / 100.0,
        'dir_win':  round(un[17] / 100.0),
        'per_win':  un[18] / 100.0,
        # Swell 1
        'swh_sw1':  un[19] / 100.0,
        'dir_sw1':  round(un[20] / 100.0),
        'per_sw1':  un[21] / 100.0,
        # Swell 2
        'swh_sw2':  un[22] / 100.0,
        'dir_sw2':  round(un[23] / 100.0),
        'per_sw2':  un[24] / 100.0,
        # Current: [7]=curr_dir [°], [11]=curr_speed [cm/s uint8]
        'curr_spd': un[11] / 100.0,
        'curr_dir': un[7],
        # Quality
        'quality':  un[25],
        # Wind: [27]=wind_dir [°], [28]=wspd×10 [0.1 m/s]
        'wind_dir': un[27],
        'wspd':     un[28] / 10.0,
        # Spectra
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

    # ── data objects (always) ─────────────────────────────────────────────────
    line1d.set_ydata(d['spec_1d'])
    fill1d.set_verts([_make_fill_verts(d['spec_1d'])])

    spec_disp = d['spec_2d'][:, :N_DISPLAY]   # (N_AREA, N_DISPLAY)
    vmax = max(float(spec_disp.max()), 1.0)
    pcm.set_array(spec_disp.T.ravel())         # (N_DISPLAY × N_AREA)
    pcm.set_clim(0, vmax)

    for ln, (key, _, _) in zip(dir_lines, _DIR_CFG):
        angle = np.radians(d.get(key, 0))
        ln.set_xdata([angle, angle])

    # ── parameter table ───────────────────────────────────────────────────────
    SEP = '─' * 46

    def wrow(name, swh, dr, per):
        if per < 0.1:
            return f"  {name:<11}  —"
        return f"  {name:<11} {swh:5.2f}m  {dr:5.0f}°  {per:5.1f}s"

    n_sys = d['ide_sys']
    txt = (
        f"  {'':11} {'SWH':>5}   {'Dir':>5}   {'T':>4}\n"
        f"  {SEP}\n"
        f"{wrow('Summary',    d['swh_sum'], d['dir_sum'], d['per_sum'])}\n"
        f"{wrow('Wind wave',  d['swh_win'], d['dir_win'], d['per_win'])}\n"
        f"{wrow('Swell 1',    d['swh_sw1'], d['dir_sw1'], d['per_sw1'])}\n"
        f"{wrow('Swell 2',    d['swh_sw2'], d['dir_sw2'], d['per_sw2'])}\n"
        f"  {SEP}\n"
        f"  {'Current:':<13} {d['curr_spd']:5.2f} m/s  →  {d['curr_dir']:4}°\n"
        f"  {'Wind:':<13}     {d['wspd']:5.2f} m/s  →  {d['wind_dir']:4}°\n"
        f"  {SEP}\n"
        f"  SOG: {d['sog']:.2f} m/s   COG: {d['cog']}°   HDG: {d['hdg']}°\n"
        f"  step: {d['step']:.3f} m   [{_PULSE.get(d['pulse'], '?')}]"
        f"   RPM: {d['rpm']:.0f}   N_sys: {n_sys}\n"
        f"  quality: {'GOOD ✓' if d['quality'] else 'BAD  ✗'}"
    )
    info.set_text(txt)

    fig.suptitle(
        f"Hs={d['swh_sum']:.2f}m   Dir={d['dir_sum']}°   T={d['per_sum']:.1f}s   "
        f"Curr={d['curr_spd']:.2f}m/s→{d['curr_dir']}°   "
        f"Wind={d['wspd']:.1f}m/s→{d['wind_dir']}°   "
        f"SOG={d['sog']:.2f}m/s   N_sys={n_sys}   [{_PULSE.get(d['pulse'], '?')}]",
        fontsize=10,
    )

    # ── throttled screen refresh ──────────────────────────────────────────────
    now = time.monotonic()
    if now - _last_render >= _RENDER_DT:
        fig.canvas.draw()
        fig.canvas.flush_events()
        _last_render = now


if __name__ == '__main__':
    IN_SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    IN_SOCK.bind((SERVER_IP, IN_PORT))

    fig, ax1, line1d, fill1d, ax2, pcm, dir_lines, info = _setup()
    while True:
        data, _ = IN_SOCK.recvfrom(_PKT_SIZE)
        d = _decode(data)
        if d is None:
            continue
        _update(fig, ax1, line1d, fill1d, ax2, pcm, dir_lines, info, d)
        print(
            f"Hs={d['swh_sum']:.2f}m  Dir={d['dir_sum']}°  T={d['per_sum']:.1f}s  "
            f"Curr={d['curr_spd']:.2f}m/s  Wind={d['wspd']:.1f}m/s  "
            f"SOG={d['sog']:.2f}m/s  COG={d['cog']}°  HDG={d['hdg']}°  "
            f"Nsys={d['ide_sys']}  [{_PULSE.get(d['pulse'], '?')}]"
        )
