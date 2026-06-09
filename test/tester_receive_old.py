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
F_DISPLAY = 0.20  # Hz — desired radial cap (clamped to Nyquist f_max if lower)
_DEFAULT_RPM = 25.0   # initial guess until the first packet sets the real rate

# ── Packet format (legacy, sv_protocol_2204.docx) ───────────────────────────────
# Header 102 B | spec_1d 64 B | spec_2d 36×36 = 1296 B → 1398 bytes total
#
# [0]   B  type = 5
# [1]   H  swh_sum  ×100  (hundredths of m)
# [2]   H  t_p_sum  ×100  (hundredths of s)
# [3]   H  dir_p_sum      (integer degrees)
# [4]   H  swh_win  ×100
# [5]   H  t_p_win  ×100
# [6]   H  dir_p_win
# [7]   H  swh_sw1  ×100
# [8]   H  t_p_sw1  ×100
# [9]   H  dir_p_sw1
# [10]  H  swh_sw2  ×100
# [11]  H  t_p_sw2  ×100
# [12]  H  dir_p_sw2
# [13]  H  wind speed  ×100  (hundredths of m/s)
# [14]  H  wind dir
# [15]  H  current speed ×1000  (thousandths of m/s)
# [16]  H  current dir
# [17]  H  rpm ×1000  (thousandths of rpm)
# [18]  B  N_FREQ count
# then  N_FREQ bytes spec_1d
# then  B N_FREQ_2D count, B N_DIRS count
# then  N_FREQ_2D×N_DIRS bytes spec_2d
_HDR_FMT  = "<B17H"
_HDR_SIZE = struct.calcsize(_HDR_FMT)                          # 35 bytes
# full header up to (and including) the spec_1d count byte
_HDR1D_OFF = _HDR_SIZE + 1                                     # 36 bytes
_SPEC2D_HDR = 2                                                # two count bytes
_PKT_SIZE = _HDR1D_OFF + N_FREQS + _SPEC2D_HDR + N_FREQ_2D * N_DIRS   # 1398 bytes

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


def _freq_geometry(rpm: float) -> dict:
    """Frequency-axis geometry derived from the antenna rate.

    The spectra span ω ∈ [0, π·rpm/60] rad/s, i.e. f ∈ [0, rpm/120] Hz.  The
    polar plot is capped at F_DISPLAY (or the Nyquist f_max, whichever is lower).
    """
    f_max  = max(rpm, 1.0) / 120.0
    f_disp = min(F_DISPLAY, f_max)
    n_disp = int(round(f_disp / f_max * N_FREQ_2D))
    n_disp = max(1, min(N_FREQ_2D, n_disp))
    f_axis = np.linspace(0, f_max, N_FREQS)
    theta_e = np.linspace(0, 2 * np.pi, N_DIRS + 1)
    r_e     = np.linspace(0, f_disp, n_disp + 1)
    t2d, r2d = np.meshgrid(theta_e, r_e)
    yticks = np.array([f for f in np.arange(0, f_max + 1e-9, 0.05) if f <= f_disp + 1e-9])
    return dict(rpm=rpm, f_max=f_max, f_disp=f_disp, n_disp=n_disp,
                f_axis=f_axis, t2d=t2d, r2d=r2d, yticks=yticks)


def _fill_verts(f_axis, ydata):
    n = len(ydata)
    return np.column_stack([
        np.concatenate([f_axis, f_axis[::-1]]),
        np.concatenate([ydata, np.zeros(n)]),
    ])


def _setup():
    plt.ion()
    fig = plt.figure(figsize=(15, 6))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.4, 0.9], wspace=0.38)
    geom = _freq_geometry(_DEFAULT_RPM)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlabel("Frequency  [Hz]")
    ax1.set_ylabel("Power  [norm.]")
    ax1.set_xlim(0, geom['f_max'])
    ax1.set_ylim(0, 260)
    ax1.grid(True, alpha=0.3)
    line1d, = ax1.plot(geom['f_axis'], np.zeros(N_FREQS), color='steelblue', lw=1.5)
    fill1d  = ax1.fill_between(geom['f_axis'], np.zeros(N_FREQS), alpha=0.25, color='steelblue')

    ax2 = fig.add_subplot(gs[1], projection='polar')
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_ylim(0, geom['f_disp'])
    ax2.set_yticks(geom['yticks'])
    ax2.set_yticklabels([f'{f:.2f}' for f in geom['yticks']], fontsize=7)
    ax2.set_rlabel_position(45)

    pcm = ax2.pcolormesh(geom['t2d'], geom['r2d'], np.zeros((geom['n_disp'], N_DIRS)),
                         cmap=_CMAP, shading='flat', vmin=0, vmax=1.0)

    dir_lines = []
    for _, clr, lbl in _DIR_CFG:
        ln, = ax2.plot([0, 0], [0, geom['f_disp'] * 0.88],
                       color=clr, lw=1.8, alpha=0.85, label=lbl)
        dir_lines.append(ln)
    ax2.legend(loc='lower right', fontsize=7,
               bbox_to_anchor=(1.35, -0.05), framealpha=0.6)

    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    info = ax3.text(0.02, 0.98, "Waiting for data…",
                    transform=ax3.transAxes, va='top', ha='left',
                    fontfamily='monospace', fontsize=9, linespacing=1.75)

    fig.suptitle("Waiting for data… (legacy protocol)", fontsize=11)
    plt.tight_layout()
    plt.pause(0.05)
    return dict(fig=fig, ax1=ax1, line1d=line1d, fill1d=fill1d, ax2=ax2,
                pcm=pcm, dir_lines=dir_lines, info=info, geom=geom)


def _apply_rpm(H, rpm: float):
    """Rebuild frequency axes from the packet rpm when it changes meaningfully.

    The polar mesh encodes the frequency→radius mapping (f_max = rpm/120), so a
    changed rate requires recreating the QuadMesh; the 1-D axis is rescaled too.
    """
    if rpm <= 0:
        return
    cur = H['geom']['rpm']
    if abs(rpm - cur) <= max(0.01, 0.005 * cur):
        return
    geom = _freq_geometry(rpm)
    H['geom'] = geom

    H['ax1'].set_xlim(0, geom['f_max'])
    H['line1d'].set_xdata(geom['f_axis'])

    ax2 = H['ax2']
    ax2.set_ylim(0, geom['f_disp'])
    ax2.set_yticks(geom['yticks'])
    ax2.set_yticklabels([f'{f:.2f}' for f in geom['yticks']], fontsize=7)
    for ln in H['dir_lines']:
        ln.set_ydata([0, geom['f_disp'] * 0.88])

    H['pcm'].remove()
    H['pcm'] = ax2.pcolormesh(
        geom['t2d'], geom['r2d'], np.zeros((geom['n_disp'], N_DIRS)),
        cmap=_CMAP, shading='flat', vmin=0, vmax=1.0)


def _decode(data: bytes):
    if len(data) < _HDR1D_OFF + N_FREQS:
        return None

    un = struct.unpack(_HDR_FMT, data[:_HDR_SIZE])
    if un[0] != 5:
        print(f"Unexpected packet type: {un[0]}")
        return None

    spec_1d = np.frombuffer(data[_HDR1D_OFF:_HDR1D_OFF + N_FREQS],
                            dtype=np.uint8).astype(np.float32)

    spec2d_off = _HDR1D_OFF + N_FREQS + _SPEC2D_HDR
    if len(data) >= _PKT_SIZE:
        spec_2d = (np.frombuffer(data[spec2d_off:_PKT_SIZE], dtype=np.uint8)
                   .reshape(N_DIRS, N_FREQ_2D).astype(np.float32))
    else:
        spec_2d = np.zeros((N_DIRS, N_FREQ_2D), dtype=np.float32)

    return {
        # summary (swh, t_p, dir_p)
        'swh_sum':  un[1]  / 100.0,
        'per_sum':  un[2]  / 100.0,
        'dir_sum':  un[3],
        # wind wave (swh, t_p, dir_p)
        'swh_win':  un[4]  / 100.0,
        'per_win':  un[5]  / 100.0,
        'dir_win':  un[6],
        # swell 1 (swh, t_p, dir_p)
        'swh_sw1':  un[7]  / 100.0,
        'per_sw1':  un[8]  / 100.0,
        'dir_sw1':  un[9],
        # swell 2 (swh, t_p, dir_p)
        'swh_sw2':  un[10] / 100.0,
        'per_sw2':  un[11] / 100.0,
        'dir_sw2':  un[12],
        # wind environment (speed ×100 → m/s, dir)
        'wspd':     un[13] / 100.0,
        'wind_dir': un[14],
        # current (speed ×1000 → m/s, dir)
        'curr_spd': un[15] / 1000.0,
        'curr_dir': un[16],
        # antenna rate (×1000 → rpm)
        'rpm':      un[17] / 1000.0,
        # spectra
        'spec_1d':  spec_1d,
        'spec_2d':  spec_2d,
    }


def _update(H, d: dict):
    global _last_render

    _apply_rpm(H, d['rpm'])
    geom = H['geom']

    H['line1d'].set_ydata(d['spec_1d'])
    H['fill1d'].set_verts([_fill_verts(geom['f_axis'], d['spec_1d'])])

    spec_disp = d['spec_2d'][:, :geom['n_disp']]    # (N_DIRS, n_disp)
    vmax = max(float(spec_disp.max()), 1.0)
    H['pcm'].set_array(spec_disp.T.ravel())           # (n_disp, N_DIRS)
    H['pcm'].set_clim(0, vmax)

    for ln, (key, _, _) in zip(H['dir_lines'], _DIR_CFG):
        angle = np.radians(d.get(key, 0))
        ln.set_xdata([angle, angle])

    SEP = '─' * 48

    def wrow(name, swh, dr, per):
        if per < 0.1:
            return f"  {name:<11}  —"
        return f"  {name:<11} {swh:5.2f}m  {dr:5.0f}°  Tp={per:.1f}s"

    txt = (
        f"  {'':11} {'SWH':>5}   {'Dir':>5}   {'T':>5}\n"
        f"  {SEP}\n"
        f"{wrow('Summary',  d['swh_sum'], d['dir_sum'], d['per_sum'])}\n"
        f"{wrow('Wind wave',d['swh_win'], d['dir_win'], d['per_win'])}\n"
        f"{wrow('Swell 1',  d['swh_sw1'], d['dir_sw1'], d['per_sw1'])}\n"
        f"{wrow('Swell 2',  d['swh_sw2'], d['dir_sw2'], d['per_sw2'])}\n"
        f"  {SEP}\n"
        f"  {'Current:':<13} {d['curr_spd']:5.2f} m/s  →  {d['curr_dir']:4}°\n"
        f"  {'Wind:':<13}     {d['wspd']:5.2f} m/s  →  {d['wind_dir']:4}°\n"
        f"  {SEP}\n"
        f"  RPM: {d['rpm']:.2f}   [legacy protocol]"
    )
    H['info'].set_text(txt)

    H['fig'].suptitle(
        f"Hs={d['swh_sum']:.2f}m   Dir={d['dir_sum']}°   Tp={d['per_sum']:.1f}s"
        f"   Curr={d['curr_spd']:.2f}m/s→{d['curr_dir']}°"
        f"   Wind={d['wspd']:.1f}m/s→{d['wind_dir']}°   RPM={d['rpm']:.2f}  [legacy]",
        fontsize=10,
    )

    now = time.monotonic()
    if now - _last_render >= _RENDER_DT:
        H['fig'].canvas.draw()
        H['fig'].canvas.flush_events()
        _last_render = now


if __name__ == '__main__':
    IN_SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    IN_SOCK.bind((MY_IP, IN_PORT))

    H = _setup()
    while True:
        data, _ = IN_SOCK.recvfrom(_PKT_SIZE)
        d = _decode(data)
        if d is None:
            continue
        _update(H, d)
        print(
            f"Hs={d['swh_sum']:.2f}m  Tp={d['per_sum']:.1f}s"
            f"  Dir={d['dir_sum']}°  Curr={d['curr_spd']:.2f}m/s"
            f"  Wind={d['wspd']:.1f}m/s  RPM={d['rpm']:.2f}  [legacy]"
        )
