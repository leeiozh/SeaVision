import logging
import numpy as np
from struct import pack
from datetime import datetime
from src.io.service import create_out_socket
from src.io.structs import Wave, Navi, ProcessResult

_log = logging.getLogger(__name__)

# ── Physical sanity limits applied before any transmission ────────────────────
# Values outside these ranges are clipped to the boundary and a warning is logged.
# They reflect hard physical upper bounds for surface gravity waves measured by
# a rotating maritime radar; exceedances indicate algorithmic artifacts.
_PHYS_SWH_MAX    = 20.0   # m   — above max reliable open-ocean Hs record (~18 m)
_PHYS_PERIOD_MAX = 30.0   # s   — surface gravity wave swell ceiling; bin-1 artifact can
                           #       reach ~600 s at 25 RPM — must be clipped
_PHYS_WSPD_MAX   = 30.0   # m/s — Beaufort 12 = 32.7 m/s; radar estimate saturates earlier
_PHYS_CURR_MAX   =  3.0   # m/s — matches processor clip (_MAX_CURRENT = 3.0 m/s)


def _phys_clip(o) -> None:
    """Clip all scalar wave parameters to physical bounds in-place.

    Called once per ProcessResult before any sink serialises the Output.
    Idempotent — safe to call from multiple sinks on the same object.
    Logs a WARNING the first time a field is clipped (use DEBUG for every event).
    """
    def _cw(w: Wave, tag: str) -> None:
        if not np.isfinite(w.swh) or w.swh < 0.0:
            w.swh = 0.0
        elif w.swh > _PHYS_SWH_MAX:
            _log.warning("SWH clip [%s]: %.2f → %.2f m", tag, w.swh, _PHYS_SWH_MAX)
            w.swh = _PHYS_SWH_MAX
        if not np.isfinite(w.t_p) or w.t_p < 0.0:
            w.t_p = 0.0
        elif w.t_p > _PHYS_PERIOD_MAX:
            _log.warning("T_p  clip [%s]: %.1f → %.1f s", tag, w.t_p, _PHYS_PERIOD_MAX)
            w.t_p = _PHYS_PERIOD_MAX
        if not np.isfinite(w.t_m) or w.t_m < 0.0:
            w.t_m = 0.0
        elif w.t_m > _PHYS_PERIOD_MAX:
            _log.warning("T_m  clip [%s]: %.1f → %.1f s", tag, w.t_m, _PHYS_PERIOD_MAX)
            w.t_m = _PHYS_PERIOD_MAX

    _cw(o.wave_sum, "sum")
    _cw(o.wave_win, "win")
    _cw(o.wave_sw1, "sw1")
    _cw(o.wave_sw2, "sw2")

    curr = getattr(o, 'curr_speed', 0.0)
    if not np.isfinite(curr):
        o.curr_speed = 0.0
    elif curr > _PHYS_CURR_MAX:
        _log.warning("curr_speed clip: %.2f → %.2f m/s", curr, _PHYS_CURR_MAX)
        o.curr_speed = _PHYS_CURR_MAX
    elif curr < 0.0:
        o.curr_speed = 0.0

    wspd = getattr(o, 'wspd', 0.0)
    if not np.isfinite(wspd):
        o.wspd = 0.0
    elif wspd > _PHYS_WSPD_MAX:
        _log.warning("wspd clip: %.2f → %.2f m/s", wspd, _PHYS_WSPD_MAX)
        o.wspd = _PHYS_WSPD_MAX
    elif wspd < 0.0:
        o.wspd = 0.0


class OutputSink:
    """Abstract base for all output destinations."""

    def send(self, result: ProcessResult):
        raise NotImplementedError

    def close(self):
        pass


def _u16(x):
    """Clip a value to the uint16 range. Returns 0 for NaN/inf."""
    if not np.isfinite(x):
        return 0
    return int(np.clip(round(x), 0, 65535))


def _dir(x):
    """Convert a direction to integer degrees [0, 359]. Returns 0 for NaN/inf."""
    if not np.isfinite(x):
        return 0
    return int(round(x)) % 360


class UdpOutputSink(OutputSink):
    """Pack a ProcessResult into a UDP datagram and send it.

    Two wire formats are supported, selected by ``protocol``:

      "new" (default) — protocol v2.0, 1412 bytes:
        52 bytes header | N_FREQ bytes spec_1d | N_DIRS×N_FREQ_2D bytes spec_2d.
        See udp_protocol.docx for the authoritative field definitions.

      "old" — legacy protocol (sv_protocol_2204.docx), 1398 bytes:
        102 bytes header | N_FREQ bytes spec_1d | N_DIRS×N_FREQ_2D bytes spec_2d.
        No pulse/step, no mean (t_m/d_m) fields, no n_sys/quality/algo_version.
        Different scaling: rpm ×1000, wind speed ×100, current speed ×1000.

    The legacy format is a temporary compatibility mode for older receivers.
    """

    def __init__(self, server_ip, server_port, n_freqs, n_dirs, n_freq_2d,
                 algo_version=1, protocol="new"):
        self.out_socket = create_out_socket(server_port, 2)
        self.server_ip = server_ip
        self.server_port = server_port
        self.n_freqs = n_freqs
        self.n_dirs = n_dirs
        self.n_freq_2d = n_freq_2d
        self.algo_version = algo_version
        self.protocol = str(protocol).lower()

    def send(self, result: ProcessResult):
        _phys_clip(result.output)
        if self.protocol == "old":
            data = self._pack_old(result.output)
        else:
            data = self._pack_new(result.output)
        self.out_socket.sendto(data, (self.server_ip, self.server_port))

    def _pack_new(self, o):
        # Header layout (52 bytes):
        #   BB HH                   — type, pulse, step_mm, rpm_x100
        #   HHHHH                   — summary: swh, t_p, t_m, dir_p, dir_m
        #   HHH HHH HHH             — wind/sw1/sw2: swh, t_p, dir_p each
        #   H H HH                  — curr_speed, curr_dir, wspd_x10, wind_dir
        #   BB HHHH                 — n_sys, quality, algo_version, reserved×3
        data = pack(
            f"<BBHHHHHHHHHHHHHHHHHHHHBBHHHH"
            f"{self.n_freqs}B{self.n_freq_2d * self.n_dirs}B",
            5, int(np.clip(o.pulse, 0, 255)),
            _u16(o.step * 1000), _u16(o.rps * 100),
            # summary — all 5 fields
            _u16(o.wave_sum.swh * 100), _u16(o.wave_sum.t_p * 100), _u16(o.wave_sum.t_m * 100),
            _dir(o.wave_sum.d_p), _dir(o.wave_sum.d_m),
            # wind wave — swh, t_p, dir_p only
            _u16(o.wave_win.swh * 100), _u16(o.wave_win.t_p * 100), _dir(o.wave_win.d_p),
            # swell 1 — swh, t_p, dir_p only
            _u16(o.wave_sw1.swh * 100), _u16(o.wave_sw1.t_p * 100), _dir(o.wave_sw1.d_p),
            # swell 2 — swh, t_p, dir_p only
            _u16(o.wave_sw2.swh * 100), _u16(o.wave_sw2.t_p * 100), _dir(o.wave_sw2.d_p),
            _u16(getattr(o, 'curr_speed', 0.0) * 100), _dir(getattr(o, 'curr_dir', 0.0)),
            _u16(getattr(o, 'wspd', 0.0) * 10), _dir(getattr(o, 'wind_dir', 0.0)),
            o.ide_sys, o.n_dis,
            self.algo_version, 0, 0, 0,
            *o.spec_1d,
            *o.spec_2d.flatten(),
        )
        return data

    def _pack_old(self, o):
        """Legacy 1398-byte layout (sv_protocol_2204.docx, little-endian).

        Header (102 bytes):
          B                  — type = 5
          H×17               — summary(swh,t_p,dir_p), wind/sw1/sw2(swh,t_p,dir),
                               wind(speed,dir), current(speed,dir), rpm
          B                  — N_FREQ count
          {N_FREQ}B          — spec_1d [0–255]
          B B                — N_FREQ_2D count, N_DIRS count
          {N_FREQ_2D×N_DIRS}B — spec_2d [0–255]

        Scaling differs from v2.0: rpm in thousandths (×1000), wind speed in
        hundredths of m/s (×100), current speed in thousandths of m/s (×1000).
        Direction values keep the same convention emitted by the processor.
        """
        data = pack(
            f"<B17HB{self.n_freqs}BBB{self.n_freq_2d * self.n_dirs}B",
            5,
            # summary — swh, t_p, dir_p
            _u16(o.wave_sum.swh * 100), _u16(o.wave_sum.t_p * 100), _dir(o.wave_sum.d_p),
            # wind wave — swh, t_p, dir_p
            _u16(o.wave_win.swh * 100), _u16(o.wave_win.t_p * 100), _dir(o.wave_win.d_p),
            # swell 1 — swh, t_p, dir_p
            _u16(o.wave_sw1.swh * 100), _u16(o.wave_sw1.t_p * 100), _dir(o.wave_sw1.d_p),
            # swell 2 — swh, t_p, dir_p
            _u16(o.wave_sw2.swh * 100), _u16(o.wave_sw2.t_p * 100), _dir(o.wave_sw2.d_p),
            # wind — speed (×100 → hundredths m/s), dir
            _u16(getattr(o, 'wspd', 0.0) * 100), _dir(getattr(o, 'wind_dir', 0.0)),
            # current — speed (×1000 → thousandths m/s), dir
            _u16(getattr(o, 'curr_speed', 0.0) * 1000), _dir(getattr(o, 'curr_dir', 0.0)),
            # antenna rate — thousandths of rpm (×1000)
            _u16(o.rps * 1000),
            # spec_1d
            self.n_freqs,
            *o.spec_1d,
            # spec_2d
            self.n_freq_2d, self.n_dirs,
            *o.spec_2d.flatten(),
        )
        return data

    def close(self):
        self.out_socket.close()


class CSVOutputSink(OutputSink):
    """Write processing results to four CSV files per session.

    Files are opened once at construction and kept open (buffering=1) for the
    session lifetime.  Filenames: {installation_id}_{timestamp}_*.csv, or
    {timestamp}_*.csv when installation_id is "default".

      _params.csv — scalar wave parameters + 1-D spectrum per output cycle.
      _port.csv   — ω-k portrait (N_SHOTS//2 × K_NUM), one block per cycle.
      _spec.csv   — directional spectrum (N_DIRS × N_FREQ_2D), one block per cycle.
      _navi.csv   — navigation snapshot per output cycle.
    """

    def __init__(self, save_path, constants):
        self.save_path = save_path
        now_time = datetime.now()
        ts = now_time.strftime("%Y%m%dT%H%M%S")
        inst = getattr(constants, "installation_id", "")
        prefix = f"{inst}_{ts}" if inst and inst != "default" else ts
        self.k_num = constants.K_NUM
        self.n_freq = constants.N_FREQ
        self.n_freq_2d = constants.N_FREQ_2D
        self.n_shots = constants.N_SHOTS
        self.n_dirs = constants.N_DIRS

        # Keep file handles open — avoids repeated open/close churn on every result
        self._f_params = open(save_path + prefix + "_params.csv", "w", buffering=1)
        self._f_port   = open(save_path + prefix + "_port.csv",   "w", buffering=1)
        self._f_spec   = open(save_path + prefix + "_spec.csv",   "w", buffering=1)
        self._f_navi   = open(save_path + prefix + "_navi.csv",   "w", buffering=1)

        self._f_params.write("datetime;pulse;step;swh;t_p;d_p;d_m;t_m;freq;\n")
        self._f_port.write(f"({self.n_shots},{self.k_num})\n")
        self._f_spec.write(f"({self.n_dirs},{self.n_freq_2d})\n")
        self._f_navi.write("datetime,lat,lon,spd,sog,cog,hdg\n")

    def send(self, result: ProcessResult):
        _phys_clip(result.output)
        o = result.output
        dtime = datetime.now().strftime("%Y%m%dT%H%M%S")
        self._save_params(dtime, o.pulse, o.step, o.wave_sum, o.spec_1d)
        self._save_spec(o.spec_2d)
        if result.port is not None:
            self._save_port(result.port)
        self._save_navi(dtime, result.navi)

    def _save_params(self, dtime, pulse: int, step: float, wave: Wave, freq: np.ndarray):
        self._f_params.write(dtime + f";{pulse};{step};" + wave.print())
        for v in freq:
            self._f_params.write(f"{v:.0f};")
        self._f_params.write("\n")

    def _save_port(self, port: np.ndarray):
        for row in port:
            self._f_port.write(";".join(f"{v:.0f}" for v in row))
            self._f_port.write(";\n")
        self._f_port.write("\n")

    def _save_spec(self, spec: np.ndarray):
        for row in spec:
            self._f_spec.write(";".join(f"{v:.0f}" for v in row))
            self._f_spec.write(";\n")
        self._f_spec.write("\n")

    def _save_navi(self, dtime, navi: Navi):
        self._f_navi.write(
            f"{dtime},{navi.lat},{navi.lon},{navi.spd},{navi.sog},{navi.cog},{navi.hdg}\n")

    def close(self):
        for f in (self._f_params, self._f_port, self._f_spec, self._f_navi):
            try:
                f.close()
            except Exception:
                pass
