import numpy as np
from typing import Optional
from struct import calcsize, unpack, error


class ProtocolError(ValueError):
    """Raised when a raw UDP packet cannot be parsed or fails validation."""

    def __init__(self, message: str, data: Optional[bytes] = None):
        super().__init__(message)
        self.data = data

    def __str__(self):
        base = super().__str__()
        if self.data is not None:
            return f"{base} (data={self.data[:32]!r}{'...' if len(self.data) > 32 else ''})"
        return base


class BackData:
    """One complete antenna rotation of backscatter data.

    step       — range resolution [m/px], typically 1.875 m.
    pulse      — pulse length code: 1=SP, 2=MP, 3=LP.
    bck        — backscatter intensity in polar coordinates, shape (AAP, ARDP), dtype uint8.
    n_received — UDP only: number of azimuth lines actually received (< AAP means packet loss).
                 -1 when not tracked (NC / BT8 sources).
    recv_time  — wall-clock time [s] when the rotation finished (time.time()).
                 Set only by the live UDP source; 0.0 for file sources. Used by
                 the processor to estimate RPM from inter-frame intervals.

    EOF sentinel: step == 0.0 signals end-of-file from NCInputSource / BT8InputSource.
    """

    def __init__(self, step: float, pulse: int, bck: np.ndarray, n_received: int = -1,
                 recv_time: float = 0.0):
        self.step = step
        self.pulse = pulse
        self.bck = bck
        self.n_received = n_received
        self.recv_time = recv_time


# Struct format:
# <  little-endian
# B  1 byte type
# H  unsigned short (num_line) 2 bytes
# H  unsigned short (step) 2 bytes
# B  signed int (raw_part_value) 1 bytes
# B  padding byte
# B  signed int (pulse) 1 bytes
# H  unsigned short (backscatter line) 1024 bytes

_BCK_PKT_SIZE = 1032
_BCK_TYPE_EXPECTED = 8
_BCK_HEADER_SIZE = 8
_BCK_PAYLOAD_SIZE = 1024


class BackPack:
    """Parsed payload of one 1032-byte backscatter UDP packet.

    Each azimuth line is split into two packets (part_index 0 and 1), each
    carrying 512 uint8 range samples (half of the full ARDP extent).
    """

    def __init__(self, num_line: int, step: float, part_index: int, part_value: int, pulse: int, payload: bytes):
        self.num_line = num_line
        self.step = step
        self.part_index = part_index
        self.part_value = part_value
        self.pulse = pulse
        self.payload = payload


def parse_back_packet(pack: bytes) -> BackPack:
    """Parse a 1032-byte backscatter UDP packet into a BackPack.

    Raises ProtocolError if the size, type byte, or part index is invalid.
    """
    if len(pack) != _BCK_PKT_SIZE:
        raise ProtocolError(f"Backscatter packet wrong size: expected {_BCK_PKT_SIZE}, got {len(pack)}", pack)

    type_field = pack[0]
    if type_field != _BCK_TYPE_EXPECTED:
        raise ProtocolError(f"Unexpected backscatter packet type: {type_field}", pack)

    try:
        num_line = unpack('<H', pack[1:3])[0]
        step = unpack('<H', pack[3:5])[0] / 1000.0
        raw_part_value = pack[5]
        pulse = pack[7]
        payload = pack[_BCK_HEADER_SIZE:_BCK_HEADER_SIZE + _BCK_PAYLOAD_SIZE]

    except error as e:
        raise ProtocolError("Failed to unpack backscatter header", pack) from e

    part_index = raw_part_value - 1

    if len(payload) != _BCK_PAYLOAD_SIZE:
        raise ProtocolError(f"Backscatter payload wrong size: {len(payload)}", payload)

    if part_index not in (0, 1):
        raise ProtocolError(f"Unexpected part index: {part_index} (raw {raw_part_value})", pack)

    return BackPack(num_line=num_line, step=step, part_index=part_index, part_value=raw_part_value, pulse=pulse,
                    payload=payload)


class Navi:
    """Navigation snapshot associated with one radar frame.

    hdg — gyro heading [°], 0–360.
    cog — course over ground [°], 0–360.
    spd — lag speed (water-relative) [m/s].
    sog — speed over ground [m/s].
    lat — latitude [°], −90 to +90.
    lon — longitude [°], −180 to +180.

    All angular fields use compass convention (0 = North, clockwise).
    sog and spd are in m/s — do not convert.
    """

    def __init__(self, hdg: float, cog: float, spd: float, sog: float, lat: float, lon: float):
        self.hdg = hdg
        self.cog = cog
        self.spd = spd
        self.sog = sog
        self.lat = lat
        self.lon = lon


# Struct format:
# <  little-endian
# B  1 byte type
# 3x  3 padding bytes (skipping bytes 1-3)
# h  signed short (spd) 2 bytes
# H  unsigned short (hdg) 2 bytes
# i  signed int (lat) 4 bytes
# i  signed int (lon) 4 bytes
# H  unsigned short (cog) 2 bytes
# H  unsigned short (sog) 2 bytes
_NAV_PKT_FMT = '<B3x h H i i H H'
_NAV_PKT_SIZE = calcsize(_NAV_PKT_FMT)  # 20


def parse_navi_packet(data: bytes) -> Navi:
    """Parse raw bytes into a Navi object. Raises ProtocolError on invalid input."""

    if len(data) < _NAV_PKT_SIZE:
        raise ProtocolError(f'Packet too short: expected {_NAV_PKT_SIZE}, got {len(data)}')

    try:
        type_field, spd_raw, hdg_raw, lat_raw, lon_raw, cog_raw, sog_raw = unpack(_NAV_PKT_FMT, data[:_NAV_PKT_SIZE])
    except error as e:
        raise ProtocolError("Failed to unpack backscatter header", data) from e

    if type_field != 1:
        raise ProtocolError(f'Unexpected packet type: {type_field}')

    spd = spd_raw / 100.0
    hdg = hdg_raw / 100.0
    lat = lat_raw / 1_000_000.0
    lon = lon_raw / 1_000_000.0
    cog = cog_raw / 100.0
    sog = sog_raw / 100.0

    # Validate ranges
    if not (-90.0 <= lat <= 90.0):
        raise ProtocolError(f'Latitude out of range: {lat}')
    if not (-180.0 <= lon <= 180.0):
        raise ProtocolError(f'Longitude out of range: {lon}')
    if not (0.0 <= cog <= 360.0):
        raise ProtocolError(f'COG out of range: {cog}')
    if not (0.0 <= hdg <= 360.0):
        raise ProtocolError(f'HDG out of range: {hdg}')

    return Navi(cog=cog, sog=sog, lat=lat, lon=lon, spd=spd, hdg=hdg)


class Wave:
    """
    Wave system descriptor.
    t_p  — peak period [s]
    t_m  — mean period [s]
    d_p  — peak direction [deg]
    d_m  — mean direction [deg]
    """
    def __init__(self, snr: float = 0.0, swh: float = 0.0,
                 t_p: float = 0.0, t_m: float = 0.0,
                 d_p: float = 0.0, d_m: float = 0.0):
        self.snr = snr
        self.swh = swh
        self.t_p = t_p
        self.t_m = t_m
        self.d_p = d_p
        self.d_m = d_m

    def sum(self, other, size):
        """Accumulate other/size into self — one step of a running mean over `size` frames."""
        t_p = other.t_p if np.isfinite(other.t_p) else 0.0
        t_m = other.t_m if np.isfinite(other.t_m) else 0.0
        return Wave(
            snr=self.snr + other.snr / size,
            swh=self.swh + other.swh / size,
            t_p=self.t_p + t_p / size,
            t_m=self.t_m + t_m / size,
            d_p=self.d_p + other.d_p / size,
            d_m=self.d_m + other.d_m / size,
        )

    def print(self):
        """Semicolon-separated string for the _params.csv row (CSVOutputSink)."""
        return (f"{self.swh:.2f};{self.t_p:.2f};"
                f"{self.d_p:.0f};{self.d_m:.0f};{self.t_m:.2f};")


class Output:
    """Averaged processing result ready for UDP/CSV transmission.

    Produced by Averager.get_mean() from the last MEAN WaveOutput frames.
    Spectral arrays are normalised to [0, 255].

    Key fields consumed by output sinks:
      pulse      — pulse length code (1/2/3 = SP/MP/LP).
      step       — range resolution [m/px].
      rps        — antenna rotation rate [rpm].
      ide_sys    — number of identified wave systems (0–3).
      n_dis      — quality flag: 1 = GOOD, 0 = BAD.
      wave_sum   — total sea state (swh, t_p, t_m, d_p, d_m).
      wave_win   — wind-sea component.
      wave_sw1/2 — swell components (sw1 shorter-period, sw2 longer-period).
      spec_1d    — 1-D frequency spectrum, shape (N_FREQ,), int [0–255].
      spec_2d    — directional spectrum, shape (N_DIRS, N_FREQ_2D), int [0–255],
                   row i = math direction i×10°, column j = frequency bin j.
      curr_speed — true ocean current speed [m/s], set after Averager by Processor.
      curr_dir   — true ocean current direction [°], compass convention.
      wspd       — wind speed estimate [m/s].
      wind_dir   — wind direction [°], math convention.
    """

    def __init__(self, pulse: int, step: float, rps: float, n_in_win: float, n_wins: float, step_area: float,
                 n_area: float, n_start: float, cog_proc: float, sog_proc: float, max_sys: int, ide_sys: int,
                 wave_sum: Wave, wave_win: Wave, wave_sw1: Wave, wave_sw2: Wave, n_dis: int, spec_1d: np.ndarray,
                 spec_2d: np.ndarray):
        self.pulse = pulse
        self.step = step
        self.rps = rps
        self.n_in_win = n_in_win
        self.n_wins = n_wins
        self.step_area = step_area
        self.n_area = n_area
        self.n_start = n_start
        self.cog_proc = cog_proc
        self.sog_proc = sog_proc
        self.max_sys = max_sys
        self.ide_sys = ide_sys
        self.wave_sum = wave_sum
        self.wave_win = wave_win
        self.wave_sw1 = wave_sw1
        self.wave_sw2 = wave_sw2
        self.n_dis = n_dis
        self.spec_1d = spec_1d
        self.spec_2d = spec_2d


class WaveOutput:
    """Single-computation result stored in the Averager ring buffer.

    Holds wave parameters and un-normalised spectra for one processor cycle.
    Averager.get_mean() accumulates MEAN of these into a final Output object.
    """

    def __init__(self, ide_sys, wave_sum, wave_win, wave_sw1, wave_sw2, spec_1d, spec_2d):
        self.ide_sys = ide_sys
        self.wave_sum = wave_sum
        self.wave_win = wave_win
        self.wave_sw1 = wave_sw1
        self.wave_sw2 = wave_sw2
        self.spec_1d = spec_1d
        self.spec_2d = spec_2d


class ProcessResult:
    """Single completed processing result passed from processor to output sinks."""
    __slots__ = ("output", "port", "navi")

    def __init__(self, output: "Output", port: np.ndarray, navi: "Navi"):
        self.output = output
        self.port = port
        self.navi = navi
