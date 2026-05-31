import numpy as np
from typing import Optional
from struct import calcsize, unpack, error


class ProtocolError(ValueError):
    def __init__(self, message: str, data: Optional[bytes] = None):
        super().__init__(message)
        self.data = data

    def __str__(self):
        base = super().__str__()
        if self.data is not None:
            return f"{base} (data={self.data[:32]!r}{'...' if len(self.data) > 32 else ''})"
        return base


class BackData:
    def __init__(self, step: float, pulse: int, bck: np.ndarray, n_received: int = -1):
        self.step = step        # range resolution
        self.pulse = pulse      # bite of pulse length
        self.bck = bck          # backscatter in polar coordinates (AAP, AREA_READ_DIST_PX)
        self.n_received = n_received  # UDP: lines actually received; -1 = not tracked (NC/BT8)


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
    def __init__(self, num_line: int, step: float, part_index: int, part_value: int, pulse: int, payload: bytes):
        self.num_line = num_line
        self.step = step
        self.part_index = part_index
        self.part_value = part_value
        self.pulse = pulse
        self.payload = payload


def parse_back_packet(pack: bytes) -> BackPack:
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
    def __init__(self, hdg: float, cog: float, spd: float, sog: float, lat: float, lon: float):
        self.hdg = hdg  # ship heading
        self.cog = cog  # ship course over ground
        self.spd = spd  # ship lag speed
        self.sog = sog  # ship speed over ground
        self.lat = lat  # latitude
        self.lon = lon  # longitude


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
        return (f"{self.swh:.2f};{self.t_p:.2f};"
                f"{self.d_p:.0f};{self.d_m:.0f};{self.t_m:.2f};")


class Output:
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
    def __init__(self, ide_sys, wave_sum, wave_win, wave_sw1, wave_sw2, spec_1d, spec_2d):
        self.ide_sys = ide_sys
        self.wave_sum = wave_sum
        self.wave_win = wave_win
        self.wave_sw1 = wave_sw1
        self.wave_sw2 = wave_sw2
        self.spec_1d = spec_1d
        self.spec_2d = spec_2d


class CurrentOutput:
    """2-D current velocity vector in geographic coordinates [m/s]."""
    def __init__(self, u_x: float = 0.0, u_y: float = 0.0):
        self.u_x = u_x
        self.u_y = u_y

    @property
    def speed(self):
        return float(np.hypot(self.u_x, self.u_y))

    @property
    def direction(self):
        return float(np.degrees(np.arctan2(self.u_y, self.u_x)) % 360)


class WindOutput:
    """Wind direction and backscatter intensity proxy."""
    def __init__(self, direction: float = 0.0, sig: float = 0.0):
        self.direction = direction
        self.sig = sig


class ProcessResult:
    """Single completed processing result passed from processor to output sinks."""
    __slots__ = ("output", "port", "navi")

    def __init__(self, output: "Output", port: np.ndarray, navi: "Navi"):
        self.output = output
        self.port = port
        self.navi = navi
