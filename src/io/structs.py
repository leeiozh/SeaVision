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
    def __init__(self, step: float, pulse: int, bck: np.ndarray):
        self.step = step  # range resolution
        self.pulse = pulse  # bite of pulse length
        self.bck = bck  # backscatter in polar coordinates (AAP, AREA_READ_DIST_PX)


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
        self.step = step  # range resolution
        self.part_index = part_index
        self.part_value = part_value
        self.pulse = pulse  # bite of pulse length
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
    def __init__(self, snr: float, swh: int, dir: float, ddir: float, per: float, len: float, vco: float, inv: bool):
        self.snr = snr
        self.swh = swh
        self.dir = dir
        self.ddir = ddir
        self.per = per
        self.len = len
        self.vco = vco
        self.inv = inv

    def sum(self, other, size):
        if np.isnan(other.per) or np.isinf(other.per):
            pp = 0
        else:
            pp = other.per
        if np.isnan(other.len) or np.isinf(other.len):
            ll = 0
        else:
            ll = other.len
        return Wave(snr=self.snr + other.snr / size, swh=self.swh + other.swh / size, dir=self.dir + other.dir / size,
                    ddir=self.ddir + other.ddir / size, per=self.per + pp / size, len=self.len + ll / size,
                    vco=self.vco + other.vco / size, inv=self.inv)

    def print(self):
        return (f"{self.snr:.2f};{self.swh:.2f};{self.per:.2f};{self.dir:.0f};{self.ddir:.0f};{self.len:.0f};"
                f"{self.vco:.2f};{self.inv};")


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
