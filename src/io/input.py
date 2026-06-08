import os
import numpy as np
from glob import glob
from time import time
from typing import Optional
from src.io.service import create_inp_socket
from src.io.structs import Navi, BackData, BackPack, \
    parse_navi_packet, parse_back_packet, ProtocolError, _NAV_PKT_SIZE, _BCK_PKT_SIZE, _BCK_PAYLOAD_SIZE
from src.runtime.logger import setup_logger

log = setup_logger("input")


class InputSource:
    """Abstract base for all radar data sources.

    Subclasses must implement get_bck() and get_navi().
    get_bck() returns BackData with step==0.0 to signal end-of-file.
    """

    def get_bck(self) -> BackData:
        raise NotImplementedError

    def get_navi(self) -> Navi:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class UdpInputSource(InputSource):
    """Live radar input over UDP.

    Assembles a full BackData frame (AAP azimuth lines × ARDP range bins) from
    the stream of 1032-byte backscatter packets.  Navigation packets are read
    non-blocking on the same call; the last valid Navi is returned if none
    arrive between two frames.

    get_bck() blocks until either all lines are received or a timeout/duplicate
    threshold is hit (indicating the next antenna rotation has started).
    """

    def __init__(self, my_ip, back_port, navi_port, aap, ardp):
        self.back_socket = create_inp_socket(my_ip, back_port, 17)
        self.navi_socket = create_inp_socket(my_ip, navi_port, 2)
        self.navi_socket.setblocking(False)
        self.curr_bck = BackData(step=1.875, pulse=0, bck=np.zeros((aap, ardp), dtype=np.uint8))
        self.curr_navi = Navi(hdg=0.0, cog=0.0, spd=0.0, sog=0.0, lat=0.0, lon=0.0)
        self.double_counter = 0
        self.ready_vec = np.zeros(aap, dtype=bool)

    def get_navi(self) -> Navi:
        """
        Keep receiving datagrams until we parse a valid Navi object.
        - timeout: overall timeout in seconds (None => wait forever).
        - max_attempts: optional limit on number of packets to try before giving up.
        Behavior:
          - If a packet is malformed, it is ignored and we continue (but count attempts).
          - If the socket times out (while waiting for a packet), raises TimeoutError.
          - If max_attempts is reached without a valid packet, raises TimeoutError.
        """

        last = None
        while True:
            try:
                data, _addr = self.navi_socket.recvfrom(_NAV_PKT_SIZE)
                last = parse_navi_packet(data)
            except BlockingIOError:
                break
            except ProtocolError:
                continue
        if last is not None:
            self.curr_navi = last
        return self.curr_navi

    def recv_back_once(self, bufsize: int = _BCK_PKT_SIZE,
                       timeout: Optional[float] = None) -> BackPack:
        """
        Receive one datagram and parse into BackPacket.
        Raises TimeoutError on socket timeout, ProtocolError on parse error.
        """
        old_timeout = self.back_socket.gettimeout()
        try:
            self.back_socket.settimeout(timeout)
            data, _addr = self.back_socket.recvfrom(bufsize)
        except TimeoutError:
            raise
        finally:
            self.back_socket.settimeout(old_timeout)

        return parse_back_packet(data)

    def proc_back_packet(self, bpack: BackPack, ready_vec: np.ndarray, max_duplicates: int) -> int:
        """
        Update self.curr_bck using a parsed BackPacket.
        ready_vec is the boolean array tracking which lines are ready.
        Returns the current count of ready lines (or len(ready_vec) if duplicates reached).
        Side effects:
          - sets self.curr_bck.step and self.curr_bck.pulse
          - writes payload data into self.curr_bck.bck at the correct indices
          - updates self.double_counter / ready_vec similar to original behaviour
        """
        if bpack.num_line < 0 or bpack.num_line >= ready_vec.size:
            raise ProtocolError(f"num_line out of range: {bpack.num_line}")

        payload_arr = np.frombuffer(bpack.payload, dtype=np.uint8)
        start = bpack.part_index * _BCK_PAYLOAD_SIZE
        end = start + _BCK_PAYLOAD_SIZE

        self.curr_bck.bck[bpack.num_line, start:end] = payload_arr
        self.curr_bck.pulse = bpack.pulse
        self.curr_bck.step = bpack.step

        if bpack.part_index == 0:
            if ready_vec[bpack.num_line]:
                self.double_counter += 1
            else:
                ready_vec[bpack.num_line] = True

        if self.double_counter >= max_duplicates:
            return ready_vec.size

        return int(np.sum(ready_vec))

    def get_bck(self, *, overall_timeout: Optional[float] = 30.0, per_recv_timeout: Optional[float] = 2.0,
                max_duplicates: int = 4, max_attempts: Optional[int] = None) -> BackData:
        """
        Collect backscatter lines until every line has both parts received.

        Two separate trackers:
          ready_vec  — set on part_index=0 arrival; used for duplicate detection
                       (when part_index=0 arrives for an already-seen line → next
                       rotation started → double_counter++).
          part1_seen — set on part_index=1 arrival; the true completion criterion.

        Exit when part1_seen.all() (all lines fully received) or
        double_counter >= max_duplicates (forced early exit: next frame started).
        """
        self.double_counter = 0
        n = self.curr_bck.bck.shape[0]
        ready_vec  = np.zeros(n, dtype=bool)  # p0 arrivals — duplicate detection
        part1_seen = np.zeros(n, dtype=bool)  # p1 arrivals — completion criterion
        attempts   = 0
        start_time = time()

        while True:
            if part1_seen.all():
                break
            if self.double_counter >= max_duplicates:
                break
            if max_attempts is not None and attempts >= max_attempts:
                raise TimeoutError(f"Reached max_attempts={max_attempts} without collecting all lines")
            if overall_timeout is None:
                recv_timeout = per_recv_timeout
            else:
                elapsed = time() - start_time
                remaining = overall_timeout - elapsed
                if remaining <= 0:
                    raise TimeoutError("Overall timeout expired while waiting for backscatter data")
                recv_timeout = remaining if per_recv_timeout is None else min(remaining, per_recv_timeout)

            attempts += 1
            try:
                backpack = self.recv_back_once(timeout=recv_timeout)
                self.proc_back_packet(backpack, ready_vec, max_duplicates)
                if backpack.part_index == 1:
                    part1_seen[backpack.num_line] = True

            except ProtocolError:
                continue
            except TimeoutError:
                continue

        n_recv = int(np.sum(part1_seen))
        if n_recv < n:
            pct = 100.0 * n_recv / n
            print()
            log.warning(
                f'Frame incomplete: {n_recv}/{n} lines ({pct:.0f}%) — packet loss.')
        # Return a copy: the next get_bck() call immediately overwrites self.curr_bck.bck
        # with new frame data while the previous frame may still be in the processing queue.
        return BackData(self.curr_bck.step, self.curr_bck.pulse,
                        self.curr_bck.bck.copy(), n_recv)

    def close(self):
        self.back_socket.close()
        self.navi_socket.close()


class NCInputSource(InputSource):
    """Read pre-recorded radar frames from a NetCDF file sequentially.

    get_bck() advances an internal frame index on each call and returns
    BackData with step=0.0 once all frames have been delivered (EOF sentinel).
    Navigation fields are read from matching dataset variables.
    """

    def __init__(self, file_path):
        from netCDF4 import Dataset
        self.file_path = file_path
        self.dataset = Dataset(file_path)
        self.curr_ind = -1

    def get_bck(self) -> BackData:
        self.curr_ind += 1
        if self.curr_ind >= self.dataset["time_radar"].shape[0]:
            return BackData(0, 0, np.array([0]))
        else:
            return BackData(1.875,  # self.dataset["step"][self.curr_ind],
                            1,  # self.dataset["pulse"][self.curr_ind]
                            self.dataset["bsktr_radar"][self.curr_ind])

    def get_navi(self) -> Navi:
        i = self.curr_ind
        return Navi(float(self.dataset["giro_radar"][i]),
                    float(self.dataset["cog_radar"][i]),
                    float(self.dataset["sog_radar"][i]),
                    float(self.dataset["sog_radar"][i]),
                    float(self.dataset["lat_radar"][i]),
                    float(self.dataset["lon_radar"][i]))

    def close(self):
        self.dataset.close()


def read_bt8(fname, radar_xdim=4096, radar_ydim=4096):
    """Parse one .bt8 binary file.

    Returns (hdg, cog, sog, step, lat, lon, image) where image has shape
    (radar_xdim, radar_ydim) uint8.  step is the range resolution [m/px]
    decoded from header byte 10 (lookup table: 0→3.75, 1→7.5, …, 7→1.875 m).
    """
    with open(fname, 'rb') as f:
        junk_chunk = f.read(64)
        head_byte = np.frombuffer(junk_chunk, dtype=np.uint8, count=23, offset=0)
        head_float = np.frombuffer(junk_chunk, dtype=float, count=-1, offset=0)

        dy = np.nan
        if head_byte[10] == 0:
            dy = 3.75
        elif head_byte[10] == 1:
            dy = 7.5
        elif head_byte[10] == 2:
            dy = 15.0
        elif head_byte[10] == 3:
            dy = 30.0
        elif head_byte[10] == 4:
            dy = 60.0
        elif head_byte[10] == 7:
            dy = 1.875

        one_turn = np.zeros((radar_xdim, radar_ydim), dtype=np.uint8)

        for i in range(radar_xdim):
            chunk = f.read(4096)
            junk_chunk = f.read(8)
            chunk_array = np.frombuffer(chunk, dtype=np.uint8, count=-1, offset=0)
            one_turn[i, :] = chunk_array[:radar_ydim]

    return head_float[3], head_float[6], head_float[7], dy, head_float[4], head_float[5], one_turn


class BT8InputSource(InputSource):
    """Read pre-recorded radar frames from a folder of .bt8 binary files.

    Files are sorted by name and processed in order from start_ind to end_ind.
    When the last file is reached get_bck() checks for newly added files
    (_rescan) and returns the EOF sentinel (step=0.0) if none are found.
    Navigation data is embedded in the .bt8 header.
    """

    def __init__(self, folder_path, aap, ardp, start_ind, end_ind, pulse):
        self.folder_path = folder_path
        self.pulse = pulse
        self.curr_ind = -1
        self.aap = aap
        self.ardp = ardp
        self.start_ind = start_ind
        self.end_ind = end_ind
        self.fnames = sorted(glob(os.path.join(folder_path, "*.bt8")))[start_ind:end_ind]
        self.gyro, self.cog, self.sog, self.dr, self.lat, self.lon = None, None, None, None, None, None
        log.info(f"BT8: found {len(self.fnames)} files in {folder_path}")

    def _rescan(self):
        new_files = sorted(glob(os.path.join(self.folder_path, "*.bt8")))[self.start_ind:self.end_ind]
        if len(new_files) > len(self.fnames):
            log.info(f"BT8: {len(new_files) - len(self.fnames)} new files found")
            self.fnames = new_files

    def get_bck(self) -> BackData:
        next_ind = self.curr_ind + 1
        if next_ind >= len(self.fnames):
            self._rescan()
        if next_ind >= len(self.fnames):
            # No new files — do not advance curr_ind, return EOF signal
            return BackData(0, 0, np.array([0]))

        self.curr_ind = next_ind
        self.gyro, self.cog, self.sog, self.dr, self.lat, self.lon, bcksctr = read_bt8(
            self.fnames[self.curr_ind], self.aap, self.ardp)
        return BackData(1.875, self.pulse, bcksctr)

    def get_navi(self) -> Navi:
        return Navi(self.gyro, self.cog, self.sog, self.sog, self.lat, self.lon)

    def close(self):
        pass
