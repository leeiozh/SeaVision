import os
import numpy as np
from glob import glob
from time import time
from socket import socket
from typing import Optional
from src.io.service import create_inp_socket
from src.io.structs import Navi, BackData, BackPack, \
    parse_navi_packet, parse_back_packet, ProtocolError, _NAV_PKT_SIZE, _BCK_PKT_SIZE, _BCK_PAYLOAD_SIZE


class InputSource:
    def get_bck(self) -> BackData:
        raise NotImplementedError

    def get_navi(self) -> Navi:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class UdpInputSource(InputSource):
    def __init__(self, my_ip, back_port, navi_port, aap, ardp):
        self.back_socket = create_inp_socket(my_ip, back_port, 17)
        self.navi_socket = create_inp_socket(my_ip, navi_port, 2)
        self.navi_socket.setblocking(False)
        self.curr_bck = BackData(step=1.875, pulse=0, bck=np.zeros((aap, ardp), dtype=np.uint8))
        self.curr_navi = None
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
        except socket.timeout as e:
            raise TimeoutError("Socket timed out while waiting for backscatter packet") from e
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

    def get_bck(self, *, overall_timeout: Optional[float] = 30.0, per_recv_timeout: Optional[float] = None,
                max_duplicates: int = 4, max_attempts: Optional[int] = None) -> BackData:
        """
        Collect backscatter lines until all lines are ready (or until timeout / duplicates).
        - overall_timeout: total seconds to wait for the full collection (None means wait forever)
        - per_recv_timeout: timeout passed to each recv; if None we compute remaining overall time per loop
        - max_duplicates: the threshold used by original code to break early when duplicate lines observed
        - max_attempts: optional hard limit on number of packets to try before failing

        Returns self.curr_bck (mutated in-place).
        Raises TimeoutError if overall_timeout elapses or max_attempts exceeded.
        """
        # initialize trackers
        self.double_counter = 0
        ready_vec = np.zeros(self.curr_bck.bck.shape[0], dtype=bool)
        attempts = 0
        start_time = time()

        while True:
            if np.sum(ready_vec) >= ready_vec.shape[0]:
                break
            if max_attempts is not None and attempts >= max_attempts:
                raise TimeoutError(f"Reached max_attempts={max_attempts} without collecting all lines")
            # compute per-recv timeout to honor overall_timeout if set
            if overall_timeout is None:
                recv_timeout = per_recv_timeout
            else:
                elapsed = time() - start_time
                remaining = overall_timeout - elapsed
                if remaining <= 0:
                    raise TimeoutError("Overall timeout expired while waiting for backscatter data")
                # if caller provided a per_recv_timeout, we use the min of remaining and that value
                recv_timeout = remaining if per_recv_timeout is None else min(remaining, per_recv_timeout)

            attempts += 1
            try:
                backpack = self.recv_back_once()  # timeout=recv_timeout)
                ready_count = self.proc_back_packet(backpack, ready_vec, max_duplicates)

                if ready_count >= ready_vec.size:
                    break

            except ProtocolError as pe:
                continue
            except TimeoutError:
                continue

        return self.curr_bck

    def close(self):
        self.back_socket.close()
        self.navi_socket.close()


class NCInputSource(InputSource):

    def __init__(self, file_path):
        from netCDF4 import Dataset
        self.file_path = file_path
        self.dataset = Dataset(file_path)
        self.curr_ind = 0

    def get_bck(self) -> BackData:
        self.curr_ind += 1
        if self.curr_ind >= self.dataset["time_radar"].shape[0]:
            return BackData(0, 0, np.array([0]))
        else:
            return BackData(1.875,  # self.dataset["step"][self.curr_ind],
                            1,  # self.dataset["pulse"][self.curr_ind]
                            self.dataset["bsktr_radar"][self.curr_ind])

    def get_navi(self) -> Navi:
        return Navi(self.dataset["giro_radar"][self.curr_ind],
                    self.dataset["cog_radar"][self.curr_ind],
                    self.dataset["sog_radar"][self.curr_ind],
                    self.dataset["sog_radar"][self.curr_ind],
                    self.dataset["lat_radar"][self.curr_ind],
                    self.dataset["lon_radar"][self.curr_ind])

    def close(self):
        self.dataset.close()


def read_bt8(fname, radar_xdim=4096, radar_ydim=4096):
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

    def __init__(self, folder_path, aap, ardp, start_ind, end_ind, pulse):
        self.folder_path = folder_path
        self.pulse = pulse
        self.curr_ind = -1
        self.aap = aap
        self.ardp = ardp
        self.fnames = sorted(glob(os.path.join(folder_path, "*.bt8")))[start_ind:end_ind]
        self.gyro, self.cog, self.sog, self.dr, self.lat, self.lon = None, None, None, None, None, None

    def get_bck(self) -> BackData:
        self.curr_ind += 1
        if self.curr_ind >= len(self.fnames):
            return BackData(0, 0, np.array([0]))
        else:

            self.gyro, self.cog, self.sog, self.dr, self.lat, self.lon, bcksctr = read_bt8(self.fnames[self.curr_ind],
                                                                                           self.aap, self.ardp)

        return BackData(1.875,  # self.dataset["step"][self.curr_ind],
                        self.pulse,  # self.dataset["pulse"][self.curr_ind]
                        bcksctr)

    def get_navi(self) -> Navi:
        return Navi(self.gyro, self.cog, self.sog, self.sog, self.lat, self.lon)

    def close(self):
        pass
