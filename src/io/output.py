import numpy as np
from struct import pack
from datetime import datetime
from src.io.structs import Output, Wave, Navi, ProcessResult
from src.io.service import create_out_socket


class OutputSink:
    def send(self, result: ProcessResult):
        raise NotImplementedError

    def close(self):
        pass


class UdpOutputSink(OutputSink):
    def __init__(self, server_ip, server_port, n_freqs, num_area):
        self.out_socket = create_out_socket(server_port, 2)
        self.server_ip = server_ip
        self.server_port = server_port
        self.n_freqs = n_freqs
        self.num_area = num_area

    def send(self, result: ProcessResult):
        o = result.output
        # un[7]  (was n_area=384)   → curr_dir, integer degrees 0-360
        # un[11] (was max_sys=3)    → curr_speed, uint8 cm/s, clipped to 255
        # un[26] (vco, signed int16) → reserved / 0
        # un[27] H (NEW)            → wind_dir, integer degrees 0-360
        # un[28] H (NEW)            → wspd × 10 (0.1 m/s resolution)
        _curr_dir   = int(round(getattr(o, 'curr_dir',   0.0))) % 360
        _curr_spd_b = int(np.clip(round(getattr(o, 'curr_speed', 0.0) * 100), 0, 255))
        _wind_dir   = int(round(getattr(o, 'wind_dir',   0.0))) % 360
        _wspd10     = int(np.clip(round(getattr(o, 'wspd', 0.0) * 10), 0, 65535))
        data = pack(
            f"<BBHHBBHHHHHBBHHHHHHHHHHHHHhHH{self.n_freqs}B{self.n_freqs * self.num_area}B",
            5,
            o.pulse,
            round(o.step * 1000),
            round(o.rps * 100),          # [3]  rps (restored)
            o.n_in_win,
            o.n_wins,
            round(o.step_area * 1000),   # [6]  step_area (restored)
            _curr_dir,                   # [7]  curr_dir [°], was n_area
            o.n_start,
            round(o.cog_proc * 100),
            round(o.sog_proc * 100),
            _curr_spd_b,                 # [11] curr_speed [cm/s], was max_sys
            o.ide_sys,
            round(o.wave_sum.swh * 100),
            round(o.wave_sum.d_p * 100),
            round(o.wave_sum.t_p * 100),
            round(o.wave_win.swh * 100),
            round(o.wave_win.d_p * 100),
            round(o.wave_win.t_p * 100),
            round(o.wave_sw1.swh * 100),
            round(o.wave_sw1.d_p * 100),
            round(o.wave_sw1.t_p * 100),
            round(o.wave_sw2.swh * 100),
            round(o.wave_sw2.d_p * 100),
            round(o.wave_sw2.t_p * 100),
            o.n_dis,
            0,                           # [26] reserved (was vco/u_proj)
            _wind_dir,                   # [27] wind_dir [°] (NEW)
            _wspd10,                     # [28] wspd × 10 [0.1 m/s] (NEW)
            *o.spec_1d,
            *o.spec_2d.flatten(),
        )
        self.out_socket.sendto(data, (self.server_ip, self.server_port))

    def close(self):
        self.out_socket.close()


class CSVOutputSink(OutputSink):
    def __init__(self, save_path, constants):
        self.save_path = save_path
        now_time = datetime.now()
        self.time = now_time.strftime("%Y%m%dT%H%M%S")
        self.path_port = self.save_path + self.time + "_port.csv"
        self.path_spec = self.save_path + self.time + "_spec.csv"
        self.path_navi = self.save_path + self.time + "_navi.csv"
        self.path_params = self.save_path + self.time + "_params.csv"
        self.k_num = constants.K_NUM
        self.rpm = constants.RPM
        self.mean = constants.MEAN
        self.n_freq = constants.N_FREQ
        self.n_shots = constants.N_SHOTS
        self.n_dirs = constants.N_DIRS

        with open(self.path_params, "w") as out:
            out.write("datetime;pulse;step;swh;t_p;d_p;d_m;t_m;freq;\n")

        with open(self.path_port, "w") as out:
            out.write(f"({self.n_shots},{self.k_num})\n")

        with open(self.path_spec, "w") as out:
            out.write(f"({self.n_dirs},{self.n_freq})\n")

        with open(self.path_navi, "w") as out:
            out.write("datetime,lat,lon,spd,sog,cog,hdg\n")

    def send(self, result: ProcessResult):
        o = result.output
        dtime = datetime.now().strftime("%Y%m%dT%H%M%S")
        self._save_params(dtime, o.pulse, o.step, o.wave_sum, o.spec_1d)
        self._save_spec(o.spec_2d)
        if result.port is not None:
            self._save_port(result.port)
        self._save_navi(dtime, result.navi)

    def _save_params(self, dtime, pulse: int, step: float, wave: Wave, freq: np.ndarray):
        with open(self.path_params, "a") as f:
            f.write(dtime + f";{pulse};{step};" + wave.print())
            for v in freq:
                f.write(f"{v:.0f};")
            f.write("\n")

    def _save_port(self, port: np.ndarray):
        with open(self.path_port, "a") as f:
            for row in port:
                f.write(";".join(f"{v:.0f}" for v in row))
                f.write(";\n")
            f.write("\n")

    def _save_spec(self, spec: np.ndarray):
        with open(self.path_spec, "a") as f:
            for row in spec:
                f.write(";".join(f"{v:.0f}" for v in row))
                f.write(";\n")
            f.write("\n")

    def _save_navi(self, dtime, navi: Navi):
        with open(self.path_navi, "a") as f:
            f.write(f"{dtime},{navi.lat},{navi.lon},{navi.spd},{navi.sog},{navi.cog},{navi.hdg}\n")
