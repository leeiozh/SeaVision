import numpy as np
from struct import pack
from datetime import datetime
from src.io.service import create_out_socket
from src.io.structs import Wave, Navi, ProcessResult


class OutputSink:
    def send(self, result: ProcessResult):
        raise NotImplementedError

    def close(self):
        pass


class UdpOutputSink(OutputSink):
    def __init__(self, server_ip, server_port, n_freqs, n_dirs, n_freq_2d):
        self.out_socket = create_out_socket(server_port, 2)
        self.server_ip = server_ip
        self.server_port = server_port
        self.n_freqs = n_freqs
        self.n_dirs = n_dirs
        self.n_freq_2d = n_freq_2d

    def send(self, result: ProcessResult):
        o = result.output
        # Header layout (52 bytes):
        #   BB HH                   — type, pulse, step_mm, rpm_x100
        #   HHHHH                   — summary: swh, t_p, t_m, dir_p, dir_m
        #   HHH HHH HHH             — wind/sw1/sw2: swh, t_p, dir_p each
        #   H H HH                  — curr_speed, curr_dir, wspd_x10, wind_dir
        #   BB HHHH                 — n_sys, quality, reserved×4
        data = pack(
            f"<BBHHHHHHHHHHHHHHHHHHHHBBHHHH"
            f"{self.n_freqs}B{self.n_freq_2d * self.n_dirs}B",
            5, o.pulse,
            round(o.step * 1000), round(o.rps * 100),
            # summary — all 5 fields
            round(o.wave_sum.swh * 100), round(o.wave_sum.t_p * 100), round(o.wave_sum.t_m * 100),
            int(round(o.wave_sum.d_p)) % 360, int(round(o.wave_sum.d_m)) % 360,
            # wind wave — swh, t_p, dir_p only
            round(o.wave_win.swh * 100), round(o.wave_win.t_p * 100),
            int(round(o.wave_win.d_p)) % 360,
            # swell 1 — swh, t_p, dir_p only
            round(o.wave_sw1.swh * 100), round(o.wave_sw1.t_p * 100),
            int(round(o.wave_sw1.d_p)) % 360,
            # swell 2 — swh, t_p, dir_p only
            round(o.wave_sw2.swh * 100), round(o.wave_sw2.t_p * 100),
            int(round(o.wave_sw2.d_p)) % 360,
            round(getattr(o, 'curr_speed', 0.0) * 100),
            int(round(getattr(o, 'curr_dir', 0.0))) % 360,
            int(np.clip(round(getattr(o, 'wspd', 0.0) * 10), 0, 65535)),
            int(round(getattr(o, 'wind_dir', 0.0))) % 360,
            o.ide_sys, o.n_dis,
            0, 0, 0, 0,
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
