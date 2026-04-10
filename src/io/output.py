import numpy as np
from struct import pack
from src.io.structs import Output, Wave, Navi
from src.io.service import create_out_socket
from datetime import datetime


class OutputSink:
    def send(self, *args):
        raise NotImplementedError


class UdpOutputSink(OutputSink):
    def __init__(self, server_ip, server_port, n_freqs, num_area):
        self.out_socket = create_out_socket(server_port, 2)
        self.server_ip = server_ip
        self.server_port = server_port
        self.n_freqs = n_freqs
        self.num_area = num_area

    def send(self, output: Output):
        data = pack(f"<BBHHBBHHHHHBBHHHHHHHHHHHHHHHHH{self.n_freqs}B{self.n_freqs * self.num_area}B",
                    5,
                    output.pulse,
                    round(output.step * 1000),
                    round(output.rps * 100),
                    output.n_in_win,
                    output.n_wins,
                    round(output.step_area * 1000),
                    output.n_area,
                    output.n_start,
                    round(output.cog_proc * 100),
                    round(output.sog_proc * 100),
                    output.max_sys,
                    output.ide_sys,
                    round(output.wave_sum.swh * 100),
                    round(output.wave_sum.dir * 100),
                    round(output.wave_sum.per * 100),
                    round(output.wave_sum.len),
                    round(output.wave_win.swh * 100),
                    round(output.wave_win.dir * 100),
                    round(output.wave_win.per * 100),
                    round(output.wave_win.len),
                    round(output.wave_sw1.swh * 100),
                    round(output.wave_sw1.dir * 100),
                    round(output.wave_sw1.per * 100),
                    round(output.wave_sw1.len),
                    round(output.wave_sw2.swh * 100),
                    round(output.wave_sw2.dir * 100),
                    round(output.wave_sw2.per * 100),
                    round(output.wave_sw2.len),
                    output.n_dis,
                    *output.spec_1d,
                    *output.spec_2d.flatten(),)

        self.out_socket.sendto(data, (self.server_ip, self.server_port))


class CSVOutputSink(OutputSink):
    def __init__(self, save_path: str, k_num: int, rpm: int, mean: int, n_freq: int, n_shots: int, num_area: int):
        self.save_path = save_path
        now_time = datetime.now()
        self.time = now_time.strftime("%Y%m%dT%H%M%S")
        self.path_port = self.save_path + self.time + "_port.csv"
        self.path_spec = self.save_path + self.time + "_spec.csv"
        self.path_navi = self.save_path + self.time + "_navi.csv"
        self.path_params = self.save_path + self.time + "_params.csv"
        self.k_num = k_num
        self.rpm = rpm
        self.mean = mean
        self.n_freq = n_freq
        self.n_shots = n_shots
        self.num_area = num_area

        with open(self.path_params, "w") as out:
            out.write("datetime;pulse;step;snr;swh;per;dir;ddir;len;vco;inv;freq;\n")

        with open(self.path_port, "w") as out:
            out.write(f"({self.n_shots},{self.k_num})\n")

        with open(self.path_spec, "w") as out:
            out.write(f"({self.num_area},{self.n_shots})\n")

        with open(self.path_navi, "w") as out:
            out.write(f"datetime,lat,lon,spd,sog,cog,hdg\n")

    def save_params(self, dtime, pulse: int, step: float, wave: Wave, freq: np.ndarray):
        with open(self.path_params, "a") as file:
            file.write(dtime + f";{pulse};{step};" + wave.print())
            for f in freq:
                file.write(f"{f:.0f};")
            file.write("\n")

    def save_port(self, port):
        with open(self.path_port, "a") as out:
            for i in range(port.shape[0]):
                for j in range(port.shape[1]):
                    out.write(f"{port[i, j]:.0f};")
                out.write("\n")
            out.write("\n")

    def save_spec(self, spec):
        with open(self.path_spec, "a") as out:
            for i in range(spec.shape[0]):
                for j in range(spec.shape[1]):
                    out.write(f"{spec[i, j]:.0f};")
                out.write("\n")
            out.write("\n")

    def save_navi(self, dtime, navi: Navi):
        with open(self.path_navi, "a") as file:
            file.write(f"{dtime},{navi.lat},{navi.lon},{navi.spd},{navi.sog},{navi.cog},{navi.hdg}\n")

    def send(self, pulse, step, wave, freq, port, spec, navi):
        dtime = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.save_params(dtime, pulse, step, wave, freq)
        self.save_spec(spec)
        self.save_port(port)
        self.save_navi(dtime, navi)
