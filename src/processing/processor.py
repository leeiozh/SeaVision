import numpy as np
from scipy.fft import fft2
from scipy.interpolate import interp1d
from src.processing.state import ProcessorState
from src.processing.averaging import Averager
from src.algorithms.area import Area
from src.algorithms.direction import calc_dir_ind
from src.algorithms.dispersion import find_curve
from src.algorithms.portrait import process_portrait
from src.algorithms.spectrum2d import calc_spec2d
from src.io.structs import Wave, WaveOutput


class Processor:

    def __init__(self, config, pics=False):
        self.pics = pics
        self.cfg = config
        self.cst = config.const
        self.om_max = 2 * np.pi / (60 / self.cfg.const.RPM)
        self.state = ProcessorState()
        self.state.init_arrays(
            n_shots=self.cst.N_SHOTS,
            mean=self.cst.MEAN,
            asp=self.cst.ASP,
            change_dir_num=self.cst.CHANGE_DIR_NUM_SHOTS,
        )

        self.averager = Averager(
            mean=self.cst.MEAN,
            n_freq=self.cst.N_FREQ,
            n_dirs=self.cst.NUM_AREA,
            n_shots=self.cst.N_SHOTS,
            cut_num=self.cst.K_NUM,
        )

        self.msh = [Area(self.cst.ASP * 2, self.cst.ADP, ang, -ang, self.cst.AAP).calc_mask() for ang in
                    np.linspace(0, 360, self.cst.NUM_AREA + 1)[:-1]]

    def get_info(self):
        return f"Ind = {self.state.index}, Pul = {self.state.curr_pulse}, CDIR = {self.state.curr_dir}"

    def stop(self):
        return 0

    def update(self, back, navi):
        s = self.state
        cst = self.cfg.const

        s.curr_step = back.step
        s.curr_pulse = back.pulse
        s.speed[s.index % cst.MEAN] = navi.sog
        s.heading[s.index % cst.MEAN] = navi.hdg

        '''main direction / trimmed segment direction calculation'''
        new_dir = calc_dir_ind(back.bck, cst.NUM_AREA, cst.AAP, cst.ADP, cst.ASP)

        if s.index > cst.MEAN:
            if new_dir - s.curr_dir > 1:
                new_dir += 1
            elif new_dir - s.curr_dir < -1:
                new_dir -= 1

        s.dir_vec[s.index % cst.CHANGE_DIR_NUM_SHOTS] = new_dir % cst.NUM_AREA

        if s.index < cst.MEAN:
            s.curr_dir = new_dir
        elif s.index == cst.MEAN:
            s.curr_dir = int(np.median(s.dir_vec))
        elif s.index % cst.CHANGE_DIR_NUM_SHOTS == 0:
            s.curr_dir = int(np.median(s.dir_vec))

        '''segment interpolation'''

        bck = back.bck
        (x, y), (wx, wy) = self.msh[s.curr_dir]
        row0 = bck[y, x] * (1.0 - wx) + bck[y, x + 1] * wx
        row1 = bck[y + 1, x] * (1.0 - wx) + bck[y + 1, x + 1] * wx
        seg = row0 * (1.0 - wy) + row1 * wy

        s.cbck[s.index % cst.N_SHOTS] = seg

        '''rolling indexing for correct welch transform'''
        s.indices = np.roll(s.indices, -1)

        result = None
        port = None

        if s.index > cst.N_SHOTS and s.index % int(self.cfg.output["out_times"]) == 0:

            if self.pics != "false":

                import matplotlib.pyplot as plt
                from matplotlib import gridspec
                fig = plt.figure(figsize=(8, 8))
                gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig, wspace=0.1, hspace=0.1, width_ratios=[1, 1])
                ax1 = fig.add_subplot(gs[0, 0])  # raw data
                ax2 = fig.add_subplot(gs[0, 1])  # portrait
                ax3 = fig.add_subplot(gs[1, 0], projection="polar")  # polar spec
                ax4 = fig.add_subplot(gs[1, 1])  # freq spec

                for ax in [ax1, ax2, ax3, ax4]:
                    ax.tick_params(direction="in")

                ax1.imshow(seg, origin="lower", cmap="binary", vmin=80, vmax=160)
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.text(30, 350, s.curr_dir, color="red")
                ax4.set_xlim(0, 0.4)
            else:
                ax1, ax2, ax3, ax4 = None, None, None, None

            '''first attempt to estimate doppler term and inversion flag'''
            sp = np.abs(fft2(np.sum(s.cbck, axis=1)))[:, :cst.K_NUM]
            sp[0, :] = 0
            _, vco, _ = find_curve(sp, np.mean(s.speed), s.curr_step, None, cst.ASP, cst.K_NUM,
                                   self.om_max, cst.N_SHOTS)

            '''omega-theta and omega-k spectrum calculation'''
            spec_2d, port_raw = calc_spec2d(s.cbck, s.curr_dir / cst.NUM_AREA * 360, vco, s.curr_step, cst.ASP,
                                            cst.K_NUM, self.om_max, cst.N_SHOTS, cst.NUM_AREA, ax3)

            '''second attempt to process omega-k spectrum'''
            inv, vco, ks = find_curve(port_raw, np.mean(s.speed), s.curr_step, None, cst.ASP, cst.K_NUM,
                                      self.om_max, cst.N_SHOTS)

            port, per, m0, spec, ks, vco = process_portrait(port_raw, inv, speed=np.mean(s.speed),
                                                            freq_win=15. / cst.N_SHOTS * self.om_max,
                                                            step=s.curr_step, asp=cst.ASP, k_num=cst.K_NUM,
                                                            om_max=self.om_max, n_shots=cst.N_SHOTS, ax_2d=ax2,
                                                            ax_1d=ax4)

            if self.pics != "false":
                import matplotlib.pyplot as plt
                plt.savefig(f"{self.pics}state.png", dpi=300, bbox_inches='tight')

            '''omega spectrum interpolation on output frequency grid'''
            spec_1d = np.interp(np.linspace(0, self.om_max, self.cst.N_FREQ),
                                np.linspace(0, self.om_max, self.cst.N_SHOTS), spec)

            f = interp1d(np.linspace(0, self.om_max, self.cst.N_SHOTS), spec_2d, axis=1, kind='cubic')
            spec_2d = f(np.linspace(0, self.om_max, self.cst.N_FREQ))

            swh = 0.01 * (cst.SNR_A + cst.SNR_B * np.sqrt(m0))

            direction = s.curr_dir / cst.NUM_AREA * 360
            if inv:
                direction = (direction + 180) % 360

            wave_sum = Wave(
                swh=swh,
                snr=m0,
                per=per,
                len=2 * np.pi / ks if ks != 0 else 0,
                dir=direction,
                ddir=direction,
                vco=vco,
                inv=inv
            )

            wave_out = WaveOutput(ide_sys=1, wave_sum=wave_sum, wave_win=wave_sum,
                                  wave_sw1=wave_sum, wave_sw2=wave_sum, spec_1d=spec_1d, spec_2d=spec_2d)

            self.averager.push(wave_out, spec_1d, spec_2d, port)

            result, spec_1d, spec_2d, port = self.averager.get_mean(
                pulse=s.curr_pulse,
                step=s.curr_step,
                rpm=cst.RPM,
                n_shots=cst.N_SHOTS,
                asp=cst.ASP,
                adp=cst.ADP,
            )

        s.index += 1
        return {"out": result, "pulse": s.curr_pulse, "step": s.curr_step, "navi": navi, "port": port}
