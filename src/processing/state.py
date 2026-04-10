import numpy as np
from dataclasses import dataclass, field


@dataclass
class ProcessorState:
    index: int = 0  # global index
    mean_index: int = 0  # index in averaging array

    indices: np.ndarray = field(default_factory=lambda: np.array([]))  # array of rolling indexes
    cbck: np.ndarray = field(default_factory=lambda: np.array([]))  # current backscatter timeseries

    curr_step: float = 1.875  # current range resolution
    curr_pulse: int = 2  # current pulse

    speed: np.ndarray = field(default_factory=lambda: np.array([]))  # current speed over ground
    heading: np.ndarray = field(default_factory=lambda: np.array([]))  # current heading

    dir_vec: np.ndarray = field(default_factory=lambda: np.array([]))  # array of directions for smooth shifting
    curr_dir: int = 0  # current assumed main direction

    vco: float = 0.0  # current multiplier in doppler term
    inv: bool = False  # current inversion flag

    def init_arrays(self,
                    n_shots: int,
                    mean: int,
                    asp: int,
                    change_dir_num: int):
        self.indices = np.arange(n_shots)
        self.cbck = np.zeros((n_shots, 2 * asp, 2 * asp))
        self.speed = np.zeros(mean)
        self.heading = np.zeros(mean)
        self.dir_vec = np.zeros(change_dir_num, dtype=int)
