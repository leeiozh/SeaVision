import numpy as np
from dataclasses import dataclass, field


@dataclass
class ProcessorState:
    index: int = 0
    mean_index: int = 0

    indices: np.ndarray = field(default_factory=lambda: np.array([]))
    # 4-D rolling buffer: (num_area, n_shots, 2*asp, 2*asp)
    cbck: np.ndarray = field(default_factory=lambda: np.array([]))

    curr_step: float = 1.875
    curr_pulse: int = 2

    speed: np.ndarray = field(default_factory=lambda: np.array([]))
    heading: np.ndarray = field(default_factory=lambda: np.array([]))
    cog: np.ndarray = field(default_factory=lambda: np.array([]))

    vco: float = 0.0

    def init_arrays(self,
                    n_shots: int,
                    num_area: int,
                    mean: int,
                    asp: int):
        self.indices = np.arange(n_shots)
        self.cbck = np.zeros((num_area, n_shots, 2 * asp, 2 * asp), dtype=np.float32)
        self.speed = np.zeros(mean)
        self.heading = np.zeros(mean)
        self.cog = np.zeros(mean)
