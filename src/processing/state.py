import numpy as np
from dataclasses import dataclass, field


@dataclass
class ProcessorState:
    """Mutable accumulation state for one Processor instance.

    index      — total frames received since last reset (monotonically increasing).
    cbck       — ring buffer of bilinearly interpolated segment patches,
                 shape (NUM_AREA, N_SHOTS, 2*ASP, 2*ASP), float32.
                 Written at position index % N_SHOTS; oldest frame is at
                 (index + 1) % N_SHOTS after the buffer is full.
    speed      — SOG ring buffer [m/s], length N_SHOTS, same window as cbck.
    cog        — COG ring buffer [°], length N_SHOTS, same window as cbck.
    heading    — HDG ring buffer [°], length MEAN (output window only).
    curr_step  — range resolution of the most recent frame [m/px].
    curr_pulse — pulse code of the most recent frame (1/2/3).
    """

    index: int = 0
    mean_index: int = 0

    indices: np.ndarray = field(default_factory=lambda: np.array([]))
    cbck: np.ndarray = field(default_factory=lambda: np.array([]))  # (NUM_AREA, N_SHOTS, 2*ASP, 2*ASP)

    curr_step: float = 1.875
    curr_pulse: int = 2

    speed: np.ndarray = field(default_factory=lambda: np.array([]))
    heading: np.ndarray = field(default_factory=lambda: np.array([]))
    cog: np.ndarray = field(default_factory=lambda: np.array([]))

    def init_arrays(self, n_shots: int, num_area: int, mean: int, asp: int):
        """Allocate all ring buffers.  Must be called once before update()."""
        self.indices = np.arange(n_shots)
        self.cbck = np.zeros((num_area, n_shots, 2 * asp, 2 * asp), dtype=np.float32)
        self.speed = np.zeros(n_shots)
        self.heading = np.zeros(mean)
        self.cog = np.zeros(n_shots)
