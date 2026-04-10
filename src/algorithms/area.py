import numpy as np


class Area:
    """
    segment for processing
    """

    def __init__(self, size, dist, azim, orient, aap):
        """
        :param size: length of segment edge [px]
        :param dist: distance from common center to segment center [px]
        :param azim: azimuth of segment center [rad]
        :param orient: angle between common north and parallel for left/right edge [rad]
        """
        self.size = size
        self.dist = dist
        self.azim = np.rad2deg(azim)
        self.orient = np.rad2deg(orient)
        self.aap = aap

    def get_vertex(self):
        """
        :return: vertex coordinates
        """
        center = np.array([
            self.dist * np.cos(self.azim),
            self.dist * np.sin(self.azim)
        ])
        diag_2 = 0.5 * self.size * np.sqrt(2)

        angle_min = 0.25 * np.pi - self.orient
        angle_plus = 0.25 * np.pi + self.orient

        lu_rd_tmp = diag_2 * np.array([-np.cos(angle_plus), np.sin(angle_plus)])
        ru_ld_tmp = diag_2 * np.array([np.cos(angle_min), np.sin(angle_min)])

        return np.array([
            center + ru_ld_tmp,
            center - lu_rd_tmp,
            center - ru_ld_tmp,
            center + lu_rd_tmp
        ])

    def calc_mask(self):
        """
        :return: coefficients for bilinear interpolation
        """
        vertex = self.get_vertex()
        mask_div = np.zeros((2, self.size, self.size), dtype=np.int32)
        mask_mod = np.zeros((2, self.size, self.size), dtype=np.float64)

        for i in range(self.size):
            x = np.linspace(vertex[2, 0] + i / self.size * (vertex[1, 0] - vertex[2, 0]),
                            vertex[3, 0] + i / self.size * (vertex[0, 0] - vertex[3, 0]),
                            self.size)
            y = np.linspace(vertex[2, 1] + i / self.size * (vertex[1, 1] - vertex[2, 1]),
                            vertex[3, 1] + i / self.size * (vertex[0, 1] - vertex[3, 1]),
                            self.size)

            r = np.sqrt(x ** 2 + y ** 2)
            r[r >= self.aap - 1] = 0
            t = self.aap * (0.25 - np.arctan2(x, y) / (2 * np.pi))

            mask_div[0, i] = np.floor(r)
            mask_div[1, i] = np.floor(t)
            mask_mod[0, i] = r - mask_div[0, i]
            mask_mod[1, i] = t - mask_div[1, i]

        return mask_div, mask_mod
