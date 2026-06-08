import numpy as np


class Area:
    """Square spatial segment extracted from a polar radar image via bilinear interpolation.

    The radar image is stored in polar coordinates (azimuth × range).  Area
    maps a Cartesian square of side `size` pixels, centred at (`dist`, `azim`)
    in polar space, back to (azimuth, range) indices with sub-pixel fractional
    weights for bilinear interpolation.

    calc_mask() is called once at startup; the returned index arrays are reused
    every frame by Processor.update() to extract segment patches without
    recomputing the geometry.
    """

    def __init__(self, size, dist, azim, orient, aap):
        """
        size   — side length of the square segment [px].
        dist   — distance from antenna to segment centre [px] (= ADP).
        azim   — azimuth of segment centre [rad].
        orient — rotation of the square relative to the radial direction [rad];
                 0 means sides are parallel/perpendicular to the radial axis.
        aap    — total number of azimuth lines in the radar image (= AAP).
        """
        self.size = size
        self.dist = dist
        self.azim = azim
        self.orient = orient
        self.aap = aap

    def get_vertex(self):
        """Return the four corner coordinates of the segment square in Cartesian space.

        Corners are ordered: [right-upper, left-upper, left-lower, right-lower].
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
        """Precompute bilinear interpolation indices and weights for this segment.

        Returns (mask_div, mask_mod) where:
          mask_div — integer (range, azimuth) indices into the polar image,
                     shape (2, size, size), int32.
          mask_mod — fractional part of each coordinate, shape (2, size, size),
                     float64.  Used by Processor.update() as:
                       (x, y), (wx, wy) = mask_div, mask_mod
                       row = bck[y, x]*(1-wx) + bck[y, x+1]*wx
                       val = row*(1-wy) + next_row*wy
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
