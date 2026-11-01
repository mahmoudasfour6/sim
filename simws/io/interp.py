"""Transfer nodal fields between non-matching 2D meshes.

`Transfer(src_points, src_triangles)` locates every destination point in the source triangulation once
(piecewise-linear barycentric interpolation, exact for fields that are linear on each source triangle);
points outside the source mesh take the value of the nearest source node and are reported in
`.outside`. Reuse one Transfer object for every field and every coupling iteration on the same pair
of meshes.
"""
from __future__ import annotations

import numpy as np


class Transfer:
    def __init__(self, src_points, src_triangles, dst_points):
        from matplotlib.tri import Triangulation  # robust point location on an arbitrary triangle list
        from scipy.spatial import cKDTree

        src = np.asarray(src_points, float)[:, :2]
        dst = np.asarray(dst_points, float)[:, :2]
        tri = np.asarray(src_triangles, int)
        self.n_src = len(src)
        finder = Triangulation(src[:, 0], src[:, 1], tri).get_trifinder()
        t_idx = finder(dst[:, 0], dst[:, 1])
        self.outside = np.where(t_idx < 0)[0]
        inside = t_idx >= 0
        self._nodes = np.zeros((len(dst), 3), int)
        self._w = np.zeros((len(dst), 3))
        corners = tri[t_idx[inside]]
        a, b, c = src[corners[:, 0]], src[corners[:, 1]], src[corners[:, 2]]
        p = dst[inside]
        det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        l1 = ((p[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (p[:, 1] - a[:, 1])) / det
        l2 = ((b[:, 0] - a[:, 0]) * (p[:, 1] - a[:, 1]) - (p[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / det
        self._nodes[inside] = corners
        self._w[inside] = np.column_stack([1 - l1 - l2, l1, l2])
        if len(self.outside):
            _, near = cKDTree(src).query(dst[self.outside])
            self._nodes[self.outside] = near[:, None]
            self._w[self.outside] = [1.0, 0.0, 0.0]

    def __call__(self, values):
        v = np.asarray(values, float)
        if v.shape[0] != self.n_src:
            raise ValueError(f"expected {self.n_src} source values, got {v.shape[0]}")
        return (v[self._nodes] * self._w).sum(axis=1)
