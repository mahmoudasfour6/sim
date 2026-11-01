"""Ion-track carrier generation for 2D (planar) device simulation.

Pairs per unit track length N(z) = LET(z) * rho / E_pair  (LET from a constant, an SRIM table or a Geant4
deposition profile). In a planar 2D simulation the device extends W (= depth_um) out of plane; the track charge
is spread over that depth, with a Gaussian lateral profile of 1/e radius a:

    G(x, y, t) = N(y) / W * exp(-(x - x0)^2 / a^2) / (sqrt(pi) a) * f(t),   f = unit-area Gaussian in time

so the total generated pairs equal int N dz exactly (enforced on the mesh by `footprint`). The track enters at
the surface y = y_entry and runs toward +y (y is depth) for `length_um`.

Planar 2D is an approximation for a cylindrical track: it represents a track repeated along the out-of-plane
direction. Use it for comparing designs and for code-to-code checks; quantitative single-event cross sections
need 3D or axisymmetric geometry (not yet supported in this package).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from simws.radiation.let import pairs_per_cm
from simws.units import Q_E

LetSpec = Union[float, Callable[[np.ndarray], np.ndarray]]


@dataclass
class IonTrack:
    let: LetSpec                     # MeV cm2/mg, constant or callable of depth below entry (um)
    x0_um: float
    length_um: float
    y_entry_um: float = 0.0
    radius_um: float = 0.05
    t_peak_s: float = 4e-12
    t_sigma_s: float = 1e-12
    depth_um: float = 1.0            # out-of-plane extent W of the 2D device

    def let_at(self, depth_um):
        d = np.asarray(depth_um, float)
        val = self.let(d) if callable(self.let) else np.full_like(d, float(self.let))
        return np.where((d >= 0) & (d <= self.length_um), val, 0.0)

    def temporal(self, t):
        return math.exp(-0.5 * ((t - self.t_peak_s) / self.t_sigma_s) ** 2) / (self.t_sigma_s * math.sqrt(2 * math.pi))

    def total_pairs(self, mat, n=4000):
        z = np.linspace(0, self.length_um, n)
        return float(np.trapezoid(pairs_per_cm(self.let_at(z), mat), z * 1e-4))

    def total_charge_fC(self, mat):
        return self.total_pairs(mat) * Q_E * 1e15

    def density_cm3(self, mat, x_um, y_um):
        """Time-integrated pair density (pairs/cm^3) in the planar 2D model, at points (x_um, y_um)."""
        x, y = np.asarray(x_um, float), np.asarray(y_um, float)
        a_cm, W_cm = self.radius_um * 1e-4, self.depth_um * 1e-4
        lateral = np.exp(-((x - self.x0_um) / self.radius_um) ** 2) / (math.sqrt(math.pi) * a_cm)
        return pairs_per_cm(self.let_at(y - self.y_entry_um), mat) / W_cm * lateral

    def footprint(self, mat, x_um, y_um, node_volume_cm2):
        """Nodal pair density rescaled so sum(footprint * NodeVolume) * W equals total_pairs exactly. The
        correction factor (returned) measures how well the mesh resolves the track; keep it within a few %."""
        dens = self.density_cm3(mat, x_um, y_um)
        on_mesh = float((dens * node_volume_cm2).sum()) * self.depth_um * 1e-4
        factor = self.total_pairs(mat) / on_mesh if on_mesh > 0 else float("nan")
        return dens * factor, factor
