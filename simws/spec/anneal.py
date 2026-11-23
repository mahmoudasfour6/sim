"""Engine-neutral description of a thermal anneal: one dopant diffusing in silicon (architecture v1, contract A16).

An AnnealProblem says WHAT is annealed: the dopant's starting depth profile, a uniform background, the isothermal
steps, the diffusivity model and the surface condition. It is the fifth problem family; no class here imports an
engine. Units are in the field names: micrometres, cm^-3, degrees Celsius, seconds.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_POINTS = 20000
MAX_INTERVALS = 20000


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DepthProfile(_M):
    """A concentration against depth. Linear between points; the first value holds from 0 to the first point;
    zero beyond the last point (A16.1)."""
    depth_um: List[float]
    concentration_cm3: List[float]

    @model_validator(mode="after")
    def _points(self):
        d, c = self.depth_um, self.concentration_cm3
        if not 2 <= len(d) <= MAX_POINTS:
            raise ValueError(f"profile: 2 to {MAX_POINTS} points, got {len(d)}")
        if len(c) != len(d):
            raise ValueError("profile: depth_um and concentration_cm3 need the same length")
        if not all(np.isfinite(d)) or not all(np.isfinite(c)):
            raise ValueError("profile: values must be finite")
        if d[0] < 0:
            raise ValueError("profile: depths must be >= 0")
        if any(b <= a for a, b in zip(d, d[1:])):
            raise ValueError("profile: depth_um must be strictly increasing")
        if any(v < 0 for v in c):
            raise ValueError("profile: concentrations must be >= 0")
        return self

    def at(self, x_um) -> np.ndarray:
        """The concentration (cm^-3) at depths x_um, by the rule of A16.1."""
        x = np.asarray(x_um, float)
        d, c = np.asarray(self.depth_um), np.asarray(self.concentration_cm3)
        out = np.interp(x, d, c, left=c[0], right=0.0)
        return np.where(x > d[-1], 0.0, out)

    def dose_cm2(self) -> float:
        """Trapezoid integral over the points plus the constant part from 0 to the first point (cm^-2)."""
        d, c = np.asarray(self.depth_um), np.asarray(self.concentration_cm3)
        return float((np.trapezoid(c, d) + c[0] * d[0]) * 1e-4)


class AnnealStep(_M):
    temperature_C: float = Field(ge=600, le=1400)
    time_s: float = Field(gt=0)


class AnnealModels(_M):
    diffusivity: Literal["intrinsic", "fermi"] = "intrinsic"


class AnnealNumerics(_M):
    dx_um: float = Field(default=0.001, gt=0)
    steps_per_anneal: int = Field(default=200, ge=10, le=100000)


class AnnealProblem(_M):
    spec_version: str = "1.0"
    kind: Literal["anneal"] = "anneal"
    dopant: Literal["B", "P", "As", "Sb"]
    initial: DepthProfile
    background_cm3: float = Field(default=0.0, ge=0)
    background_type: Literal["donor", "acceptor"] = "acceptor"
    thickness_um: float = Field(gt=0)
    steps: List[AnnealStep] = Field(min_length=1)
    models: AnnealModels = Field(default_factory=AnnealModels)
    surface: Literal["reflecting", "fixed"] = "reflecting"
    surface_concentration_cm3: Optional[float] = Field(default=None, gt=0)
    numerics: AnnealNumerics = Field(default_factory=AnnealNumerics)

    @model_validator(mode="after")
    def _consistent(self):
        if self.initial.depth_um[-1] > self.thickness_um * (1 + 1e-12):
            raise ValueError("initial: the profile reaches beyond thickness_um")
        if (self.surface == "fixed") != (self.surface_concentration_cm3 is not None):
            raise ValueError("surface: 'fixed' needs surface_concentration_cm3, and only 'fixed' takes it")
        n = self.thickness_um / self.numerics.dx_um
        if n > MAX_INTERVALS * (1 + 1e-9):
            raise ValueError(f"numerics.dx_um: {n:.0f} intervals is more than {MAX_INTERVALS}")
        if abs(n - round(n)) > 1e-6 * max(1.0, n):
            raise ValueError("numerics.dx_um must divide thickness_um into a whole number of intervals")
        return self

    @property
    def dopant_type(self) -> str:
        return "acceptor" if self.dopant == "B" else "donor"

    @property
    def n_intervals(self) -> int:
        return int(round(self.thickness_um / self.numerics.dx_um))

    def output_depths_um(self) -> np.ndarray:
        """x_k = k dx, k = 0..M: where every engine reports the concentration (A16.3)."""
        return np.arange(self.n_intervals + 1) * self.numerics.dx_um

    def time_points_s(self, step: AnnealStep) -> np.ndarray:
        """t_k = time_s (k/N)^2, k = 1..N (A16.1)."""
        N = self.numerics.steps_per_anneal
        return step.time_s * (np.arange(1, N + 1) / N) ** 2

    def problem_id(self) -> str:
        from simws.hub.canonical import job_id
        return job_id(self.model_dump(mode="json"))

    def with_changes(self, changes: Dict[str, object]) -> "AnnealProblem":
        d = self.model_dump(mode="json")
        for path, value in changes.items():
            node = d
            parts = path.split(".")
            for k in parts[:-1]:
                node = node[int(k)] if isinstance(node, list) else node[k]
            last = parts[-1]
            if isinstance(node, list):
                node[int(last)] = value
            else:
                node[last] = value
        return AnnealProblem.model_validate(d)
