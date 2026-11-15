"""Engine layer base types (architecture v1, section 4.1): result container, capability sheet, errors.

Nothing here imports an engine. Units follow simws.spec: micrometres, cm^-3, V, K.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from simws.spec.schema import DCSweep, Problem


class EngineNotAvailable(RuntimeError):
    """The engine or its executable is not installed on this machine."""


class Unsupported(ValueError):
    """The engine cannot run this problem. `reasons` has one human-readable string per violated rule."""

    def __init__(self, engine: str, reasons: List[str]):
        self.engine = engine
        self.reasons = list(reasons)
        super().__init__(f"{engine} cannot run this problem: " + "; ".join(self.reasons))


class EngineResult(BaseModel):
    """Engine-neutral result. Currents are positive INTO the device, in `current_unit`."""
    model_config = ConfigDict(extra="forbid")

    engine: str
    problem_id: str
    analysis: Literal["equilibrium", "dc_sweep"]
    sweep_contact: Optional[str] = None            # None for equilibrium
    sweep_V: List[float]                           # [0.0] for equilibrium
    currents: Dict[str, List[float]]               # contact name -> current into the device, one per sweep value
    current_unit: Literal["A/cm**2", "A/cm"]       # 1D: per cm^2 of cross-section; 2D: per cm of depth
    flags: List[str] = []
    diagnostics: Dict[str, Any] = {}               # JSON-serialisable only


def _fmt(v: Any) -> str:
    return repr(v)


class Capability(BaseModel):
    """What an engine's translator can run. Every list holds the ALLOWED values."""
    model_config = ConfigDict(extra="forbid")

    engine: str
    env: str                                       # conda env name on the laptop
    description: str
    in_process: bool                               # True: run() works inside Python
    dimensions: List[int]
    max_regions: int
    materials: List[str]
    grading_elements: List[str]
    contact_kinds: List[str]
    statistics: List[str]
    mobility: List[str]
    velocity_saturation: List[bool]
    srh: List[str]
    auger: List[bool]
    band_gap_narrowing: List[str]
    incomplete_ionization: List[str]
    impact_ionization: List[str]
    analyses: List[str]
    tested_temperature_K: Tuple[float, float]      # runs outside this range are flagged, not refused
    mesh_study: bool = False                       # A4.2: can rerun a problem on refined meshes
    notes: str = ""

    def unsupported(self, problem: Problem) -> List[str]:
        """One reason per violated field, each starting with the field name. Empty list = supported."""
        out: List[str] = []

        def check(field: str, value: Any, allowed: List[Any]) -> None:
            if value not in allowed:
                out.append(f"{field}: {_fmt(value)} not in {_fmt(list(allowed))}")

        dev, m = problem.device, problem.models
        check("dimensions", dev.dimension, self.dimensions)
        if len(dev.regions) > self.max_regions:
            out.append(f"max_regions: {len(dev.regions)} > {self.max_regions}")
        for mat in dict.fromkeys(r.material for r in dev.regions):
            check("materials", mat, self.materials)
        for el in dict.fromkeys(g.element for g in dev.grading):
            check("grading_elements", el, self.grading_elements)
        for kind in dict.fromkeys(c.kind for c in dev.contacts):
            check("contact_kinds", kind, self.contact_kinds)
        check("statistics", m.statistics, self.statistics)
        check("mobility", m.mobility, self.mobility)
        check("velocity_saturation", m.velocity_saturation, self.velocity_saturation)
        check("srh", m.srh, self.srh)
        check("auger", m.auger, self.auger)
        check("band_gap_narrowing", m.band_gap_narrowing, self.band_gap_narrowing)
        check("incomplete_ionization", m.incomplete_ionization, self.incomplete_ionization)
        check("impact_ionization", m.impact_ionization, self.impact_ionization)
        check("analyses", problem.analysis.kind, self.analyses)
        return out

    def outside_tested_range(self, problem: Problem) -> List[str]:
        """['temperature_K: 77 outside tested [250, 400]'] when the temperature is outside the range (bounds
        included), else []."""
        lo, hi = self.tested_temperature_K
        t = problem.models.temperature_K
        if lo <= t <= hi:
            return []
        return [f"temperature_K: {t:g} outside tested [{lo:g}, {hi:g}]"]
