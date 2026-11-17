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
    analysis: Literal["equilibrium", "dc_sweep", "small_signal"]
    sweep_contact: Optional[str] = None            # None for equilibrium
    sweep_V: List[float]                           # [0.0] for equilibrium
    currents: Dict[str, List[float]]               # contact name -> current into the device, one per sweep value
    current_unit: Literal["A/cm**2", "A/cm"]       # 1D: per cm^2 of cross-section; 2D: per cm of depth
    flags: List[str] = []
    diagnostics: Dict[str, Any] = {}               # JSON-serialisable only
    # small_signal only (contract A6.2): y_ij = current INTO the device at port i / voltage at port j
    frequencies_Hz: List[float] = []
    y_real: Dict[str, List[float]] = {}            # keys "11", "12", "21", "22": one value per frequency
    y_imag: Dict[str, List[float]] = {}
    y_unit: Optional[Literal["S/cm**2", "S/cm"]] = None    # 1D: per cm^2 of cross-section; 2D: per cm of depth


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


# ---------------------------------------------------------------------------------------------- implantation
class ImplantResult(BaseModel):
    """Engine-neutral result of an ImplantProblem (contract A7.3): where every implanted ion came to rest."""
    model_config = ConfigDict(extra="forbid")

    engine: str
    problem_id: str
    n_ions: int
    n_reflected: int                               # left through the front surface
    n_transmitted: int                             # left through the back of the last layer
    depth_um: List[float]                          # one per implanted ion, in launch order: depth below the surface
    lateral_um: List[float]                        # same ions: displacement from the entry point along the axis
    #                                                perpendicular to the plane of incidence
    flags: List[str] = []
    diagnostics: Dict[str, Any] = {}               # JSON-serialisable only


class ImplantCapability(BaseModel):
    """What an implant engine's translator can run. Every list holds the ALLOWED values."""
    model_config = ConfigDict(extra="forbid")

    engine: str
    env: str
    description: str
    in_process: bool
    structures: List[str]
    potentials: List[str]
    electronic_stopping: List[str]
    recoils: List[bool]
    max_layers: int
    max_keV_per_amu: float                         # beam energy / ion mass above this is refused
    tested_keV_per_amu: Tuple[float, float]        # runs outside are flagged, not refused
    tested_target_symbols: List[str]
    tested_ion_symbols: List[str] = []             # A7.9: ions the acceptance checks passed for (empty: any ion)
    notes: str = ""

    def unsupported(self, problem) -> List[str]:
        """One reason per violated field, each starting with the field name. Empty list = supported."""
        out: List[str] = []

        def check(field: str, value: Any, allowed: List[Any]) -> None:
            if value not in allowed:
                out.append(f"{field}: {_fmt(value)} not in {_fmt(list(allowed))}")

        check("structures", problem.target.structure, self.structures)
        check("potentials", problem.models.potential, self.potentials)
        check("electronic_stopping", problem.models.electronic_stopping, self.electronic_stopping)
        check("recoils", problem.models.recoils, self.recoils)
        if len(problem.target.layers) > self.max_layers:
            out.append(f"max_layers: {len(problem.target.layers)} > {self.max_layers}")
        e = problem.beam.energy_keV / problem.beam.ion.mass_amu
        if e > self.max_keV_per_amu:
            out.append(f"max_keV_per_amu: {e:.4g} keV/amu > {self.max_keV_per_amu:g}")
        return out

    def outside_tested_range(self, problem) -> List[str]:
        """Strings for what lies outside the range the acceptance checks covered (bounds included); [] inside."""
        out: List[str] = []
        lo, hi = self.tested_keV_per_amu
        e = problem.beam.energy_keV / problem.beam.ion.mass_amu
        if not lo <= e <= hi:
            out.append(f"energy_keV_per_amu: {e:.4g} outside tested [{lo:g}, {hi:g}]")
        syms = sorted({s for la in problem.target.layers for s in la.density_cm3})
        if any(s not in self.tested_target_symbols for s in syms):
            out.append(f"target: {syms} not all in tested {sorted(self.tested_target_symbols)}")
        ion = problem.beam.ion.symbol
        if self.tested_ion_symbols and ion not in self.tested_ion_symbols:
            out.append(f"ion: {_fmt(ion)} not in tested {sorted(self.tested_ion_symbols)}")
        return out


# ---------------------------------------------------------------------------------- single-event deposition (A8)
class SeeResult(BaseModel):
    """Engine-neutral result of a SeeProblem (contract A8.3): the histogram of the energy each event left in the SV."""
    model_config = ConfigDict(extra="forbid")

    engine: str
    problem_id: str
    n_events: int
    edges_MeV: List[float]                         # bins + 1, increasing, from the description
    counts: List[float]                            # events with edges[i] <= Edep < edges[i+1]
    underflow: float                               # Edep < low_MeV
    overflow: float                                # Edep >= high_MeV
    flags: List[str] = []
    diagnostics: Dict[str, Any] = {}


class SeeCapability(BaseModel):
    """What a single-event engine's translator can run. Every list holds the ALLOWED values."""
    model_config = ConfigDict(extra="forbid")

    engine: str
    env: str
    description: str
    in_process: bool
    particles: List[str]
    em: List[str]
    hadronic: List[str]
    max_beol_layers: int
    tested_particles: List[str]
    tested_energy_MeV: Tuple[float, float]
    tested_materials: List[str]
    tested_em: List[str]
    tested_hadronic: List[str]
    notes: str = ""

    def unsupported(self, problem) -> List[str]:
        out: List[str] = []

        def check(field: str, value: Any, allowed: List[Any]) -> None:
            if value not in allowed:
                out.append(f"{field}: {_fmt(value)} not in {_fmt(list(allowed))}")

        check("particles", problem.source.particle, self.particles)
        check("em", problem.physics.em, self.em)
        check("hadronic", problem.physics.hadronic, self.hadronic)
        if len(problem.geometry.beol) > self.max_beol_layers:
            out.append(f"max_beol_layers: {len(problem.geometry.beol)} > {self.max_beol_layers}")
        return out

    def outside_tested_range(self, problem) -> List[str]:
        out: List[str] = []
        if problem.source.particle not in self.tested_particles:
            out.append(f"particle: {_fmt(problem.source.particle)} not in tested {self.tested_particles}")
        lo, hi = self.tested_energy_MeV
        e = problem.source.energy_MeV
        if not lo <= e <= hi:
            out.append(f"energy_MeV: {e:g} outside tested [{lo:g}, {hi:g}]")
        mats = sorted({problem.geometry.bulk_material} | {b.material for b in problem.geometry.beol})
        if any(m not in self.tested_materials for m in mats):
            out.append(f"materials: {mats} not all in tested {self.tested_materials}")
        if problem.physics.em not in self.tested_em:
            out.append(f"em: {_fmt(problem.physics.em)} not in tested {self.tested_em}")
        if problem.physics.hadronic not in self.tested_hadronic:
            out.append(f"hadronic: {_fmt(problem.physics.hadronic)} not in tested {self.tested_hadronic}")
        return out
