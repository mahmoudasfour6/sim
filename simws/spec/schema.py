"""Engine-neutral problem description (simws 0.10, architecture v1).

A Problem says WHAT to simulate, never HOW a particular engine does it:

    Problem = DeviceSpec (shape, doping, alloy grading, contacts, mesh hints)
            + ModelSpec  (physics models, each named once)
            + analysis   (what to compute: equilibrium, DC sweep)
            + Numerics   (solver tolerances)

Every engine gets a translator (simws.engines.<engine>) that turns a Problem into that engine's input, and a
capability sheet that says which Problems it can run. No class in this file imports an engine.

Units are fixed and written in the field names or docstrings: lengths in micrometres, concentrations in cm^-3,
voltages in V, temperature in K, mobility in cm^2/V/s, lifetimes in s, energies in eV. (Results leave the hub in
canonical SI; that conversion belongs to the adapters, not to the description.)

Interval rule for boxes: a bound of None is unbounded. A point c is inside (lo, hi) when lo <= c < hi; with
include_upper=True the upper bound is included (lo <= c <= hi). The legacy HBT layout used closed intervals, so
its library entry sets include_upper=True; new devices should keep the default.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

SPEC_VERSION = "1.0"

Bound = Optional[float]
Interval = Tuple[Bound, Bound]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _inside(c, iv: Interval, include_upper: bool):
    c = np.asarray(c, dtype=float)
    lo, hi = iv
    m = np.ones(c.shape, dtype=bool)
    if lo is not None:
        m &= c >= lo
    if hi is not None:
        m &= (c <= hi) if include_upper else (c < hi)
    return m


# ------------------------------------------------------------------------------------------------ geometry
class Box(_M):
    """Axis-aligned box in micrometres. y is None for a 1D device."""
    x: Interval = (None, None)
    y: Optional[Interval] = None

    @model_validator(mode="after")
    def _ordered(self):
        for name, iv in (("x", self.x), ("y", self.y)):
            if iv is not None and iv[0] is not None and iv[1] is not None and not iv[0] < iv[1]:
                raise ValueError(f"box {name} interval must have lower < upper, got {iv}")
        return self

    def contains(self, x, y=None, include_upper: bool = False):
        m = _inside(x, self.x, include_upper)
        if self.y is not None:
            if y is None:
                raise ValueError("a 2D box needs y coordinates")
            m = m & _inside(y, self.y, include_upper)
        return m


class Region(_M):
    """A material region. Its box must be bounded on every side."""
    name: str
    material: str = "silicon"            # a name known to simws.materials.load
    box: Box

    @model_validator(mode="after")
    def _bounded(self):
        ivs = [self.box.x] + ([self.box.y] if self.box.y is not None else [])
        if any(b is None for iv in ivs for b in iv):
            raise ValueError(f"region '{self.name}': the box must be bounded on every side")
        return self


Side = Literal["x_min", "x_max", "y_min", "y_max"]


class Contact(_M):
    """A contact on one side of a region. span limits it along that side (micrometres, the other axis);
    None = the whole side. 1D devices use x_min / x_max only."""
    name: str
    region: str
    side: Side
    span: Optional[Tuple[float, float]] = None
    kind: Literal["ohmic", "schottky"] = "ohmic"
    barrier_eV: Optional[float] = None    # electron barrier height, schottky only

    @model_validator(mode="after")
    def _kind(self):
        if self.kind == "schottky" and self.barrier_eV is None:
            raise ValueError(f"contact '{self.name}': a schottky contact needs barrier_eV")
        if self.kind == "ohmic" and self.barrier_eV is not None:
            raise ValueError(f"contact '{self.name}': barrier_eV is for schottky contacts only")
        if self.span is not None and not self.span[0] < self.span[1]:
            raise ValueError(f"contact '{self.name}': span must have lower < upper")
        return self


class DopingTerm(_M):
    """A uniform concentration of one dopant type inside a box. Terms add: the donor density at a point is the sum
    of every donor term whose box contains it (likewise acceptors)."""
    species: Literal["donor", "acceptor"]
    concentration_cm3: float = Field(gt=0)
    box: Box = Field(default_factory=Box)
    include_upper: bool = False
    label: str = ""


class GradeTerm(_M):
    """Linear alloy grading along one axis: the mole fraction goes from x_start at `start` to x_stop at `stop`
    (micrometres) and is zero outside [start, stop] (both ends included)."""
    element: Literal["Ge"] = "Ge"
    axis: Literal["x", "y"] = "y"
    start_um: float
    stop_um: float
    x_start: float = Field(ge=0, le=1)
    x_stop: float = Field(ge=0, le=1)
    mode: Literal["strained", "relaxed"] = "strained"

    @model_validator(mode="after")
    def _ordered(self):
        if not self.start_um < self.stop_um:
            raise ValueError("grade: start_um must be below stop_um")
        return self


class MeshHints(_M):
    """Tensor-product mesh hints: (position, spacing) pairs in micrometres per axis; engines interpolate the
    spacing between lines. scale multiplies every spacing (0.5 = twice as fine), for refinement studies. An engine
    with its own mesher may use these as target sizes."""
    x_lines: List[Tuple[float, float]]
    y_lines: Optional[List[Tuple[float, float]]] = None
    scale: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _lines(self):
        for name, lines in (("x_lines", self.x_lines), ("y_lines", self.y_lines)):
            if lines is None:
                continue
            if len(lines) < 2:
                raise ValueError(f"mesh {name}: need at least two lines")
            pos = [p for p, _ in lines]
            if pos != sorted(pos) or len(set(pos)) != len(pos):
                raise ValueError(f"mesh {name}: positions must be strictly increasing")
            if any(h <= 0 for _, h in lines):
                raise ValueError(f"mesh {name}: spacings must be positive")
        return self


class DeviceSpec(_M):
    name: str
    dimension: Literal[1, 2]
    regions: List[Region]
    contacts: List[Contact]
    doping: List[DopingTerm] = Field(default_factory=list)
    grading: List[GradeTerm] = Field(default_factory=list)
    mesh: MeshHints
    description: str = ""

    @model_validator(mode="after")
    def _consistent(self):
        if not self.regions:
            raise ValueError("a device needs at least one region")
        rn = [r.name for r in self.regions]
        if len(set(rn)) != len(rn):
            raise ValueError("region names must be unique")
        cn = [c.name for c in self.contacts]
        if len(set(cn)) != len(cn):
            raise ValueError("contact names must be unique")
        for r in self.regions:
            if (r.box.y is None) != (self.dimension == 1):
                raise ValueError(f"region '{r.name}': a {self.dimension}D device needs "
                                 f"{'no y interval' if self.dimension == 1 else 'a y interval'}")
        for c in self.contacts:
            if c.region not in rn:
                raise ValueError(f"contact '{c.name}': unknown region '{c.region}'")
            if self.dimension == 1 and (c.side not in ("x_min", "x_max") or c.span is not None):
                raise ValueError(f"contact '{c.name}': a 1D contact sits on x_min or x_max and has no span")
        for t in list(self.doping):
            if (t.box.y is not None) and self.dimension == 1:
                raise ValueError("doping: a 1D device has no y interval")
        for g in self.grading:
            if g.axis == "y" and self.dimension == 1:
                raise ValueError("grading: a 1D device grades along x")
        if (self.mesh.y_lines is None) != (self.dimension == 1):
            raise ValueError("mesh: y_lines are required for 2D and forbidden for 1D")
        return self

    # -------------------------------------------------------------------------------------- reference evaluators
    # These define the meaning of the description. A translator is correct when its engine sees the same values
    # at every mesh node.
    def region(self, name: str) -> Region:
        for r in self.regions:
            if r.name == name:
                return r
        raise KeyError(f"no region '{name}'")

    def contact(self, name: str) -> Contact:
        for c in self.contacts:
            if c.name == name:
                return c
        raise KeyError(f"no contact '{name}'")

    def extent(self) -> Box:
        xs = [b for r in self.regions for b in r.box.x]
        if self.dimension == 1:
            return Box(x=(min(xs), max(xs)))
        ys = [b for r in self.regions for b in r.box.y]
        return Box(x=(min(xs), max(xs)), y=(min(ys), max(ys)))

    def doping_at(self, x, y=None) -> Tuple[np.ndarray, np.ndarray]:
        """(donors, acceptors) in cm^-3 at the given points (micrometres)."""
        x = np.asarray(x, dtype=float)
        nd, na = np.zeros(x.shape), np.zeros(x.shape)
        for t in self.doping:
            m = t.box.contains(x, y, t.include_upper)
            if t.species == "donor":
                nd = nd + t.concentration_cm3 * m
            else:
                na = na + t.concentration_cm3 * m
        return nd, na

    def mole_fraction_at(self, x, y=None, element: str = "Ge") -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros(x.shape)
        for g in self.grading:
            if g.element != element:
                continue
            c = x if g.axis == "x" else np.asarray(y, dtype=float)
            m = (c >= g.start_um) & (c <= g.stop_um)
            out = out + m * (g.x_start + (g.x_stop - g.x_start) * (c - g.start_um) / (g.stop_um - g.start_um))
        return out


# ------------------------------------------------------------------------------------------------ models
class ModelSpec(_M):
    """Physics models. Each choice is a name every engine translates (or refuses in its capability sheet)."""
    temperature_K: float = Field(default=300.0, gt=0)
    statistics: Literal["boltzmann", "fermi_dirac"] = "boltzmann"
    mobility: Literal["constant", "masetti", "klaassen"] = "masetti"
    mu_n_cm2_Vs: Optional[float] = Field(default=None, gt=0)       # mobility == "constant"
    mu_p_cm2_Vs: Optional[float] = Field(default=None, gt=0)
    velocity_saturation: bool = False
    srh: Literal["off", "constant", "scharfetter"] = "scharfetter"
    tau_n_s: Optional[float] = Field(default=None, gt=0)           # srh == "constant"
    tau_p_s: Optional[float] = Field(default=None, gt=0)
    auger: bool = True
    band_gap_narrowing: Literal["off", "slotboom"] = "slotboom"
    incomplete_ionization: Literal["off", "altermatt", "constant_energy"] = "off"
    ii_energy_eV: Optional[float] = Field(default=None, gt=0)      # constant_energy
    ii_critical_doping_cm3: Optional[float] = Field(default=None, gt=0)
    impact_ionization: Literal["off", "van_overstraeten"] = "off"
    donor_species: Literal["P", "As"] = "P"
    acceptor_species: Literal["B"] = "B"
    extended_precision: Optional[bool] = None                      # None = engine decides

    @model_validator(mode="after")
    def _paired(self):
        const_mu = self.mobility == "constant"
        if const_mu != (self.mu_n_cm2_Vs is not None and self.mu_p_cm2_Vs is not None) or \
                (not const_mu and (self.mu_n_cm2_Vs is not None or self.mu_p_cm2_Vs is not None)):
            raise ValueError("mobility 'constant' needs mu_n_cm2_Vs and mu_p_cm2_Vs; other models take neither")
        const_tau = self.srh == "constant"
        if const_tau != (self.tau_n_s is not None and self.tau_p_s is not None) or \
                (not const_tau and (self.tau_n_s is not None or self.tau_p_s is not None)):
            raise ValueError("srh 'constant' needs tau_n_s and tau_p_s; other choices take neither")
        const_ii = self.incomplete_ionization == "constant_energy"
        if const_ii != (self.ii_energy_eV is not None and self.ii_critical_doping_cm3 is not None) or \
                (not const_ii and (self.ii_energy_eV is not None or self.ii_critical_doping_cm3 is not None)):
            raise ValueError("incomplete_ionization 'constant_energy' needs ii_energy_eV and "
                             "ii_critical_doping_cm3; other choices take neither")
        return self


# ------------------------------------------------------------------------------------------------ analyses
class Equilibrium(_M):
    """Zero bias on every contact. Result: terminal currents (must vanish)."""
    kind: Literal["equilibrium"] = "equilibrium"


class Track(_M):
    """A contact that follows the swept one: V = V_sweep + offset_V, ramped in steps of at most step_V."""
    contact: str
    offset_V: float = 0.0
    step_V: float = Field(default=0.1, gt=0)


class DCSweep(_M):
    """Step one contact through `values_V`. At every value the tracking contacts are ramped first, in the order
    given, then the swept contact. Contacts in `fixed_V` are ramped once before the sweep; every other contact
    stays at 0 V. Result: the current into every contact at every value."""
    kind: Literal["dc_sweep"] = "dc_sweep"
    contact: str
    values_V: List[float] = Field(min_length=1)
    step_V: float = Field(default=0.05, gt=0)
    tracking: List[Track] = Field(default_factory=list)
    fixed_V: Dict[str, float] = Field(default_factory=dict)


Analysis = Union[Equilibrium, DCSweep]


class Numerics(_M):
    relative_error: float = Field(default=1e-8, gt=0)
    absolute_error: float = Field(default=1e10, gt=0)       # carriers / s per cm of depth (DEVSIM convention)
    max_iterations: int = Field(default=40, gt=0)


# ------------------------------------------------------------------------------------------------ problem
class Problem(_M):
    spec_version: str = SPEC_VERSION
    device: DeviceSpec
    models: ModelSpec = Field(default_factory=ModelSpec)
    analysis: Analysis = Field(default_factory=Equilibrium, discriminator="kind")
    numerics: Numerics = Field(default_factory=Numerics)

    @model_validator(mode="after")
    def _contacts_exist(self):
        names = {c.name for c in self.device.contacts}
        a = self.analysis
        if isinstance(a, DCSweep):
            used = [a.contact] + [t.contact for t in a.tracking] + list(a.fixed_V)
            for n in used:
                if n not in names:
                    raise ValueError(f"analysis: unknown contact '{n}'")
            if len(set(used)) != len(used):
                raise ValueError("analysis: a contact may be swept, tracking or fixed, not several of these")
        return self

    def problem_id(self) -> str:
        """Content hash of the whole description (32 hex characters): same Problem, same id, on any machine."""
        from simws.hub.canonical import job_id
        return job_id(self.model_dump(mode="json"))

    def with_changes(self, changes: Dict[str, object]) -> "Problem":
        """A copy with dotted-path overrides, validated again:
        p.with_changes({"models.temperature_K": 77.0, "device.mesh.scale": 0.5})"""
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
                if last not in node:
                    raise KeyError(f"no field '{path}' in the problem description")
                node[last] = value
        return Problem.model_validate(d)
