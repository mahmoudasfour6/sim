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
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

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
    # A23b: a finite hole recombination velocity at an ohmic contact (cm/s): the hole flux out of the device there is
    # q S (p - p_eq) instead of p = p_eq (a polysilicon emitter's effective contact). None = ohmic as before
    hole_recombination_velocity_cm_s: Optional[float] = None

    @model_serializer(mode="wrap")
    def _drop_unset(self, handler):
        d = handler(self)
        if isinstance(d, dict) and d.get("hole_recombination_velocity_cm_s") is None:   # A23b: no id changes
            d.pop("hole_recombination_velocity_cm_s", None)
        return d

    @model_validator(mode="after")
    def _kind(self):
        if self.hole_recombination_velocity_cm_s is not None:
            if self.kind != "ohmic":
                raise ValueError(f"contact '{self.name}': a hole recombination velocity is for ohmic contacts only")
            if not self.hole_recombination_velocity_cm_s > 0:
                raise ValueError(f"contact '{self.name}': the hole recombination velocity must be > 0")
        if self.kind == "schottky" and self.barrier_eV is None:
            raise ValueError(f"contact '{self.name}': a schottky contact needs barrier_eV")
        if self.kind == "ohmic" and self.barrier_eV is not None:
            raise ValueError(f"contact '{self.name}': barrier_eV is for schottky contacts only")
        if self.span is not None and not self.span[0] < self.span[1]:
            raise ValueError(f"contact '{self.name}': span must have lower < upper")
        return self


class Interface(_M):
    """A25.2: a Si/SiO2 interface on one side of a region (2D): interface traps at midgap (sigma_n = sigma_p =
    sigma_cm2) adding the surface recombination q s (n p - n_ie^2) / (n + p + 2 n_ie), s = sigma v_th N_it, and a
    fixed positive sheet charge q N_ot. span limits it along the side (micrometres); None = the whole side."""
    name: str
    region: str
    side: Side
    span: Optional[Tuple[float, float]] = None
    N_it_cm2: float = Field(default=0.0, ge=0.0)
    N_ot_cm2: float = Field(default=0.0, ge=0.0)
    sigma_cm2: float = Field(default=1e-15, gt=0.0)

    @model_validator(mode="after")
    def _span(self):
        if self.span is not None and not self.span[0] < self.span[1]:
            raise ValueError(f"interface '{self.name}': span must have lower < upper")
        return self


class DopingTerm(_M):
    """A uniform concentration of one dopant type inside a box. Terms add: the donor density at a point is the sum
    of every donor term whose box contains it (likewise acceptors)."""
    species: Literal["donor", "acceptor"]
    concentration_cm3: float = Field(gt=0)
    box: Box = Field(default_factory=Box)
    include_upper: bool = False
    label: str = ""


class ProfileTerm(_M):
    """A tabulated concentration of one dopant type along one axis (contract A16.5), the same along the other axis:
    linear between points, zero outside [first, last]. Terms add to the boxes."""
    species: Literal["donor", "acceptor"]
    axis: Literal["x", "y"] = "x"
    position_um: List[float]
    concentration_cm3: List[float]
    label: str = ""

    @model_validator(mode="after")
    def _points(self):
        x, c = self.position_um, self.concentration_cm3
        if not 2 <= len(x) <= 20000:
            raise ValueError(f"profile term: 2 to 20000 points, got {len(x)}")
        if len(c) != len(x):
            raise ValueError("profile term: position_um and concentration_cm3 need the same length")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(c))):
            raise ValueError("profile term: values must be finite")
        if any(b <= a for a, b in zip(x, x[1:])):
            raise ValueError("profile term: position_um must be strictly increasing")
        if any(v < 0 for v in c):
            raise ValueError("profile term: concentrations must be >= 0")
        return self

    def at(self, coord) -> np.ndarray:
        coord = np.asarray(coord, dtype=float)
        x, c = np.asarray(self.position_um), np.asarray(self.concentration_cm3)
        out = np.interp(coord, x, c, left=0.0, right=0.0)
        return np.where((coord < x[0]) | (coord > x[-1]), 0.0, out)


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
    profiles: List[ProfileTerm] = Field(default_factory=list)     # A16.5: tabulated doping (e.g. after an anneal)
    interfaces: List[Interface] = Field(default_factory=list)     # A25.2: interface traps and oxide charge

    @model_serializer(mode="wrap")
    def _drop_unset(self, handler):
        d = handler(self)
        if isinstance(d, dict) and not d.get("profiles"):          # A16.5: no id changes without profiles
            d.pop("profiles", None)
        if isinstance(d, dict) and not d.get("interfaces"):        # A25.2: no id changes without interfaces
            d.pop("interfaces", None)
        return d

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
        for t in self.profiles:
            if t.axis == "y" and self.dimension == 1:
                raise ValueError("profiles: a 1D device has profiles along x only")
        if (self.mesh.y_lines is None) != (self.dimension == 1):
            raise ValueError("mesh: y_lines are required for 2D and forbidden for 1D")
        names = [i.name for i in self.interfaces]
        if len(set(names)) != len(names):
            raise ValueError("interfaces: names must be unique")
        for i in self.interfaces:
            if self.dimension != 2:
                raise ValueError(f"interface '{i.name}': interfaces are for 2D devices (A25.2)")
            if i.region not in rn:
                raise ValueError(f"interface '{i.name}': unknown region '{i.region}'")
            r = self.region(i.region)
            along = r.box.x if i.side in ("y_min", "y_max") else r.box.y
            if i.span is not None and (i.span[0] < along[0] - 1e-9 or i.span[1] > along[1] + 1e-9):
                raise ValueError(f"interface '{i.name}': span {i.span} is off its side {i.side} "
                                 f"({along[0]:g} to {along[1]:g} um)")
            for c in self.contacts:
                if c.region != i.region or c.side != i.side:
                    continue
                a = i.span or along
                b = c.span or along
                if min(a[1], b[1]) - max(a[0], b[0]) > 1e-9:
                    raise ValueError(f"interface '{i.name}' overlaps contact '{c.name}' on {i.side}: an interface "
                                     "cannot sit on a contact's nodes")
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
        for t in self.profiles:
            v = t.at(x if t.axis == "x" else np.asarray(y, dtype=float))
            if t.species == "donor":
                nd = nd + v
            else:
                na = na + v
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


class Bias(_M):
    """One contact of a DC operating point: ramped to value_V in steps of at most step_V."""
    contact: str
    value_V: float
    step_V: float = Field(default=0.05, gt=0)


class SmallSignal(_M):
    """Two-port AC small-signal analysis about a DC operating point (contract A6). The bias entries are ramped in
    the order given; every other contact stays at 0 V. ports = (port 1, port 2); every other contact is AC ground.
    Result: the operating-point currents and the Y parameters at every frequency."""
    kind: Literal["small_signal"] = "small_signal"
    bias: List[Bias] = Field(default_factory=list)
    ports: Tuple[str, str]
    frequencies_Hz: List[float] = Field(min_length=1)

    @model_validator(mode="after")
    def _ok(self):
        names = [b.contact for b in self.bias]
        if len(set(names)) != len(names):
            raise ValueError("small_signal: a contact may appear once in bias")
        if self.ports[0] == self.ports[1]:
            raise ValueError("small_signal: the two ports must be different contacts")
        f = self.frequencies_Hz
        if any(x <= 0 for x in f) or any(b <= a for a, b in zip(f, f[1:])):
            raise ValueError("small_signal: frequencies must be positive and strictly increasing")
        return self


class StrikeLoad(_M):
    """A18.1: the struck contact on a supply through R, with C to ground, solved together with the device. The 1D
    device stands for `area_um2`."""
    contact: str
    supply_V: float
    R_ohm: float = Field(gt=0)
    C_F: float = Field(default=0.0, ge=0)
    area_um2: float = Field(gt=0)
    supply_step_V: float = Field(default=0.1, gt=0)


class GenerationProfile(_M):
    """A19.1: relative pairs per unit depth, linear between points, zero outside."""
    depth_um: List[float]
    weight: List[float]

    @model_validator(mode="after")
    def _ok(self):
        d, w = self.depth_um, self.weight
        if not 2 <= len(d) <= 20000 or len(w) != len(d):
            raise ValueError("profile: 2 to 20000 points, depth_um and weight of the same length")
        if not (np.all(np.isfinite(d)) and np.all(np.isfinite(w))):
            raise ValueError("profile: values must be finite")
        if any(b <= a for a, b in zip(d, d[1:])):
            raise ValueError("profile: depth_um must be strictly increasing")
        if any(v < 0 for v in w) or not any(v > 0 for v in w):
            raise ValueError("profile: weights must be >= 0 and not all 0")
        return self

    def at(self, x_um) -> np.ndarray:
        x = np.asarray(x_um, float)
        d, w = np.asarray(self.depth_um), np.asarray(self.weight)
        return np.where((x < d[0]) | (x > d[-1]), 0.0, np.interp(x, d, w))


class StrikeTransient(_M):
    """A particle strike as a generation pulse at a fixed bias (contract A17.1), 1D devices: `pairs_cm2` pairs per cm^2
    of device area generated uniformly over depth_um = (start, stop), in time a Gaussian (t_peak_s, sigma_s)
    truncated to t_peak +- 4 sigma and renormalised to unit area. The bias entries are ramped first, in the order
    given; every other contact stays at 0 V. Result: the current into every contact at every time point."""
    kind: Literal["strike"] = "strike"
    bias: List[Bias] = Field(default_factory=list)
    pairs_cm2: float = Field(gt=0)
    depth_um: Tuple[float, float]
    t_peak_s: float = Field(gt=0)
    sigma_s: float = Field(gt=0)
    t_stop_s: float = Field(gt=0)
    steps_per_sigma: int = Field(default=10, ge=2, le=1000)
    points_per_decade: int = Field(default=20, ge=2, le=1000)
    load: Optional[StrikeLoad] = None                  # A18.1
    profile: Optional[GenerationProfile] = None        # A19.1

    @model_serializer(mode="wrap")
    def _drop_unset(self, handler):
        d = handler(self)
        if isinstance(d, dict):                         # A18.1 / A19.1: no A17 id changes
            for k in ("load", "profile"):
                if d.get(k) is None:
                    d.pop(k, None)
        return d

    @model_validator(mode="after")
    def _ok(self):
        names = [b.contact for b in self.bias]
        if len(set(names)) != len(names):
            raise ValueError("strike: a contact may appear once in bias")
        if self.load is not None and self.load.contact in names:
            raise ValueError(f"strike: the load contact '{self.load.contact}' may not also be in bias")
        if self.profile is not None:
            ends = (self.profile.depth_um[0], self.profile.depth_um[-1])
            if any(abs(a - b) > 1e-9 for a, b in zip(ends, self.depth_um)):
                raise ValueError(f"strike: depth_um {tuple(self.depth_um)} must equal the profile's ends {ends}")
        if not self.depth_um[0] < self.depth_um[1]:
            raise ValueError("strike: depth_um must have start < stop")
        if self.t_peak_s < 4.0 * self.sigma_s * (1 - 1e-12):
            raise ValueError("strike: t_peak_s must be at least 4 sigma_s (the pulse starts at t_peak - 4 sigma)")
        if not self.t_stop_s > self.t_peak_s + 4.0 * self.sigma_s:
            raise ValueError("strike: t_stop_s must be after t_peak_s + 4 sigma_s")
        return self

    def window(self) -> Tuple[float, float]:
        return self.t_peak_s - 4.0 * self.sigma_s, self.t_peak_s + 4.0 * self.sigma_s

    def time_points_s(self) -> np.ndarray:
        """A17.1: 10 points geometric from 1e-3 (t_peak - 4 sigma) to t_peak - 4 sigma (none when that is 0), steps
        of sigma / steps_per_sigma to t_peak + 4 sigma, then points_per_decade geometric to t_stop_s."""
        a, b = self.window()
        pts: List[float] = []
        if a > 0:
            pts += list(np.geomspace(1e-3 * a, a, 10))
        n = int(round((b - a) / (self.sigma_s / self.steps_per_sigma)))
        pts += list(a + (b - a) * np.arange(1, n + 1) / n)
        decades = np.log10(self.t_stop_s / b)
        m = max(1, int(np.ceil(decades * self.points_per_decade - 1e-9)))
        pts += list(b * (self.t_stop_s / b) ** (np.arange(1, m + 1) / m))
        out = np.asarray(pts, float)
        return out[np.concatenate(([True], np.diff(out) > 0))]

    def g(self, t) -> np.ndarray:
        """The truncated, renormalised unit-area Gaussian (1/s) at times t."""
        from math import erf, sqrt
        t = np.asarray(t, float)
        a, b = self.window()
        norm = erf(4.0 / sqrt(2.0))
        out = np.exp(-0.5 * ((t - self.t_peak_s) / self.sigma_s) ** 2) / (self.sigma_s * np.sqrt(2 * np.pi) * norm)
        return np.where((t >= a) & (t <= b), out, 0.0)


Analysis = Union[Equilibrium, DCSweep, SmallSignal, StrikeTransient]


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
        if isinstance(a, SmallSignal):
            for n in [b.contact for b in a.bias] + list(a.ports):
                if n not in names:
                    raise ValueError(f"analysis: unknown contact '{n}'")
        if isinstance(a, StrikeTransient):
            for n in [b.contact for b in a.bias] + ([a.load.contact] if a.load else []):
                if n not in names:
                    raise ValueError(f"analysis: unknown contact '{n}'")
            ext = self.device.extent().x
            if a.depth_um[0] < ext[0] - 1e-12 or a.depth_um[1] > ext[1] + 1e-12:
                raise ValueError(f"strike: depth_um {a.depth_um} is not inside the device {ext}")
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
