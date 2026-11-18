"""Engine-neutral description of a circuit (architecture v1, contract amendment A9).

A CircuitProblem is a netlist (elements, models, model libraries), one analysis (op, dc, ac or tran), the probes
to report, the temperatures and the solver tolerances. No class here imports an engine. SI units in the field
names; element, model and node names are case-insensitive, as in SPICE; node "0" is ground.
"""
from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

NAME = r"^[A-Za-z][A-Za-z0-9_]*$"
NODE = r"^[A-Za-z0-9_]+$"
MAX_DC_POINTS = 100001
_PROBE_V = re.compile(r"^v\(\s*([A-Za-z0-9_]+)\s*(?:,\s*([A-Za-z0-9_]+)\s*)?\)$", re.I)
_PROBE_I = re.compile(r"^i\(\s*([A-Za-z][A-Za-z0-9_]*)\s*\)$", re.I)

_TWO = {"resistor", "capacitor", "inductor", "vsource", "isource", "diode"}
_VALUED = {"resistor", "capacitor", "inductor"}
_SOURCES = {"vsource", "isource"}
_MODELLED = {"diode", "bjt", "va", "subckt"}
_MODEL_KIND = {"diode": {"diode"}, "bjt": {"npn", "pnp"}, "va": {"va"}}


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Pulse(_M):
    v1: float
    v2: float
    delay_s: float = Field(default=0.0, ge=0)
    rise_s: float = Field(gt=0)
    fall_s: float = Field(gt=0)
    width_s: float = Field(gt=0)
    period_s: float = Field(gt=0)


class Pwl(_M):
    points: List[Tuple[float, float]] = Field(min_length=2)

    @model_validator(mode="after")
    def _times(self):
        t = [p[0] for p in self.points]
        if t[0] < 0 or any(b <= a for a, b in zip(t, t[1:])):
            raise ValueError("pwl: times must be >= 0 and strictly increasing")
        return self


class Element(_M):
    name: str = Field(pattern=NAME)
    kind: Literal["resistor", "capacitor", "inductor", "vsource", "isource", "diode", "bjt", "va", "subckt"]
    nodes: List[str]
    value: Optional[float] = None
    ac: Optional[float] = None
    pulse: Optional[Pulse] = None
    pwl: Optional[Pwl] = None
    model: Optional[str] = None
    params: Dict[str, float] = {}

    @model_validator(mode="after")
    def _rules(self):
        k, n = self.kind, len(self.nodes)
        for node in self.nodes:
            if not re.match(NODE, node):
                raise ValueError(f"element '{self.name}': bad node name '{node}'")
        if k in _TWO and n != 2:
            raise ValueError(f"element '{self.name}': a {k} has 2 nodes, got {n}")
        if k == "bjt" and n not in (3, 4):
            raise ValueError(f"element '{self.name}': a bjt has 3 or 4 nodes, got {n}")
        if k in ("va", "subckt") and n < 1:
            raise ValueError(f"element '{self.name}': a {k} needs at least one node")
        if k in _VALUED:
            if self.value is None or not self.value > 0:
                raise ValueError(f"element '{self.name}': a {k} needs a value > 0")
        elif k not in _SOURCES and self.value is not None:
            raise ValueError(f"element '{self.name}': a {k} takes no value")
        if k not in _SOURCES and (self.ac is not None or self.pulse is not None or self.pwl is not None):
            raise ValueError(f"element '{self.name}': ac, pulse and pwl are for sources only")
        if self.pulse is not None and self.pwl is not None:
            raise ValueError(f"element '{self.name}': give pulse or pwl, not both")
        if k in _MODELLED and not self.model:
            raise ValueError(f"element '{self.name}': a {k} needs a model")
        if k not in _MODELLED and self.model is not None:
            raise ValueError(f"element '{self.name}': a {k} takes no model")
        return self


class VerilogA(_M):
    module: str = Field(pattern=NAME)
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _module(self):
        if not re.search(rf"\bmodule\s+{re.escape(self.module)}\b", self.source):
            raise ValueError(f"verilog-a: the source has no 'module {self.module}'")
        return self


class Model(_M):
    name: str = Field(pattern=NAME)
    kind: Literal["diode", "npn", "pnp", "va"]
    params: Dict[str, float] = {}
    va: Optional[VerilogA] = None

    @model_validator(mode="after")
    def _va(self):
        if (self.kind == "va") != (self.va is not None):
            raise ValueError(f"model '{self.name}': a Verilog-A source is needed for kind 'va' and only for it")
        return self


class Library(_M):
    path: str = Field(min_length=1)
    section: Optional[str] = None


class Op(_M):
    kind: Literal["op"] = "op"


class DcSweep(_M):
    kind: Literal["dc"] = "dc"
    source: str
    start: float
    stop: float
    step: float

    @model_validator(mode="after")
    def _steps(self):
        if self.step == 0 or (self.stop - self.start) * self.step < 0:
            raise ValueError("dc: step must be non-zero with the sign of stop - start")
        if self.n_points() < 2 or self.n_points() > MAX_DC_POINTS:
            raise ValueError(f"dc: between 2 and {MAX_DC_POINTS} points, got {self.n_points()}")
        return self

    def n_points(self) -> int:
        return int(round((self.stop - self.start) / self.step)) + 1


class AcSweep(_M):
    kind: Literal["ac"] = "ac"
    points_per_decade: int = Field(ge=1, le=1000)
    f_start_Hz: float = Field(gt=0)
    f_stop_Hz: float

    @model_validator(mode="after")
    def _range(self):
        if not self.f_stop_Hz > self.f_start_Hz:
            raise ValueError("ac: f_stop_Hz must be > f_start_Hz")
        return self


class Transient(_M):
    kind: Literal["tran"] = "tran"
    step_s: float = Field(gt=0)
    stop_s: float
    max_step_s: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _range(self):
        if not self.stop_s > self.step_s:
            raise ValueError("tran: stop_s must be > step_s")
        return self


class CircuitNumerics(_M):
    reltol: float = Field(default=1e-6, ge=1e-9, le=1e-2)
    abstol_A: float = Field(default=1e-15, gt=0)
    vntol_V: float = Field(default=1e-9, gt=0)
    gmin_S: float = Field(default=1e-15, ge=0)


Analysis = Union[Op, DcSweep, AcSweep, Transient]


def parse_probe(p: str):
    """("v", node, node2 or None) or ("i", source name); raises ValueError on anything else."""
    m = _PROBE_V.match(p.strip())
    if m:
        return ("v", m.group(1).lower(), m.group(2).lower() if m.group(2) else None)
    m = _PROBE_I.match(p.strip())
    if m:
        return ("i", m.group(1).lower())
    raise ValueError(f"probe '{p}': use v(node), v(node,node) or i(vsource)")


class CircuitProblem(_M):
    spec_version: str = "1.0"
    kind: Literal["circuit"] = "circuit"
    title: str = "simws circuit"
    elements: List[Element] = Field(min_length=1)
    models: List[Model] = []
    libraries: List[Library] = []
    analysis: Analysis = Field(discriminator="kind")
    probes: List[str] = Field(min_length=1)
    temperature_C: float = Field(default=27.0, gt=-273.15)
    tnom_C: float = Field(default=27.0, gt=-273.15)
    numerics: CircuitNumerics = Field(default_factory=CircuitNumerics)

    @model_validator(mode="after")
    def _consistent(self):
        names = [e.name.lower() for e in self.elements]
        if len(set(names)) != len(names):
            raise ValueError("circuit: element names must be unique (case-insensitive)")
        mnames = [m.name.lower() for m in self.models]
        if len(set(mnames)) != len(mnames):
            raise ValueError("circuit: model names must be unique (case-insensitive)")
        models = {m.name.lower(): m for m in self.models}
        nodes = {n.lower() for e in self.elements for n in e.nodes}
        if "0" not in nodes:
            raise ValueError("circuit: no element is connected to ground (node '0')")
        kinds = {e.name.lower(): e.kind for e in self.elements}
        for e in self.elements:
            if e.kind in _MODEL_KIND:
                m = models.get(e.model.lower())
                if m is None:
                    raise ValueError(f"element '{e.name}': no model '{e.model}'")
                if m.kind not in _MODEL_KIND[e.kind]:
                    raise ValueError(f"element '{e.name}': a {e.kind} cannot use a model of kind '{m.kind}'")
            if e.kind == "subckt" and not self.libraries:
                raise ValueError(f"element '{e.name}': a subckt needs a library")
        for p in self.probes:
            pr = parse_probe(p)
            if pr[0] == "v":
                for n in pr[1:]:
                    if n is not None and n not in nodes:
                        raise ValueError(f"probe '{p}': node '{n}' is not in the circuit")
            elif kinds.get(pr[1]) != "vsource":
                raise ValueError(f"probe '{p}': '{pr[1]}' is not a vsource")
        if isinstance(self.analysis, DcSweep) and kinds.get(self.analysis.source.lower()) not in _SOURCES:
            raise ValueError(f"dc: '{self.analysis.source}' is not a vsource or isource")
        return self

    def problem_id(self) -> str:
        from simws.hub.canonical import job_id
        return job_id(self.model_dump(mode="json"))

    def with_changes(self, changes: Dict[str, object]) -> "CircuitProblem":
        """A copy with dotted-path overrides, validated again: p.with_changes({"temperature_C": 50.0})"""
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
        return CircuitProblem.model_validate(d)
