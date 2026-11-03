"""Typed objects that cross process boundaries (interface specification: envelopes and schemas sections).

Quantity, Uncertainty, Axis, FieldRef, MaterialLawSpec, JobKey, JobEnvelope, ResultEnvelope, Manifest, OutputSpec.
Every model refuses unknown fields, so a typo or a stale producer fails at the boundary instead of downstream.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from simws.units import Q_

SCHEMA_VERSION = "1.0"
CONVENTIONS_VERSION = "1.0"

Status = Literal["ok", "not_converged", "timeout", "out_of_memory", "invalid_input", "outside_validity",
                 "engine_error", "check_failed"]
Label = Literal["none", "calibration", "validation"]
Fidelity = Literal["screening", "paper"]
Retention = Literal["permanent", "until_validated", "scratch"]

Number = Union[float, int]
Array = Union[Number, List[Any]]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)


def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _check_unit(u: str) -> str:
    Q_(1.0, u)   # raises on a malformed unit
    return u


class Axis(_M):
    name: str
    unit: str
    values: List[float]

    _u = field_validator("unit")(lambda cls, v: _check_unit(v))


class Uncertainty(_M):
    """none | sigma (one standard uncertainty, independent Gaussian) | ensemble (values per member ID) |
    pce (polynomial-chaos coefficients over named variables)."""
    kind: Literal["none", "sigma", "ensemble", "pce"] = "none"
    sigma: Optional[Array] = None
    members: Optional[Dict[str, Array]] = None
    pce: Optional[Dict[str, Any]] = None
    components: Dict[str, Array] = Field(default_factory=dict)   # budget breakdown: numerical, parametric, ...

    @model_validator(mode="after")
    def _consistent(self):
        if self.kind == "sigma" and self.sigma is None:
            raise ValueError("uncertainty kind 'sigma' needs a sigma")
        if self.kind == "ensemble" and not self.members:
            raise ValueError("uncertainty kind 'ensemble' needs members")
        if self.kind == "pce" and not self.pce:
            raise ValueError("uncertainty kind 'pce' needs coefficients")
        return self


class Quantity(_M):
    name: str
    value: Array
    unit: str
    axes: List[Axis] = Field(default_factory=list)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)
    validity: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    flags: List[str] = Field(default_factory=list)
    conventions: Dict[str, str] = Field(default_factory=dict)
    source: Optional[str] = None
    label: Label = "none"
    member: str = "base"

    _u = field_validator("unit")(lambda cls, v: _check_unit(v))

    @field_validator("value", mode="before")
    @classmethod
    def _numpy(cls, v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    def array(self) -> np.ndarray:
        return np.asarray(self.value, dtype=float)

    def to(self, unit: str) -> np.ndarray:
        return Q_(self.array(), self.unit).to(unit).magnitude


class FieldRef(_M):
    name: str
    path: str                       # relative to the result's data/ directory
    mesh_id: str
    location: Literal["node", "edge", "cell"] = "node"
    unit: str
    conventions: Dict[str, str] = Field(default_factory=dict)
    retention: Retention = "until_validated"
    transfer: Optional[Dict[str, Any]] = None   # {"source_mesh": id, "error": float} for transferred fields


class PhysicsEdge(_M):
    output: str
    input: str
    mechanism: str
    sensitivity: Optional[float] = None         # d ln(out) / d ln(in)
    method: Literal["finite_difference", "adjoint", "surrogate", "declared"] = "declared"


class MaterialLawSpec(_M):
    material: str
    parameter: str
    form: Literal["constant", "table", "expression"]
    expression: Optional[str] = None            # name of a registered formula, e.g. "varshni"
    coefficients: Dict[str, Quantity] = Field(default_factory=dict)
    table: Optional[Dict[str, Any]] = None
    unit: str
    validity: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    source: str
    status: Literal["verified", "provisional", "computed", "calibrated"]


class JobKey(_M):
    adapter: str
    adapter_version: str
    engine: str
    engine_version: str
    env_lock: str
    schema_version: str = SCHEMA_VERSION
    conventions_version: str = CONVENTIONS_VERSION
    inputs: Dict[str, Any] = Field(default_factory=dict)     # result/quantity IDs or canonical literal quantities
    params: Dict[str, Any] = Field(default_factory=dict)
    member: str = "base"
    seed: Optional[int] = None


class Budget(_M):
    wall_s: float = 3600.0
    memory_gb: float = 8.0
    threads: int = 1
    ranks: int = 1


class JobEnvelope(_M):
    id: str
    key: JobKey
    env: str
    budget: Budget = Field(default_factory=Budget)
    priority: Literal["interactive", "batch", "overnight"] = "interactive"
    requested_by: str = "user"
    fidelity: Fidelity = "screening"
    created_utc: str = Field(default_factory=now_utc)


def _jsonable(x):
    """numpy scalars/arrays and tuples -> plain JSON values (lenient: non-finite floats become None)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if np.isfinite(f) else None
    return x


class ResultEnvelope(_M):
    id: str = ""
    job_id: str
    status: Status
    quantities: List[Quantity] = Field(default_factory=list)
    fields: List[FieldRef] = Field(default_factory=list)
    physics_edges: List[PhysicsEdge] = Field(default_factory=list)
    verification: Dict[str, Any] = Field(default_factory=dict)
    validity: Dict[str, Any] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    citations: List[str] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)
    message: str = ""
    finished_utc: str = Field(default_factory=now_utc)

    @field_validator("verification", "validity", "diagnostics", "provenance", mode="before")
    @classmethod
    def _plain_dicts(cls, v):
        return _jsonable(v)

    def q(self, name: str, member: Optional[str] = None) -> Quantity:
        for x in self.quantities:
            if x.name == name and (member is None or x.member == member):
                return x
        raise KeyError(f"no quantity '{name}' in result {self.id or self.job_id}")


class OutputSpec(_M):
    name: str
    unit: str                                   # any unit of the required dimension
    constraint: Literal["any", "positive", "nonnegative", "finite"] = "finite"
    description: str = ""

    _u = field_validator("unit")(lambda cls, v: _check_unit(v))


class Manifest(_M):
    name: str
    version: str                                # code version (log text, refactors): provenance only
    physics_version: Optional[str] = None       # enters the job key; None = same as version (0.8 behaviour)
    batch_members: bool = False                 # cheap adapters: all ensemble members in one job
    env: str
    engine: str
    engine_version: str = "unknown"
    description: str = ""
    inputs: List[str] = Field(default_factory=list)
    outputs: List[OutputSpec] = Field(default_factory=list)
    native_conventions: Dict[str, str] = Field(default_factory=dict)
    validity: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    cost_model: Dict[str, Any] = Field(default_factory=dict)
    verification: Dict[str, Any] = Field(default_factory=dict)
    partner: Optional[str] = None
    physics_edges: List[PhysicsEdge] = Field(default_factory=list)
    fallbacks: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    citations: List[str] = Field(default_factory=list)
