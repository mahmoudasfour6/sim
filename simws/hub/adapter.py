"""Adapter contract (interface specification: adapter lifecycle).

An adapter wraps one engine operation. It runs in its own conda env as an isolated process
(`python -m simws.run <adapter> <staging dir>`), reads job.json, writes data/ and result.json, and exits.

    class HbtGummel(Adapter):
        manifest = Manifest(name="devsim.hbt_gummel", version="1.0", env="tcad", engine="devsim", ...)
        def run(self, job, ctx):           # call the engine; may raise the exceptions below
            ...
            return {"IC": ic, ...}
        def collect(self, job, ctx, raw):  # convert to canonical quantities
            return [ctx.quantity("hbt.IC", raw["IC"], "A", ...)]

Lifecycle: prepare -> run -> collect -> self_check. Exceptions map to failure categories:
NotConverged -> not_converged, OutsideValidity -> outside_validity, InvalidInput -> invalid_input,
TimeoutError -> timeout, MemoryError -> out_of_memory, anything else -> engine_error.

The registry maps adapter names to "module:Class" so the hub can read a manifest without importing the engine:
engine imports belong inside run().
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from simws.hub.schema import (Axis, FieldRef, JobEnvelope, Manifest, PhysicsEdge, Quantity, ResultEnvelope,
                              Uncertainty)


class AdapterError(Exception):
    status = "engine_error"


class NotConverged(AdapterError):
    status = "not_converged"


class OutsideValidity(AdapterError):
    status = "outside_validity"


class InvalidInput(AdapterError):
    status = "invalid_input"


def status_for(exc: BaseException) -> str:
    if isinstance(exc, AdapterError):
        return exc.status
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, MemoryError):
        return "out_of_memory"
    return "engine_error"


class Context:
    """What an adapter sees of its staging directory, plus helpers that tag outputs consistently."""

    def __init__(self, staging: Path, job: JobEnvelope, manifest: Manifest):
        self.staging = Path(staging)
        self.data = self.staging / "data"
        self.logs = self.staging / "logs"
        self.data.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.job = job
        self.manifest = manifest
        self.fields: List[FieldRef] = []
        self.physics_edges: List[PhysicsEdge] = list(manifest.physics_edges)
        self.verification: Dict[str, Any] = {}
        self.diagnostics: Dict[str, Any] = {}
        self.flags: List[str] = []
        self.citations: List[str] = list(manifest.citations)

    @property
    def params(self) -> Dict[str, Any]:
        return self.job.key.params

    def param(self, name, default=None, unit: Optional[str] = None):
        """A parameter, converted to `unit` when it was given as a canonical quantity {value, unit}."""
        v = self.job.key.params.get(name, self.job.key.inputs.get(name, default))
        if isinstance(v, dict) and "value" in v and "unit" in v:
            if unit is None:
                return v["value"]
            from simws.units import Q_
            import numpy as np
            return Q_(np.asarray(v["value"], float), v["unit"]).to(unit).magnitude
        return v

    def quantity(self, name, value, unit, *, sigma=None, members=None, axes=None, conventions=None,
                 validity=None, source=None, flags=None, member=None) -> Quantity:
        if members:
            unc = Uncertainty(kind="ensemble", members=members)
        elif sigma is not None:
            unc = Uncertainty(kind="sigma", sigma=sigma)
        else:
            unc = Uncertainty()
        return Quantity(name=name, value=value, unit=unit, uncertainty=unc,
                        axes=[a if isinstance(a, Axis) else Axis(**a) for a in (axes or [])],
                        conventions=conventions or {}, validity=validity or {}, source=source or self.manifest.name,
                        flags=list(flags or []), member=member or self.job.key.member)

    def field(self, name, path, unit, mesh_id, location="node", retention="until_validated", conventions=None):
        f = FieldRef(name=name, path=str(path), unit=unit, mesh_id=mesh_id, location=location,
                     retention=retention, conventions=conventions or {})
        self.fields.append(f)
        return f


class Adapter:
    manifest: Manifest

    def resolve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Runs at submit time in the hub's process: fill every default that changes the result (pseudopotential
        set and file hashes, cutoffs from config) into the params, so the job key captures it. Must not import
        the engine."""
        return params

    def prepare(self, job: JobEnvelope, ctx: Context) -> None:
        for name in self.manifest.inputs:
            if name not in job.key.params and name not in job.key.inputs:
                raise InvalidInput(f"missing input '{name}'")

    def run(self, job: JobEnvelope, ctx: Context) -> Any:
        raise NotImplementedError

    def collect(self, job: JobEnvelope, ctx: Context, raw: Any) -> List[Quantity]:
        raise NotImplementedError

    def self_check(self, job: JobEnvelope, ctx: Context, quantities: List[Quantity]) -> List[str]:
        """Engine-specific sanity checks; return a list of problems (empty = fine)."""
        return []

    # ------------------------------------------------------------------ full lifecycle
    def execute(self, job: JobEnvelope, staging: Path) -> ResultEnvelope:
        members = job.key.params.get(MEMBERS_KEY)
        if members:
            return self._execute_members(job, staging, members)
        ctx = Context(staging, job, self.manifest)
        try:
            self.prepare(job, ctx)
            raw = self.run(job, ctx)
            qs = self.collect(job, ctx, raw)
            problems = self.self_check(job, ctx, qs)
            status = "check_failed" if problems else "ok"
            msg = "; ".join(problems)
        except BaseException as exc:  # noqa: BLE001  (KeyboardInterrupt re-raised below)
            if isinstance(exc, KeyboardInterrupt):
                raise
            import traceback
            (ctx.logs / "traceback.txt").write_text(traceback.format_exc())
            qs, status, msg = [], status_for(exc), f"{type(exc).__name__}: {exc}"
        return ResultEnvelope(job_id=job.id, status=status, quantities=qs, fields=ctx.fields,
                              physics_edges=ctx.physics_edges, verification=ctx.verification,
                              diagnostics=ctx.diagnostics, flags=ctx.flags, message=msg,
                              citations=ctx.citations,
                              validity={"box": {k: list(v) for k, v in self.manifest.validity.items()}})


    def _execute_members(self, job: JobEnvelope, staging: Path, members: Dict[str, Dict[str, Any]]) -> ResultEnvelope:
        """Batched ensemble: run every member in this one process; quantities carry their member IDs. The whole
        job fails if any member fails (a partial ensemble would bias the spread)."""
        base = {k: v for k, v in job.key.params.items() if k != MEMBERS_KEY}
        qs, flags, problems, diag, verif = [], [], [], {}, {}
        for mid in sorted(members):
            sub = job.model_copy(deep=True)
            sub.key.params = {**base, **members[mid]}
            sub.key.member = mid
            r = self.execute(sub, staging)
            if r.status != "ok":
                return ResultEnvelope(job_id=job.id, status=r.status, message=f"member {mid}: {r.message}",
                                      citations=r.citations, diagnostics={"failed_member": mid})
            for q in r.quantities:
                q.member = mid
            qs += r.quantities
            flags += [f for f in r.flags if f not in flags]
            diag[mid], verif[mid] = r.diagnostics, r.verification
        return ResultEnvelope(job_id=job.id, status="ok", quantities=qs, flags=flags,
                              diagnostics={"members": diag, "batched": True}, verification=verif,
                              physics_edges=list(self.manifest.physics_edges), citations=list(self.manifest.citations),
                              validity={"box": {k: list(v) for k, v in self.manifest.validity.items()}},
                              message="; ".join(problems))


MEMBERS_KEY = "_members"      # params key for batched ensemble members {member id: param overrides}


# ---------------------------------------------------------------------------------------------- registry
REGISTRY: Dict[str, str] = {}


def register(name: str, target: str):
    """register('devsim.hbt_gummel', 'simws.adapters.devsim_hbt:HbtGummel')"""
    REGISTRY[name] = target


def _load_builtin():
    try:
        importlib.import_module("simws.adapters")      # registers the built-in adapters
    except ModuleNotFoundError as e:
        if e.name != "simws.adapters":
            raise


def get(name: str) -> Adapter:
    _load_builtin()
    if name not in REGISTRY:
        raise KeyError(f"unknown adapter '{name}'; known: {sorted(REGISTRY)}")
    mod, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(mod), cls)()


def manifest(name: str) -> Manifest:
    return get(name).manifest


def names() -> List[str]:
    _load_builtin()
    return sorted(REGISTRY)
