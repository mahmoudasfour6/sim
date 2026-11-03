"""Gatekeeper (interface specification: gatekeeper checks).

Runs before a result is committed. A failed hard check turns the result into status 'check_failed' with the reasons
in `message`; soft checks add flags. The gatekeeper never edits values.

 1  schema           envelope validates (pydantic did this on load) and job_id matches
 2  declared outputs every manifest output is present (status ok only)
 3  units            each quantity's unit has the dimension the manifest declares
 4  finite           values are finite; 'positive'/'nonnegative' constraints hold
 5  conventions      every axis tagged on a quantity is a known axis with the canonical value
 6  validity         inputs outside the manifest validity box -> flag 'extrapolated' (not rejected; spec decision)
 7  uncertainty      paper fidelity requires uncertainty on every quantity; screening flags 'unverified'
 8  fields           every field file exists and its unit parses
 9  verification     paper fidelity requires the manifest's verification entries to be reported
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from simws import conventions as cv
from simws.hub.schema import JobEnvelope, Manifest, ResultEnvelope
from simws.units import Q_


class GateReport:
    def __init__(self):
        self.errors: List[str] = []
        self.flags: List[str] = []

    @property
    def ok(self) -> bool:
        return not self.errors

    def __repr__(self):
        return f"GateReport(ok={self.ok}, errors={self.errors}, flags={self.flags})"


def _same_dim(u1: str, u2: str) -> bool:
    return Q_(1.0, u1).dimensionality == Q_(1.0, u2).dimensionality


def _input_ranges(job: JobEnvelope) -> dict:
    """(min, max) of every numeric literal input/param, in its canonical unit (validity boxes use canonical units)."""
    out = {}
    for src in (job.key.params, job.key.inputs):
        for k, v in src.items():
            if isinstance(v, dict) and "value" in v and "unit" in v:
                v = v["value"]
            if isinstance(v, bool) or isinstance(v, str) or v is None:
                continue
            try:
                a = np.asarray(v, dtype=float)
            except (TypeError, ValueError):
                continue
            if a.size:
                out[k] = (float(a.min()), float(a.max()))
    return out


def check(job: JobEnvelope, res: ResultEnvelope, manifest: Manifest, staging: Optional[Path] = None) -> GateReport:
    r = GateReport()
    # 1 schema / identity
    if res.job_id != job.id:
        r.errors.append(f"result job_id {res.job_id} does not match job {job.id}")

    by_name = {}
    for q in res.quantities:
        by_name.setdefault(q.name, []).append(q)

    # 2 declared outputs
    if res.status == "ok":
        for o in manifest.outputs:
            if o.name not in by_name:
                r.errors.append(f"declared output '{o.name}' missing")

    specs = {o.name: o for o in manifest.outputs}
    for q in res.quantities:
        spec = specs.get(q.name)
        # 3 units
        if spec is not None and not _same_dim(q.unit, spec.unit):
            r.errors.append(f"{q.name}: unit '{q.unit}' has the wrong dimension (expected that of '{spec.unit}')")
        # 4 finite / constraints
        try:
            a = q.array()
        except (TypeError, ValueError):
            r.errors.append(f"{q.name}: value is not numeric")
            continue
        if not np.all(np.isfinite(a)):
            r.errors.append(f"{q.name}: non-finite values")
        elif spec is not None:
            if spec.constraint == "positive" and np.any(a <= 0):
                r.errors.append(f"{q.name}: must be positive")
            if spec.constraint == "nonnegative" and np.any(a < 0):
                r.errors.append(f"{q.name}: must be non-negative")
        # 5 conventions
        for axis, val in q.conventions.items():
            if axis not in cv.CANONICAL:
                r.errors.append(f"{q.name}: unknown convention axis '{axis}'")
            elif axis == "dose":
                if not val.startswith("Gy("):
                    r.errors.append(f"{q.name}: dose must be Gy(<material>), got '{val}'")
            elif val != cv.CANONICAL[axis]:
                r.errors.append(f"{q.name}: {axis} is '{val}', canonical is '{cv.CANONICAL[axis]}' (convert at the adapter)")
        # 7 uncertainty
        if q.uncertainty.kind == "none":
            if job.fidelity == "paper":
                r.errors.append(f"{q.name}: paper fidelity requires an uncertainty")
            elif "unverified" not in q.flags:
                q.flags.append("unverified")

    # 6 validity box (job inputs against manifest validity)
    rng = _input_ranges(job)
    for name, (lo, hi) in manifest.validity.items():
        if name in rng and (rng[name][0] < lo or rng[name][1] > hi):
            r.flags.append(f"extrapolated:{name}=[{rng[name][0]:g},{rng[name][1]:g}] outside [{lo:g},{hi:g}]")
    for q in res.quantities:
        for name, (lo, hi) in q.validity.items():
            if name in rng and (rng[name][0] < lo or rng[name][1] > hi) and "extrapolated" not in q.flags:
                q.flags.append("extrapolated")

    # 8 fields
    for f in res.fields:
        try:
            Q_(1.0, f.unit)
        except Exception:  # noqa: BLE001
            r.errors.append(f"field {f.name}: bad unit '{f.unit}'")
        if staging is not None and f.retention != "scratch" and not (staging / "data" / f.path).is_file():
            r.errors.append(f"field {f.name}: file {f.path} missing")

    # 9 verification
    if job.fidelity == "paper" and res.status == "ok":
        for k in manifest.verification:
            if k not in res.verification:
                r.errors.append(f"paper fidelity: verification '{k}' not reported")

    if any(f.startswith("extrapolated") for f in r.flags) and "extrapolated" not in res.flags:
        r.flags.append("extrapolated")
    return r


def apply(job: JobEnvelope, res: ResultEnvelope, manifest: Manifest, staging: Optional[Path] = None) -> Tuple[ResultEnvelope, GateReport]:
    """Run the checks and fold the outcome into the envelope."""
    rep = check(job, res, manifest, staging)
    for f in rep.flags:
        if f not in res.flags:
            res.flags.append(f)
    if not rep.ok and res.status == "ok":
        res.status = "check_failed"
        res.message = "; ".join(rep.errors)
    res.diagnostics["gatekeeper"] = {"errors": rep.errors, "flags": rep.flags}
    return res, rep
