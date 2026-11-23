"""Anneal adapters (architecture v1, contract A16.4): anneal.devsim_anneal, anneal.fipy.

One base class runs any simws.spec AnnealProblem through the engine module named by the subclass. Canonical SI:
depth in m, concentration in 1/m**3, dose in 1/m**2, Dt in m**2. Deterministic: no uncertainty is attached; the
uncertainty of the diffusivity tables is not included (flag "budget:model_not_included").
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from simws.hub.adapter import Adapter, AdapterError, InvalidInput
from simws.hub.schema import Manifest, OutputSpec

UM = 1e-6
BUDGET_FLAGS = ["budget:none", "budget:model_not_included"]
NO_JUNCTION_FLAG = "anneal:no_junction"
DOSE_TOL = 1e-6                    # A16.7 R93: reflecting surface keeps the dose
JUNCTION = OutputSpec(name="anneal.junction_depth", unit="m", constraint="positive",
                      description="first depth where the dopant falls to a background of the other type (A16.4)")
_OUTPUTS = [
    OutputSpec(name="anneal.concentration", unit="1/m**3", constraint="nonnegative",
               description="dopant concentration after the anneal, on the axis 'depth' (x_k = k dx)"),
    OutputSpec(name="anneal.dose", unit="1/m**2", constraint="nonnegative",
               description="trapezoid integral of the concentration over the reported points"),
    OutputSpec(name="anneal.Dt", unit="m**2", constraint="positive",
               description="sum over the steps of the intrinsic D times the time (context)"),
]


def _problem(params: Dict[str, Any]):
    from simws.spec.anneal import AnnealProblem
    p = params.get("problem")
    if p is None:
        raise InvalidInput("missing input 'problem'")
    try:
        return AnnealProblem.model_validate(p.model_dump(mode="json") if hasattr(p, "model_dump") else p)
    except Exception as e:  # noqa: BLE001  (pydantic ValidationError)
        raise InvalidInput(f"invalid problem description: {e}") from e


def junction_depth_um(depth_um, conc_cm3, background_cm3: float) -> Optional[float]:
    """First depth where C falls to the background, linear between points; None when C starts at or below it or
    never falls to it (A16.4)."""
    x = np.asarray(depth_um, float)
    c = np.asarray(conc_cm3, float)
    if background_cm3 <= 0 or c.size < 2 or c[0] <= background_cm3:
        return None
    below = np.nonzero(c <= background_cm3)[0]
    if below.size == 0:
        return None
    i = int(below[0])
    return float(x[i - 1] + (x[i] - x[i - 1]) * (c[i - 1] - background_cm3) / (c[i - 1] - c[i]))


class AnnealAdapter(Adapter):
    ENGINE = ""

    def resolve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(params)
        out["problem"] = _problem(params).model_dump(mode="json")
        return out

    def prepare(self, job, ctx) -> None:
        super().prepare(job, ctx)
        from simws import engines
        reasons = engines.unsupported(self.ENGINE, _problem(job.key.params))
        if reasons:
            raise InvalidInput(f"unsupported by {self.ENGINE}: " + "; ".join(reasons))

    def run(self, job, ctx):
        from simws import engines
        from simws.engines.base import EngineNotAvailable, Unsupported
        problem = _problem(job.key.params)
        mod = engines.module(self.ENGINE)
        try:
            res = mod.run(problem, ctx.staging)
        except EngineNotAvailable as e:
            raise AdapterError(f"engine '{self.ENGINE}' is not installed on this machine: {e}") from e
        except Unsupported as e:
            raise InvalidInput(f"unsupported by {self.ENGINE}: " + "; ".join(e.reasons)) from e
        return {"result": res, "problem": problem, "outside": mod.CAPABILITY.outside_tested_range(problem)}

    def collect(self, job, ctx, raw):
        from simws.process import diffusivity as dif
        res, problem = raw["result"], raw["problem"]
        ctx.flags += [f for f in res.flags if f not in ctx.flags]
        ctx.flags += ["extrapolated:" + s for s in raw["outside"]]
        ctx.flags += [f for f in BUDGET_FLAGS if f not in ctx.flags]
        x = np.asarray(res.depth_um, float)
        c = np.asarray(res.concentration_cm3, float)
        dose_cm2 = float(np.trapezoid(c, x * 1e-4))
        ctx.diagnostics.update({"problem_id": problem.problem_id(), "engine": res.engine, "dopant": problem.dopant,
                                "model": problem.models.diffusivity, "surface": problem.surface,
                                "initial_dose_cm2": res.initial_dose_cm2, "final_dose_cm2": dose_cm2,
                                "non_finite": int(np.count_nonzero(~np.isfinite(c))), **res.diagnostics})
        axes = [{"name": "depth", "unit": "m", "values": (x * UM).tolist()}]
        qs = [ctx.quantity("anneal.concentration", (np.where(np.isfinite(c), c, 0.0) * 1e6).tolist(), "1/m**3",
                           axes=axes),
              ctx.quantity("anneal.dose", dose_cm2 * 1e4, "1/m**2"),
              ctx.quantity("anneal.Dt", dif.Dt_cm2(problem) * 1e-4, "m**2")]
        own = "acceptor" if problem.dopant == "B" else "donor"
        if problem.background_cm3 > 0 and problem.background_type != own:
            xj = junction_depth_um(x, c, problem.background_cm3)
            if xj is None:
                ctx.flags.append(NO_JUNCTION_FLAG)
            else:
                qs.append(ctx.quantity("anneal.junction_depth", xj * UM, "m"))
                ctx.diagnostics["junction_depth_um"] = xj
        return qs

    def self_check(self, job, ctx, qs) -> List[str]:
        out: List[str] = []
        d = ctx.diagnostics
        if d.get("non_finite"):
            out.append(f"concentration: {d['non_finite']} non-finite values")
        if d.get("surface") == "reflecting":
            q0, q1 = float(d.get("initial_dose_cm2", 0.0)), float(d.get("final_dose_cm2", 0.0))
            if q0 > 0 and abs(q1 / q0 - 1) > DOSE_TOL:
                out.append(f"dose: {q1:.6e} cm^-2 after, {q0:.6e} before (reflecting surface; limit {DOSE_TOL:g})")
        return out


_VERIFICATION = {"analytic": "Gaussian and erfc solutions of the intrinsic equation (A16.7 R90-R92)",
                 "engine_pair": "the other anneal engine on different code (R94)"}


class AnnealDevsim(AnnealAdapter):
    ENGINE = "devsim_anneal"
    manifest = Manifest(
        name="anneal.devsim_anneal", version="1.0", env="tcad", engine="devsim_anneal", engine_version="A16",
        description="One dopant diffusing in 1D silicon (Plummer-Deal-Griffin diffusivities, intrinsic or Fermi) "
                    "on DEVSIM",
        verification=_VERIFICATION, inputs=["problem"], outputs=_OUTPUTS,
        citations=["plummer2000", "morin1954", "devsim"])


class AnnealFipy(AnnealAdapter):
    ENGINE = "fipy"
    manifest = Manifest(
        name="anneal.fipy", version="1.0", env="tcad", engine="fipy", engine_version="A16",
        description="One dopant diffusing in 1D silicon (Plummer-Deal-Griffin diffusivities, intrinsic or Fermi) "
                    "on FiPy",
        verification=_VERIFICATION, inputs=["problem"], outputs=_OUTPUTS,
        citations=["plummer2000", "morin1954", "fipy"])
