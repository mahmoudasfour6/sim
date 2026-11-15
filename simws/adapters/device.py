"""Engine-neutral device adapters (architecture v1, section 5): device.devsim and device.charon.

One base class runs any simws.spec Problem through the engine module named by the subclass (simws.engines). The
engine is loaded inside run(); resolve() and the manifest never import it. Quantities are named dc.I.<contact>
in canonical SI (A/m**2 for 1D, A/m for 2D), positive into the device.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from simws.hub import guards as G
from simws.hub.adapter import Adapter, AdapterError, InvalidInput
from simws.hub.schema import Manifest, OutputSpec

CUR = {"terminal_current": "into_device+"}
# (unit in the engine result, canonical unit, factor)
_UNITS = {"A/cm**2": ("A/m**2", 1e4), "A/cm": ("A/m", 1e2)}
_FLOOR = {"A/m**2": 1e-4, "A/m": 1e-8}      # current noise floor of the Newton tolerance, per canonical unit
KIRCHHOFF = "dc.kirchhoff_residual"         # max over the sweep of |sum of terminal currents| / (max |I| + floor)
_OUTPUTS = [OutputSpec(name=KIRCHHOFF, unit="dimensionless", constraint="nonnegative",
                       description="largest |sum of terminal currents| / (largest |terminal current| + noise floor); "
                                   "the currents themselves are dc.I.<contact>, one per contact of the device")]


def _problem(params: Dict[str, Any]):
    from simws.spec.schema import Problem
    p = params.get("problem")
    if p is None:
        raise InvalidInput("missing input 'problem'")
    try:
        return Problem.model_validate(p.model_dump(mode="json") if hasattr(p, "model_dump") else p)
    except Exception as e:  # noqa: BLE001  (pydantic ValidationError)
        raise InvalidInput(f"invalid problem description: {e}") from e


class DeviceAdapter(Adapter):
    ENGINE = ""

    def resolve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(params)
        out["problem"] = _problem(params).model_dump(mode="json")
        return out

    @staticmethod
    def _study(job) -> bool:
        """A4.1: a mesh study runs at paper fidelity or when the job asks for it."""
        return job.fidelity == "paper" or bool(job.key.params.get("mesh_study", False))

    def prepare(self, job, ctx) -> None:
        super().prepare(job, ctx)
        from simws import engines
        if self._study(job) and not engines.module(self.ENGINE).CAPABILITY.mesh_study:      # A4.2
            raise InvalidInput(f"unsupported by {self.ENGINE}: mesh_study: {self.ENGINE} cannot refine its mesh "
                               f"(so it cannot give paper-fidelity results)")
        reasons = engines.module(self.ENGINE).unsupported(_problem(job.key.params))
        if reasons:
            raise InvalidInput(f"unsupported by {self.ENGINE}: " + "; ".join(reasons))

    def run(self, job, ctx):
        from simws import engines
        from simws.engines.base import EngineNotAvailable, Unsupported
        problem = _problem(job.key.params)
        mod = engines.module(self.ENGINE)
        levels = None
        try:
            if self._study(job):                             # A4.3: three meshes, coarse to fine
                from simws.hub import mesh_study as MS
                levels = [mod.run(problem.with_changes({"device.mesh.scale": sc}), ctx.staging)
                          for sc in MS.scales(problem.device.mesh.scale)]
                res = levels[-1]
            else:
                res = mod.run(problem, ctx.staging)
        except EngineNotAvailable as e:
            raise AdapterError(f"engine '{self.ENGINE}' is not installed on this machine: {e}") from e
        except Unsupported as e:
            raise InvalidInput(f"unsupported by {self.ENGINE}: " + "; ".join(e.reasons)) from e
        return {"result": res, "problem": problem, "outside": mod.CAPABILITY.outside_tested_range(problem),
                "levels": levels}

    @staticmethod
    def _residual(res, factor: float, floor: float) -> float:
        I = np.array([np.atleast_1d(np.asarray(v, float)) * factor for v in res.currents.values()])
        return float(np.max(np.abs(I.sum(axis=0)) / (np.abs(I).max(axis=0) + floor)))

    def collect(self, job, ctx, raw):
        res, problem, levels = raw["result"], raw["problem"], raw.get("levels")
        unit, factor = _UNITS[res.current_unit]
        floor = _FLOOR[unit]
        scalar = res.analysis == "equilibrium"
        ctx.flags += [f for f in res.flags if f not in ctx.flags]
        ctx.flags += ["extrapolated:" + s for s in raw["outside"]]
        ctx.diagnostics.update({"problem_id": problem.problem_id(), "engine": res.engine,
                                "device": problem.device.name, "temperature_K": problem.models.temperature_K,
                                **res.diagnostics})
        axes = [] if scalar else [{"name": f"V_{res.sweep_contact}", "unit": "V", "values": list(res.sweep_V)}]
        sig, per = {}, {}
        if levels is not None:                               # A4.4: error bar per contact and sweep point
            from simws.hub import mesh_study as MS
            if any(list(L.sweep_V) != list(res.sweep_V) or set(L.currents) != set(res.currents) for L in levels):
                raise AdapterError("mesh study: the three meshes did not return the same sweep")
            for name in res.currents:
                sig[name], per[f"dc.I.{name}"] = MS.series(
                    [np.abs(np.asarray(L.currents[name], float)) * factor for L in levels], floor,
                    None if scalar else res.sweep_V)

        def _sigma(name):
            if name not in sig:
                return None
            return float(sig[name][0]) if scalar else sig[name].tolist()

        qs = []
        for name, vals in res.currents.items():
            v = np.asarray(vals, float) * factor
            qs.append(ctx.quantity(f"dc.I.{name}", float(v[0]) if scalar else v, unit, axes=axes, conventions=CUR,
                                   sigma=_sigma(name)))
        # the one output every problem has (amendment A1.8): how well the terminal currents sum to zero
        # A2.4: when the engine derived a terminal current from the others, the residual is zero by construction
        derived = any(f.endswith("_current_from_kirchhoff") for f in res.flags)
        rs = None
        if levels is not None:                               # uniform interval over the three meshes
            r3 = [self._residual(L, factor, floor) for L in levels]
            rs = float((max(r3) - min(r3)) / (2 * 3 ** 0.5))
        qs.append(ctx.quantity(KIRCHHOFF, self._residual(res, factor, floor), "dimensionless", sigma=rs,
                               flags=["not_independent"] if derived else []))
        if levels is not None:
            for q in qs:
                q.uncertainty.components["numerical"] = q.uncertainty.sigma
            rec = MS.record(problem.device.mesh.scale, per)
            ctx.verification["mesh"] = rec
            ctx.diagnostics["mesh_scale_reported"] = rec["mesh_scales"][-1]
            ctx.flags.append(MS.NUMERICAL_ONLY_FLAG)         # A4.5: mesh error only, nothing borrowed
            if not rec["passed"]:
                ctx.flags.append(MS.NOT_ASYMPTOTIC_FLAG)
        return qs

    def self_check(self, job, ctx, qs) -> List[str]:
        cur = [q for q in qs if q.name.startswith("dc.I.")]
        if not cur:
            return ["no currents"]
        unit = cur[0].unit
        floor = _FLOOR[unit]
        qs = [q for q in qs if q.name.startswith("dc.I.")]
        I = np.array([np.atleast_1d(q.array()) for q in qs])
        if not qs[0].axes:                                   # equilibrium: every current vanishes
            worst = float(np.max(np.abs(I)))
            return [] if worst < floor else [f"equilibrium current {worst:.3e} {unit} above the {floor:g} floor"]
        return G.kirchhoff(list(I), rtol=1e-4, atol=floor)


class DeviceDevsim(DeviceAdapter):
    ENGINE = "devsim"
    manifest = Manifest(
        name="device.devsim", version="1.1", env="tcad", engine="devsim", engine_version="env-locked",
        description="Any simws.spec Problem (equilibrium or DC sweep) on the DEVSIM translator",
        verification={"mesh": "paper fidelity: GCI per sweep point over 3 meshes (spacings x1, 1/sqrt2, 1/2)"},
        inputs=["problem"], outputs=_OUTPUTS, native_conventions=CUR, citations=["devsim"])


class DeviceCharon(DeviceAdapter):
    ENGINE = "charon"
    manifest = Manifest(
        name="device.charon", version="1.1", env="tcad", engine="charon", engine_version="env-locked",
        description="Any simws.spec Problem the Charon translator supports (1D pn DC sweeps)",
        # no citation key yet: "charon" is not in simws.hub.citations, and a key that is not registered makes
        # citations.bibliography() raise. Add the checked reference there first, then cite it here.
        inputs=["problem"], outputs=_OUTPUTS, native_conventions=CUR, citations=[])
