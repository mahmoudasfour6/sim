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
_Y_UNITS = {"S/cm**2": ("S/m**2", 1e4), "S/cm": ("S/m", 1e2)}
_ROLLOFF_MIN = 1e-6                         # smallest relative fall of |h21| that counts as a roll-off (A6.8)
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
        if self._study(job) and _problem(job.key.params).analysis.kind in ("small_signal", "strike"):   # A6.6, A17.3
            raise InvalidInput(f"unsupported by {self.ENGINE}: mesh_study: not available for "
                               f"{_problem(job.key.params).analysis.kind} yet (screening fidelity)")
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
        if res.analysis == "strike":
            return self._strike(ctx, raw)
        unit, factor = _UNITS[res.current_unit]
        floor = _FLOOR[unit]
        scalar = res.analysis in ("equilibrium", "small_signal")       # one operating point: scalars
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
        if res.analysis == "small_signal":
            qs += self._ac_quantities(ctx, res)
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

    def _strike(self, ctx, raw) -> list:
        """A17.4: tr.I.<contact> over time, tr.Q.<contact> (backward sum of I - I(0)), tr.Q_generated (exact), and the
        Kirchhoff residual as the largest |sum of currents| over the largest |current|."""
        from simws.chain.see import Q_E
        res, problem = raw["result"], raw["problem"]
        unit, factor = _UNITS[res.current_unit]
        ctx.flags += [f for f in res.flags if f not in ctx.flags]
        ctx.flags += ["extrapolated:" + s for s in raw["outside"]]
        ctx.diagnostics.update({"problem_id": problem.problem_id(), "engine": res.engine, "analysis": "strike",
                                "device": problem.device.name, "temperature_K": problem.models.temperature_K,
                                **res.diagnostics})
        t = np.asarray(res.time_s, float)
        axes = [{"name": "time", "unit": "s", "values": t.tolist()}]
        qunit = {"A/m**2": "C/m**2", "A/m": "C/m"}[unit]
        qs, I = [], []
        for name, vals in res.currents.items():
            v = np.asarray(vals, float) * factor
            I.append(v)
            qs.append(ctx.quantity(f"tr.I.{name}", v.tolist(), unit, axes=axes, conventions=CUR))
            # A17.4 (corrected): the backward-Euler sum, sum_k (I_k - I_0)(t_k - t_{k-1}), the quadrature the
            # time stepping conserves charge with
            qs.append(ctx.quantity(f"tr.Q.{name}", float(np.sum((v[1:] - v[0]) * np.diff(t))), qunit,
                                   conventions=CUR))
        an = problem.analysis
        qs.append(ctx.quantity("tr.Q_generated", Q_E * an.pairs_cm2 * 1e4, qunit))
        I = np.array(I)
        worst = float(np.max(np.abs(I.sum(axis=0))) / (np.max(np.abs(I)) + _FLOOR[unit]))
        qs.append(ctx.quantity(KIRCHHOFF, worst, "dimensionless"))
        return qs

    @staticmethod
    def _ac_quantities(ctx, res) -> list:
        """A6.5: Y parameters over frequency in canonical SI, |h21|, and the scalars f_T and f_max when defined."""
        from simws.analysis import two_port as TP
        yunit, yfac = _Y_UNITS[res.y_unit]
        f = [float(x) for x in res.frequencies_Hz]
        ax = [{"name": "f", "unit": "Hz", "values": f}]
        qs = []
        for k in ("11", "12", "21", "22"):
            qs.append(ctx.quantity(f"ac.y{k}_re", (np.asarray(res.y_real[k], float) * yfac).tolist(), yunit, axes=ax))
            qs.append(ctx.quantity(f"ac.y{k}_im", (np.asarray(res.y_imag[k], float) * yfac).tolist(), yunit, axes=ax))
        Y = [{k: complex(res.y_real[k][n], res.y_imag[k][n]) for k in ("11", "12", "21", "22")} for n in range(len(f))]
        h = [abs(TP.h21(y["11"], y["12"], y["21"], y["22"])) for y in Y]
        qs.append(ctx.quantity("ac.h21_abs", h, "dimensionless", axes=ax))
        # A6.8: a fall of |h21| below 1e-6 relative is solver noise, not a roll-off (a passive one-port gives 1.0)
        ft = TP.f_t_single_pole(f[-1], h[-1], h[0]) if len(f) >= 2 and h[0] > h[-1] * (1 + _ROLLOFF_MIN) else None
        if ft is not None:
            qs.append(ctx.quantity("ac.fT", float(ft), "Hz"))
        elif len(f) >= 2:
            ctx.flags.append("ac:no_rolloff")
        y = Y[-1]
        fm = TP.f_max(f[-1], TP.mason_u(y["11"], y["12"], y["21"], y["22"]))
        if fm is not None:
            qs.append(ctx.quantity("ac.fmax", float(fm), "Hz"))
        else:
            ctx.flags.append("ac:U_unbounded")
        return qs

    def self_check(self, job, ctx, qs) -> List[str]:
        if ctx.diagnostics.get("analysis") == "strike":        # A17.4: Kirchhoff is acceptance item R104, not a guard
            bad = [q.name for q in qs if not np.all(np.isfinite(np.atleast_1d(np.asarray(q.array(), float))))]
            return [f"{n}: non-finite values" for n in bad]
        cur = [q for q in qs if q.name.startswith("dc.I.")]
        if not cur:
            return ["no currents"]
        unit = cur[0].unit
        floor = _FLOOR[unit]
        biased = any(q.name.startswith("ac.") for q in qs)   # small-signal: scalars at a biased operating point
        qs = [q for q in qs if q.name.startswith("dc.I.")]
        I = np.array([np.atleast_1d(q.array()) for q in qs])
        if not qs[0].axes and not biased:                    # equilibrium: every current vanishes
            worst = float(np.max(np.abs(I)))
            return [] if worst < floor else [f"equilibrium current {worst:.3e} {unit} above the {floor:g} floor"]
        return G.kirchhoff(list(I), rtol=1e-4, atol=floor)


class DeviceDevsim(DeviceAdapter):
    ENGINE = "devsim"
    manifest = Manifest(
        name="device.devsim", version="1.2", env="tcad", engine="devsim", engine_version="env-locked",
        description="Any simws.spec Problem (equilibrium, DC sweep, small-signal or strike) on the DEVSIM translator",
        verification={"mesh": "paper fidelity: GCI per sweep point over 3 meshes (spacings x1, 1/sqrt2, 1/2)"},
        inputs=["problem"], outputs=_OUTPUTS, native_conventions=CUR, citations=["devsim"])


class DeviceCharon(DeviceAdapter):
    ENGINE = "charon"
    manifest = Manifest(
        name="device.charon", version="1.2", env="tcad", engine="charon", engine_version="env-locked",
        description="Any simws.spec Problem the Charon translator supports (1D pn DC sweeps)",
        # no citation key yet: "charon" is not in simws.hub.citations, and a key that is not registered makes
        # citations.bibliography() raise. Add the checked reference there first, then cite it here.
        inputs=["problem"], outputs=_OUTPUTS, native_conventions=CUR, citations=[])


class DeviceDevsimEb(DeviceAdapter):
    """A11: the same Problem with an electron energy balance (different physics, same code as device.devsim)."""
    ENGINE = "devsim_eb"
    manifest = Manifest(
        name="device.devsim_eb", version="1.0", env="tcad", engine="devsim_eb", engine_version="env-locked",
        description="1D silicon Problems with an electron energy balance on DEVSIM (Baccarani-Wordeman 1985): "
                    "a model-form check of drift-diffusion",
        verification={"mesh": "paper fidelity: GCI per sweep point over 3 meshes (spacings x1, 1/sqrt2, 1/2)"},
        inputs=["problem"], outputs=_OUTPUTS, native_conventions=CUR, citations=["devsim"])
