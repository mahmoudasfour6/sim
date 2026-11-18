"""Circuit adapters (architecture v1, contract amendment A9.5): circuit.ngspice.

One base class runs any simws.spec CircuitProblem through the engine module named by the subclass and turns each
probe into a hub quantity: circuit.v.<node>, circuit.v.<a>-<b> (V), circuit.i.<source> (A, SPICE sign: the
current entering the source at its first node). AC probes give four quantities: .re, .im, .mag and .phase (rad).
A circuit solve is deterministic: no uncertainty is attached; its error is set by the solver tolerances.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from simws.hub.adapter import Adapter, AdapterError, InvalidInput
from simws.hub.schema import Manifest, OutputSpec

BUDGET_FLAG = "budget:none"
N_POINTS = "circuit.n_points"
_OUTPUTS = [OutputSpec(name=N_POINTS, unit="dimensionless", constraint="positive",
                       description="number of analysis points (1 for an operating point); the probes are "
                                   "circuit.v.<node>, circuit.v.<a>-<b> and circuit.i.<source>")]


def _problem(params: Dict[str, Any]):
    from simws.spec.circuit import CircuitProblem
    p = params.get("problem")
    if p is None:
        raise InvalidInput("missing input 'problem'")
    try:
        return CircuitProblem.model_validate(p.model_dump(mode="json") if hasattr(p, "model_dump") else p)
    except Exception as e:  # noqa: BLE001  (pydantic ValidationError)
        raise InvalidInput(f"invalid problem description: {e}") from e


def quantity_name(probe: str) -> str:
    from simws.spec.circuit import parse_probe
    pr = parse_probe(probe)
    if pr[0] == "v":
        return f"circuit.v.{pr[1]}" if pr[2] is None else f"circuit.v.{pr[1]}-{pr[2]}"
    return f"circuit.i.{pr[1]}"


class CircuitAdapter(Adapter):
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
        res, problem = raw["result"], raw["problem"]
        ctx.flags += [f for f in res.flags if f not in ctx.flags]
        ctx.flags += ["extrapolated:" + s for s in raw["outside"]]
        ctx.flags.append(BUDGET_FLAG)
        n = 1 if res.analysis == "op" else len(res.axis)
        ctx.diagnostics.update({"problem_id": problem.problem_id(), "engine": res.engine, "analysis": res.analysis,
                                "temperature_C": problem.temperature_C, "n_points": n,
                                "expected_points": problem.analysis.n_points() if res.analysis == "dc" else None,
                                **res.diagnostics})
        scalar = res.analysis == "op"
        axes = [] if scalar else [{"name": res.axis_name.lower(), "unit": res.axis_unit, "values": list(res.axis)}]
        qs = []
        bad = ctx.diagnostics.setdefault("non_finite", [])
        if not np.all(np.isfinite(np.asarray(res.axis, float))):
            bad.append("axis")
            axes = [] if scalar else [{"name": res.axis_name.lower(), "unit": res.axis_unit,
                                       "values": [float(x) if np.isfinite(x) else 0.0 for x in res.axis]}]
        for probe, vals in res.signals.items():
            name, unit = quantity_name(probe), res.units[probe]
            # A9.11: a non-finite value cannot be stored (the hub refuses NaN); the probe is left out and the
            # self-check fails the result
            parts = [np.asarray(vals, float)] + ([np.asarray(res.signals_imag.get(probe, []), float)]
                                                if res.analysis == "ac" else [])
            if not all(np.all(np.isfinite(x)) for x in parts):
                bad.append(probe)
                continue
            if res.analysis == "ac":
                z = np.asarray(vals, float) + 1j * np.asarray(res.signals_imag[probe], float)
                for suffix, part, u in ((".re", z.real, unit), (".im", z.imag, unit), (".mag", np.abs(z), unit),
                                        (".phase", np.angle(z), "rad")):
                    qs.append(ctx.quantity(name + suffix, part.tolist(), u, axes=axes))
            else:
                v = np.asarray(vals, float)
                qs.append(ctx.quantity(name, float(v[0]) if scalar else v.tolist(), unit, axes=axes))
        qs.append(ctx.quantity(N_POINTS, float(n), "dimensionless"))
        return qs

    def self_check(self, job, ctx, qs) -> List[str]:
        out: List[str] = []
        for name in ctx.diagnostics.get("non_finite", []):
            out.append(f"{name}: non-finite values (left out of the result)")
        exp = ctx.diagnostics.get("expected_points")
        if exp is not None and ctx.diagnostics.get("n_points") != exp:
            out.append(f"dc sweep: {ctx.diagnostics.get('n_points')} points, the description has {exp}")
        return out


class CircuitNgspice(CircuitAdapter):
    ENGINE = "ngspice"
    manifest = Manifest(
        name="circuit.ngspice", version="1.0", env="tcad", engine="ngspice", engine_version="47 + OpenVAF osdi_0.3",
        description="Any simws.spec CircuitProblem: op, dc, ac or tran on ngspice, Verilog-A models through OpenVAF",
        inputs=["problem"], outputs=_OUTPUTS, citations=["ngspice47", "openvaf"])
