"""Single-event adapters (architecture v1, contract amendment A8.5): see.g4see.

One base class runs any simws.spec SeeProblem through the engine module named by the subclass. The engine returns
the histogram of the energy each event left in the sensitive volume; everything derived from it (spectrum,
probability above a threshold, mean, spread, error bars) is computed here, so every single-event engine reports
the same quantities in the same way. Energies are in eV (the hub's canonical energy unit).

Error bars: counting statistics, plus for the mean a binning term (the mean is computed from bin centres).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from simws.hub.adapter import Adapter, AdapterError, InvalidInput
from simws.hub.schema import Manifest, OutputSpec

MEV = 1e6                               # eV per MeV
PAPER_MIN_EVENTS = 10000
BUDGET_FLAG = "budget:statistical_and_binning"

_OUTPUTS = [
    OutputSpec(name="see.edep_spectrum", unit="1/eV", constraint="nonnegative",
               description="probability per event and per eV of depositing that energy in the SV, axis 'edep'"),
    OutputSpec(name="see.p_above", unit="dimensionless", constraint="nonnegative",
               description="probability per event of depositing at least the threshold in the SV, axis 'threshold'"),
    OutputSpec(name="see.fraction_underflow", unit="dimensionless", constraint="nonnegative",
               description="events below the histogram range / events"),
    OutputSpec(name="see.fraction_overflow", unit="dimensionless", constraint="nonnegative",
               description="events at or above the histogram's upper limit / events"),
]


def _problem(params: Dict[str, Any]):
    from simws.spec.see import SeeProblem
    p = params.get("problem")
    if p is None:
        raise InvalidInput("missing input 'problem'")
    try:
        return SeeProblem.model_validate(p.model_dump(mode="json") if hasattr(p, "model_dump") else p)
    except Exception as e:  # noqa: BLE001  (pydantic ValidationError)
        raise InvalidInput(f"invalid problem description: {e}") from e


def _p_sigma(k: np.ndarray, n: int) -> np.ndarray:
    # clipped: a count above n is an inconsistent result (the self-check fails it), not a reason for a NaN
    p = np.clip((np.asarray(k, float) + 1) / (n + 2), 0.0, 1.0)
    return np.sqrt(p * (1 - p) / n)


class SeeAdapter(Adapter):
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
        N = int(res.n_events)
        e = np.asarray(res.edges_MeV, float) * MEV
        c = np.asarray(res.counts, float)
        under, over = float(res.underflow), float(res.overflow)
        w = np.diff(e)
        mid = np.sqrt(e[:-1] * e[1:]) if problem.histogram.scale == "log" else 0.5 * (e[:-1] + e[1:])
        ctx.flags += [f for f in res.flags if f not in ctx.flags]
        ctx.flags += ["extrapolated:" + s for s in raw["outside"]]
        ctx.flags.append(BUDGET_FLAG)
        ctx.diagnostics.update({"problem_id": problem.problem_id(), "engine": res.engine,
                                "particle": problem.source.particle, "energy_MeV": problem.source.energy_MeV,
                                "n_events": N, "counted": float(c.sum() + under + over), **res.diagnostics})

        def q(name, value, unit, sigma, axes=None, components=None):
            qq = ctx.quantity(name, value, unit, sigma=sigma, axes=axes)
            for k, v in (components or {"statistical": qq.uncertainty.sigma}).items():
                qq.uncertainty.components[k] = v
            return qq

        qs = [q("see.edep_spectrum", (c / (N * w)).tolist(), "1/eV", (np.sqrt(np.maximum(c, 1.0)) / (N * w)).tolist(),
                axes=[{"name": "edep", "unit": "eV", "values": mid.tolist()}])]
        k = np.cumsum(c[::-1])[::-1] + over
        qs.append(q("see.p_above", (k / N).tolist(), "dimensionless", _p_sigma(k, N).tolist(),
                    axes=[{"name": "threshold", "unit": "eV", "values": e[:-1].tolist()}]))
        qs.append(q("see.fraction_underflow", under / N, "dimensionless", float(_p_sigma(under, N))))
        qs.append(q("see.fraction_overflow", over / N, "dimensionless", float(_p_sigma(over, N))))
        if under == 0 and over == 0:
            mean = float((c * mid).sum() / N)
            s = math.sqrt(float((c * (mid - mean) ** 2).sum()) / (N - 1))
            occupied = w[c > 0]
            b = float(occupied.max()) / math.sqrt(12) if occupied.size else 0.0
            stat = s / math.sqrt(N)
            qs.append(q("see.edep_mean", mean, "eV", math.sqrt(stat ** 2 + b ** 2),
                        components={"statistical": stat, "binning": b}))
            if s > 0:
                m4 = float((c * (mid - mean) ** 4).sum()) / N
                s_sig = 0.5 * s * math.sqrt(max(m4 / s ** 4 - (N - 3) / (N - 1), 0.0) / N)
            else:
                s_sig = 0.0
            qs.append(q("see.edep_std", s, "eV", s_sig))
        else:
            ctx.flags.append(f"see:mean_undefined: {int(under)} events below and {int(over)} events above the "
                             f"histogram range; widen the histogram for a mean")
        return qs

    def self_check(self, job, ctx, qs) -> List[str]:
        d = ctx.diagnostics
        out: List[str] = []
        if abs(d["counted"] - d["n_events"]) > 1e-6 * d["n_events"]:
            out.append(f"event count: the histogram holds {d['counted']:g} events, {d['n_events']} were run")
        if job.fidelity == "paper" and d["n_events"] < PAPER_MIN_EVENTS:
            out.append(f"paper fidelity needs at least {PAPER_MIN_EVENTS} events, got {d['n_events']}")
        return out


class SeeG4see(SeeAdapter):
    ENGINE = "g4see"
    manifest = Manifest(
        name="see.g4see", version="1.0", env="tcad", engine="g4see", engine_version="v1.0 / Geant4 11.2.2",
        description="Any simws.spec SeeProblem the G4SEE translator supports: energy deposited per event in one "
                    "sensitive volume",
        inputs=["problem"], outputs=_OUTPUTS, citations=["lucsanyi2022"])
