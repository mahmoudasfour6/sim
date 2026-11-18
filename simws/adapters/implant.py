"""Implant adapters (architecture v1, contract amendments A7.5, A13.4, A14.4): implant.rustbca, implant.g4implant.

One base class runs any simws.spec ImplantProblem through the engine module named by the subclass. The engine
returns where each implanted ion stopped; everything derived from that (moments, profile, error bars) is computed
here, so every implant engine reports the same quantities in the same way. Canonical SI: lengths in m, the profile
in 1/m (stopping probability per ion and per metre of depth), the concentration in 1/m**3.

The error bars are counting statistics only (flag "budget:statistical_only"): they say how well the Monte Carlo
sample determines the number, not how good the collision physics is.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from simws.hub.adapter import Adapter, AdapterError, InvalidInput, OutsideValidity
from simws.hub.schema import Manifest, OutputSpec

UM = 1e-6
MIN_IMPLANTED = 100                 # fewer stopped ions than this: no range statistics
PAPER_MIN_IMPLANTED = 10000
Z_MAX = 4.0                         # split-halves check: |difference of the two half means| / its standard error
STATISTICAL_ONLY_FLAG = "budget:statistical_only"
# A14.4: present only with recoils, so not in the manifest (the gatekeeper requires every declared output)
DAMAGE_OUTPUTS = [
    OutputSpec(name="implant.vacancies_per_ion", unit="dimensionless", constraint="nonnegative",
               description="A14: target atoms displaced (given more than their Ed) per launched ion, full cascade"),
    OutputSpec(name="implant.vacancy_profile", unit="1/m", constraint="nonnegative",
               description="A14: vacancies per launched ion and per metre of depth, on the axis 'depth'"),
    OutputSpec(name="implant.vacancy_depth_mean", unit="m", constraint="positive",
               description="A14: mean depth at which the vacancies were made"),
]
DAMAGE_FLAG = "implant:damage_full_cascade: no recombination or annealing; an upper bound on surviving defects (A14.1)"

_OUTPUTS = [
    OutputSpec(name="implant.Rp", unit="m", constraint="positive", description="projected range: mean depth of the implanted ions"),
    OutputSpec(name="implant.dRp", unit="m", constraint="positive", description="range straggle: standard deviation of the depth"),
    OutputSpec(name="implant.lateral_straggle", unit="m", constraint="nonnegative",
               description="standard deviation of the displacement perpendicular to the plane of incidence"),
    OutputSpec(name="implant.fraction_implanted", unit="dimensionless", constraint="nonnegative",
               description="implanted ions / launched ions"),
    OutputSpec(name="implant.fraction_reflected", unit="dimensionless", constraint="nonnegative",
               description="ions that left through the front surface / launched ions"),
    OutputSpec(name="implant.fraction_transmitted", unit="dimensionless", constraint="nonnegative",
               description="ions that left through the back of the target / launched ions"),
    OutputSpec(name="implant.profile", unit="1/m", constraint="nonnegative",
               description="stopping probability per launched ion and per metre of depth, on the axis 'depth'"),
]


def _problem(params: Dict[str, Any]):
    from simws.spec.implant import ImplantProblem
    p = params.get("problem")
    if p is None:
        raise InvalidInput("missing input 'problem'")
    try:
        return ImplantProblem.model_validate(p.model_dump(mode="json") if hasattr(p, "model_dump") else p)
    except Exception as e:  # noqa: BLE001  (pydantic ValidationError)
        raise InvalidInput(f"invalid problem description: {e}") from e


def _std_and_sigma(v: np.ndarray):
    """Sample standard deviation s (k-1) and its standard uncertainty (s/2) sqrt((m4/s^4 - (k-3)/(k-1)) / k)."""
    k = v.size
    s = float(np.std(v, ddof=1))
    if s == 0.0:
        return 0.0, 0.0
    m4 = float(np.mean((v - v.mean()) ** 4))
    return s, 0.5 * s * math.sqrt(max(m4 / s ** 4 - (k - 3) / (k - 1), 0.0) / k)


def _fraction_sigma(count: int, n: int) -> float:
    p = (count + 1) / (n + 2)
    return math.sqrt(p * (1 - p) / n)


class ImplantAdapter(Adapter):
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
        n = int(res.n_ions)
        x = np.asarray(res.depth_um, float) * UM
        lat = np.asarray(res.lateral_um, float) * UM
        k = int(x.size)
        ctx.flags += [f for f in res.flags if f not in ctx.flags]
        ctx.flags += ["extrapolated:" + s for s in raw["outside"]]
        ctx.flags.append(STATISTICAL_ONLY_FLAG)
        ctx.diagnostics.update({"problem_id": problem.problem_id(), "engine": res.engine,
                                "ion": problem.beam.ion.symbol, "energy_keV": problem.beam.energy_keV,
                                "tilt_deg": problem.beam.tilt_deg, "n_ions": n, "n_implanted": k,
                                "n_reflected": int(res.n_reflected), "n_transmitted": int(res.n_transmitted),
                                **res.diagnostics})
        if res.n_transmitted > 0:                             # A7.9: set before any failure below
            ctx.flags.append(f"implant:transmitted: {int(res.n_transmitted)} of {n} ions left through the back of "
                             f"the target; the profile is cut off there")
        if k < MIN_IMPLANTED:
            raise OutsideValidity(f"only {k} of {n} ions stopped in the target (at least {MIN_IMPLANTED} are "
                                  f"needed for range statistics): thicker target or more ions")

        def q(name, value, unit, sigma, axes=None):
            qq = ctx.quantity(name, value, unit, sigma=sigma, axes=axes)
            qq.uncertainty.components["statistical"] = qq.uncertainty.sigma
            return qq

        rp = float(x.mean())
        s, s_sig = _std_and_sigma(x)
        ls, ls_sig = _std_and_sigma(lat)
        qs = [q("implant.Rp", rp, "m", s / math.sqrt(k)),
              q("implant.dRp", s, "m", s_sig),
              q("implant.lateral_straggle", ls, "m", ls_sig)]
        for name, count in (("implanted", k), ("reflected", int(res.n_reflected)),
                            ("transmitted", int(res.n_transmitted))):
            qs.append(q(f"implant.fraction_{name}", count / n, "dimensionless", _fraction_sigma(count, n)))

        b = problem.depth_bin_um * UM
        nb = int(math.ceil(problem.target.thickness_um / problem.depth_bin_um - 1e-9))
        idx = np.minimum(np.maximum(np.floor(x / b).astype(int), 0), nb - 1)
        counts = np.bincount(idx, minlength=nb).astype(float)
        axes = [{"name": "depth", "unit": "m", "values": ((np.arange(nb) + 0.5) * b).tolist()}]
        prof = counts / (n * b)
        prof_sig = np.sqrt(np.maximum(counts, 1.0)) / (n * b)
        qs.append(q("implant.profile", prof.tolist(), "1/m", prof_sig.tolist(), axes=axes))
        if problem.beam.dose_cm2 is not None:
            dose = problem.beam.dose_cm2 * 1e4                # ions per m^2
            qs.append(q("implant.concentration", (dose * prof).tolist(), "1/m**3", (dose * prof_sig).tolist(),
                        axes=axes))

        if problem.models.recoils:
            qs += self._damage(ctx, res, problem, q, axes, n, b, nb)

        # split halves: the first k//2 implanted ions against the rest (are the ions independent samples?)
        h = k // 2
        a, c = x[:h], x[h:]
        se = math.sqrt(float(np.var(a, ddof=1)) / a.size + float(np.var(c, ddof=1)) / c.size)
        z = float((a.mean() - c.mean()) / se) if se > 0 else 0.0
        ctx.verification["statistics"] = {"kind": "solution", "method": "split halves", "n_implanted": k,
                                          "rp_halves_um": [float(a.mean() / UM), float(c.mean() / UM)],
                                          "z": z, "passed": bool(abs(z) <= Z_MAX)}
        return qs

    @staticmethod
    def _damage(ctx, res, problem, q, axes, n, b, nb):
        """A14.4: vacancy quantities, uncertainties from the means of the batches."""
        h = np.asarray(res.vacancy_batches, float)
        ions = np.asarray(res.vacancy_batch_ions, float)
        if h.ndim != 2 or h.shape[1] != nb or h.shape[0] != ions.size or int(ions.sum()) != n or h.shape[0] < 2:
            raise AdapterError(f"damage histograms: shape {h.shape}, ions {ions.tolist()}; expected (batches, {nb}) "
                               f"for {n} ions")
        nbat = h.shape[0]
        x = (np.arange(nb) + 0.5) * b
        total = h.sum(axis=1)                                    # vacancies per batch
        per_ion_b = total / ions
        prof_b = h / (ions[:, None] * b)
        mean_b = np.array([float((hb * x).sum() / hb.sum()) if hb.sum() > 0 else np.nan for hb in h])
        se = lambda v: float(np.std(v, ddof=1) / math.sqrt(v.size))     # noqa: E731
        vtot = float(h.sum())
        if not vtot > 0:
            raise OutsideValidity("no vacancy was made: the ion never gave a target atom more than its Ed")
        ctx.flags.append(DAMAGE_FLAG)
        ctx.diagnostics.update({"n_vacancies": int(vtot), "damage_batches": nbat})
        prof = h.sum(axis=0) / (n * b)
        prof_sig = np.std(prof_b, axis=0, ddof=1) / math.sqrt(nbat)
        out = [q("implant.vacancies_per_ion", vtot / n, "dimensionless", se(per_ion_b)),
               q("implant.vacancy_profile", prof.tolist(), "1/m", prof_sig.tolist(), axes=axes),
               q("implant.vacancy_depth_mean", float((h.sum(axis=0) * x).sum() / vtot), "m",
                 se(mean_b[np.isfinite(mean_b)]))]
        if problem.beam.dose_cm2 is not None:
            dose = problem.beam.dose_cm2 * 1e4
            out.append(q("implant.vacancy_concentration", (dose * prof).tolist(), "1/m**3",
                         (dose * prof_sig).tolist(), axes=axes))
        return out

    def self_check(self, job, ctx, qs) -> List[str]:
        d = ctx.diagnostics
        out: List[str] = []
        if d["n_implanted"] + d["n_reflected"] + d["n_transmitted"] != d["n_ions"]:
            out.append(f"ion count: {d['n_implanted']} implanted + {d['n_reflected']} reflected + "
                       f"{d['n_transmitted']} transmitted is not the {d['n_ions']} launched")
        st = ctx.verification.get("statistics", {})
        if not st.get("passed", False):
            out.append(f"split halves: the two half-sample ranges differ by {st.get('z', float('nan')):.2f} standard "
                       f"errors (limit {Z_MAX:g}): the ions do not look like independent samples")
        if job.fidelity == "paper" and d["n_implanted"] < PAPER_MIN_IMPLANTED:
            out.append(f"paper fidelity needs at least {PAPER_MIN_IMPLANTED} implanted ions, got {d['n_implanted']}")
        return out


class ImplantRustbca(ImplantAdapter):
    ENGINE = "rustbca"
    manifest = Manifest(
        name="implant.rustbca", version="1.0", env="tcad", engine="rustbca", engine_version="3.0.0",
        description="Any simws.spec ImplantProblem the RustBCA translator supports: ion range profiles in layered "
                    "amorphous targets",
        verification={"statistics": "split halves: the range of the first and second half of the ions agree "
                                    "within 4 standard errors"},
        inputs=["problem"], outputs=_OUTPUTS, citations=["drobny2021"])


class ImplantG4implant(ImplantAdapter):
    ENGINE = "g4implant"
    manifest = Manifest(
        name="implant.g4implant", version="1.0", env="tcad", engine="g4implant", engine_version="11.2.2",
        description="Any simws.spec ImplantProblem the Geant4 translator supports: ion range profiles and "
                    "displacement counts in layered amorphous targets (G4ScreenedNuclearRecoil)",
        verification={"statistics": "split halves: the range of the first and second half of the ions agree "
                                    "within 4 standard errors"},
        inputs=["problem"], outputs=_OUTPUTS, citations=["geant4_2016", "mendenhall2005"])
