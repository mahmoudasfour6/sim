"""Verification records for result envelopes (interface specification: verification).

Two kinds, both reported in ResultEnvelope.verification and required by paper fidelity when the manifest lists
them:
  code verification      the discretised equations converge at the designed order (MMS; simws.tcad.mms)
  solution verification  the numerical error of this result, from a refinement study (simws.verify): GCI, observed
                         order, Richardson estimate; becomes the 'numerical' budget component

    from simws.hub import verification as V
    rec = V.solution(lambda lv: run(level=lv)["beta"], levels=3)         # in an adapter's run()
    ctx.verification["mesh"] = rec;  sigma_num = rec["sigma_num"]
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from simws.hub.uncertainty import numerical_from_refinement
from simws.verify import analyse, refinement_study


def from_refinement(rr, value: Optional[float] = None) -> Dict:
    d = {k: v for k, v in rr.as_dict().items()}
    d["kind"] = "solution"
    d["sigma_num"] = numerical_from_refinement(rr, value)
    d["asymptotic"] = rr.asymptotic_ratio is not None and 0.9 < rr.asymptotic_ratio < 1.1
    d["passed"] = d["sigma_num"] is not None and not (rr.oscillatory and rr.gci_fine is None)
    return d


def solution(run: Callable[[int], float], levels: int = 3, ratio: float = 2.0, start: int = 0) -> Dict:
    return from_refinement(refinement_study(run, levels, ratio, start))


def solution_from_values(levels, values, seconds=None, ratio=2.0) -> Dict:
    return from_refinement(analyse(levels, values, seconds or [0.0] * len(values), ratio))


def code(observed_order: float, expected_order: float, tol: float = 0.1, **detail) -> Dict:
    return {"kind": "code", "observed_order": observed_order, "expected_order": expected_order,
            "passed": bool(abs(observed_order - expected_order) <= tol), **detail}


def closed_form(numerical: float, analytic: float, rtol: float, **detail) -> Dict:
    """Engine result against an exact solution of the same problem (code verification without refinement)."""
    rel = numerical / analytic - 1.0
    return {"kind": "code", "numerical": numerical, "analytic": analytic, "rel_diff": rel, "rtol": rtol,
            "passed": bool(abs(rel) <= rtol), **detail}
