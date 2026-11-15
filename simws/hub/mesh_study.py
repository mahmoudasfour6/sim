"""Mesh error bars for sweep results (contract amendment A4).

The rule is the one the legacy HBT adapter has used since 0.9a (simws.adapters.tcad.HbtGummel._mesh_uncertainty):
three meshes with every spacing multiplied by 1, 1/sqrt(2), 1/2; per point a grid convergence index when the three
values behave (observed order between P_MIN and P_MAX, near the asymptotic range), otherwise a conservative bound
that assumes an order no better than P_MIN. New here: a point whose finest value is below the solver's noise floor
is not analysed at all; its error bar is the floor.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

RATIO = 2 ** 0.5
LEVELS = 3
P_MIN, P_MAX = 0.5, 4.0          # plausible observed orders (second-order scheme; corners lower it)
FS = 1.25                        # Roache's safety factor, as in simws.verify
NOT_ASYMPTOTIC_FLAG = "mesh_not_asymptotic: conservative mesh bound used for some points"
NUMERICAL_ONLY_FLAG = "budget:numerical_only"


def scales(s0: float) -> List[float]:
    """Mesh scales coarse to fine: s0, s0/r, s0/r^2."""
    return [float(s0) / RATIO ** k for k in range(LEVELS)]


def point(values_coarse_to_fine: Sequence[float], floor: float) -> Dict:
    """Error bar of one point from its magnitudes on the three meshes. Returns the per-point record with `sigma`
    (absolute, same unit as the values)."""
    from simws.hub import verification as Vf
    from simws.hub.uncertainty import GCI_COVERAGE_K
    vals = [float(abs(v)) for v in values_coarse_to_fine]
    f2, f1 = vals[-2], vals[-1]
    out = {"values_coarse_to_fine": vals, "order": None, "gci_fine": None, "asymptotic_ratio": None,
           "conservative_bound": False, "below_floor": False}
    if f1 < floor:                                   # solver noise: nothing to analyse
        out.update(below_floor=True, sigma=float(floor))
    else:
        r = Vf.solution_from_values(list(range(len(vals))), vals, ratio=RATIO)
        p, ar, sn = r.get("order"), r.get("asymptotic_ratio"), r.get("sigma_num")
        good = (sn is not None and p is not None and P_MIN <= p <= P_MAX and ar is not None and 0.5 < ar < 2.0) \
            or (sn == 0.0)
        sigma = float(sn) if good else FS * abs(f1 - f2) / (RATIO ** P_MIN - 1) / GCI_COVERAGE_K
        out.update(order=p, gci_fine=r.get("gci_fine"), asymptotic_ratio=ar, conservative_bound=not good,
                   sigma=float(sigma))
    out["rel_sigma"] = out["sigma"] / f1 if f1 else None
    return out


def series(levels: Sequence[np.ndarray], floor: float,
           axis_values: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, List[Dict]]:
    """levels: the same quantity on the three meshes (coarse to fine), each of shape (n points,). Returns
    (sigma array of shape (n,), per-point records with `axis_value`)."""
    L = [np.atleast_1d(np.asarray(a, float)) for a in levels]
    if len(L) != LEVELS or any(a.shape != L[0].shape for a in L):
        raise ValueError(f"mesh study needs {LEVELS} levels of the same shape")
    n = L[0].shape[0]
    ax = list(axis_values) if axis_values is not None else [None] * n
    per = []
    for j in range(n):
        rec = point([a[j] for a in L], floor)
        per.append({"axis_value": None if ax[j] is None else float(ax[j]), **rec})
    return np.array([r.pop("sigma") for r in per]), per


def record(s0: float, quantities: Dict[str, List[Dict]]) -> Dict:
    """The verification record for ResultEnvelope.verification['mesh']."""
    passed = not any(pt["conservative_bound"] for per in quantities.values() for pt in per)
    return {"kind": "solution", "ratio": RATIO, "levels": LEVELS, "mesh_scales": scales(s0),
            "quantities": quantities, "passed": bool(passed)}
