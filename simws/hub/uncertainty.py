"""Uncertainty: correlated ensembles and error budgets (interface specification: uncertainty model).

Member IDs name one realisation of every uncertain choice, e.g. "functional=PBE|sample=3". A quantity computed in
several members keeps the member ID, so quantities from the same member stay correlated when combined:

    from simws.hub import uncertainty as U
    mid = U.member_id(functional="PBE", sample=3)              # 'functional=PBE|sample=3'
    ens = U.Ensemble.from_quantities(quantities)               # one name, many members
    beta = U.propagate(lambda IC, IB: IC / IB, IC=ens_ic, IB=ens_ib)   # member by member (correlation kept)

Budget composition (independent components added in quadrature, every component a standard uncertainty):
    numerical    from a refinement study: GCI / k with k = 2 (GCI is a ~95 % band; Roache 1998, Eca & Hoekstra 2014)
    parametric   spread of a parameter-sampling ensemble (sample standard deviation)
    model_form   spread over discrete model alternatives (functionals, mobility models): range / sqrt(12), i.e. a
                 uniform distribution over the range of the alternatives (no alternative is preferred)
    calibration  posterior standard deviation of calibrated parameters propagated to the output (0.9+)
    u = U.budget(value, numerical=..., parametric=..., model_form=...)  -> Uncertainty(kind="sigma", components=...)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

from simws.hub.schema import Quantity, Uncertainty

GCI_COVERAGE_K = 2.0


# ------------------------------------------------------------------------------------------------ member IDs
def member_id(**choices) -> str:
    if not choices:
        return "base"
    for k, v in choices.items():
        if any(c in f"{k}{v}" for c in "|="):
            raise ValueError(f"member choice {k}={v} must not contain '|' or '='")
    return "|".join(f"{k}={choices[k]}" for k in sorted(choices))


def parse_member(mid: str) -> Dict[str, str]:
    if mid == "base":
        return {}
    return dict(part.split("=", 1) for part in mid.split("|"))


def sub_member(mid: str, *axes) -> str:
    """Restrict a member ID to some axes (e.g. the functional part of 'functional=PBE|sample=3')."""
    d = parse_member(mid)
    return member_id(**{k: d[k] for k in axes if k in d})


# ------------------------------------------------------------------------------------------------ ensembles
@dataclass
class Ensemble:
    name: str
    unit: str
    members: Dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_quantities(cls, qs: Iterable[Quantity], unit: Optional[str] = None) -> "Ensemble":
        qs = list(qs)
        if not qs:
            raise ValueError("empty ensemble")
        names = {q.name for q in qs}
        if len(names) != 1:
            raise ValueError(f"ensemble mixes quantities {names}")
        unit = unit or qs[0].unit
        out = cls(qs[0].name, unit)
        for q in qs:
            if q.member in out.members:
                raise ValueError(f"member {q.member} appears twice")
            out.members[q.member] = q.to(unit)
        return out

    def ids(self) -> List[str]:
        return sorted(self.members)

    def stack(self, ids=None) -> np.ndarray:
        return np.array([self.members[i] for i in (ids or self.ids())], dtype=float)

    def mean(self):
        return self.stack().mean(axis=0)

    def std(self):
        a = self.stack()
        return a.std(axis=0, ddof=1) if len(a) > 1 else np.zeros_like(a[0])

    def range_sigma(self):
        """Model-form standard uncertainty: range / sqrt(12) over discrete alternatives."""
        a = self.stack()
        return (a.max(axis=0) - a.min(axis=0)) / math.sqrt(12.0)

    def percentile(self, q):
        return np.percentile(self.stack(), q, axis=0)

    def group(self, axis: str) -> Dict[str, "Ensemble"]:
        """Split by one member axis, e.g. group('functional') -> {'PBE': Ensemble, 'LDA': Ensemble}."""
        out: Dict[str, Ensemble] = {}
        for mid, v in self.members.items():
            key = parse_member(mid).get(axis, "")
            out.setdefault(key, Ensemble(self.name, self.unit)).members[mid] = v
        return out

    def to_quantity(self, value=None, **kw) -> Quantity:
        v = self.mean() if value is None else value
        return Quantity(name=self.name, value=np.asarray(v).tolist(), unit=self.unit,
                        uncertainty=Uncertainty(kind="ensemble",
                                                members={k: np.asarray(x).tolist() for k, x in self.members.items()},
                                                components={"ensemble_std": np.asarray(self.std()).tolist()}),
                        **kw)


def propagate(f: Callable, name: str = "derived", unit: str = "dimensionless", **ens: Ensemble) -> Ensemble:
    """Evaluate f member by member. All ensembles must share the same member IDs: that is what keeps correlated
    inputs correlated (IC and IB from the same mesh/functional/sample are divided together)."""
    ids = None
    for e in ens.values():
        s = set(e.members)
        if ids is not None and s != ids:
            raise ValueError(f"member sets differ: {sorted(ids ^ s)[:5]} ...; propagate needs matched members")
        ids = s
    out = Ensemble(name, unit)
    for mid in sorted(ids or ()):
        out.members[mid] = np.asarray(f(**{k: e.members[mid] for k, e in ens.items()}), dtype=float)
    return out


def correlation(a: Ensemble, b: Ensemble) -> float:
    ids = sorted(set(a.members) & set(b.members))
    if len(ids) < 3:
        raise ValueError("need at least 3 shared members for a correlation")
    x = np.array([np.mean(a.members[i]) for i in ids]); y = np.array([np.mean(b.members[i]) for i in ids])
    return float(np.corrcoef(x, y)[0, 1])


# ------------------------------------------------------------------------------------------------ budgets
def numerical_from_refinement(rr, value=None) -> Optional[float]:
    """Standard numerical uncertainty (absolute) from a simws.verify.RefinementResult: GCI_fine |f1| / k."""
    if rr.gci_fine is None:
        return None
    f1 = rr.values[-1] if value is None else value
    return float(rr.gci_fine * abs(f1) / GCI_COVERAGE_K)


def budget(value, numerical=None, parametric=None, model_form=None, calibration=None, **other) -> Uncertainty:
    """Combine independent standard-uncertainty components in quadrature. Components may be scalars or arrays of
    the value's shape; None components are omitted (and listed as 'missing' when the caller needs to know)."""
    comps = {k: v for k, v in dict(numerical=numerical, parametric=parametric, model_form=model_form,
                                   calibration=calibration, **other).items() if v is not None}
    if not comps:
        return Uncertainty()
    shape = np.shape(value)
    tot = np.zeros(shape)
    for v in comps.values():
        a = np.broadcast_to(np.asarray(v, float), shape)
        if np.any(a < 0) or not np.all(np.isfinite(a)):
            raise ValueError("uncertainty components must be finite and non-negative")
        tot = tot + a ** 2
    tot = np.sqrt(tot)
    return Uncertainty(kind="sigma", sigma=tot.tolist() if shape else float(tot),
                       components={k: (np.asarray(v, float).tolist() if np.ndim(v) else float(v))
                                   for k, v in comps.items()})


def dominant(u: Uncertainty) -> Optional[str]:
    """Name of the largest budget component (first place to spend more compute or measurement)."""
    if not u.components:
        return None
    return max(u.components, key=lambda k: float(np.max(np.abs(np.asarray(u.components[k], float)))))


def missing_components(u: Uncertainty, required=("numerical", "parametric", "model_form")) -> List[str]:
    return [c for c in required if c not in u.components]
