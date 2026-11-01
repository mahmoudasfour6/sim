"""Solution verification: systematic refinement studies (mesh size or time step).

    from simws.verify import refinement_study
    res = refinement_study(lambda level: run_case(nx=25 * 2**level)["T_max_K"], levels=4, ratio=2)
    print(res.table())

For three or more levels with constant refinement ratio r it reports, using the finest three solutions
f1 (finest), f2, f3:
    observed order        p   = ln((f3 - f2) / (f2 - f1)) / ln r
    Richardson estimate   f_h0 = f1 + (f1 - f2) / (r^p - 1)
    grid convergence index GCI_fine = Fs |(f1 - f2) / f1| / (r^p - 1),  Fs = 1.25   (Roache 1998)
and checks asymptotic range: GCI_23 / (r^p GCI_12) ~ 1. Oscillatory convergence (sign change between successive
differences) is flagged, in which case p and the GCI are not meaningful.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class RefinementResult:
    levels: List[int]
    values: List[float]
    seconds: List[float]
    ratio: float
    order: Optional[float] = None
    richardson: Optional[float] = None
    gci_fine: Optional[float] = None
    asymptotic_ratio: Optional[float] = None
    oscillatory: bool = False
    notes: List[str] = field(default_factory=list)

    def table(self, name="quantity"):
        out = [f"{'level':>5} {name:>22} {'diff to finer':>14} {'seconds':>9}"]
        for i, (lv, v, s) in enumerate(zip(self.levels, self.values, self.seconds)):
            d = "" if i + 1 >= len(self.values) else f"{self.values[i + 1] - v:+.4e}"
            out.append(f"{lv:5d} {v:22.10g} {d:>14} {s:9.1f}")
        if self.order is not None:
            out.append(f"observed order p = {self.order:.3f}; Richardson extrapolate = {self.richardson:.10g}; "
                       f"GCI_fine = {100 * self.gci_fine:.2e} %; asymptotic check = {self.asymptotic_ratio:.3f}")
        out += self.notes
        return "\n".join(out)

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def analyse(levels, values, seconds, ratio=2.0, Fs=1.25, noise_rel=1e-4) -> RefinementResult:
    res = RefinementResult(list(levels), list(values), list(seconds), ratio)
    if len(values) < 3:
        res.notes.append("need at least 3 levels for an observed order")
        return res
    f3, f2, f1 = values[-3], values[-2], values[-1]
    e32, e21 = f3 - f2, f2 - f1
    if e21 == 0:
        res.notes.append("finest two levels identical: converged to round-off")
        res.order, res.richardson, res.gci_fine, res.asymptotic_ratio = float("inf"), f1, 0.0, 1.0
        return res
    if e32 * e21 < 0:
        res.oscillatory = True
        spread = (max(values[-3:]) - min(values[-3:])) / abs(f1)
        if spread < noise_rel:
            # differences sit at the nonlinear-solver / time-integration noise level: discretization error is
            # below it. Report a conservative uncertainty (Fs x spread of the finest three) instead of an order.
            res.gci_fine = Fs * spread
            res.notes.append(f"converged to the solver-noise level: finest three levels agree within {spread:.1e} "
                             f"(relative); uncertainty bound Fs x spread = {res.gci_fine:.1e}")
        else:
            res.notes.append("oscillatory convergence: order and GCI not meaningful; refine further")
        return res
    p = math.log(abs(e32 / e21)) / math.log(ratio)
    res.order = p
    res.richardson = f1 + (f1 - f2) / (ratio ** p - 1)
    gci12 = Fs * abs(e21 / f1) / (ratio ** p - 1)
    gci23 = Fs * abs(e32 / f2) / (ratio ** p - 1)
    res.gci_fine = gci12
    res.asymptotic_ratio = gci23 / (ratio ** p * gci12)
    if not 0.9 < res.asymptotic_ratio < 1.1:
        res.notes.append("not yet in the asymptotic range (check should be ~1): add a finer level")
    return res


def refinement_study(run: Callable[[int], float], levels: int = 4, ratio: float = 2.0, start: int = 0) -> RefinementResult:
    """run(level) -> scalar; level 0 is the coarsest, each level refines by `ratio`."""
    vals, secs = [], []
    for lv in range(start, start + levels):
        t0 = time.perf_counter()
        vals.append(float(run(lv)))
        secs.append(time.perf_counter() - t0)
    return analyse(list(range(start, start + levels)), vals, secs, ratio)
