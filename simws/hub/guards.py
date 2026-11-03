"""Physics guards: cheap law-of-physics checks an adapter runs in self_check (interface specification: physics
guards). Each returns a list of problems (empty = passed) so adapters can concatenate them.

    problems = G.positive("n", n) + G.mass_action(n, p, n_ie, rtol=1e-3) + G.kirchhoff([IC, IB, IE])
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


def positive(name, x) -> List[str]:
    a = np.asarray(x, float)
    return [] if np.all(a > 0) else [f"{name}: {int(np.sum(a <= 0))} non-positive values"]


def finite(name, x) -> List[str]:
    a = np.asarray(x, float)
    return [] if np.all(np.isfinite(a)) else [f"{name}: non-finite values"]


def monotonic(name, x, increasing=True, strict=False) -> List[str]:
    d = np.diff(np.asarray(x, float))
    ok = (d > 0 if strict else d >= 0) if increasing else (d < 0 if strict else d <= 0)
    return [] if np.all(ok) else [f"{name}: not {'strictly ' if strict else ''}{'in' if increasing else 'de'}creasing"]


def mass_action(n, p, n_ie, rtol=1e-2) -> List[str]:
    """Equilibrium only: n p = n_ie^2 at every node."""
    n, p, ni = (np.asarray(v, float) for v in (n, p, n_ie))
    err = np.max(np.abs(n * p / ni ** 2 - 1.0))
    return [] if err <= rtol else [f"mass action violated: max |np/n_ie^2 - 1| = {err:.2e}"]


def kirchhoff(currents: Sequence, rtol=1e-6, atol=0.0) -> List[str]:
    """Terminal currents (passive sign, into the device) sum to zero."""
    I = np.asarray(currents, float)
    s = np.abs(I.sum(axis=0))
    scale = np.max(np.abs(I), axis=0)
    ok = s <= rtol * scale + atol
    return [] if np.all(ok) else [f"Kirchhoff: terminal currents sum to {np.max(s):.3e} (scale {np.max(scale):.3e})"]


def charge_neutral(total_charge_C, scale_C, rtol=1e-6) -> List[str]:
    return [] if abs(total_charge_C) <= rtol * abs(scale_C) else [f"net charge {total_charge_C:.3e} C is not zero"]


def bounded(name, x, lo=None, hi=None) -> List[str]:
    a = np.asarray(x, float)
    out = []
    if lo is not None and np.any(a < lo):
        out.append(f"{name}: below {lo}")
    if hi is not None and np.any(a > hi):
        out.append(f"{name}: above {hi}")
    return out


def ideality(name, V, I, n_min=0.95, n_max=2.2, T=300.0) -> List[str]:
    """Diode/transistor current ideality factor n = (1/Vt) dV/dln I within [n_min, n_max] where I > 0."""
    from simws.units import K_B_EV
    V, I = np.asarray(V, float), np.asarray(I, float)
    m = I > 0
    if m.sum() < 3:
        return []
    n = np.gradient(V[m], np.log(I[m])) / (K_B_EV * T)
    bad = (n < n_min) | (n > n_max)
    return [] if not np.any(bad) else [f"{name}: ideality outside [{n_min}, {n_max}] (range {n.min():.2f}-{n.max():.2f})"]


def all_of(*lists: Iterable[str]) -> List[str]:
    out: List[str] = []
    for l in lists:
        out += list(l)
    return out
