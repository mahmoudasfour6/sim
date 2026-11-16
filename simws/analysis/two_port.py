"""Two-port small-signal figures of merit from Y parameters (contract amendment A6.3). Pure numpy: no engine.

The formulas are those of the legacy simws.tcad.small_signal.figures:
    h21  = y21 / y11                                   short-circuit current gain
    U    = |y21 - y12|^2 / (4 (Re y11 Re y22 - Re y12 Re y21))      Mason's unilateral gain
    f_T  from the single-pole form h21 = beta0 / (1 + j f / f_beta):  f_T = beta0 * f_beta
    f_max = f sqrt(U(f))
"""
from __future__ import annotations

import math
from typing import Optional


def h21(y11: complex, y12: complex, y21: complex, y22: complex) -> complex:
    return complex(y21) / complex(y11)


def mason_u(y11: complex, y12: complex, y21: complex, y22: complex) -> float:
    """Mason's unilateral power gain; inf when the denominator is <= 0 (the two-port is not passive-bounded)."""
    y11, y12, y21, y22 = complex(y11), complex(y12), complex(y21), complex(y22)
    den = 4.0 * (y11.real * y22.real - y12.real * y21.real)
    if not den > 0:
        return math.inf
    return abs(y21 - y12) ** 2 / den


def f_t_single_pole(f: float, h21_abs: float, beta0: float) -> Optional[float]:
    """Transit frequency from |h21| at frequency f and the low-frequency gain beta0, assuming one pole. None
    unless beta0 > |h21| (no roll-off seen at f)."""
    if not (beta0 > h21_abs > 0):
        return None
    return beta0 * f / math.sqrt((beta0 / h21_abs) ** 2 - 1.0)


def f_max(f: float, U: float) -> Optional[float]:
    """Maximum oscillation frequency estimate f sqrt(U); None unless U is finite and > 0."""
    if not (math.isfinite(U) and U > 0):
        return None
    return f * math.sqrt(U)
