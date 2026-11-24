"""Transient enhanced diffusion (TED), the effective "+1" model of contract A22.

While the {311} clusters from the implant damage last, the interstitial supersaturation is S(T), uniform in depth;
every diffusivity is multiplied by f_I S + f_V / S (handout #12 eq. 22 with C_V/C_V* = 1/S). The clusters last

    tau(T) = TAU_REF (Q / Q_REF) (Rp / RP_REF) exp[E_TAU/k (1/T - 1/T_REF)]

Sources (A22.0): Stolk et al., J. Appl. Phys. 81, 6031 (1997): the +1 rule, E_TAU = 3.8 eV, about 40 s at 815 C
for 40 keV Si at 5e13 cm^-2, marker enhancements about 200 (10 min) and about 60 (30 min) at 790 C. EE212 handout
#14 eqs. (20)-(21): tau proportional to Rp Q; S a function of T only, its plot (1.3e5 at 600 C, 3.5e2 at 1100 C)
giving E_S. Handout #12's table of f_I. RP_REF: RustBCA (A7), 40 keV 28Si into amorphous Si, Kr-C potential,
Lindhard-Scharff stopping, 20 000 ions, seed 1: 59.02 +- 0.18 nm (computed 24 Nov 2026).
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from simws.process.diffusivity import K_B_EV, kelvin

TAU_REF_S = 40.0
T_REF_K = kelvin(815.0)
Q_REF_CM2 = 5e13
RP_REF_UM = 0.05902
E_TAU_EV = 3.8

T_S_K = kelvin(790.0)
_S_PLOT = ((600.0, 1.3e5), (1100.0, 3.5e2))          # handout #14 page 23, the curve's two end points
E_S_EV = math.log(_S_PLOT[0][1] / _S_PLOT[1][1]) * K_B_EV / (1.0 / kelvin(_S_PLOT[0][0]) - 1.0 / kelvin(_S_PLOT[1][0]))
_MARKERS = ((200.0, 600.0), (60.0, 1800.0))          # Stolk Sec. IV: (time-averaged enhancement, anneal time s)

F_I = {"B": 1.0, "P": 1.0, "As": 0.4, "Sb": 0.02}    # handout #12 page 24

NOT_MODELLED_WITH_TED = ("anneal:not_modelled: clustering, solid solubility, oxidation and segregation (TED by the "
                         "effective model of A22)")
TED_FLAG = ("anneal:ted_effective: the +1 model with a uniform supersaturation; tau and S from Stolk et al. 1997 and "
            "the EE212 handout (simws's calibration); no clustering (boron above about 1e18 cm^-3 is immobile in "
            "reality), no depth dependence")


def tau_s(T_K: float, Q_cm2: float, Rp_um: float) -> float:
    """How long the clusters last at T (s), A22.2."""
    return TAU_REF_S * (Q_cm2 / Q_REF_CM2) * (Rp_um / RP_REF_UM) * math.exp(E_TAU_EV / K_B_EV * (1.0 / T_K - 1.0 / T_REF_K))


def _s_ref() -> float:
    tau = tau_s(T_S_K, Q_REF_CM2, RP_REF_UM)
    return float(np.mean([(E * t - t + tau) / tau for E, t in _MARKERS]))


S_REF = _s_ref()


def supersaturation(T_K: float) -> float:
    """S(T) during the clusters' dissolution, A22.2."""
    return S_REF * math.exp(E_S_EV / K_B_EV * (1.0 / T_K - 1.0 / T_S_K))


def enhancement(dopant: str, T_K: float) -> float:
    """f_I S + f_V / S."""
    S = supersaturation(T_K)
    fi = F_I[dopant]
    return fi * S + (1.0 - fi) / S


def damage(problem) -> Tuple[float, float]:
    """(Q cm^-2, Rp um) of the initial profile: its dose and mean depth (A22.1)."""
    x = np.asarray(problem.initial.depth_um, float)
    c = np.asarray(problem.initial.concentration_cm3, float)
    Q = problem.initial.dose_cm2()
    # A22.9 (after the checker): exact moments of A16.1's profile (c[0] held from 0 to x[0], linear between points)
    h = np.diff(x)
    m0 = c[0] * x[0] + float(np.sum(0.5 * h * (c[:-1] + c[1:])))
    m1 = 0.5 * c[0] * x[0] ** 2 + float(np.sum(h / 6.0 * (c[:-1] * (2 * x[:-1] + x[1:]) + c[1:] * (x[:-1] + 2 * x[1:]))))
    rp = float(m1 / m0) if m0 > 0 else 0.0
    return Q, rp


def schedule(problem) -> List[Tuple[np.ndarray, float]]:
    """For each step: (time points, t_end) where t_end is the time within the step at which TED ends (inf if it
    lasts the whole step, 0 if it ended before). The time points are A16.1's plus t_end when inside the step."""
    out = []
    use = getattr(problem.models, "ted", "none") == "plus_one"
    remaining = 1.0
    if use:
        Q, rp = damage(problem)
    for step in problem.steps:
        pts = problem.time_points_s(step)
        if not use or remaining <= 0.0:
            out.append((pts, 0.0))
            continue
        tau = tau_s(kelvin(step.temperature_C), Q, rp)
        t_end = remaining * tau
        if t_end < step.time_s:
            pts = np.unique(np.concatenate([pts, [t_end]]))
            remaining = 0.0
        else:
            remaining -= step.time_s / tau
            t_end = math.inf
        out.append((pts, t_end))
    return out


def stolk_fig1():
    """Stolk et al. 1997 Fig. 1 as traced (simws/data/ted/stolk1997_fig1.json): (initial DepthProfile for the
    anneal: the as-implanted trace with its exponential tail extension to 0.7 um, A22.7; the diffused trace as
    (depth_um, conc_cm3) arrays; the source dict)."""
    import json
    from pathlib import Path
    from simws.spec.anneal import DepthProfile
    d = json.loads((Path(__file__).resolve().parent.parent / "data" / "ted" / "stolk1997_fig1.json").read_text())
    x = np.asarray(d["as_implanted"]["depth_um"], float)
    c = np.asarray(d["as_implanted"]["conc_cm3"], float)
    order = np.argsort(x)
    x, c = x[order], c[order]
    keep = np.concatenate(([True], np.diff(x) > 0))
    x, c = x[keep], c[keep]
    slope = d["source"]["tail_extension"]["as_implanted_log10_slope_per_um"]
    xe = np.arange(x[-1] + 0.002, 0.7 + 1e-9, 0.002)
    ce = c[-1] * 10.0 ** (slope * (xe - x[-1]))
    ini = DepthProfile(depth_um=np.concatenate([x, xe]).tolist(), concentration_cm3=np.concatenate([c, ce]).tolist())
    dx = np.asarray(d["diffused_800C_35min"]["depth_um"], float)
    dc = np.asarray(d["diffused_800C_35min"]["conc_cm3"], float)
    o = np.argsort(dx)
    return ini, (dx[o], dc[o]), d["source"]


def depth_at(x_um, c_cm3, level: float) -> float:
    """The deepest depth where the profile is still at or above `level`, linear between the points (um)."""
    x = np.asarray(x_um, float)
    c = np.asarray(c_cm3, float)
    above = np.nonzero(c >= level)[0]
    if above.size == 0:
        return float("nan")
    i = int(above[-1])
    if i + 1 >= x.size:
        return float(x[i])
    return float(x[i] + (x[i + 1] - x[i]) * (c[i] - level) / (c[i] - c[i + 1]))
