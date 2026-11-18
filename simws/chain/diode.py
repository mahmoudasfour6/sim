"""Device to circuit, first link (contract A9.6): a TCAD diode I-V turned into a SPICE diode.

    fit = fit_shockley(V, J, T_K)                 # J: current density in A/m^2 from a device result
    problem = circuit_from_fit(fit, area_m2=1e-12, start=0.30, stop=0.60, step=0.05)

The fit is an ideal Shockley diode J = Js (exp(q V / (N k T)) - 1), fitted as ln J = ln Js + q V / (N k T)
(the -1 is negligible at forward bias above a few kT/q). `max_rel_residual` says how well an ideal diode describes
the device over the points given: it is model-form error, not solver error.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

K_B = 1.380649e-23           # J/K, CODATA 2018 (exact)
Q_E = 1.602176634e-19        # C, CODATA 2018 (exact)


def fit_shockley(V: Sequence[float], J: Sequence[float], T_K: float) -> Dict[str, float]:
    V = np.asarray(V, float)
    J = np.asarray(J, float)
    if V.size < 2 or V.size != J.size:
        raise ValueError("fit_shockley: at least two (V, J) points of equal length")
    if np.any(J <= 0):
        raise ValueError("fit_shockley: every J must be > 0 (forward bias)")
    slope, intercept = np.polyfit(V, np.log(J), 1)
    vt = K_B * T_K / Q_E
    N = 1.0 / (slope * vt)
    Js = float(np.exp(intercept))
    J_fit = Js * np.exp(V / (N * vt))                 # the fitted line (A9.11), without the -1 of the diode
    return {"Js_A_m2": Js, "N": float(N), "T_K": float(T_K), "max_rel_residual": float(np.max(np.abs(J_fit / J - 1)))}


def circuit_from_fit(fit: Dict[str, float], area_m2: float = 1e-12, start: float = 0.30, stop: float = 0.60,
                     step: float = 0.05, gmin_S: float = 0.0):
    """The SPICE diode of the fit: IS = Js x area, at the device's temperature (tnom too, so IS is not rescaled).
    gmin_S = 0 (A9.11): SPICE puts gmin across every junction; with IS of order 1e-18 A the gmin current
    (gmin x V) is a visible part of the low-bias current and is not part of the fitted diode."""
    from simws.spec.library import diode_iv
    t_c = float(f"{fit['T_K'] - 273.15:.12g}")
    p = diode_iv(IS_A=float(f"{fit['Js_A_m2'] * area_m2:.12g}"), N=float(f"{fit['N']:.12g}"), start=start,
                 stop=stop, step=step, model="builtin", temperature_C=t_c, tnom_C=t_c)
    return p.with_changes({"numerics.gmin_S": float(gmin_S)})
