"""Dopant diffusivities in silicon and the Fermi-level model (architecture v1, contract A16.2).

Sources (A16.0):
  * EE212 class handout #12, J. D. Plummer, M. D. Deal, P. B. Griffin, Stanford 1999-00 (the basis of Table 7-5 of
    Plummer, Deal and Griffin, Silicon VLSI Technology, 2000): eq. (13), page 10, intrinsic D0 and E_A; eq. (19),
    page 18, the charge-state components.
  * F. J. Morin and J. P. Maita, Phys. Rev. 96, 28 (1954), eq. (3): n_i^2 = 1.5e33 T^3 exp(-1.21 eV/kT)
    (fitted below 700 K; used here at process temperatures, so extrapolated).

Both anneal engines take every number from this module, so they solve the same equation:
    dC/dt = d/dx (K(C) dC/dx),   K = D_eff(C) h(C)
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

K_B_EV = 8.617333262e-5                         # eV/K, CODATA 2018
NI_PREFACTOR = math.sqrt(1.5e33)                # cm^-3 K^-1.5 (= 3.873e16), Morin and Maita eq. (3)
NI_EG_HALF_EV = 0.605                           # half of their 1.21 eV

# eq. (13): intrinsic D = D0 exp(-E/kT)  (cm^2/s, eV)
INTRINSIC: Dict[str, Tuple[float, float]] = {"B": (1.0, 3.5), "As": (9.17, 3.99), "Sb": (4.58, 3.88),
                                             "P": (4.70, 3.68)}
# eq. (19): charge-state components; "plus" is for the acceptor B (p/ni), "minus"/"double" for donors (n/ni)
CHARGE_STATES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "B": {"neutral": (0.05, 3.5), "plus": (0.95, 3.5)},
    "As": {"neutral": (0.011, 3.44), "minus": (31.0, 4.15)},
    "Sb": {"neutral": (0.214, 3.65), "minus": (15.0, 4.08)},
    "P": {"neutral": (3.85, 3.66), "minus": (4.44, 4.0), "double": (44.2, 4.37)},
}
NI_FLAG = "model:ni_morin_maita_extrapolated"
NOT_MODELLED_FLAG = ("anneal:not_modelled: transient enhanced diffusion from implant damage, clustering, solid "
                     "solubility, oxidation and segregation")


def kelvin(T_C: float) -> float:
    return float(T_C) + 273.15


def ni_cm3(T_K: float) -> float:
    """Intrinsic carrier density (cm^-3), Morin and Maita."""
    return NI_PREFACTOR * T_K ** 1.5 * math.exp(-NI_EG_HALF_EV / (K_B_EV * T_K))


def arrhenius(d0: float, e: float, T_K: float) -> float:
    return d0 * math.exp(-e / (K_B_EV * T_K))


def intrinsic_D(dopant: str, T_K: float) -> float:
    """eq. (13), cm^2/s."""
    return arrhenius(*INTRINSIC[dopant], T_K)


def components(dopant: str, T_K: float) -> Dict[str, float]:
    """eq. (19) components at T (cm^2/s): neutral, and plus / minus / double where the dopant has them."""
    return {k: arrhenius(d0, e, T_K) for k, (d0, e) in CHARGE_STATES[dopant].items()}


def sign_of_background(dopant: str, background_type: str) -> float:
    """+1 when the background is the dopant's own type, -1 when it is the other type."""
    own = "acceptor" if dopant == "B" else "donor"
    return 1.0 if background_type == own else -1.0


def majority_over_ni(C, Nb_signed: float, ni: float):
    """n/ni (donors) or p/ni (B) from neutrality, N = C + Nb_signed: N/2 + sqrt(N^2/4 + ni^2), over ni."""
    N = np.asarray(C, float) + Nb_signed
    return (0.5 * N + np.sqrt(0.25 * N * N + ni * ni)) / ni


def field_factor(C, Nb_signed: float, ni: float):
    """h = 1 + C / sqrt(N^2 + 4 ni^2)."""
    C = np.asarray(C, float)
    N = C + Nb_signed
    return 1.0 + C / np.sqrt(N * N + 4.0 * ni * ni)


def fermi_D(dopant: str, C, Nb_signed: float, T_K: float):
    """D_eff (cm^2/s) of the Fermi model at concentration C."""
    comp = components(dopant, T_K)
    r = majority_over_ni(C, Nb_signed, ni_cm3(T_K))
    out = comp["neutral"] + 0.0 * r
    for key, power in (("plus", 1), ("minus", 1), ("double", 2)):
        if key in comp:
            out = out + comp[key] * r ** power
    return out


def K(problem, C, T_K: float):
    """K = D_eff h (cm^2/s) for the problem's model at concentration C (cm^-3)."""
    C = np.asarray(C, float)
    if problem.models.diffusivity == "intrinsic":
        return intrinsic_D(problem.dopant, T_K) + 0.0 * C
    nb = sign_of_background(problem.dopant, problem.background_type) * problem.background_cm3
    return fermi_D(problem.dopant, C, nb, T_K) * field_factor(C, nb, ni_cm3(T_K))


def K_expression(problem, T_K: float, var: str = "C") -> str:
    """The same K as a DEVSIM expression in `var` (cm^-3), for the DEVSIM engine."""
    if problem.models.diffusivity == "intrinsic":
        return repr(intrinsic_D(problem.dopant, T_K))
    ni = ni_cm3(T_K)
    nb = sign_of_background(problem.dopant, problem.background_type) * problem.background_cm3
    N = f"({var} + {nb!r})"
    r = f"((0.5*{N} + pow(0.25*{N}*{N} + {ni * ni!r}, 0.5)) / {ni!r})"
    comp = components(problem.dopant, T_K)
    terms = [repr(comp["neutral"])]
    for key, power in (("plus", 1), ("minus", 1), ("double", 2)):
        if key in comp:
            terms.append(f"{comp[key]!r}*{r}" + ("" if power == 1 else f"*{r}"))
    h = f"(1 + {var} / pow({N}*{N} + {4 * ni * ni!r}, 0.5))"
    return f"(({' + '.join(terms)}) * {h})"


def Dt_cm2(problem) -> float:
    """Sum over the steps of the intrinsic D times the time (cm^2), A16.4."""
    return float(sum(intrinsic_D(problem.dopant, kelvin(s.temperature_C)) * s.time_s for s in problem.steps))
