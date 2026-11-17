"""NIST ASTAR / PSTAR stopping powers for silicon, and the slowing-down integral used as a reference (contract A8.7).

The tables (simws/data/nist_star/*.csv) are NIST Standard Reference Database 124, retrieved 17 Nov 2026, copied
unchanged; their header lines give the source. They rest on ICRU Report 49.

    from simws.radiation import nist_star
    nist_star.energy_after("alpha", 5.0, 10.0)          # MeV left after 10 um of silicon
    nist_star.deposit("proton", 10.0, 0.0, 100.0)       # MeV lost between depths 0 and 100 um along the path

This is a continuous-slowing-down estimate of the MEAN energy loss: no straggling, no delta electrons leaving
the layer, a straight path. It is a reference for checking translations, not a transport code.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "nist_star"
TABLES = {"alpha": "astar_silicon.csv", "proton": "pstar_silicon.csv"}
RHO_SI_G_CM3 = 2.33          # density used by the NIST tables for silicon and by Geant4's G4_Si
STEP_UM = 0.01


@lru_cache(maxsize=None)
def table(particle: str) -> Tuple[np.ndarray, np.ndarray]:
    """(kinetic energy in MeV, total stopping power in MeV cm2/g)."""
    if particle not in TABLES:
        raise ValueError(f"no NIST table for '{particle}'; known: {sorted(TABLES)}")
    a = np.loadtxt(DATA / TABLES[particle], delimiter=",", comments="#", skiprows=7)
    return a[:, 0], a[:, 3]


def stopping_MeV_per_um(particle: str, E_MeV: float, rho_g_cm3: float = RHO_SI_G_CM3) -> float:
    """Total stopping power at E, interpolated linearly in log S against log E, in MeV per micrometre."""
    E, S = table(particle)
    if not E[0] <= E_MeV <= E[-1]:
        raise ValueError(f"{E_MeV:g} MeV is outside the table ({E[0]:g} to {E[-1]:g} MeV)")
    return float(np.exp(np.interp(np.log(E_MeV), np.log(E), np.log(S)))) * rho_g_cm3 * 1e-4


def energy_after(particle: str, E0_MeV: float, path_um: float, step_um: float = STEP_UM,
                 rho_g_cm3: float = RHO_SI_G_CM3) -> float:
    """Energy left after `path_um` of silicon: dE/dx = -S(E) rho integrated with RK4. Raises when the particle
    would slow below the table's lowest energy (it stops inside the path)."""
    E, _ = table(particle)
    f = lambda e: -stopping_MeV_per_um(particle, e, rho_g_cm3)   # noqa: E731
    n = int(round(path_um / step_um))
    h = path_um / n if n else 0.0
    e = float(E0_MeV)
    for _ in range(n):
        k1 = f(e)
        k2 = f(e + 0.5 * h * k1)
        k3 = f(e + 0.5 * h * k2)
        k4 = f(e + h * k3)
        e += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if e < E[0] * 1.5:
            raise ValueError(f"the {particle} stops within {path_um:g} um")
    return e


def deposit(particle: str, E0_MeV: float, entry_um: float, exit_um: float, **kw) -> float:
    """Mean energy lost between path lengths entry_um and exit_um (both measured from where E = E0_MeV)."""
    e_in = energy_after(particle, E0_MeV, entry_um, **kw) if entry_um > 0 else float(E0_MeV)
    return e_in - energy_after(particle, e_in, exit_um - entry_um, **kw)
