"""Electronic stopping correction factors from measured data (contract A12).

    from simws.radiation import empirical_stopping as es
    e = es.entry("B")            # {"ck": 1.6.., "u": .., "n": .., "publications": [...], ...}, or None
    es.lindhard_scharff_MeV_cm2_mg(5, 11.009, 50e3)   # RustBCA 3.0.0's LS stopping in silicon

The rule (A12.2): measured rows of the IAEA stopping database for ion X in solid silicon at or below 10 keV/u
(SRIM2013 rows excluded), r_i = S_i / S_LS(E_i), inverse-variance weighted mean with a Birge-ratio uncertainty;
an entry needs at least 3 rows from at least 2 publications.
"""
from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

DATA = Path(__file__).resolve().parent.parent / "data" / "stopping"
SOURCE = ("IAEA Electronic Stopping Power of Matter for Ions, release 2026-09 (Montanari et al., NIM B 551 (2024) "
          "165336, doi 10.1016/j.nimb.2024.165336)")
E_MAX_MEV_U = 0.010
DEFAULT_ERROR = 0.10
EXCLUDED_REFS = ("SRIM2013",)
SI_MASS_U = 28.0855
SI_Z = 14
_Q = 1.602176634e-19
_AMU_G = 1.66053906660e-24
_Z = {"B": 5, "P": 15, "As": 33}


def lindhard_scharff_cross_section_eV_m2(Z1: float, Z2: float, E_eV: float, M1_u: float) -> float:
    """RustBCA 3.0.0 src/material.rs: 1.212 A^2 eV x Z1^(7/6) Z2 / (Z1^(2/3) + Z2^(2/3))^(3/2) x sqrt(E/M1),
    E in eV and M1 in u (eV m^2 per target atom)."""
    k = 1.212e-20 * Z1 ** (7.0 / 6.0) * Z2 / (Z1 ** (2.0 / 3.0) + Z2 ** (2.0 / 3.0)) ** 1.5
    return k * math.sqrt(E_eV / M1_u)


def lindhard_scharff_MeV_cm2_mg(Z1: float, M1_u: float, E_eV: float) -> float:
    """The LS stopping of silicon in MeV/(mg/cm^2)."""
    s = lindhard_scharff_cross_section_eV_m2(Z1, SI_Z, E_eV, M1_u)          # eV m^2 per atom
    return s * 1e4 * 1e-6 / (SI_MASS_U * _AMU_G * 1e3)                       # MeV cm^2 per mg


def rows(ion: str) -> List[Dict[str, str]]:
    path = DATA / f"iaea_{ion.lower()}_si.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def selected(ion: str) -> List[Dict[str, object]]:
    """The rows that pass A12.2(a), each with its ratio r and relative error e."""
    out = []
    for r in rows(ion):
        if r["projectile_name"] != ion or r["target_name"] != "Si" or r["phase"] != "s":
            continue
        if r["energy_unit"] != "MeV/u" or r["stopping_unit_converted"] != "MeV/(mg/cm2)":
            continue
        if r["ref_id"] in EXCLUDED_REFS:
            continue
        e_u = float(r["energy"])
        if e_u > E_MAX_MEV_U:
            continue
        m1 = float(r["ion_isotope"])
        e_ev = e_u * m1 * 1e6
        s = float(r["stopping_power_converted"])
        err = float(r["percentage_error"]) / 100.0 if r["percentage_error"].strip() else DEFAULT_ERROR
        r_i = s / lindhard_scharff_MeV_cm2_mg(_Z[ion], m1, e_ev)
        out.append({"ref_id": r["ref_id"], "citation": r["citation_reference"], "doi": r["doi"],
                    "E_MeV_u": e_u, "E_eV": e_ev, "S_MeV_cm2_mg": s, "r": r_i, "e": err})
    return out


def _weighted(sel: List[Dict[str, object]]):
    w = [1.0 / (float(s["r"]) * float(s["e"])) ** 2 for s in sel]
    ck = sum(wi * float(s["r"]) for wi, s in zip(w, sel)) / sum(w)
    chi2 = sum(wi * (float(s["r"]) - ck) ** 2 for wi, s in zip(w, sel))
    n = len(sel)
    birge = math.sqrt(chi2 / (n - 1)) if n > 1 else 1.0
    return ck, math.sqrt(1.0 / sum(w)) * max(1.0, birge), chi2, birge


@lru_cache(maxsize=None)
def entry(ion: str) -> Optional[dict]:
    """The A12.2 entry for `ion` in silicon, or None when (d) fails."""
    if ion not in _Z:
        return None
    sel = selected(ion)
    pubs = sorted({str(s["ref_id"]) for s in sel})
    if len(sel) < 3 or len(pubs) < 2:
        return None
    ck, u, chi2, birge = _weighted(sel)
    loo = {}
    for p in pubs:
        rest = [s for s in sel if s["ref_id"] != p]
        if rest:
            loo[p] = _weighted(rest)[0]
    x = [float(s["E_MeV_u"]) * 1e3 for s in sel]        # keV/u
    y = [float(s["r"]) for s in sel]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxx = sum((a - mx) ** 2 for a in x)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx if sxx > 0 else 0.0
    return {"ion": ion, "target": "Si", "ck": ck, "u": u, "n": len(sel), "publications": pubs, "chi2": chi2,
            "birge_ratio": birge, "leave_one_out": loo, "slope_per_keV_u": slope,
            "E_range_keV_u": (min(x), max(x)), "source": SOURCE}
