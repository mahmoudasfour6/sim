"""Checks of the first-principles parameters (results of simws.studies.dft_params) against measurements.
Tolerances are set from the known accuracy of semilocal DFT, stated before the comparison, not fitted:

masses           m_l, m_t (cyclotron resonance, Hensel et al., Phys. Rev. 138, A225 (1965)) within 10 %;
                 heavy/light/split-off hole masses along [100] and [111] from the Luttinger parameters
                 gamma1 4.285, gamma2 0.339, gamma3 1.446 (Hensel & Suzuki, PRB 9, 4219 (1974)) within 15 %;
                 spin-orbit splitting 44 meV (Yu & Cardona) within 20 %
intrinsic        n_i(T) through the device code path (simws.tcad.physics.intrinsic_density with the computed
                 'dft' profile: DFT Nc(T), Nv(T) and the measured gap) vs the measured fit of Misiakos & Tsamakis,
                 J. Appl. Phys. 74, 3293 (1993), 250-400 K, within 10 %
gap              HSE06-corrected gap (0 K) vs 1.170 eV within 0.1 eV
elastic          (C11 + 2 C12)/3 vs the equation-of-state bulk modulus within 3 % (internal consistency, per
                 functional); ensemble C11, C12, C44 vs McSkimin & Andreatch (1964) within 15 %
deformation      Xi_u in 8.0-10.0 eV (measured 8.6-9.2), b in -2.6 to -1.9 eV (measured -2.1 to -2.35)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

G1, G2, G3 = 4.285, 0.339, 1.446
MEAS = {"m_l": 0.9163, "m_t": 0.1905,
        "m_hh_100": 1 / (G1 - 2 * G2), "m_hh_111": 1 / (G1 - 2 * G3),
        "m_lh_100": 1 / (G1 + 2 * G2), "m_lh_111": 1 / (G1 + 2 * G3), "m_so_100": 1 / G1,
        "Delta_so": 0.044, "C11": 165.7, "C12": 63.9, "C44": 79.6}
TOL = {"m_l": 0.10, "m_t": 0.10, "m_hh_100": 0.15, "m_hh_111": 0.15, "m_lh_100": 0.15, "m_lh_111": 0.15,
       "m_so_100": 0.15, "Delta_so": 0.20, "C11": 0.15, "C12": 0.15, "C44": 0.15}


def misiakos(T):
    return 5.29e19 * (T / 300.0) ** 2.54 * math.exp(-6726.0 / T)


def check(results_dir):
    R = Path(results_dir)
    res = json.loads((R / "params.json").read_text())
    res = res.get("results", res)
    P, per = res["parameters"], res["per_functional"]
    out = {"parameters": {}}
    ok = True
    for k, ref in MEAS.items():
        if k not in P:
            continue
        v, s = P[k]["value"], P[k]["sigma"]
        rel = v / ref - 1
        passed = abs(rel) <= TOL[k]
        ok &= passed
        out["parameters"][k] = {"dft": v, "sigma": s, "measured": ref, "rel_dev": rel, "tol": TOL[k],
                                "z": (v - ref) / s if s > 0 else None, "pass": bool(passed)}
    # n_i(T) through the device code path
    from simws.materials import load
    from simws.materials.semiconductor import intrinsic_density
    m = load("silicon", profile="dft", computed_dir=str(R))
    base = load("silicon")
    rows = []
    for T in (250.0, 300.0, 350.0, 400.0):
        rows.append({"T": T, "dft": intrinsic_density(m, T), "base_model": intrinsic_density(base, T),
                     "measured_fit": misiakos(T)})
    worst = max(abs(r["dft"] / r["measured_fit"] - 1) for r in rows)
    out["intrinsic"] = {"rows": rows, "worst_rel_dev": worst, "pass": bool(worst < 0.10)}
    ok &= worst < 0.10
    if "Eg0_hybrid" in P:
        d = P["Eg0_hybrid"]["value"] - 1.170
        out["gap"] = {"Eg0_hybrid": P["Eg0_hybrid"]["value"], "sigma": P["Eg0_hybrid"]["sigma"], "dev_eV": d,
                      "pass": bool(abs(d) < 0.1)}
        ok &= abs(d) < 0.1
    cons = {f: abs((v["C11"] + 2 * v["C12"]) / 3 / v["B0"] - 1) for f, v in per.items()}
    out["elastic_consistency"] = {"by_functional": cons, "pass": bool(max(cons.values()) < 0.03)}
    ok &= max(cons.values()) < 0.03
    dp = {"Xi_u": (P["Xi_u"]["value"], 8.0, 10.0)}
    if "b" in P:
        dp["b"] = (P["b"]["value"], -2.6, -1.9)
    out["deformation"] = {k: {"value": v, "range": [lo, hi], "pass": bool(lo <= v <= hi)} for k, (v, lo, hi) in dp.items()}
    ok &= all(v["pass"] for v in out["deformation"].values())
    out["pass"] = bool(ok and all(v["pass"] for v in out["parameters"].values()))
    return out
