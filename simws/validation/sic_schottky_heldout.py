"""0.9c gate: 4H-SiC Schottky forward I-V versus temperature against a published dataset held out from fitting.

Status: the harness is complete; the dataset is NOT chosen yet. The gate refuses to run until a dataset directory
with `approved: true` in its meta.json exists (the user approves the dataset and the fit / held-out split before
the run, so the held-out temperatures are never looked at while fitting).

Dataset directory (simws/validation/data/sic_schottky/<name>/):
    meta.json   {"citation": ..., "doi_or_arxiv": ..., "figure": ..., "digitized_by": ..., "approved": true,
                 "area_cm2": ..., "N_D_cm3": ..., "fit_T_K": [...], "heldout_T_K": [...], "metal": ...}
    iv.csv      T_K, V, I_A   (forward branch, as digitized; current or current density with area_cm2 = 1)

Model (every term has a physical origin; nothing is a free ideality factor):
    J = A* T^2 exp(-phi_eff / kT) (1 + Gamma) (exp((V - I R_s) / kT) - 1)
    image-force lowering   d_phi = sqrt(q E_s / 4 pi eps),  E_s = sqrt(2 q N_D (phi_B - xi - V') / eps)
    tunnelling             Gamma: the WKB thermionic-field integral of simws.tcad.contacts (m* = 0.42, eps_r = 9.7)
    inhomogeneity          Gaussian barrier distribution (Werner & Guettler, J. Appl. Phys. 69, 1522 (1991)):
                           phi_eff = phi_B0 - sigma_s^2 / (2 kT); voltage dependence of the mean barrier neglected
Fitted on the fit temperatures only: phi_B0, sigma_s, R_s. A* = 146 A cm^-2 K^-2 (4H-SiC, m* 0.42 per valley x 3
valleys, theoretical; PROVISIONAL, flagged) and N_D from the paper.
Held-out acceptance, fixed before any held-out comparison (proposal, to be confirmed with the dataset):
    |ln(I_pred / I_meas)| < ln(1.3) for every held-out point with 1e-7 A/cm^2 < J < 1e-1 A/cm^2 (below series-
    resistance saturation and above the instrument floor).

A device-level version (DEVSIM SiC region with the simws.tcad.contacts Schottky boundary) is the follow-up once the
SiC region carries the band-edge models that boundary needs (dEc, n_i bookkeeping).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from simws.units import K_B_EV, Q_E

DATA = Path(__file__).resolve().parent / "data" / "sic_schottky"
A_STAR = 146.0                     # A cm^-2 K^-2, PROVISIONAL (theoretical 4H-SiC)
EPS_R, M_T = 9.7, 0.42
NC300 = 1.7e19                     # cm^-3, 4H-SiC (3 valleys); PROVISIONAL, scaled as T^1.5
TOL_LN = math.log(1.3)
J_WINDOW = (1e-7, 1e-1)


def load(name):
    d = DATA / name
    meta = json.loads((d / "meta.json").read_text())
    rows = []
    with open(d / "iv.csv") as fh:
        for r in csv.DictReader(fh):
            rows.append((float(r["T_K"]), float(r["V"]), float(r["I_A"])))
    return meta, np.array(rows)


def current(V, T, phi_B0, sigma_s, R_s, N_D, area=1.0, tunneling=True):
    """Forward current (A) at applied V (array), solved self-consistently for the series-resistance drop."""
    from simws.tcad.contacts import e00, tunnel_gamma
    kT = K_B_EV * T
    eps = EPS_R * 8.8541878128e-14
    Nc = NC300 * (T / 300.0) ** 1.5
    xi = kT * math.log(Nc / N_D)
    phi = phi_B0 - sigma_s ** 2 / (2 * kT)
    E00 = e00(N_D, M_T, EPS_R)
    def I_of(vi):
        band = max(phi - xi - vi, 1e-3)
        Es = math.sqrt(2 * Q_E * N_D * band / eps)
        dphi = math.sqrt(Q_E * Es / (4 * math.pi * eps))
        g = float(tunnel_gamma(min(band, phi), kT, E00)[0]) if tunneling else 0.0
        return area * A_STAR * T * T * math.exp(-(phi - dphi) / kT) * (1 + g) * math.expm1(vi / kT)

    from scipy.optimize import brentq
    out = []
    for v in np.atleast_1d(V):
        if R_s <= 0 or v <= 0:
            out.append(I_of(v))
            continue
        # vi + I(vi) R_s = v is monotone in vi on [0, v]
        vi = brentq(lambda x: x + I_of(x) * R_s - v, 0.0, float(v), xtol=1e-14, rtol=1e-12)
        out.append(I_of(vi))
    return np.array(out)


def fit(meta, data):
    from scipy.optimize import least_squares
    fitT = set(meta["fit_T_K"])
    sel = np.array([t in fitT for t in data[:, 0]])
    d = data[sel]
    A = meta.get("area_cm2", 1.0)
    ok = (d[:, 2] / A > J_WINDOW[0]) & (d[:, 2] / A < J_WINDOW[1])
    d = d[ok]

    def res(p):
        phi, sig, lrs = p
        r = []
        for T in sorted(set(d[:, 0])):
            m = d[:, 0] == T
            I = current(d[m, 1], T, phi, sig, 10 ** lrs, meta["N_D_cm3"], A)
            r.append(np.log(np.maximum(I, 1e-300)) - np.log(d[m, 2]))
        return np.concatenate(r)
    sol = least_squares(res, x0=[1.2, 0.05, 0.0], bounds=([0.3, 0.0, -6], [2.5, 0.3, 4]))
    return {"phi_B0_eV": sol.x[0], "sigma_s_eV": sol.x[1], "R_s_ohm": 10 ** sol.x[2],
            "rms_ln_fit": float(np.sqrt(np.mean(sol.fun ** 2))), "n_points": int(len(d))}


# ---------------------------------------------------------------------------------------------- protocol 2
# Fixed on 2026-11-05, after protocol 1 (Vivona 2020) failed in the series-resistance knee and BEFORE the second
# dataset was opened (see data/sic_schottky/PREREGISTRATION_2.md). Changes from protocol 1: the split comes from a
# rule applied to the temperatures in the data; the held-out window ends where the fitted series-resistance drop
# reaches kT/q; each held-out temperature needs >= 10 points in its window, otherwise the gate is inconclusive.
MIN_POINTS_2 = 10


def split_rule_2(temps):
    """Fit at the lowest temperature and at the one closest to the middle of the measured range (excluding the
    lowest); hold out every other temperature."""
    T = sorted(set(float(t) for t in temps))
    if len(T) < 3:
        raise ValueError(f"protocol 2 needs >= 3 temperatures, the data has {len(T)}")
    mid = 0.5 * (T[0] + T[-1])
    second = min(T[1:], key=lambda t: (abs(t - mid), t))
    fit_T = [T[0], second]
    return fit_T, [t for t in T if t not in fit_T]


def window_2(T, p):
    """Held-out window of protocol 2 in current density (A/cm^2): from 1e-7 to where J R_s = kT/q (capped at 1e-1)."""
    kT = K_B_EV * T
    return J_WINDOW[0], min(J_WINDOW[1], kT / max(p["R_s_ohm"], 1e-30))


def predict(meta, data, p):
    A = meta.get("area_cm2", 1.0)
    proto = int(meta.get("protocol", 1))
    rows, worst = [], 0.0
    for T in meta["heldout_T_K"]:
        m = data[:, 0] == T
        V, I = data[m, 1], data[m, 2]
        Ip = current(V, T, p["phi_B0_eV"], p["sigma_s_eV"], p["R_s_ohm"], meta["N_D_cm3"], A)
        lo, hi = window_2(T, p) if proto == 2 else J_WINDOW
        inwin = (I / A > lo) & (I / A < hi)
        ln = np.abs(np.log(Ip[inwin] / I[inwin]))
        w = float(ln.max()) if ln.size else float("nan")
        worst = max(worst, w) if ln.size else worst
        rows.append({"T_K": T, "n_points": int(inwin.sum()), "max_abs_ln_ratio": w, "window_A_cm2": [lo, hi],
                     "max_ratio": float(math.exp(w)) if ln.size else None})
    need = MIN_POINTS_2 if proto == 2 else 1
    conclusive = all(r["n_points"] >= need for r in rows)
    return {"rows": rows, "worst_abs_ln_ratio": worst, "tolerance_abs_ln": TOL_LN, "protocol": proto,
            "conclusive": bool(conclusive), "pass": bool(conclusive and worst < TOL_LN)}


def run(name) -> dict:
    meta, data = load(name)
    if not meta.get("approved"):
        raise PermissionError(f"dataset {name}: not approved for the held-out gate (meta.json 'approved')")
    if int(meta.get("protocol", 1)) == 2:
        fit_T, held_T = split_rule_2(data[:, 0])
        for key, v in (("fit_T_K", fit_T), ("heldout_T_K", held_T)):
            if key in meta and sorted(meta[key]) != sorted(v):
                raise ValueError(f"protocol 2: meta {key} {meta[key]} differs from the rule's {v}")
        meta = {**meta, "fit_T_K": fit_T, "heldout_T_K": held_T}
    overlap = set(meta["fit_T_K"]) & set(meta["heldout_T_K"])
    if overlap:
        raise ValueError(f"fit and held-out temperatures overlap: {sorted(overlap)}")
    p = fit(meta, data)
    pr = predict(meta, data, p)
    return {"dataset": name, "citation": meta.get("citation"), "fit": p, "heldout": pr,
            "provisional": ["A_star", "Nc300 (4H-SiC)"], "pass": pr["pass"]}


def self_test(seed=0) -> dict:
    """Synthetic round trip: data from the model with known parameters and 5 % noise, fit on two temperatures,
    predict three others; checks the harness, not the physics."""
    rng = np.random.default_rng(seed)
    truth = dict(phi_B0=1.25, sigma_s=0.07, R_s=2.0, N_D=1e16)
    rows = []
    for T in (298.0, 323.0, 348.0, 373.0, 398.0):
        V = np.linspace(0.2, 1.4, 25)
        I = current(V, T, truth["phi_B0"], truth["sigma_s"], truth["R_s"], truth["N_D"], 1e-3)
        rows += [(T, v, i * math.exp(rng.normal(0, 0.05))) for v, i in zip(V, I)]
    meta = {"N_D_cm3": 1e16, "area_cm2": 1e-3, "fit_T_K": [298.0, 348.0], "heldout_T_K": [323.0, 373.0, 398.0]}
    data = np.array(rows)
    p = fit(meta, data)
    pr = predict(meta, data, p)
    return {"truth": truth, "fit": p, "heldout": pr,
            "pass": bool(abs(p["phi_B0_eV"] - truth["phi_B0"]) < 0.02 and pr["pass"])}
