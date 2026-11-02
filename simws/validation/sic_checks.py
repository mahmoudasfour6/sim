"""Checks of the 4H-SiC models (simws.tcad.sic).

single_level_closed_form  the neutrality solver vs the closed form for one donor level, no compensation:
                          n = n1/(2g) (sqrt(1 + 4 g N_D/n1) - 1)
devsim_ionization         DEVSIM bulk carrier density in uniform n (N) and p (Al) slabs = the independent solver
anisotropic_mobility      resistor along vs across c: current ratio = mu_par/mu_perp = 1.2 (electrons), and the
                          absolute current = q n_ionized mu_CT(N) V / L
breakdown                 p+/n diode with injected electrons, impact ionization along <0001>: multiplication from
                          DEVSIM vs McIntyre's integral on the same field (within 5 %), breakdown voltage from the
                          1/M extrapolation vs the ionization-integral extrapolation (within 1 %) and vs
                          eps E_c^2 / (2 q N_D) from the simulated critical field (non-punch-through, within 5 %).
                          The critical field is reported; it should sit in the 2.5-4.5 MV/cm range quoted for 4H-SiC
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from simws.materials import load
from simws.tcad.sic import SiCPhysics, alpha_sic, band, equilibrium, ionization_constants, setup_region_sic
from simws.tcad.physics import contact_current
from simws.units import K_B_EV, Q_E


def _mesh(L, lines):
    from simws.validation.physics_checks import _mesh as m
    m(L, lines)


def single_level_closed_form():
    m = load("sic_4h")
    m1 = m.with_overrides(ion_N_k=(m.get("ion_N_h", "eV"), "eV"))
    c = ionization_constants(m1, 300.0)
    out = []
    for N in (1e15, 1e17, 1e19):
        n_cf = c["n1h"] / (2 * c["gD"]) * (math.sqrt(1 + 4 * c["gD"] * N / c["n1h"]) - 1)
        n_num = equilibrium(m1, N, 0.0)["n"]
        out.append(n_num / n_cf - 1)
    return {"worst_rel_diff": float(max(map(abs, out))), "pass": bool(max(map(abs, out)) < 1e-9)}


def devsim_ionization():
    from devsim import get_node_model_values
    m = load("sic_4h")
    res = {}
    for label, ND, NA in (("N 1e18", 1e18, 0.0), ("Al 1e18", 0.0, 1e18)):
        _mesh(5e-4, [(0, 5e-6, "L"), (5e-4, 5e-6, "R")])
        setup_region_sic("d", "si", m, f"{ND}", f"{NA}", SiCPhysics(), {"left": (False, ND, NA), "right": (False, ND, NA)})
        e = equilibrium(m, ND, NA)
        x = np.array(get_node_model_values(device="d", region="si", name="x"))
        n = np.array(get_node_model_values(device="d", region="si", name="Electrons" if ND else "Holes"))
        mid = n[int(np.argmin(np.abs(x - 2.5e-4)))]
        ref = e["n"] if ND else e["p"]
        res[label] = {"devsim": float(mid), "solver": ref, "ionized_fraction": (e["ND+"] / ND) if ND else (e["NA-"] / NA),
                      "rel_diff": float(mid / ref - 1)}
    worst = max(abs(v["rel_diff"]) for v in res.values())
    return {"cases": res, "worst_rel_diff": worst, "pass": bool(worst < 1e-8)}


def anisotropic_mobility():
    from devsim import set_parameter, solve
    m = load("sic_4h")
    N, L, V = 1e17, 1e-3, 0.01
    I = {}
    for axis in ("x", None):
        _mesh(L, [(0, 1e-5, "L"), (L, 1e-5, "R")])
        setup_region_sic("d", "si", m, f"{N}", "0", SiCPhysics(c_axis=axis), {"left": (False, N, 0), "right": (False, N, 0)})
        set_parameter(device="d", name="left_bias", value=V)
        solve(type="dc", absolute_error=1e10, relative_error=1e-12, maximum_iterations=40)
        I[axis or "perp"] = abs(contact_current("d", "left"))
    e = equilibrium(m, N, 0.0)
    mu = 40.0 + (950.0 - 40.0) / (1 + (N / 1.94e17) ** 0.61)
    I_an = Q_E * e["n"] * mu * V / L
    ratio = I["x"] / I["perp"]
    return {"I_par": I["x"], "I_perp": I["perp"], "ratio": ratio, "I_perp_analytic": I_an,
            "abs_rel_diff": I["perp"] / I_an - 1, "pass": bool(abs(ratio - 1.2) < 1e-6 and abs(I["perp"] / I_an - 1) < 1e-3)}


def breakdown(ND=2e17, NA=1e19):
    from devsim import get_edge_model_values, get_node_model_values
    from devsim.python_packages.model_create import CreateNodeModel
    from simws.tcad.solve import ramp_bias
    m = load("sic_4h")
    xj, L = 0.5e-4, 6e-4
    W_est = math.sqrt(2 * 9.66 * 8.854e-14 * 600 / (Q_E * ND))
    _mesh(L, [(0, 2e-6, "L"), (xj - 0.05e-4, 2e-7, None), (xj, 5e-8, None), (xj + 1.2 * W_est, 5e-7, None), (L, 2e-6, "R")])
    dop_n, dop_p = f"{ND}*step(x-{xj})", f"{NA}*step({xj}-x)"
    gen = "1e22*step(x-0.25e-4)*step(0.40e-4-x)"            # injected electrons in the neutral p+ side
    setup_region_sic("d", "si", m, dop_n, dop_p, SiCPhysics(c_axis="x", impact=True, tau_n=1e-7, tau_p=1e-7),
                     {"left": (False, 0.0, NA), "right": (False, ND, 0.0)}, extra_generation=gen)
    rows = []

    class _Done(RuntimeError):
        pass

    def record(v):
        x = np.array(get_node_model_values(device="d", region="si", name="x"))
        E = np.array(get_edge_model_values(device="d", region="si", name="ElectricField"))
        o = np.argsort(x); dx = np.diff(x[o])
        an, ap = alpha_sic(m, E, 1.0)
        inner = np.cumsum((an - ap) * dx) - 0.5 * (an - ap) * dx
        rows.append((-v, abs(contact_current("d", "left")), float(np.sum(an * np.exp(-inner) * dx)), float(np.max(np.abs(E)))))
        if rows[-1][2] > 0.995:                      # past breakdown: nothing more to learn, stop the ramp
            raise _Done()

    ramp_bias("d", "left", -5.0, step=1.0, callback=record, relative_error=1e-8)
    try:
        ramp_bias("d", "left", -2000.0, step=5.0, max_step=20.0, min_step=1e-3, callback=record, relative_error=1e-8)
    except RuntimeError:                             # _Done (past breakdown) or a stalled ramp
        pass
    r = np.array(rows)
    I0 = r[r[:, 0] <= 5.0 + 1e-9][-1, 1]
    M_dd, M_int = r[:, 1] / I0, 1.0 / (1.0 - np.minimum(r[:, 2], 0.999999))
    eps = m.get("eps_r_par", "dimensionless") * 8.8541878128e-14
    sel = (M_int > 1.05) & (M_int < 20)
    worst = float(np.max(np.abs(M_dd[sel] / M_int[sel] - 1))) if sel.any() else float("nan")
    fit_sel = (M_dd > 2) & (M_dd < 50)
    v_dd = float(-np.polyfit(r[fit_sel, 0], 1 / M_dd[fit_sel], 1)[1] / np.polyfit(r[fit_sel, 0], 1 / M_dd[fit_sel], 1)[0]) if fit_sel.sum() > 2 else None
    isel = (r[:, 2] > 0.5) & (r[:, 2] < 0.98)
    v_int = float(-np.polyfit(r[isel, 0], 1 - r[isel, 2], 1)[1] / np.polyfit(r[isel, 0], 1 - r[isel, 2], 1)[0]) if isel.sum() > 2 else None
    ec = float(np.interp(v_int, r[:, 0], r[:, 3])) if v_int else float("nan")
    bv_ec = eps * ec ** 2 / (2 * Q_E * ND)
    return {"ND": ND, "M_worst_rel_diff": worst, "BV_devsim": v_dd, "BV_ionization_integral": v_int,
            "E_crit_MV_cm": ec / 1e6, "BV_from_Ec": bv_ec, "last_bias": float(r[-1, 0]),
            "pass": bool(worst < 0.05 and v_dd and v_int and abs(v_dd / v_int - 1) < 1e-2 and abs(v_int / bv_ec - 1) < 0.05
                         and 2.5 < ec / 1e6 < 4.5)}


def run_all():
    res = {"single_level_closed_form": single_level_closed_form(), "devsim_ionization": devsim_ionization(),
           "anisotropic_mobility": anisotropic_mobility(), "breakdown": breakdown()}
    res["pass"] = all(v["pass"] for v in res.values())
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), indent=1, default=float))
