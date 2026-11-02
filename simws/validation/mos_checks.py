"""Closed-form checks of the MOS capacitor (simws.tcad.mos).

flatband_exact        V_fb from the solution = gate offset + bulk potential (n = n_ie exp(psi/Vt) at the doping)
debye_flatband_cap    C(V_fb) = 1 / (1/C_ox + L_D/eps_si), L_D = sqrt(eps_si V_t / (q N_A))
fixed_charge_shift    interface sheet Q_f shifts V_fb by exactly -q Q_f / C_ox
oxide_centroid_shift  a sheet N at distance d from the Si interface shifts V_fb by -q N (t_ox - d) / eps_ox
interface_trap_shift  at equal surface potential, D_it moves the gate voltage by -Q_it/C_ox,
                      Q_it = -q D_it V_t ln(n_s / n_ie)  (uniform D_it, neutral level at midgap)
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from simws.materials import load
from simws.tcad.mos import MOSCase, build, cv, flatband, gate_voltage_at_surface_potential
from simws.units import EPS0_F_PER_CM, K_B_EV, Q_E

NM = 1e-7


def _bulk_psi(case, mat):
    from simws.tcad.physics import intrinsic_density
    N = abs(case.NA)
    ni = intrinsic_density(mat, case.physics.T)
    if case.physics.bgn:
        E0, Nr, C = mat.get("bgn_E0", "eV"), mat.get("bgn_N_ref", "cm**-3"), mat.get("bgn_C", "dimensionless")
        dEg = E0 * (math.log(N / Nr) + math.sqrt(math.log(N / Nr) ** 2 + C))
    else:
        dEg = 0.0
    nie = ni * math.exp(dEg / (2 * K_B_EV * case.physics.T))
    Vt = K_B_EV * case.physics.T
    maj = 0.5 * (N + math.sqrt(N * N + 4 * nie * nie))
    n0 = nie * nie / maj if case.NA > 0 else maj
    return Vt * math.log(n0 / ni) - 0.5 * dEg, nie, Vt


def run_all():
    mat, ox = load("silicon"), load("sio2")
    base = MOSCase()
    eps_si = mat.get("eps_r", "dimensionless") * EPS0_F_PER_CM
    eps_ox = ox.get("eps_r", "dimensionless") * EPS0_F_PER_CM
    cox = eps_ox / (base.t_ox_nm * NM)
    out = {}

    psi_b, nie, Vt = _bulk_psi(base, mat)
    vfb0 = flatband(base)
    out["flatband_exact"] = {"V_fb_sim": vfb0, "V_fb_analytic": base.gate_offset_V + psi_b, "abs_diff_V": abs(vfb0 - base.gate_offset_V - psi_b)}

    c = cv(base, [vfb0])["C_F_cm2"][0]
    LD = math.sqrt(eps_si * Vt / (Q_E * base.NA))
    cfb = 1 / (1 / cox + LD / eps_si)
    out["debye_flatband_cap"] = {"C_sim": c, "C_analytic": cfb, "rel_diff": c / cfb - 1}

    Qf = 5e11
    vfb_q = flatband(dataclasses.replace(base, Q_f=Qf))
    exp_q = -Q_E * Qf / cox
    out["fixed_charge_shift"] = {"dV_sim": vfb_q - vfb0, "dV_analytic": exp_q, "rel_diff": (vfb_q - vfb0) / exp_q - 1}

    d, N = 2.0, 1e12
    vfb_o = flatband(dataclasses.replace(base, oxide=[(d, N)]))
    from devsim import get_node_model_values
    xo = np.array(get_node_model_values(device="d", region="ox", name="x"))
    d_node = -xo[int(np.argmin(np.abs(xo + d * NM)))]                  # where the sheet actually sits
    exp_o = -Q_E * N * (base.t_ox_nm * NM - d_node) / eps_ox
    out["oxide_centroid_shift"] = {"dV_sim": vfb_o - vfb0, "dV_analytic": exp_o, "rel_diff": (vfb_o - vfb0) / exp_o - 1}

    Dit = 2e11
    psis = np.array([-0.1, 0.2, 0.4, 0.6, 0.85])
    Va = gate_voltage_at_surface_potential(base, psis)
    Vb = gate_voltage_at_surface_potential(dataclasses.replace(base, D_it=Dit), psis)
    # surface electrons for band bending psi_s: n_s = n_i exp((psi_b + psi_s + dEg/2) / Vt), E_F - E_i = Vt ln(n_s/n_ie)
    from simws.tcad.physics import intrinsic_density
    ni = intrinsic_density(mat, base.physics.T)
    dEg = 2 * (Vt * math.log(nie / ni))
    ns = ni * np.exp((psi_b + psis + 0.5 * dEg) / Vt)
    Qit = -Q_E * Dit * Vt * np.log(ns / nie)
    exp_d = -Qit / cox
    rel = (Vb - Va) / exp_d - 1
    out["interface_trap_shift"] = {"psi_s": psis, "dV_sim": Vb - Va, "dV_analytic": exp_d,
                                   "worst_rel_diff": float(np.max(np.abs(rel[np.abs(exp_d) > 1e-3])))}
    out["pass"] = bool(out["flatband_exact"]["abs_diff_V"] < 1e-6 and abs(out["debye_flatband_cap"]["rel_diff"]) < 0.02
                       and abs(out["fixed_charge_shift"]["rel_diff"]) < 1e-6 and abs(out["oxide_centroid_shift"]["rel_diff"]) < 1e-6
                       and out["interface_trap_shift"]["worst_rel_diff"] < 1e-3)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
