"""0.9b checks of the Schottky contact models (simws.tcad.contacts).

te_closed_form      thermionic emission against J = A* T^2 exp(-phi_B/kT)(exp(V/kT) - 1): exact in the limit where
                    drift-diffusion behind the contact is lossless. The deviation is v_R / v_D ~ 1/mu, so the check
                    runs at mu = 1e8 cm^2/Vs and also confirms the 1/mu scaling (gate: 1e-4)
detailed_balance    zero terminal current at zero bias with image-force lowering and tunnelling on (gate: below
                    1e-12 of the 0.1 V forward current)
image_force         forward current with image-force lowering against the closed form with
                    d_phi = sqrt(q E_s / 4 pi eps), E_s from the depletion approximation (gate: 2 %)
tunnel_quadrature   the device's 32-point Gauss-Legendre Gamma against adaptive quadrature (gate: 1e-4)
padovani_stratton   tunnelling enhancement against Padovani & Stratton's thermionic-field formula (a saddle-point
                    evaluation of the same integral; reported, with the TFE-regime spread as a model-form error)
"""
from __future__ import annotations

import math

import numpy as np

from simws.units import K_B_EV, Q_E

N_D, PHI_B, A_N, A_P = 1e16, 0.70, 112.0, 32.0


def _diode(mu=1400.0, T=300.0, N=N_D, image_force=False, tunneling=False, A_p=A_P, Vs=(0.1, 0.2, 0.3), L_um=1.0):
    import simws.tcad  # noqa: F401
    from devsim import (add_1d_contact, add_1d_mesh_line, add_1d_region, create_1d_mesh, create_device,
                        finalize_mesh, get_contact_current, get_node_model_values, reset_devsim, solve)
    from simws.materials import load
    from simws.tcad.contacts import relax_schottky, schottky_contact
    from simws.tcad.physics import Physics, setup_region
    from simws.tcad.solve import ramp_bias
    reset_devsim()
    L = L_um * 1e-4
    create_1d_mesh(mesh="m")
    add_1d_mesh_line(mesh="m", pos=0.0, ps=1e-8, tag="a")
    add_1d_mesh_line(mesh="m", pos=0.3 * L, ps=2e-7)
    add_1d_mesh_line(mesh="m", pos=L, ps=1e-6, tag="b")
    add_1d_contact(mesh="m", name="anode", tag="a", material="metal")
    add_1d_contact(mesh="m", name="cathode", tag="b", material="metal")
    add_1d_region(mesh="m", material="Si", region="r", tag1="a", tag2="b")
    finalize_mesh(mesh="m"); create_device(mesh="m", device="d")
    mat = load("silicon")
    phys = Physics(T=T, bgn=False, srh=False, auger=False, mu_n=mu, mu_p=mu / 3)
    setup_region("d", "r", mat, f"{N!r}", f"{N!r}", phys, {"anode": False, "cathode": False})
    used = schottky_contact("d", "r", "anode", mat, phys, phi_Bn=PHI_B, A_n=A_N, A_p=A_p,
                            image_force=image_force, tunneling=tunneling)
    opts = dict(absolute_error=1e10, relative_error=1e-12, maximum_iterations=80)
    solve(type="dc", **opts)
    relax = (lambda _v=None: relax_schottky("d", "r", opts)) if (image_force or tunneling) else None
    if relax:
        relax()
    cur = lambda e: get_contact_current(device="d", contact="anode", equation=e)
    out = {"used": used, "J0_n": cur("ElectronContinuityEquation"), "J0_p": cur("HoleContinuityEquation"),
           "V": [], "Jn": [], "Es": []}
    for V in Vs:
        ramp_bias("d", "anode", V, step=0.05, callback=relax, retry_callback=True, **opts)
        out["V"].append(V)
        out["Jn"].append(abs(cur("ElectronContinuityEquation")))
        if image_force or tunneling:
            out["Es"].append(float(abs(np.array(get_node_model_values(device="d", region="r", name="Es_node"))[0])))
    return out


def te_closed_form(mus=(1e6, 1e8)) -> dict:
    from simws.tcad.contacts import te_current
    rows, dev = [], {}
    for mu in mus:
        r = _diode(mu=mu, A_p=0.0)
        rel = [j / float(te_current(PHI_B, v, 300.0, A_N)) - 1 for v, j in zip(r["V"], r["Jn"])]
        dev[mu] = rel
        rows.append({"mu": mu, "rel_diff": rel})
    scaling = [dev[mus[0]][k] / dev[mus[1]][k] for k in range(len(dev[mus[0]]))]
    worst = max(abs(x) for x in dev[mus[-1]])
    return {"rows": rows, "deviation_ratio_mu": scaling, "expected_ratio": mus[1] / mus[0], "worst_rel_diff": worst,
            "pass": bool(worst < 1e-4 and all(abs(s / (mus[1] / mus[0]) - 1) < 0.05 for s in scaling))}


def detailed_balance() -> dict:
    r = _diode(image_force=True, tunneling=True, Vs=(0.1,))
    zero = max(abs(r["J0_n"]), abs(r["J0_p"]))
    rel = zero / r["Jn"][0]
    return {"J_zero_bias": zero, "J_0p1V": r["Jn"][0], "ratio": rel, "pass": bool(rel < 1e-12)}


def image_force(mu=1e8) -> dict:
    from simws.materials import load
    from simws.materials.semiconductor import intrinsic_density
    from simws.tcad import cryo as C
    r = _diode(mu=mu, image_force=True, A_p=0.0)
    T = 300.0
    kT = K_B_EV * T
    mat = load("silicon")
    Nc, _, _ = C.band_params(mat, T, intrinsic_density(mat, T))
    eps = mat.get("eps_r", "dimensionless") * 8.8541878128e-14
    xi = kT * math.log(Nc / N_D)
    rows, worst = [], 0.0
    for V, J, Es_dev in zip(r["V"], r["Jn"], r["Es"]):
        Vbb = PHI_B - xi - V - kT                                   # depletion approximation with the kT tail
        Es = math.sqrt(2 * Q_E * N_D * Vbb / eps)
        dphi = math.sqrt(Q_E * Es / (4 * math.pi * eps))
        Jref = A_N * T * T * math.exp(-(PHI_B - dphi) / kT) * (math.exp(V / kT) - 1)
        rel = J / Jref - 1
        worst = max(worst, abs(rel))
        rows.append({"V": V, "J": J, "J_closed_form": Jref, "rel_diff": rel, "dphi_meV": 1e3 * dphi,
                     "Es_device": Es_dev, "Es_depletion": Es})
    return {"rows": rows, "worst_rel_diff": worst, "pass": bool(worst < 0.02)}


def tunnel_device(N=1e18, mu=1e8) -> dict:
    """Device current with tunnelling vs TE x (1 + Gamma) evaluated (numpy) at the device's own interface field:
    checks the DEVSIM expression, the field iteration and the contact bookkeeping together (gate 1e-3)."""
    from simws.tcad.contacts import e00, tunnel_gamma
    r = _diode(mu=mu, N=N, tunneling=True, A_p=0.0, Vs=(0.05, 0.1, 0.15))
    T, kT = 300.0, K_B_EV * 300.0
    eps = r["used"]["eps_F_cm"] if "eps_F_cm" in r["used"] else 11.7 * 8.8541878128e-14
    E = e00(N, 0.26, eps / 8.8541878128e-14)
    rows, worst = [], 0.0
    for V, J, Es in zip(r["V"], r["Jn"], r["Es"]):
        Ub = min(eps * Es * Es / (2 * Q_E * N), PHI_B)
        Jref = A_N * T * T * math.exp(-PHI_B / kT) * (1 + float(tunnel_gamma(Ub, kT, E)[0])) * (math.exp(V / kT) - 1)
        rel = J / Jref - 1
        worst = max(worst, abs(rel))
        rows.append({"V": V, "J": J, "J_ref": Jref, "rel_diff": rel, "Ub_eV": Ub})
    return {"N": N, "rows": rows, "worst_rel_diff": worst, "pass": bool(worst < 1e-3)}


def tunnel_quadrature() -> dict:
    from simws.tcad.contacts import e00, tunnel_gamma, tunnel_gamma_reference
    kT, worst, rows = K_B_EV * 300.0, 0.0, []
    for N in (1e16, 1e17, 1e18, 1e19):
        E = e00(N, 0.26, 11.7)
        for Ub in (0.2, 0.4, 0.6):
            a, b = float(tunnel_gamma(Ub, kT, E)[0]), tunnel_gamma_reference(Ub, kT, E)
            worst = max(worst, abs(a / b - 1))
            rows.append({"N": N, "Ub": Ub, "gamma": a, "reference": b})
    return {"rows": rows, "worst_rel_diff": worst, "pass": bool(worst < 1e-4)}


def padovani_stratton() -> dict:
    """Thermionic + tunnelling (ours) vs Padovani-Stratton TFE, forward 0.05-0.15 V, Si, phi_B 0.70 eV."""
    from simws.tcad.contacts import e00, tunnel_gamma
    kT, T, Nc = K_B_EV * 300.0, 300.0, 2.86e19
    rows = []
    for N in (1e17, 3e17, 1e18, 3e18, 1e19):
        E = e00(N, 0.26, 11.7)
        xi = kT * math.log(Nc / N)
        E0 = E / math.tanh(E / kT)
        for V in (0.05, 0.15):
            ours = (1 + float(tunnel_gamma(PHI_B - xi - V, kT, E)[0]))
            Js = A_N * T * math.sqrt(math.pi * E * (PHI_B - V - xi)) / (K_B_EV * math.cosh(E / kT)) \
                * math.exp(-xi / kT - (PHI_B - xi) / E0)
            te = A_N * T * T * math.exp(-PHI_B / kT) * (math.exp(V / kT) - 1)
            rows.append({"N": N, "kT_over_E00": kT / E, "V": V, "ours_over_TE": ours,
                         "PS_over_TE": Js * math.exp(V / E0) / te, "ours_over_PS": ours * te / (Js * math.exp(V / E0))})
    tfe = [r["ours_over_PS"] for r in rows if 1.0 <= r["kT_over_E00"] <= 3.0]
    return {"rows": rows, "tfe_regime_ratio_range": [min(tfe), max(tfe)],
            "note": "Padovani-Stratton is a saddle-point approximation valid near kT ~ E00; the spread is model-form"}


def run_all() -> dict:
    res = {"te_closed_form": te_closed_form(), "detailed_balance": detailed_balance(), "image_force": image_force(),
           "tunnel_quadrature": tunnel_quadrature(), "tunnel_device": tunnel_device(),
           "padovani_stratton": padovani_stratton()}
    res["pass"] = all(res[k]["pass"] for k in ("te_closed_form", "detailed_balance", "image_force", "tunnel_quadrature",
                                                "tunnel_device"))
    return res
