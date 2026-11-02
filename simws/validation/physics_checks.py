"""Checks of simws.tcad.physics against closed-form results (1D devices, seconds each).

equilibrium_bgn      graded n+/p junction (1e20 -> 1e16) with band-gap narrowing at zero bias: terminal current
                     must vanish and n p = n_ie^2 at every node (fails if BGN entered only n_i, not the currents)
mobility_resistors   uniform resistors at several dopings: R = L / (q N mu_Masetti(N)) and, at 400 K, the
                     lattice-mobility scaling (T/300)^-zeta at low doping
intrinsic_vs_measured n_i(T) against the Misiakos-Tsamakis measured fit, 250-400 K (within 8 %)
lifetime_bars        long uniformly illuminated bars: excess minority density in the middle = G tau_eff with
                     1/tau_eff = 1/tau_SRH(N) + C N^2 (SRH + Auger, low injection)
"""
from __future__ import annotations

import math

import numpy as np

import simws.tcad  # noqa: F401
from simws.materials import load
from simws.tcad.physics import Physics, contact_current, masetti_numpy, setup_region
from simws.units import K_B_EV, Q_E


def _mesh(length_cm, points, name="m"):
    from devsim import add_1d_contact, add_1d_mesh_line, add_1d_region, create_1d_mesh, create_device, finalize_mesh
    from devsim import reset_devsim
    reset_devsim()
    create_1d_mesh(mesh=name)
    for pos, ps, tag in points:
        add_1d_mesh_line(mesh=name, pos=pos, ps=ps, **({"tag": tag} if tag else {}))
    add_1d_contact(mesh=name, name="left", tag="L", material="metal")
    add_1d_contact(mesh=name, name="right", tag="R", material="metal")
    add_1d_region(mesh=name, material="Si", region="si", tag1="L", tag2="R")
    finalize_mesh(mesh=name)
    create_device(mesh=name, device="d")


def equilibrium_bgn(mat=None):
    from devsim import get_node_model_values
    mat = mat or load("silicon")
    L = 2e-4
    _mesh(L, [(0, 1e-7, "L"), (0.3e-4, 1e-7, None), (L, 1e-6, "R")])
    nd = "1e20*exp(-pow(x/0.1e-4, 2))"                            # Gaussian n+ implant, 0.1 um (DEVSIM: -a^2 = (-a)^2)
    setup_region("d", "si", mat, f"{nd} - 1e16", f"{nd} + 1e16", Physics(), {"left": False, "right": False})
    n = np.array(get_node_model_values(device="d", region="si", name="Electrons"))
    p = np.array(get_node_model_values(device="d", region="si", name="Holes"))
    nie = np.array(get_node_model_values(device="d", region="si", name="n_ie"))
    dEg = np.array(get_node_model_values(device="d", region="si", name="DeltaEg"))
    I = abs(contact_current("d", "left"))
    np_dev = float(np.max(np.abs(n * p / nie ** 2 - 1)))
    # scale of the individual drift and diffusion fluxes that must cancel: q mu Vt n_max / h_min
    scale = Q_E * 100.0 * 0.02585 * 1e20 / 1e-7
    return {"current_A_per_cm2": I, "flux_scale_A_per_cm2": scale, "current_over_scale": I / scale,
            "np_over_nie2_worst_rel_dev": np_dev, "DeltaEg_max_meV": float(dEg.max() * 1e3),
            "pass": bool(I / scale < 1e-12 and np_dev < 1e-6)}


def mobility_resistors(mat=None):
    from devsim import get_node_model_values, set_parameter, solve
    mat = mat or load("silicon")
    out, worst = [], 0.0
    L, V = 1e-3, 0.01
    for N, T in ((1e14, 300.0), (1e17, 300.0), (1e19, 300.0), (1e14, 400.0)):
        _mesh(L, [(0, 1e-5, "L"), (L, 1e-5, "R")])
        setup_region("d", "si", mat, f"{N}", f"{N}", Physics(T=T, srh=False, auger=False),
                     {"left": False, "right": False})
        set_parameter(device="d", name="left_bias", value=V)
        solve(type="dc", absolute_error=1e10, relative_error=1e-12, maximum_iterations=40)
        I = abs(contact_current("d", "left"))
        mu = float(masetti_numpy(mat, N, T, "n"))
        mup = float(masetti_numpy(mat, N, T, "p"))
        nie = get_node_model_values(device="d", region="si", name="n_ie")[0]
        n0 = 0.5 * (N + math.sqrt(N * N + 4 * nie * nie)); p0 = nie * nie / n0   # thermal carriers matter at 400 K
        I_an = Q_E * (n0 * mu + p0 * mup) * V / L
        rel = I / I_an - 1
        worst = max(worst, abs(rel))
        out.append({"N_cm3": N, "T_K": T, "mu_masetti": mu, "I_rel_diff": rel})
    scaling = out[3]["mu_masetti"] / out[0]["mu_masetti"]
    zeta = mat.get("mu_n_T_exponent", "dimensionless")
    return {"cases": out, "worst_rel_diff": worst, "mu_ratio_400_300": scaling,
            "mu_ratio_expected": (400 / 300) ** -zeta,
            "pass": bool(worst < 1e-3 and abs(scaling / (400 / 300) ** -zeta - 1) < 0.02)}


def lifetime_bars(mat=None):
    from devsim import get_node_model_values
    mat = mat or load("silicon")
    G = 1e18
    out, worst = [], 0.0
    for NA in (1e17, 1e19):
        L = 400e-4 if NA < 1e18 else 40e-4
        _mesh(L, [(0, L / 2000, "L"), (L, L / 2000, "R")])
        setup_region("d", "si", mat, f"-{NA}", f"{NA}", Physics(bgn=False), {"left": False, "right": False},
                     extra_generation=f"{G}")
        x = np.array(get_node_model_values(device="d", region="si", name="x"))
        n = np.array(get_node_model_values(device="d", region="si", name="Electrons"))
        nie = np.array(get_node_model_values(device="d", region="si", name="n_ie"))
        dn = float(np.interp(L / 2, x, n) - nie[0] ** 2 / NA)
        tn = mat.get("srh_tau_max_n", "s") / (1 + (NA / mat.get("srh_N_ref", "cm**-3")) ** mat.get("srh_gamma", "dimensionless"))
        rate = 1 / tn + mat.get("auger_Cp", "cm**6/s") * NA ** 2      # p-type, low injection
        # ohmic contacts at both ends pin dn = 0: dn(L/2) = G tau (1 - 1/cosh(L / 2 L_D)), L_D = sqrt(D tau)
        LD = math.sqrt(K_B_EV * 300.0 * float(masetti_numpy(mat, NA, 300.0, "n")) / rate)
        dn_an = G / rate * (1 - 1 / math.cosh(L / 2 / LD))
        rel = dn / dn_an - 1
        worst = max(worst, abs(rel))
        out.append({"NA_cm3": NA, "dn_sim": dn, "dn_analytic": dn_an, "rel_diff": rel, "L_over_LD": L / LD,
                    "auger_share": mat.get("auger_Cp", "cm**6/s") * NA ** 2 / rate})
    return {"cases": out, "worst_rel_diff": worst, "pass": bool(worst < 1e-2)}


def intrinsic_vs_measured(mat=None):
    """n_i(T) model vs the measured fit of Misiakos & Tsamakis, J. Appl. Phys. 74, 3293 (1993):
    n_i = 5.29e19 (T/300)^2.54 exp(-6726/T). The T^1.5 density-of-states scaling drifts low above 400 K."""
    from simws.tcad.physics import intrinsic_density
    mat = mat or load("silicon")
    rows = []
    for T in (250.0, 300.0, 350.0, 400.0):
        ref = 5.29e19 * (T / 300) ** 2.54 * math.exp(-6726 / T)
        rows.append({"T_K": T, "model": intrinsic_density(mat, T), "measured_fit": ref,
                     "rel_diff": intrinsic_density(mat, T) / ref - 1})
    worst = max(abs(r["rel_diff"]) for r in rows)
    return {"cases": rows, "worst_rel_diff": worst, "pass": bool(worst < 0.08)}


def run_all():
    res = {"equilibrium_bgn": equilibrium_bgn(), "mobility_resistors": mobility_resistors(),
           "lifetime_bars": lifetime_bars(), "intrinsic_vs_measured": intrinsic_vs_measured()}
    res["pass"] = all(v["pass"] for v in res.values())
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), indent=1, default=float))
