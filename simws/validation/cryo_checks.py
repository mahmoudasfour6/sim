"""Closed-form checks of the 0.9a cryogenic physics (simws.tcad.cryo + simws.tcad.physics).

resistor_equilibrium  uniformly doped 1D Si bar, freeze-out + Fermi-Dirac on: the electron density DEVSIM finds
                      equals the local-neutrality solution with the same statistics (gate: 1e-6 relative)
resistor_ohmic        small bias across the bar: I = q n mu A V / L with n from neutrality and mu from Klaassen
                      evaluated at that n (gate: 1e-4 relative; tests transport, Klaassen and contacts together)
statistics_error      the device's Fermi-Dirac approximation (Nilsson inverse) against the exact integral: a
                      model error recorded in the budget, not a pass/fail check
einstein_graded       donors graded 1e15 -> 1e20 cm^-3 (degenerate end) at zero bias: the current must vanish. With
                      Fermi-Dirac densities only the generalized Einstein relation in the SG flux (the ln gamma
                      terms) gives detailed balance; Boltzmann diffusion would drive a current (gate: |J| < 1e-10 of
                      the drift scale q mu n V_t / L)
onsager               three-contact 2D bar at equilibrium: the small-signal conductance matrix is symmetric,
                      G_ab = G_ba (Onsager reciprocity; central differences, gate 1e-6 relative)
"""
from __future__ import annotations

import numpy as np

from simws.units import Q_E

_count = [0]


def _bar(T, ND=0.0, NA=0.0, L_um=1.0, n_cells=40, bias=0.0, donor="P", acceptor="B", fd=True, ii=True,
         mobility="klaassen", ii_model="altermatt", ii_N_crit=None, ii_E_const=None):
    import simws.tcad  # noqa: F401
    from devsim import (add_1d_contact, add_1d_mesh_line, add_1d_region, create_1d_mesh, create_device,
                        finalize_mesh, get_node_model_values, reset_devsim)
    from simws.materials import load
    from simws.tcad.physics import Physics, contact_current, setup_region
    from simws.tcad.solve import ramp_bias
    reset_devsim()
    _count[0] += 1
    L = L_um * 1e-4
    create_1d_mesh(mesh="m")
    add_1d_mesh_line(mesh="m", pos=0.0, ps=L / n_cells, tag="a")
    add_1d_mesh_line(mesh="m", pos=L, ps=L / n_cells, tag="b")
    add_1d_contact(mesh="m", name="a", tag="a", material="metal")
    add_1d_contact(mesh="m", name="b", tag="b", material="metal")
    add_1d_region(mesh="m", material="Si", region="r", tag1="a", tag2="b")
    finalize_mesh(mesh="m")
    create_device(mesh="m", device="d")
    mat = load("silicon")
    phys = Physics(T=T, bgn=False, srh=False, auger=False, incomplete_ionization=ii, fermi_dirac=fd,
                   mobility=mobility, donor_species=donor, acceptor_species=acceptor, ii_model=ii_model,
                   ii_N_crit=ii_N_crit, ii_E_const=ii_E_const)
    used = setup_region("d", "r", mat, f"{ND - NA!r}", f"{ND + NA!r}", phys, {"a": False, "b": False})
    out = {"used": used, "n": np.array(get_node_model_values(device="d", region="r", name="Electrons")),
           "p": np.array(get_node_model_values(device="d", region="r", name="Holes"))}
    if bias:
        opts = dict(absolute_error=1e10, relative_error=1e-10, maximum_iterations=60)
        ramp_bias("d", "a", bias, step=bias, **opts)
        if mobility == "klaassen":
            from simws.tcad.physics import relax_mobility
            relax_mobility("d", "r", opts)
        out["I_A_per_cm2"] = contact_current("d", "a")
        out["mu_n"] = np.array(get_node_model_values(device="d", region="r", name="mu_n_node"))
    return out


def resistor_equilibrium(Ts=(300.0, 150.0, 77.0), dopings=(1e15, 1e16, 1e17, 1e18, 1e19)) -> dict:
    from simws.materials import load
    from simws.tcad import cryo as C
    mat = load("silicon")
    rows, worst = [], 0.0
    for T in Ts:
        for N in dopings:
            r = _bar(T, ND=N)
            u = r["used"]
            n_ref, p_ref, _, _ = C.device_equilibrium(mat, N, 0.0, u["n_i_cm3"], T, u["Nc_eff_cm3"], u["Nv_eff_cm3"])
            rel = float(np.max(np.abs(r["n"] / n_ref[0] - 1)))
            worst = max(worst, rel)
            rows.append({"T_K": T, "ND": N, "n_devsim": float(r["n"][len(r["n"]) // 2]), "n_neutral": float(n_ref[0]),
                         "ionized": float(n_ref[0] / N), "rel_diff": rel})
    return {"rows": rows, "worst_rel_diff": worst, "pass": bool(worst < 1e-6)}


def resistor_ohmic(Ts=(300.0, 77.0), dopings=(1e16, 1e18), V=1e-3, L_um=1.0) -> dict:
    from simws.materials import load
    from simws.tcad import cryo as C
    mat = load("silicon")
    rows, worst = [], 0.0
    for T in Ts:
        for N in dopings:
            r = _bar(T, ND=N, bias=V, L_um=L_um)
            u = r["used"]
            n, p, npr, _ = C.device_equilibrium(mat, N, 0.0, u["n_i_cm3"], T, u["Nc_eff_cm3"], u["Nv_eff_cm3"])
            Dp = N * C.ionized_fraction(C.ii_params(mat, "P"), N, npr, u["Nc_eff_cm3"], T)
            mu = float(C.klaassen_mobility(mat, "n", 0.0, Dp, n, p, T, "P")[0])
            # degenerate carriers: conduction current is q mu n grad(phi_n); the small-bias resistance is unchanged
            I_ref = Q_E * float(n[0]) * mu * V / (L_um * 1e-4)
            rel = abs(abs(r["I_A_per_cm2"]) / I_ref - 1)
            worst = max(worst, rel)
            rows.append({"T_K": T, "ND": N, "I_devsim": float(r["I_A_per_cm2"]), "I_ref": I_ref, "mu_n": mu,
                         "rel_diff": float(rel)})
    return {"rows": rows, "worst_rel_diff": float(worst), "pass": bool(worst < 1e-4)}


def statistics_error(us=np.geomspace(1e-6, 100, 61)) -> dict:
    from simws.tcad import cryo as C
    exact = C.fd12_exact(C.eta_from_u_nilsson(us))
    rel = exact / us - 1
    return {"max_density_error": float(np.max(np.abs(rel))), "u_at_max": float(us[int(np.argmax(np.abs(rel)))])}


def einstein_graded(Ts=(300.0, 77.0), L_um=1.0, n_cells=200) -> dict:
    import simws.tcad  # noqa: F401
    from devsim import (add_1d_contact, add_1d_mesh_line, add_1d_region, create_1d_mesh, create_device,
                        finalize_mesh, get_node_model_values, reset_devsim)
    from simws.materials import load
    from simws.tcad.physics import Physics, contact_current, setup_region
    rows, worst = [], 0.0
    for T in Ts:
        reset_devsim()
        L = L_um * 1e-4
        create_1d_mesh(mesh="m")
        add_1d_mesh_line(mesh="m", pos=0.0, ps=L / n_cells, tag="a")
        add_1d_mesh_line(mesh="m", pos=L, ps=L / n_cells, tag="b")
        add_1d_contact(mesh="m", name="a", tag="a", material="metal")
        add_1d_contact(mesh="m", name="b", tag="b", material="metal")
        add_1d_region(mesh="m", material="Si", region="r", tag1="a", tag2="b")
        finalize_mesh(mesh="m")
        create_device(mesh="m", device="d")
        phys = Physics(T=T, bgn=False, srh=False, auger=False, incomplete_ionization=True, fermi_dirac=True,
                       mobility="klaassen")
        ND = f"1e15*pow(1e5, x/{L!r})"
        used = setup_region("d", "r", load("silicon"), ND, ND, phys, {"a": False, "b": False})
        n = np.array(get_node_model_values(device="d", region="r", name="Electrons"))
        mu = np.array(get_node_model_values(device="d", region="r", name="mu_n_node"))
        J = max(abs(contact_current("d", "a")), abs(contact_current("d", "b")))
        scale = Q_E * float(np.max(mu * n)) * used["V_t"] / L
        rel = J / scale
        worst = max(worst, rel)
        rows.append({"T_K": T, "J_A_cm2": float(J), "drift_scale_A_cm2": scale, "rel": float(rel),
                     "u_max": float(n.max() / used["Nc_eff_cm3"])})
    return {"rows": rows, "worst_rel": float(worst), "pass": bool(worst < 1e-10)}


def onsager(Ts=(300.0, 77.0), N=1e17, W_um=1.0, H_um=0.5) -> dict:
    """G_ij = dI_i/dV_j at equilibrium by central differences (delta = 1e-4 V_t); symmetric to 1e-6."""
    import simws.tcad  # noqa: F401
    from devsim import add_2d_contact, create_device, finalize_mesh, reset_devsim, set_parameter, solve
    from simws.materials import load
    from simws.tcad.mixed_mode import _mesh_2d
    from simws.tcad.physics import Physics, contact_current, relax_mobility, setup_region
    UM = 1e-4
    names = ("a", "b", "c")
    rows, worst = [], 0.0
    for T in Ts:
        reset_devsim()
        _mesh_2d(W_um, H_um, [(0, 0.05), (0.4 * W_um, 0.02), (0.6 * W_um, 0.02), (W_um, 0.05)],
                 [(0, 0.01), (H_um, 0.05)])
        add_2d_contact(mesh="m", name="a", material="metal", region="si", xl=0, xh=0, yl=0, yh=H_um * UM, bloat=1e-10)
        add_2d_contact(mesh="m", name="b", material="metal", region="si", xl=W_um * UM, xh=W_um * UM, yl=0,
                       yh=H_um * UM, bloat=1e-10)
        add_2d_contact(mesh="m", name="c", material="metal", region="si", xl=0.4 * W_um * UM, xh=0.6 * W_um * UM,
                       yl=0, yh=0, bloat=1e-10)
        finalize_mesh(mesh="m")
        create_device(mesh="m", device="d")
        phys = Physics(T=T, bgn=False, srh=True, auger=False, tau_n=1e-7, tau_p=1e-7, incomplete_ionization=True,
                       fermi_dirac=True, mobility="klaassen")
        used = setup_region("d", "si", load("silicon"), f"{N!r}", f"{N!r}", phys, {k: False for k in names})
        opts = dict(type="dc", absolute_error=1e10, relative_error=1e-12, maximum_iterations=60)
        d = 1e-4 * used["V_t"]
        G = np.zeros((3, 3))
        for j, cj in enumerate(names):
            I = []
            for sgn in (1, -1):
                set_parameter(device="d", name=f"{cj}_bias", value=sgn * d)
                solve(**opts)
                relax_mobility("d", "si", dict(absolute_error=1e10, relative_error=1e-12, maximum_iterations=60),
                               tol=1e-12)
                I.append([contact_current("d", ci) for ci in names])
            set_parameter(device="d", name=f"{cj}_bias", value=0.0)
            G[:, j] = (np.array(I[0]) - np.array(I[1])) / (2 * d)
        solve(**opts)
        asym = max(abs(G[i, j] - G[j, i]) / max(abs(G[i, j]), abs(G[j, i])) for i in range(3) for j in range(3) if i < j)
        worst = max(worst, asym)
        rows.append({"T_K": T, "G_S_per_cm": G.tolist(), "max_rel_asymmetry": float(asym),
                     "row_sums_rel": float(np.max(np.abs(G.sum(axis=0))) / np.max(np.abs(G)))})
    return {"rows": rows, "worst_rel": float(worst), "pass": bool(worst < 1e-6)}


def run_all() -> dict:
    res = {"resistor_equilibrium": resistor_equilibrium(), "resistor_ohmic": resistor_ohmic(),
           "statistics_error": statistics_error(), "einstein_graded": einstein_graded(), "onsager": onsager()}
    res["pass"] = all(res[k]["pass"] for k in ("resistor_equilibrium", "resistor_ohmic", "einstein_graded", "onsager"))
    return res
