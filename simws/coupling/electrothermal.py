"""In-process two-way electro-thermal coupling: DEVSIM drift-diffusion <-> scikit-fem lattice heat equation.

Both solvers live in one Python process and share one triangle mesh (DEVSIM's), so no files and no field
interpolation are needed inside the loop. The fixed-point iteration on the temperature field uses Aitken
dynamic relaxation and stops when max|T_new - T| < tol_K.

Electrical side (DEVSIM, per node / edge):
    n_i(T)  = n_i(300) (T/300)^1.5 exp(Eg(300)/(2 k 300) - Eg(T)/(2 k T)),  Eg(T) Varshni   [node model]
    mu(T)   = mu(300) (T_edge/300)^-zeta                                                     [edge model]
Thermal side (scikit-fem, P1):  -div(k(T) grad T) = H,  k(T) = k300 (300/T)^a,  T = T_sink on the heat-sink
    contact, adiabatic elsewhere. H is the Joule power J.E integrated over each node's control volume, so
    the total heat source equals the electrical power exactly (checked: energy_balance_rel_error).

Known limitation, stated in every result: the thermal voltage V_t stays at 300 K inside DEVSIM's
Scharfetter-Gummel fluxes (DEVSIM edge expressions cannot reference node-valued V_t). This matters for
diffusion-dominated currents at large temperature rises; drift-dominated devices (resistors, the default
case) are affected weakly. Results report `vt_isothermal: true`.
"""
from __future__ import annotations

import simws.tcad  # noqa: F401  (safe devsim.reset_devsim)

from dataclasses import dataclass, field

import numpy as np

from simws.units import K_B_EV

UM = 1e-4


@dataclass
class ResistorCase:
    length_um: float = 10.0
    height_um: float = 1.0
    donors: float = 1.0e17          # cm^-3, uniform n-type
    nx: int = 100
    ny: int = 10
    T_sink: float = 300.0           # K, applied at the cathode contact (x = L)
    biases: list = field(default_factory=lambda: [0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    couple_mobility: bool = True
    couple_ni: bool = True
    couple_k: bool = True
    tol_K: float = 1e-4
    max_iter: int = 60


def _ni_expr(mat, T_sym="TempK"):
    ni = mat.get("n_i", "cm**-3")
    Eg0, a, b = mat.get("varshni_Eg0", "eV"), mat.get("varshni_alpha", "eV/K"), mat.get("varshni_beta", "K")
    Eg = lambda T: f"({Eg0} - {a}*{T}^2/({T} + {b}))"
    return (f"{ni}*({T_sym}/300)^1.5*exp({Eg('300')}/(2*{K_B_EV}*300) - {Eg(T_sym)}/(2*{K_B_EV}*{T_sym}))")


def build_electrical(case: ResistorCase, mat):
    """DEVSIM side of the resistor: returns a namespace with px, py (cm), tris, n_nodes, set_T(T), solve(),
    set_bias(V), nodal_power() (W/cm per node, sums to I*V), contact_current() and edge-node arrays."""
    from types import SimpleNamespace
    from devsim import (add_2d_contact, add_2d_mesh_line, add_2d_region, create_2d_mesh, create_device,
                        edge_model, finalize_mesh, get_contact_current, get_edge_model_values,
                        get_element_node_list, get_node_model_values, node_model, set_node_values,
                        set_parameter, solve)
    from devsim.python_packages import simple_physics as sp
    from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
    import skfem

    from simws.tcad import devsim_materials

    from devsim import reset_devsim
    reset_devsim()                      # each run starts from a clean DEVSIM state
    L, H = case.length_um * UM, case.height_um * UM
    dev, reg = "res", "si"
    create_2d_mesh(mesh="r")
    for i in range(case.nx + 1):
        add_2d_mesh_line(mesh="r", dir="x", pos=i * L / case.nx, ps=L / case.nx)
    for j in range(case.ny + 1):
        add_2d_mesh_line(mesh="r", dir="y", pos=j * H / case.ny, ps=H / case.ny)
    add_2d_mesh_line(mesh="r", dir="x", pos=-1e-8, ps=1e-8)
    add_2d_mesh_line(mesh="r", dir="x", pos=L + 1e-8, ps=1e-8)
    add_2d_region(mesh="r", material="Si", region=reg, xl=0, xh=L)
    add_2d_region(mesh="r", material="Si", region="pad_l", xl=-1e-8, xh=0)
    add_2d_region(mesh="r", material="Si", region="pad_r", xl=L, xh=L + 1e-8)
    for name, xc in (("anode", 0.0), ("cathode", L)):
        add_2d_contact(mesh="r", name=name, material="metal", region=reg, xl=xc, xh=xc, yl=0, yh=H, bloat=1e-10)
    finalize_mesh(mesh="r")
    create_device(mesh="r", device=dev)
    devsim_materials.apply(dev, reg, mat)
    mu_n300, mu_p300 = mat.get("mu_n", "cm**2/V/s"), mat.get("mu_p", "cm**2/V/s")
    zn, zp = mat.get("mu_n_T_exponent", "dimensionless"), mat.get("mu_p_T_exponent", "dimensionless")

    CreateNodeModel(dev, reg, "NetDoping", f"{case.donors}")
    CreateSolution(dev, reg, "TempK")
    n_nodes = len(get_node_model_values(device=dev, region=reg, name="x"))
    set_node_values(device=dev, region=reg, name="TempK", values=[300.0] * n_nodes)
    if case.couple_ni:
        for nm in ("n_i", "n1", "p1"):         # node models override the scalar parameters of the same name
            node_model(device=dev, region=reg, name=nm, equation=_ni_expr(mat))
    Tedge = "(0.5*(TempK@n0 + TempK@n1)/300)"
    edge_model(device=dev, region=reg, name="mu_n_T",
               equation=f"{mu_n300}*{Tedge}^(-{zn})" if case.couple_mobility else f"{mu_n300}")
    edge_model(device=dev, region=reg, name="mu_p_T",
               equation=f"{mu_p300}*{Tedge}^(-{zp})" if case.couple_mobility else f"{mu_p300}")

    sp.CreateSiliconPotentialOnly(dev, reg)
    for c in ("anode", "cathode"):
        set_parameter(device=dev, name=sp.GetContactBiasName(c), value=0.0)
        sp.CreateSiliconPotentialOnlyContact(dev, reg, c)
    solve(type="dc", absolute_error=1.0, relative_error=1e-6, maximum_iterations=100)
    CreateSolution(dev, reg, "Electrons"); CreateSolution(dev, reg, "Holes")
    set_node_values(device=dev, region=reg, name="Electrons", init_from="IntrinsicElectrons")
    set_node_values(device=dev, region=reg, name="Holes", init_from="IntrinsicHoles")
    sp.CreateSiliconDriftDiffusion(dev, reg, mu_n="mu_n_T", mu_p="mu_p_T")
    for c in ("anode", "cathode"):
        sp.CreateSiliconDriftDiffusionAtContact(dev, reg, c)
    opts = dict(absolute_error=1e10, relative_error=1e-8, maximum_iterations=60)
    solve(type="dc", **opts)
    # Joule heat from quasi-Fermi-potential gradients, H = Jn.grad(-phi_n) + Jp.grad(-phi_p). Using the
    # electrostatic field instead would fold temperature-dependent n_i offsets (thermoelectric terms) into the
    # heat source; with quasi-Fermi potentials the discrete heat equals the terminal power I*V exactly.
    from devsim import edge_from_node_model
    node_model(device=dev, region=reg, name="QFn", equation="Potential - V_t*log(Electrons/n_i)")
    node_model(device=dev, region=reg, name="QFp", equation="Potential + V_t*log(Holes/n_i)")
    for nmod in ("QFn", "QFp"):
        edge_from_node_model(device=dev, region=reg, node_model=nmod)
    edge_model(device=dev, region=reg, name="JouleDensity",
               equation="(ElectronCurrent*(QFn@n0 - QFn@n1) + HoleCurrent*(QFp@n0 - QFp@n1))*EdgeInverseLength")

    # --- shared mesh for the thermal solver -------------------------------------------------------------
    px = np.array(get_node_model_values(device=dev, region=reg, name="x"))
    py = np.array(get_node_model_values(device=dev, region=reg, name="y"))
    tris = np.array(get_element_node_list(device=dev, region=reg), dtype=int)
    mesh = skfem.MeshTri(np.vstack([px, py]), tris.T)
    basis = skfem.Basis(mesh, skfem.ElementTriP1())
    sink = np.where(np.isclose(px, L))[0]
    edges = np.array(get_edge_model_values(device=dev, region=reg, name="EdgeCouple"))
    elen = np.array(get_edge_model_values(device=dev, region=reg, name="EdgeLength"))
    # edge -> node index map: DEVSIM exposes edge endpoints only through @n0/@n1 expressions
    from devsim import edge_from_node_model
    for nmod in ("x", "y"):
        edge_from_node_model(device=dev, region=reg, node_model=nmod)
    key = {(round(a / UM, 9), round(b / UM, 9)): i for i, (a, b) in enumerate(zip(px, py))}
    ends = [np.array(get_edge_model_values(device=dev, region=reg, name=n)) / UM for n in ("x@n0", "y@n0", "x@n1", "y@n1")]
    enodes = np.array([[key[(round(a, 9), round(b, 9))], key[(round(c, 9), round(d, 9))]]
                       for a, b, c, d in zip(*ends)], dtype=int)

    def nodal_power():
        jd = np.array(get_edge_model_values(device=dev, region=reg, name="JouleDensity"))
        # Discrete power identity of the finite-volume scheme: each edge carries current I_e = J_e * EdgeCouple
        # across potential drop E_e * EdgeLength, and sum_e I_e dV_e equals the terminal power I*V exactly.
        p_edge = jd * edges * elen                           # W per cm of depth
        p = np.zeros(n_nodes)
        np.add.at(p, enodes[:, 0], 0.5 * p_edge)
        np.add.at(p, enodes[:, 1], 0.5 * p_edge)
        return p

    def contact_current():
        return (get_contact_current(device=dev, contact="anode", equation="ElectronContinuityEquation")
                + get_contact_current(device=dev, contact="anode", equation="HoleContinuityEquation"))

    def set_T(T):
        set_node_values(device=dev, region=reg, name="TempK", values=list(T))

    def set_bias(V):
        set_parameter(device=dev, name=sp.GetContactBiasName("anode"), value=float(V))

    return SimpleNamespace(px=px, py=py, tris=tris, n_nodes=n_nodes, mesh=mesh, basis=basis, sink=sink,
                           set_T=set_T, set_bias=set_bias, solve=lambda: solve(type="dc", **opts),
                           nodal_power=nodal_power, contact_current=contact_current, L=L, H=H)


def thermal_solver(mesh, sink_nodes, case: ResistorCase, mat):
    """scikit-fem P1 steady heat solver on `mesh` (cm): thermal(p_node, T_guess) -> (T, heat_to_sink), where p_node is
    the nodal heat load (W/cm per node, an integral quantity) and T = T_sink on sink_nodes."""
    import skfem
    basis = skfem.Basis(mesh, skfem.ElementTriP1())
    n = mesh.p.shape[1]
    k300, ka = mat.get("thermal_k", "W/cm/K"), mat.get("thermal_k_T_exponent", "dimensionless")

    @skfem.BilinearForm
    def conduction(u, v, w):
        return w["k"] * (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1])

    def thermal(p_node, T_guess):
        T = T_guess.copy()
        for _ in range(50):                                  # inner Picard for k(T)
            kq = k300 * (300.0 / basis.interpolate(T).value) ** (ka if case.couple_k else 0.0)
            K = skfem.asm(conduction, basis, k=kq)
            T_new = skfem.solve(*skfem.condense(K, p_node, x=np.full(n, case.T_sink), D=sink_nodes))
            if np.max(np.abs(T_new - T)) < 1e-9:
                T = T_new
                break
            T = T_new
        heat_to_sink = float(-(K @ T - p_node)[sink_nodes].sum())   # reaction at the Dirichlet nodes
        return T, heat_to_sink
    return thermal


def run(case: ResistorCase = ResistorCase(), material=None, verbose=False):
    from simws.materials import load
    mat = material or load("silicon")
    e = build_electrical(case, mat)
    n_nodes, px, py, tris = e.n_nodes, e.px, e.py, e.tris
    set_T, nodal_power, contact_current = e.set_T, e.nodal_power, e.contact_current
    thermal = thermal_solver(e.mesh, e.sink, case, mat)
    from devsim import solve
    opts = dict(absolute_error=1e10, relative_error=1e-8, maximum_iterations=60)

    results = []
    T = np.full(n_nodes, case.T_sink)
    v_prev = 0.0
    for V in case.biases:
        for vv in np.linspace(v_prev, V, max(2, int(abs(V - v_prev) / 0.25) + 1))[1:]:
            e.set_bias(vv)
            solve(type="dc", **opts)
        v_prev = V
        # isothermal reference at this bias
        set_T(np.full(n_nodes, case.T_sink)); solve(type="dc", **opts)
        I_iso = contact_current()
        set_T(T); solve(type="dc", **opts)
        omega, r_prev, hist = 0.5, None, []
        for it in range(1, case.max_iter + 1):
            P = nodal_power()
            T_new, q_sink = thermal(P, T)
            r = T_new - T
            if r_prev is not None:
                dr = r - r_prev
                omega = float(np.clip(-omega * np.dot(r_prev, dr) / max(np.dot(dr, dr), 1e-300), 0.05, 1.0))
            T = T + omega * r
            r_prev = r
            hist.append(float(np.max(np.abs(r))))
            set_T(T); solve(type="dc", **opts)
            if hist[-1] < case.tol_K:
                break
        I = contact_current()
        P = nodal_power()
        T_fin, q_sink = thermal(P, T)
        res = {
            "bias_V": V, "current_A_per_cm": abs(I), "current_isothermal_A_per_cm": abs(I_iso),
            "power_W_per_cm": abs(I) * V, "joule_W_per_cm": float(P.sum()), "heat_to_sink_W_per_cm": q_sink,
            "T_max_K": float(T.max()), "iterations": len(hist), "converged": hist[-1] < case.tol_K,
            "residual_history_K": hist,
            "energy_balance_rel_error": float(abs(q_sink - P.sum()) / max(P.sum(), 1e-300)),
            "joule_vs_IV_rel_error": float(abs(P.sum() - abs(I) * V) / max(abs(I) * V, 1e-300)),
            "vt_isothermal": True,
        }
        if verbose:
            print(f"V={V:4.2f} V  I={res['current_A_per_cm']:.5f} (iso {res['current_isothermal_A_per_cm']:.5f}) A/cm  "
                  f"Tmax={res['T_max_K']:.2f} K  its={res['iterations']}  energy err={res['energy_balance_rel_error']:.1e}  "
                  f"Joule vs IV err={res['joule_vs_IV_rel_error']:.1e}")
        results.append(res)
    return {"case": case.__dict__, "material": mat.name, "points": results,
            "nodes": {"x_cm": px, "y_cm": py, "triangles": tris, "T_K": T}}


def analytic_max_temperature(power_W_per_cm, case: ResistorCase, k_W_per_cmK):
    """Uniform heating, sink at x = L, adiabatic elsewhere, constant k: T_max = T_sink + q L^2 / (2 k)."""
    L, H = case.length_um * UM, case.height_um * UM
    q = power_W_per_cm / (L * H)
    return case.T_sink + q * L**2 / (2 * k_W_per_cmK)
