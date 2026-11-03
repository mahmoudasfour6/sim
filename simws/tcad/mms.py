"""Method of manufactured solutions for the production Poisson-Boltzmann equation (code verification).

The equation solved is the one every device uses (simws.tcad.physics.setup_region, carriers=False):
    div(eps grad psi) + q (p - n + N + extra) = 0,   n = n_i exp(psi/Vt), p = n_i exp(-psi/Vt)
DEVSIM's expression language has no sin(), so the source is set as node values (exact at every node).
Manufactured solution on a 1D silicon bar of length L with uniform donor doping N:
    psi_m(x) = psi_bi + A sin(pi x / L)
which matches the ohmic-contact value psi_bi at both ends. The source enters through the same `extra_charge` hook
the MOS oxide charge uses:
    extra(x) = eps A (pi/L)^2 sin(pi x/L) / q - (p_m(x) - n_m(x) + N)
so psi_m solves the continuous problem exactly; the discrete error must fall as h^2 (finite volumes, uniform mesh).

    from simws.tcad.mms import poisson_order
    r = poisson_order(levels=4)        # {'h_cm': [...], 'err_V': [...], 'order': [...], 'observed_order': ~2}
"""
from __future__ import annotations

import math

import numpy as np

from simws.units import EPS0_F_PER_CM, K_B_EV, Q_E

_count = [0]


def poisson_error(n_cells: int, L_cm=1e-4, N_cm3=1e16, A_V=0.1, T=300.0, material=None) -> dict:
    from devsim import (add_1d_contact, add_1d_mesh_line, add_1d_region, create_1d_mesh, create_device,
                        delete_device, delete_mesh, finalize_mesh, get_node_model_values)
    import simws.tcad  # noqa: F401  (installs the safe reset_devsim)
    from devsim import reset_devsim
    # DEVSIM's solve() works on every device in memory: a device left by an earlier calculation in the same
    # process (e.g. an HBT) would join this Newton solve and stop it converging
    reset_devsim()

    from simws.materials import load
    from simws.tcad.physics import Physics, setup_region

    mat = material or load("silicon")
    _count[0] += 1
    dev, mesh = f"mms{_count[0]}", f"mmsm{_count[0]}"
    h = L_cm / n_cells
    create_1d_mesh(mesh=mesh)
    add_1d_mesh_line(mesh=mesh, pos=0.0, ps=h, tag="a")
    add_1d_mesh_line(mesh=mesh, pos=L_cm, ps=h, tag="b")
    add_1d_contact(mesh=mesh, name="a", tag="a", material="metal")
    add_1d_contact(mesh=mesh, name="b", tag="b", material="metal")
    add_1d_region(mesh=mesh, material="Si", region="r", tag1="a", tag2="b")
    finalize_mesh(mesh=mesh)
    create_device(mesh=mesh, device=dev)

    phys = Physics(T=T, bgn=False)
    from simws.materials.semiconductor import intrinsic_density
    ni = intrinsic_density(mat, T)
    Vt = K_B_EV * T
    eps = mat.get("eps_r", "dimensionless") * EPS0_F_PER_CM
    # contact value used by setup_region: Vt ln(n0/n_i), n0 = N/2 + sqrt(N^2/4 + n_i^2)
    n0 = 0.5 * N_cm3 + math.sqrt(0.25 * N_cm3 ** 2 + ni ** 2)
    psi_bi = Vt * math.log(n0 / ni)
    k = math.pi / L_cm
    # DEVSIM expressions have no sin(): the source is a node solution filled from numpy (exact at the nodes)
    from devsim import node_solution, set_node_values
    x = np.array(get_node_model_values(device=dev, region="r", name="x"))
    psi_m = psi_bi + A_V * np.sin(k * x)
    src = eps * A_V * k * k / Q_E * np.sin(k * x) - (ni * np.exp(-psi_m / Vt) - ni * np.exp(psi_m / Vt) + N_cm3)
    node_solution(device=dev, region="r", name="MMSSource")
    set_node_values(device=dev, region="r", name="MMSSource", values=src.tolist())
    try:
        setup_region(dev, "r", mat, str(N_cm3), str(N_cm3), phys, {"a": False, "b": False},
                     extra_charge=lambda n_expr: "MMSSource", carriers=False)
        psi = np.array(get_node_model_values(device=dev, region="r", name="Potential"))
    finally:
        delete_device(device=dev)
        delete_mesh(mesh=mesh)
    exact = psi_m
    return {"h_cm": h, "n_cells": n_cells, "err_max_V": float(np.max(np.abs(psi - exact))),
            "err_l2_V": float(np.sqrt(np.mean((psi - exact) ** 2))), "psi_bi_V": psi_bi}


def poisson_order(levels=4, n0=20, **kw) -> dict:
    rows = [poisson_error(n0 * 2 ** i, **kw) for i in range(levels)]
    e = np.array([r["err_max_V"] for r in rows])
    orders = np.log(e[:-1] / e[1:]) / math.log(2.0)
    return {"h_cm": [r["h_cm"] for r in rows], "err_max_V": e.tolist(),
            "err_l2_V": [r["err_l2_V"] for r in rows], "order": orders.tolist(),
            "observed_order": float(orders[-1]), "expected_order": 2.0}
