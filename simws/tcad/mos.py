"""1D MOS capacitor (metal gate / SiO2 / p-Si / back contact) with oxide charge and interface traps.

    from simws.tcad.mos import MOSCase, cv
    r = cv(MOSCase(t_ox_nm=10, NA=1e17, Q_f=5e11, D_it=1e11), V=np.linspace(-3, 3, 61))
    r["C_F_cm2"], r["psi_s_V"], r["V_fb"]

Charges (all sheet densities in cm^-2, positive = positive charge):
  Q_f        fixed charge at the Si/SiO2 interface
  oxide      list of (distance from the Si interface in nm, sheet density) placed at the nearest oxide node
             (radiation-induced oxide-trapped charge, simws.radiation.tid)
  D_it       interface-trap density (cm^-2 eV^-1), uniform over the gap with its charge-neutrality level at
             midgap: Q_it = -q D_it (E_F - E_i)_surface. Quasi-static: the traps follow the Fermi level.
Capacitance is quasi-static (low frequency): C = dQ_gate / dV_gate from the gate contact charge, the inversion
layer responds. High-frequency C-V (minority carriers frozen) is not modelled.

gate_offset_V is the gate work function relative to the silicon intrinsic level, phi_m - (chi + Eg/2) in volts
(0 = midgap metal; n+ poly ~ -0.56 V; p+ poly ~ +0.56 V). The flat-band voltage is reported from the solution
(the bias at which the surface potential equals the bulk potential), not from a formula.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import simws.tcad  # noqa: F401
from simws.tcad.physics import Physics, setup_region
from simws.units import EPS0_F_PER_CM, Q_E

NM = 1e-7


@dataclass
class MOSCase:
    t_ox_nm: float = 10.0
    NA: float = 1e17                      # p-type substrate (cm^-3); negative value -> n-type with |NA| donors
    si_um: float = 2.0
    gate_offset_V: float = 0.0
    Q_f: float = 0.0                      # cm^-2 at the interface
    oxide: List[Tuple[float, float]] = field(default_factory=list)   # (nm from Si interface, cm^-2)
    D_it: float = 0.0                     # cm^-2 eV^-1
    physics: Physics = field(default_factory=lambda: Physics(srh=False, auger=False))


def build(case: MOSCase, material=None, oxide_material=None):
    from devsim import (add_1d_contact, add_1d_interface, add_1d_mesh_line, add_1d_region, contact_equation,
                        create_1d_mesh, create_device, equation, finalize_mesh, get_node_model_values,
                        reset_devsim, set_node_values, set_parameter)
    from devsim.python_packages.model_create import (CreateContactNodeModel, CreateContactNodeModelDerivative,
                                                      CreateEdgeModel, CreateEdgeModelDerivatives, CreateNodeModel,
                                                      CreateSolution)
    from devsim.python_packages.simple_physics import CreateSiliconOxideInterface
    from simws.materials import load

    mat = material or load("silicon")
    ox = oxide_material or load("sio2")
    reset_devsim()
    tox, L = case.t_ox_nm * NM, case.si_um * 1e-4
    create_1d_mesh(mesh="m")
    add_1d_mesh_line(mesh="m", pos=-tox, ps=tox / 40, tag="G")
    add_1d_mesh_line(mesh="m", pos=0.0, ps=0.05 * NM, tag="I")
    add_1d_mesh_line(mesh="m", pos=L, ps=L / 50, tag="B")
    add_1d_region(mesh="m", material="SiO2", region="ox", tag1="G", tag2="I")
    add_1d_region(mesh="m", material="Si", region="si", tag1="I", tag2="B")
    add_1d_interface(mesh="m", tag="I", name="if")
    add_1d_contact(mesh="m", name="gate", tag="G", material="metal")
    add_1d_contact(mesh="m", name="sub", tag="B", material="metal")
    finalize_mesh(mesh="m")
    create_device(mesh="m", device="d")

    # oxide: Poisson with a node charge for trapped oxide charge
    set_parameter(device="d", region="ox", name="Permittivity", value=ox.get("eps_r", "dimensionless") * EPS0_F_PER_CM)
    set_parameter(device="d", region="ox", name="ElectronCharge", value=Q_E)
    CreateSolution("d", "ox", "Potential")
    CreateSolution("d", "ox", "OxideCharge")                         # positive charges / cm^3
    xo = np.array(get_node_model_values(device="d", region="ox", name="x"))
    vo = np.array(get_node_model_values(device="d", region="ox", name="NodeVolume"))
    qo = np.zeros_like(xo)
    for d_nm, N in case.oxide:
        i = int(np.argmin(np.abs(xo + d_nm * NM)))
        qo[i] += N / vo[i]
    set_node_values(device="d", region="ox", name="OxideCharge", values=list(qo))
    CreateNodeModel("d", "ox", "OxideNodeCharge", "-ElectronCharge*OxideCharge")
    for name, expr in (("ElectricField", "(Potential@n0 - Potential@n1)*EdgeInverseLength"),
                       ("PotentialEdgeFlux", "Permittivity*ElectricField")):
        CreateEdgeModel("d", "ox", name, expr)
        CreateEdgeModelDerivatives("d", "ox", name, expr, "Potential")
    equation(device="d", region="ox", name="PotentialEquation", variable_name="Potential",
             node_model="OxideNodeCharge", edge_model="PotentialEdgeFlux")

    # silicon: interface-node sheet charges enter as charge densities on the interface control volume
    def extra(n):
        return f"(FixedSheet + ItMask*(-{case.D_it}*V_t*log({n}/n_ie)))/NodeVolume"

    N = case.NA
    net, tot = (f"{-N}", f"{abs(N)}")

    xs = np.array(get_node_model_values(device="d", region="si", name="x"))
    at_if = (np.abs(xs) < 1e-12).astype(float)
    CreateSolution("d", "si", "ItMask"); CreateSolution("d", "si", "FixedSheet")
    set_node_values(device="d", region="si", name="ItMask", values=list(at_if))
    set_node_values(device="d", region="si", name="FixedSheet", values=list(at_if * case.Q_f))

    def pre():
        CreateSiliconOxideInterface("d", "if")
        set_parameter(device="d", name="gate_bias", value=0.0)
        expr = f"Potential - gate_bias + {case.gate_offset_V}"
        CreateContactNodeModel("d", "gate", "gatenodemodel", expr)
        CreateContactNodeModelDerivative("d", "gate", "gatenodemodel", expr, "Potential")
        contact_equation(device="d", contact="gate", name="PotentialEquation", node_model="gatenodemodel",
                         edge_charge_model="PotentialEdgeFlux")

    # no DC current flows in a MOS capacitor, so the Fermi level is flat: equilibrium (Poisson-Boltzmann) carriers
    # are exact for quasi-static C-V and avoid the ill-conditioned continuity equations in inversion
    used = setup_region("d", "si", mat, net, tot, case.physics, {"sub": False}, extra_charge=extra, pre_solve=pre,
                        carriers=False)
    return used


def _state():
    from devsim import get_contact_charge, get_node_model_values
    xs = np.array(get_node_model_values(device="d", region="si", name="x"))
    psi = np.array(get_node_model_values(device="d", region="si", name="Potential"))
    i0, ib = int(np.argmin(np.abs(xs))), int(np.argmax(xs))
    return (get_contact_charge(device="d", contact="gate", equation="PotentialEquation"), psi[i0] - psi[ib])


def cv(case: MOSCase, V, dV=0.01, material=None):
    """Quasi-static C-V. Returns V, C (F/cm^2), surface potential relative to bulk (V), gate charge (C/cm^2),
    C_ox, and the flat-band voltage (interpolated where psi_s = 0)."""
    from simws.materials import load
    from simws.tcad.solve import ramp_bias
    build(case, material)
    opts = dict(absolute_error=1.0, relative_error=1e-12, maximum_iterations=60)
    V = np.asarray(V, float)
    C, psi, Q = [], [], []
    for v in V:
        q = []
        for vv in (v - dV / 2, v + dV / 2):
            ramp_bias("d", "gate", vv, step=0.1, **opts)
            q.append(_state()[0])
        ramp_bias("d", "gate", v, step=0.1, **opts)
        qg, ps = _state()
        C.append(abs(q[1] - q[0]) / dV); psi.append(ps); Q.append(qg)
    psi = np.array(psi)
    cox = load("sio2").get("eps_r", "dimensionless") * EPS0_F_PER_CM / (case.t_ox_nm * NM)
    o = np.argsort(psi)
    vfb = float(np.interp(0.0, psi[o], V[o])) if psi.min() < 0 < psi.max() else None
    return {"V": V, "C_F_cm2": np.array(C), "psi_s_V": psi, "Q_gate_C_cm2": np.array(Q), "C_ox_F_cm2": cox, "V_fb": vfb}


def oxide_field_MV_cm(case: MOSCase, V_gate, material=None):
    """Mean oxide field (MV/cm, positive = pointing from gate to silicon) at a gate bias, from the solution."""
    from devsim import get_node_model_values
    from simws.tcad.solve import ramp_bias
    build(case, material)
    ramp_bias("d", "gate", V_gate, step=0.1, absolute_error=1e10, relative_error=1e-12, maximum_iterations=60)
    x = np.array(get_node_model_values(device="d", region="ox", name="x"))
    p = np.array(get_node_model_values(device="d", region="ox", name="Potential"))
    o = np.argsort(x)
    return float((p[o][0] - p[o][-1]) / (x[o][-1] - x[o][0]) / 1e6)


def flatband(case: MOSCase, material=None, tol=1e-9, v0=-1.0, v1=0.0):
    """Gate voltage at which the surface potential equals the bulk potential (secant/bisection on the solution)."""
    from devsim import set_parameter, solve
    build(case, material)
    opts = dict(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=60)

    def psi_at(v):
        from simws.tcad.solve import ramp_bias
        ramp_bias("d", "gate", v, step=0.1, absolute_error=1.0, relative_error=1e-12, maximum_iterations=60)
        return _state()[1]

    a, b = v0, v1
    fa, fb = psi_at(a), psi_at(b)
    while fa * fb > 0:                     # widen the bracket
        a, b = a - 1.0, b + 1.0
        fa, fb = psi_at(a), psi_at(b)
    for _ in range(100):
        m = b - fb * (b - a) / (fb - fa) if fb != fa else 0.5 * (a + b)
        if not (min(a, b) < m < max(a, b)):
            m = 0.5 * (a + b)
        fm = psi_at(m)
        if abs(fm) < tol or abs(b - a) < tol:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return m


def gate_voltage_at_surface_potential(case: MOSCase, psi_targets, V_lo=-4.0, V_hi=4.0, n=161, material=None):
    """Invert psi_s(V_gate) on a dense sweep (monotonic), for comparing structures at the same band bending."""
    from simws.tcad.solve import ramp_bias
    build(case, material)
    V = np.linspace(V_lo, V_hi, n)
    psi = []
    for v in V:
        ramp_bias("d", "gate", v, step=0.1, absolute_error=1.0, relative_error=1e-12, maximum_iterations=60)
        psi.append(_state()[1])
    psi = np.array(psi)
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(psi, V)(np.asarray(psi_targets, float))
