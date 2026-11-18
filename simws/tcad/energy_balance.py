"""Electron energy balance on top of simws.tcad.physics (contract A11.1, A11.2).

    from simws.tcad.energy_balance import add_energy_balance
    setup_region(...)                                  # drift-diffusion device, equilibrium solved
    vs = add_energy_balance("d", "si", mat, T, contacts=["left", "right"])
    solve(...)                                         # the same equilibrium, now with Tn

Model: G. Baccarani and M. R. Wordeman, Solid-State Electron. 28, 407 (1985), energy-balance form (no
kinetic-energy term in tau_w): mu_n = mu_0 T / Tn, tau_w = (3 k mu_0 / (2 q v_s^2)) T Tn / (T + Tn); energy flux
S = -(5/2) u J - kappa grad(Tn), kappa = (5/2)(k^2/q) mu n Tn; div S = J.E - W, W = n q v_s^2 (Tn^2 - T^2) /
(mu_0 T Tn). Holes stay isothermal drift-diffusion. Units as the rest of simws.tcad: cm, V, A, K.
"""
from __future__ import annotations

from typing import Iterable

from simws.units import K_B_EV

TN = "Tn"
EQUATION = "ElectronEnergyEquation"


def add_energy_balance(device: str, region: str, mat, T: float, contacts: Iterable[str]) -> float:
    """Replace the electron current by its energy-balance form and add the electron temperature equation.
    Returns the saturation velocity used (cm/s)."""
    from devsim import contact_equation, edge_from_node_model, equation, node_solution, set_node_value
    from devsim.python_packages.model_create import (CreateContactNodeModel, CreateEdgeModel,
                                                     CreateEdgeModelDerivatives, CreateNodeModel,
                                                     CreateNodeModelDerivative)
    from simws.tcad.physics import saturation_velocity

    vs = float(saturation_velocity(mat, T)[0])
    node_solution(device=device, region=region, name=TN)
    edge_from_node_model(device=device, region=region, node_model=TN)
    set_node_value(device=device, region=region, name=TN, value=float(T))
    k = K_B_EV                                            # V/K: u = k Tn / q in volts
    # edge averages and the hot-electron mobility mu_0 T / Tn
    # intermediate edge models carry their own derivatives: the symbolic differentiation of the current and the
    # flux below refers to them by name (Tn_e:Tn@n0, delta_eb:Potential@n1, ...)
    for name, expr, deps in (
            ("Tn_e", f"0.5*({TN}@n0 + {TN}@n1)", (TN,)),
            ("mu_n_eb", f"0.5*(mu_n_node@n0 + mu_n_node@n1)*{T}/Tn_e", (TN,)),
            ("u_e", f"{k}*Tn_e", (TN,)),
            ("delta_eb", "(Potential@n0 - Potential@n1 + (dEc@n0 - dEc@n1))/u_e", ("Potential", TN))):
        CreateEdgeModel(device, region, name, expr)
        for v in deps:
            CreateEdgeModelDerivatives(device, region, name, expr, v)
    jn = (f"ElectronCharge*mu_n_eb*EdgeInverseLength*(Electrons@n1*{k}*{TN}@n1*B(-delta_eb)"
          f" - Electrons@n0*{k}*{TN}@n0*B(delta_eb))")
    CreateEdgeModel(device, region, "ElectronCurrent", jn)
    for v in ("Potential", "Electrons", "Holes", TN):
        CreateEdgeModelDerivatives(device, region, "ElectronCurrent", jn, v)
    # energy flux (W/cm^2), Joule term on the edge volume, relaxation on the node volume
    kappa = f"2.5*{k}*{k}*ElectronCharge*mu_n_eb*0.5*(Electrons@n0 + Electrons@n1)*Tn_e"
    flux = f"-2.5*u_e*ElectronCurrent - {kappa}*({TN}@n1 - {TN}@n0)*EdgeInverseLength"
    CreateEdgeModel(device, region, "ElectronEnergyFlux", flux)
    for v in ("Potential", "Electrons", "Holes", TN):
        CreateEdgeModelDerivatives(device, region, "ElectronEnergyFlux", flux, v)
    joule = "-ElectronCurrent*ElectricField"
    CreateEdgeModel(device, region, "ElectronJoule", joule)
    for v in ("Potential", "Electrons", "Holes", TN):
        CreateEdgeModelDerivatives(device, region, "ElectronJoule", joule, v)
    loss = f"Electrons*ElectronCharge*{vs * vs}*({TN}*{TN} - {T * T})/(mu_n_node*{T}*{TN})"
    CreateNodeModel(device, region, "ElectronEnergyLoss", loss)
    for v in ("Electrons", TN):
        CreateNodeModelDerivative(device, region, "ElectronEnergyLoss", loss, v)
    equation(device=device, region=region, name=EQUATION, variable_name=TN, node_model="ElectronEnergyLoss",
             edge_model="ElectronEnergyFlux", edge_volume_model="ElectronJoule", variable_update="positive")
    for c in contacts:                                    # ohmic contacts hold the lattice temperature
        name = f"{c}node_tn"
        CreateContactNodeModel(device, c, name, f"{TN} - {T}")
        CreateContactNodeModel(device, c, f"{name}:{TN}", "1")
        contact_equation(device=device, contact=c, name=EQUATION, node_model=name)
    return vs
