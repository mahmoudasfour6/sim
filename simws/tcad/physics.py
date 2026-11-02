"""Full drift-diffusion physics for a DEVSIM silicon region, beyond DEVSIM's simple_physics.

    from simws.tcad.physics import Physics, setup_region
    phys = Physics(T=350.0)                                   # every model evaluated at the lattice temperature T
    setup_region(dev, "si", mat, net_doping="...", total_doping="...", phys=phys,
                 contacts={"anode": False, "cathode": False})  # name -> is_circuit

Models (each switchable in `Physics`; parameters and references in simws/materials/data/silicon.yaml):
  doping_mobility  Masetti (1983): mu(N, T), lattice part mu_max (T/300)^-zeta
  field_mobility   Caughey-Thomas velocity saturation with v_sat(T) (Canali 1975), on top of mu(N, T)
  srh              Shockley-Read-Hall with Scharfetter lifetime tau(N) (T/300)^a
  auger            Dziewior-Schmid (1977) Auger recombination
  bgn              Slotboom band-gap narrowing (Klaassen 1992 parameters)
  impact           Van Overstraeten-de Man impact ionization with the optical-phonon temperature factor
  n_i(T)           Varshni band gap + T^1.5 density-of-states scaling

Band-gap narrowing is applied consistently, not only to n_i: the gap shrinks symmetrically (dEg/2 on each band),
so electrons see the effective potential psi + dEg/2 and holes psi - dEg/2 in the Scharfetter-Gummel currents,
the equilibrium satisfies n p = n_ie^2 = n_i^2 exp(dEg/kT) everywhere, and the ohmic contacts use n_ie. A device
in equilibrium therefore carries no current even with strongly graded doping (checked in simws.validation).

Temperature is uniform within a run (isothermal). Self-consistent self-heating is in simws.coupling.electrothermal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import simws.tcad  # noqa: F401  (safe reset_devsim, solver selection)
from simws.units import EPS0_F_PER_CM, K_B_EV, Q_E


@dataclass
class Physics:
    T: float = 300.0
    doping_mobility: bool = True
    field_mobility: bool = False
    srh: bool = True
    auger: bool = True
    bgn: bool = True
    impact: bool = False
    tau_n: Optional[float] = None        # constant lifetimes override the Scharfetter model (s)
    tau_p: Optional[float] = None
    mu_n: Optional[float] = None         # constant mobilities override the Masetti model (cm^2/Vs)
    mu_p: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)


def EnsureEdgeFromNodeModelExists(device, region, nodemodel):
    """Create <model>@n0 / <model>@n1 edge models. Own version: in older DEVSIM releases the helper of the same
    name tests `"m@n0" and "m@n1"` (always true) and never creates them."""
    from devsim import edge_from_node_model, get_edge_model_list
    have = get_edge_model_list(device=device, region=region)
    if f"{nodemodel}@n0" not in have or f"{nodemodel}@n1" not in have:
        edge_from_node_model(device=device, region=region, node_model=nodemodel)


def intrinsic_density(mat, T):
    """n_i(T) from n_i(300 K), Varshni Eg(T) and Nc Nv ~ T^3."""
    ni = mat.get("n_i", "cm**-3")
    Eg0, a, b = mat.get("varshni_Eg0", "eV"), mat.get("varshni_alpha", "eV/K"), mat.get("varshni_beta", "K")
    Eg = lambda t: Eg0 - a * t * t / (t + b)
    return ni * (T / 300.0) ** 1.5 * math.exp(Eg(300.0) / (2 * K_B_EV * 300.0) - Eg(T) / (2 * K_B_EV * T))


def saturation_velocity(mat, T):
    return (mat.get("vsat_n_300", "cm/s") * (300.0 / T) ** mat.get("vsat_n_T_exponent", "dimensionless"),
            mat.get("vsat_p_300", "cm/s") * (300.0 / T) ** mat.get("vsat_p_T_exponent", "dimensionless"))


def impact_gamma(mat, T):
    hw = mat.get("ii_hbar_omega", "eV")
    return math.tanh(hw / (2 * K_B_EV * 300.0)) / math.tanh(hw / (2 * K_B_EV * T))


def masetti_numpy(mat, N, T=300.0, carrier="n"):
    """Masetti mobility (cm^2/Vs) for total doping N (cm^-3), for checks and plots."""
    import numpy as np
    c = carrier
    g = lambda k: mat.get(f"masetti_{c}_{k}", "cm**2/V/s" if k.startswith("mu") else
                          ("cm**-3" if k in ("Pc", "Cr", "Cs") else "dimensionless"))
    zeta = mat.get(f"mu_{c}_T_exponent", "dimensionless")
    mu_L = mat.get(f"mu_{c}", "cm**2/V/s") * (T / 300.0) ** (-zeta)
    N = np.maximum(np.asarray(N, float), 1.0)
    Pc = g("Pc")
    first = g("mumin1") * (np.exp(-Pc / N) if Pc > 0 else 1.0)
    return first + (mu_L - g("mumin2")) / (1 + (N / g("Cr")) ** g("alpha")) - g("mu1") / (1 + (g("Cs") / N) ** g("beta"))


def _masetti_expr(mat, T, c):
    g = lambda k, u: mat.get(f"masetti_{c}_{k}", u)
    zeta = mat.get(f"mu_{c}_T_exponent", "dimensionless")
    mu_L = mat.get(f"mu_{c}", "cm**2/V/s") * (T / 300.0) ** (-zeta)
    Pc = g("Pc", "cm**-3")
    first = f"{g('mumin1', 'cm**2/V/s')}" + (f"*exp(-{Pc}/Ntot)" if Pc > 0 else "")
    return (f"({first} + ({mu_L} - {g('mumin2', 'cm**2/V/s')})/(1 + pow(Ntot/{g('Cr', 'cm**-3')}, "
            f"{g('alpha', 'dimensionless')})) - {g('mu1', 'cm**2/V/s')}/(1 + pow({g('Cs', 'cm**-3')}/Ntot, "
            f"{g('beta', 'dimensionless')})))")


def setup_region(device, region, mat, net_doping: str, total_doping: str, phys: Physics = Physics(),
                 contacts: Optional[Dict[str, bool]] = None, extra_generation: Optional[str] = None,
                 solve_equilibrium=True):
    """Create every model and equation on `region`, the contacts in `contacts` (name -> is_circuit), and
    (optionally) solve the zero-bias equilibrium. `extra_generation` is a node expression in pairs/cm^3/s added
    to both continuity equations (e.g. an ion track). Bias parameters are '<contact>_bias' (simple_physics
    naming); set them with set_parameter or simws.tcad.solve.ramp_bias. Returns a dict of the values used."""
    from devsim import equation, set_node_values, set_parameter, solve
    from devsim.python_packages.model_create import (CreateEdgeModel, CreateEdgeModelDerivatives, CreateNodeModel,
                                                      CreateNodeModelDerivative, CreateSolution)

    contacts = contacts or {}
    T = phys.T
    V_t = K_B_EV * T
    n_i = intrinsic_density(mat, T)
    used = {"T_K": T, "n_i_cm3": n_i, "V_t": V_t}
    for name, val in {"Permittivity": mat.get("eps_r", "dimensionless") * EPS0_F_PER_CM, "ElectronCharge": Q_E,
                      "V_t": V_t, "n_i": n_i, "T": T}.items():
        set_parameter(device=device, region=region, name=name, value=val)

    # doping, band-gap narrowing, effective intrinsic density
    CreateNodeModel(device, region, "NetDoping", net_doping)
    CreateNodeModel(device, region, "Ntot", f"max({total_doping}, 1.0)")
    if phys.bgn:
        E0, Nr, C = mat.get("bgn_E0", "eV"), mat.get("bgn_N_ref", "cm**-3"), mat.get("bgn_C", "dimensionless")
        CreateNodeModel(device, region, "DeltaEg", f"{E0}*(log(Ntot/{Nr}) + pow(log(Ntot/{Nr})^2 + {C}, 0.5))")
    else:
        CreateNodeModel(device, region, "DeltaEg", "0")
    CreateNodeModel(device, region, "n_ie", "n_i*exp(DeltaEg/(2*V_t))")

    # potential-only start: n = n_i exp((psi + dEg/2)/Vt), p = n_i exp(-(psi - dEg/2)/Vt)
    CreateSolution(device, region, "Potential")
    for name, expr in (("IntrinsicElectrons", "n_i*exp((Potential + 0.5*DeltaEg)/V_t)"),
                       ("IntrinsicHoles", "n_i*exp(-(Potential - 0.5*DeltaEg)/V_t)"),
                       ("PotentialIntrinsicCharge", "-ElectronCharge*kahan3(IntrinsicHoles, -IntrinsicElectrons, NetDoping)")):
        CreateNodeModel(device, region, name, expr)
        CreateNodeModelDerivative(device, region, name, expr, "Potential")
    for name, expr in (("ElectricField", "(Potential@n0 - Potential@n1)*EdgeInverseLength"),
                       ("PotentialEdgeFlux", "Permittivity*ElectricField")):
        CreateEdgeModel(device, region, name, expr)
        CreateEdgeModelDerivatives(device, region, name, expr, "Potential")
    equation(device=device, region=region, name="PotentialEquation", variable_name="Potential",
             node_model="PotentialIntrinsicCharge", edge_model="PotentialEdgeFlux", variable_update="log_damp")
    for c, circ in contacts.items():
        if not circ:
            try:
                from devsim import get_parameter
                get_parameter(device=device, name=_bias(c))
            except Exception:  # noqa: BLE001
                set_parameter(device=device, name=_bias(c), value=0.0)
        _potential_contact(device, region, c, circ)
    if solve_equilibrium:
        solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=60)

    # carriers
    CreateSolution(device, region, "Electrons"); CreateSolution(device, region, "Holes")
    set_node_values(device=device, region=region, name="Electrons", init_from="IntrinsicElectrons")
    set_node_values(device=device, region=region, name="Holes", init_from="IntrinsicHoles")
    for v in ("Potential", "Electrons", "Holes", "DeltaEg"):
        EnsureEdgeFromNodeModelExists(device, region, v)

    # mobility: Masetti per node -> edge average; optional velocity saturation on top
    if phys.doping_mobility and phys.mu_n is None:
        CreateNodeModel(device, region, "mu_n_node", _masetti_expr(mat, T, "n"))
        CreateNodeModel(device, region, "mu_p_node", _masetti_expr(mat, T, "p"))
    else:
        zn, zp = mat.get("mu_n_T_exponent", "dimensionless"), mat.get("mu_p_T_exponent", "dimensionless")
        CreateNodeModel(device, region, "mu_n_node", str(phys.mu_n or mat.get("mu_n", "cm**2/V/s") * (T / 300) ** -zn))
        CreateNodeModel(device, region, "mu_p_node", str(phys.mu_p or mat.get("mu_p", "cm**2/V/s") * (T / 300) ** -zp))
    for c in ("n", "p"):
        EnsureEdgeFromNodeModelExists(device, region, f"mu_{c}_node")
        CreateEdgeModel(device, region, f"mu_{c}_lf", f"0.5*(mu_{c}_node@n0 + mu_{c}_node@n1)")
    mu_n, mu_p = "mu_n_lf", "mu_p_lf"
    if phys.field_mobility:
        from simws.tcad import models
        vn, vp = saturation_velocity(mat, T)
        used.update(vsat_n=vn, vsat_p=vp)
        mu_n, mu_p = models.field_dependent_mobility(device, region, mat, mu_n0=mu_n, mu_p0=mu_p, vsat=(vn, vp))

    # Scharfetter-Gummel currents on the band-edge (BGN-shifted) potentials. vdiff and B(vdiff) are separate edge
    # models with hand-written derivatives (as DEVSIM's simple_dd does): inlining them makes DEVSIM's symbolic
    # differentiation take ~30 s per device.
    for c, sgn in (("n", "+"), ("p", "-")):
        vd = f"vdiff_{c}"
        CreateEdgeModel(device, region, vd, f"(Potential@n0 - Potential@n1 {sgn} 0.5*(DeltaEg@n0 - DeltaEg@n1))/V_t")
        CreateEdgeModel(device, region, f"{vd}:Potential@n0", "1/V_t")
        CreateEdgeModel(device, region, f"{vd}:Potential@n1", "-1/V_t")
        CreateEdgeModel(device, region, f"Bern_{c}", f"B({vd})")
        CreateEdgeModel(device, region, f"Bern_{c}:Potential@n0", f"dBdx({vd})/V_t")
        CreateEdgeModel(device, region, f"Bern_{c}:Potential@n1", f"-dBdx({vd})/V_t")
    Jn = (f"ElectronCharge*{mu_n}*EdgeInverseLength*V_t*kahan3(Electrons@n1*Bern_n, Electrons@n1*vdiff_n, "
          f"-Electrons@n0*Bern_n)")
    Jp = (f"-ElectronCharge*{mu_p}*EdgeInverseLength*V_t*kahan3(Holes@n1*Bern_p, -Holes@n0*Bern_p, "
          f"-Holes@n0*vdiff_p)")
    for name, expr in (("ElectronCurrent", Jn), ("HoleCurrent", Jp)):
        CreateEdgeModel(device, region, name, expr)
        for v in ("Potential", "Electrons", "Holes"):
            CreateEdgeModelDerivatives(device, region, name, expr, v)

    # recombination
    terms = []
    if phys.srh:
        if phys.tau_n is None:
            tn, tp = mat.get("srh_tau_max_n", "s"), mat.get("srh_tau_max_p", "s")
            Nr, gm = mat.get("srh_N_ref", "cm**-3"), mat.get("srh_gamma", "dimensionless")
            fT = (T / 300.0) ** mat.get("srh_T_exponent", "dimensionless")
            CreateNodeModel(device, region, "tau_n", f"{tn * fT}/(1 + pow(Ntot/{Nr}, {gm}))")
            CreateNodeModel(device, region, "tau_p", f"{tp * fT}/(1 + pow(Ntot/{Nr}, {gm}))")
        else:
            CreateNodeModel(device, region, "tau_n", str(phys.tau_n))
            CreateNodeModel(device, region, "tau_p", str(phys.tau_p if phys.tau_p is not None else phys.tau_n))
        terms.append("(Electrons*Holes - n_ie^2)/(tau_p*(Electrons + n_ie) + tau_n*(Holes + n_ie))")
    if phys.auger:
        terms.append(f"({mat.get('auger_Cn', 'cm**6/s')}*Electrons + {mat.get('auger_Cp', 'cm**6/s')}*Holes)"
                     "*(Electrons*Holes - n_ie^2)")
    U = " + ".join(terms) if terms else "0"
    CreateNodeModel(device, region, "URecomb", U)
    for v in ("Electrons", "Holes"):
        CreateNodeModelDerivative(device, region, "URecomb", U, v)
    gen = f" - ({extra_generation})" if extra_generation else ""
    for name, sign in (("ElectronGeneration", "-"), ("HoleGeneration", "+")):
        expr = f"{sign}ElectronCharge*(URecomb{gen})"
        CreateNodeModel(device, region, name, expr)
        for v in ("Electrons", "Holes"):
            CreateNodeModelDerivative(device, region, name, expr, v)
    for name, expr, var in (("NCharge", "-ElectronCharge*Electrons", "Electrons"),
                            ("PCharge", "ElectronCharge*Holes", "Holes")):
        CreateNodeModel(device, region, name, expr)
        CreateNodeModelDerivative(device, region, name, expr, var)
    pne = "-ElectronCharge*kahan3(Holes, -Electrons, NetDoping)"
    CreateNodeModel(device, region, "PotentialNodeCharge", pne)
    for v in ("Electrons", "Holes"):
        CreateNodeModelDerivative(device, region, "PotentialNodeCharge", pne, v)
    equation(device=device, region=region, name="PotentialEquation", variable_name="Potential",
             node_model="PotentialNodeCharge", edge_model="PotentialEdgeFlux", time_node_model="",
             variable_update="log_damp")
    equation(device=device, region=region, name="ElectronContinuityEquation", variable_name="Electrons",
             time_node_model="NCharge", edge_model="ElectronCurrent", node_model="ElectronGeneration",
             variable_update="positive")
    equation(device=device, region=region, name="HoleContinuityEquation", variable_name="Holes",
             time_node_model="PCharge", edge_model="HoleCurrent", node_model="HoleGeneration",
             variable_update="positive")
    if phys.impact:
        from simws.tcad import models
        used["impact_gamma"] = impact_gamma(mat, T)
        models.impact_ionization(device, region, mat, gamma=used["impact_gamma"])
    for c, circ in contacts.items():
        _carrier_contact(device, region, c, circ)
    if solve_equilibrium:
        solve(type="dc", absolute_error=1e10, relative_error=1e-8, maximum_iterations=60)
    return used


# ohmic contacts: equilibrium densities with n p = n_ie^2, in a form that never cancels (DEVSIM evaluates both
# branches of ifelse, so 0.5*(-N + sqrt(N^2 + 4 n_ie^2)) would underflow to 0 at N = 1e20 and break the log)
_MAJ = "(0.5*(abs(NetDoping) + pow(NetDoping^2 + 4*n_ie^2, 0.5)))"
_N0 = f"ifelse(NetDoping > 0, {_MAJ}, n_ie^2/{_MAJ})"
_P0 = f"ifelse(NetDoping > 0, n_ie^2/{_MAJ}, {_MAJ})"


def _bias(contact):
    from devsim.python_packages.simple_physics import GetContactBiasName
    return GetContactBiasName(contact)


def _potential_contact(device, region, contact, is_circuit):
    from devsim import contact_equation
    from devsim.python_packages.model_create import (CreateContactNodeModel, CreateEdgeModel,
                                                      CreateEdgeModelDerivatives, InEdgeModelList)
    if not InEdgeModelList(device, region, "contactcharge_edge"):
        CreateEdgeModel(device, region, "contactcharge_edge", "Permittivity*ElectricField")
        CreateEdgeModelDerivatives(device, region, "contactcharge_edge", "Permittivity*ElectricField", "Potential")
    b = _bias(contact)
    # psi = bias + Vt ln(n0/n_i) - dEg/2  (equivalently bias - Vt ln(p0/n_i) + dEg/2, since n0 p0 = n_ie^2)
    expr = f"Potential - {b} - V_t*log({_N0}/n_i) + 0.5*DeltaEg"
    name = f"{contact}nodemodel"
    CreateContactNodeModel(device, contact, name, expr)
    CreateContactNodeModel(device, contact, f"{name}:Potential", "1")
    kw = dict(device=device, contact=contact, name="PotentialEquation", node_model=name,
              edge_charge_model="contactcharge_edge")
    if is_circuit:
        CreateContactNodeModel(device, contact, f"{name}:{b}", "-1")
        kw["circuit_node"] = b
    contact_equation(**kw)


def _carrier_contact(device, region, contact, is_circuit):
    from devsim import contact_equation
    from devsim.python_packages.model_create import CreateContactNodeModel
    for eq, var, edge, expr in (
            ("ElectronContinuityEquation", "Electrons", "ElectronCurrent",
             f"Electrons - {_N0}"),
            ("HoleContinuityEquation", "Holes", "HoleCurrent",
             f"Holes - {_P0}")):
        name = f"{contact}node{var.lower()}"
        CreateContactNodeModel(device, contact, name, expr)
        CreateContactNodeModel(device, contact, f"{name}:{var}", "1")
        kw = dict(device=device, contact=contact, name=eq, node_model=name, edge_current_model=edge)
        if is_circuit:
            kw["circuit_node"] = _bias(contact)
        contact_equation(**kw)


def contact_current(device, contact):
    """Total (electron + hole) particle current at a contact, A (per cm of depth in 2D, total in axisymmetric)."""
    from devsim import get_contact_current
    return (get_contact_current(device=device, contact=contact, equation="ElectronContinuityEquation")
            + get_contact_current(device=device, contact=contact, equation="HoleContinuityEquation"))
