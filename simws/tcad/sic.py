"""4H-SiC drift-diffusion in DEVSIM: incomplete ionization, anisotropic Caughey-Thomas mobility, anisotropic
Chynoweth impact ionization (Hatakeyama 2004), SRH. Parameters and sources: simws/materials/data/sic_4h.yaml.

    from simws.tcad.sic import SiCPhysics, setup_region_sic, equilibrium
    setup_region_sic(dev, "sic", mat, ND="...", NA="...", phys=SiCPhysics(T=300, c_axis="x"),
                     contacts={"anode": (False, 0.0, 1e19), "cathode": (False, 1e16, 0.0)})   # (is_circuit, ND, NA)

Incomplete ionization (Gibbs statistics, steady state): nitrogen occupies hexagonal and cubic sites in equal
parts (4H), each with its own level; aluminium is a single acceptor level:
    N_D+ = N_D [ 0.5 / (1 + g_D n / n_1h) + 0.5 / (1 + g_D n / n_1k) ],  n_1 = N_c exp(-dE_d / kT)
    N_A- = N_A / (1 + g_A p / p_1),                                      p_1 = N_v exp(-dE_a / kT)
The ionized charge enters Poisson with its dependence on n and p (full Newton derivatives). Ionization is assumed
instantaneous, which is exact in DC and approximate for transients faster than the donor capture time.

Anisotropy: the c-axis direction is given per device (`c_axis` "x" or "y", or None for isotropic perpendicular
values). On each mesh edge, mu = mu_perp (1 + (r - 1) cos^2 t) with t the angle to c and r = mu_par/mu_perp
(1.2 for electrons, 1 for holes); impact-ionization a and b interpolate the same way between the <11-20> and
<0001> values of Hatakeyama et al. The edge-projection form is the usual approximation for edge-based schemes.

Numerics: n_i ~ 1e-8 cm^-3, so minority densities reach 1e-30 cm^-3. Leakage currents are therefore far below
the solver's resolution; probe breakdown with injected carriers (simws.validation.sic_checks), not leakage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

import simws.tcad  # noqa: F401
from simws.tcad.physics import EnsureEdgeFromNodeModelExists
from simws.units import EPS0_F_PER_CM, K_B_EV, Q_E

H_PLANCK, M0, KB_J = 6.62607015e-34, 9.1093837015e-31, 1.380649e-23


@dataclass
class SiCPhysics:
    T: float = 300.0
    c_axis: Optional[str] = "x"          # direction of the crystal c-axis in the mesh, or None (isotropic, perp)
    incomplete_ionization: bool = True
    tau_n: float = 1e-7                  # s (SiC epilayer lifetimes are process dependent: calibrate)
    tau_p: float = 1e-7
    impact: bool = False
    permittivity: Optional[str] = None   # "par" (field along c), "perp", or None = "par" if c_axis else "perp"


def band(mat, T):
    """(Eg, Nc, Nv, n_i) at T: Varshni exciton gap + exciton binding; Nc, Nv from DOS masses (Nc incl. 3 valleys)."""
    Eg = (mat.get("varshni_Eg0", "eV") - mat.get("varshni_alpha", "eV/K") * T * T / (T + mat.get("varshni_beta", "K"))
          + mat.get("exciton_binding", "eV"))
    pref = 2 * (2 * math.pi * M0 * KB_J * T / H_PLANCK ** 2) ** 1.5 * 1e-6          # cm^-3 for m = m0
    Nc = pref * mat.get("m_dos_n", "dimensionless") ** 1.5 * mat.get("valleys_n", "dimensionless")
    Nv = pref * mat.get("m_dos_p", "dimensionless") ** 1.5
    return Eg, Nc, Nv, math.sqrt(Nc * Nv) * math.exp(-Eg / (2 * K_B_EV * T))


def ionization_constants(mat, T):
    Eg, Nc, Nv, ni = band(mat, T)
    kT = K_B_EV * T
    return {"n1h": Nc * math.exp(-mat.get("ion_N_h", "eV") / kT), "n1k": Nc * math.exp(-mat.get("ion_N_k", "eV") / kT),
            "p1A": Nv * math.exp(-mat.get("ion_Al", "eV") / kT), "gD": mat.get("g_D", "dimensionless"),
            "gA": mat.get("g_A", "dimensionless"), "ni": ni, "Nc": Nc, "Nv": Nv, "Eg": Eg}


def ionized(mat, T, n, p, ND, NA, incomplete=True):
    c = ionization_constants(mat, T)
    if not incomplete:
        return ND, NA
    Dp = ND * (0.5 / (1 + c["gD"] * n / c["n1h"]) + 0.5 / (1 + c["gD"] * n / c["n1k"]))
    Am = NA / (1 + c["gA"] * p / c["p1A"])
    return Dp, Am


def equilibrium(mat, ND, NA, T=300.0, incomplete=True):
    """Neutral-bulk n, p, N_D+, N_A- (independent of DEVSIM: bracketed root of the neutrality equation in log n)."""
    from scipy.optimize import brentq
    ni = band(mat, T)[3]

    def f(ln_n):
        n = math.exp(ln_n); p = ni * ni / n
        Dp, Am = ionized(mat, T, n, p, ND, NA, incomplete)
        return p - n + Dp - Am
    ln_n = brentq(f, math.log(ni * ni / (NA + ND + 1e30)) - 5, math.log(ND + NA + 1.0) + 5, xtol=1e-14, rtol=1e-15)
    n = math.exp(ln_n); p = ni * ni / n
    Dp, Am = ionized(mat, T, n, p, ND, NA, incomplete)
    return {"n": n, "p": p, "ND+": Dp, "NA-": Am, "ni": ni}


def _ionized_expr(n, p, incomplete):
    if not incomplete:
        return "(NDexpr - NAexpr)"
    return (f"(NDexpr*(0.5/(1 + gD*{n}/n1h) + 0.5/(1 + gD*{n}/n1k)) - NAexpr/(1 + gA*{p}/p1A))")


def setup_region_sic(device, region, mat, ND: str, NA: str, phys: SiCPhysics = SiCPhysics(),
                     contacts: Optional[Dict[str, Tuple[bool, float, float]]] = None, extra_generation=None,
                     solve_equilibrium=True):
    from devsim import contact_equation, equation, set_node_values, set_parameter, solve
    from devsim.python_packages.model_create import (CreateContactNodeModel, CreateEdgeModel,
                                                      CreateEdgeModelDerivatives, CreateNodeModel,
                                                      CreateNodeModelDerivative, CreateSolution)
    contacts = contacts or {}
    T = phys.T
    c = ionization_constants(mat, T)
    Vt = K_B_EV * T
    which = phys.permittivity or ("par" if phys.c_axis else "perp")
    eps = mat.get("eps_r_par" if which == "par" else "eps_r", "dimensionless") * EPS0_F_PER_CM   # scalar: 1D along c
    for k, v in {"Permittivity": eps, "ElectronCharge": Q_E, "V_t": Vt, "n_i": c["ni"], "T": T, "n1h": c["n1h"],
                 "n1k": c["n1k"], "p1A": c["p1A"], "gD": c["gD"], "gA": c["gA"]}.items():
        set_parameter(device=device, region=region, name=k, value=v)
    CreateNodeModel(device, region, "NDexpr", ND)
    CreateNodeModel(device, region, "NAexpr", NA)
    CreateNodeModel(device, region, "NetDoping", "NDexpr - NAexpr")
    CreateNodeModel(device, region, "Ntot", "max(NDexpr + NAexpr, 1.0)")

    CreateSolution(device, region, "Potential")
    for name, expr in (("IntrinsicElectrons", "n_i*exp(Potential/V_t)"), ("IntrinsicHoles", "n_i*exp(-Potential/V_t)")):
        CreateNodeModel(device, region, name, expr)
        CreateNodeModelDerivative(device, region, name, expr, "Potential")
    pic = f"-ElectronCharge*(IntrinsicHoles - IntrinsicElectrons + {_ionized_expr('IntrinsicElectrons', 'IntrinsicHoles', phys.incomplete_ionization)})"
    CreateNodeModel(device, region, "PotentialIntrinsicCharge", pic)
    CreateNodeModelDerivative(device, region, "PotentialIntrinsicCharge", pic, "Potential")
    for name, expr in (("ElectricField", "(Potential@n0 - Potential@n1)*EdgeInverseLength"),
                       ("PotentialEdgeFlux", "Permittivity*ElectricField")):
        CreateEdgeModel(device, region, name, expr)
        CreateEdgeModelDerivatives(device, region, name, expr, "Potential")
    equation(device=device, region=region, name="PotentialEquation", variable_name="Potential",
             node_model="PotentialIntrinsicCharge", edge_model="PotentialEdgeFlux", variable_update="log_damp")
    eq = {}
    for cname, (circ, cND, cNA) in contacts.items():
        e = equilibrium(mat, cND, cNA, T, phys.incomplete_ionization)
        eq[cname] = e
        b = f"{cname}_bias"
        if not circ:
            set_parameter(device=device, name=b, value=0.0)
        expr = f"Potential - {b} - V_t*log({e['n']}/n_i)"
        CreateContactNodeModel(device, cname, f"{cname}nodemodel", expr)
        CreateContactNodeModel(device, cname, f"{cname}nodemodel:Potential", "1")
        CreateEdgeModel(device, region, "contactcharge_edge", "Permittivity*ElectricField")
        CreateEdgeModelDerivatives(device, region, "contactcharge_edge", "Permittivity*ElectricField", "Potential")
        kw = dict(device=device, contact=cname, name="PotentialEquation", node_model=f"{cname}nodemodel",
                  edge_charge_model="contactcharge_edge")
        if circ:
            CreateContactNodeModel(device, cname, f"{cname}nodemodel:{b}", "-1")
            kw["circuit_node"] = b
        contact_equation(**kw)
    if solve_equilibrium:
        solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=80)

    CreateSolution(device, region, "Electrons"); CreateSolution(device, region, "Holes")
    set_node_values(device=device, region=region, name="Electrons", init_from="IntrinsicElectrons")
    set_node_values(device=device, region=region, name="Holes", init_from="IntrinsicHoles")
    for v in ("Potential", "Electrons", "Holes", "x"):
        EnsureEdgeFromNodeModelExists(device, region, v)

    # Caughey-Thomas mobility (perpendicular values) with temperature exponents, edge-projected anisotropy
    for cr in ("n", "p"):
        mumax = mat.get(f"mu_{cr}", "cm**2/V/s") * (T / 300) ** mat.get(f"ct_Tmax_{cr}", "dimensionless")
        mumin = mat.get(f"ct_mumin_{cr}", "cm**2/V/s") * (T / 300) ** mat.get(f"ct_Tmin_{cr}", "dimensionless")
        Nr, al = mat.get(f"ct_Nref_{cr}", "cm**-3"), mat.get(f"ct_alpha_{cr}", "dimensionless")
        CreateNodeModel(device, region, f"mu_{cr}_node", f"{mumin} + ({mumax} - {mumin})/(1 + pow(Ntot/{Nr}, {al}))")
        EnsureEdgeFromNodeModelExists(device, region, f"mu_{cr}_node")
    if phys.c_axis:
        ax = phys.c_axis
        if ax == "y":
            EnsureEdgeFromNodeModelExists(device, region, "y")
        CreateEdgeModel(device, region, "cos2c", f"pow(({ax}@n1 - {ax}@n0)*EdgeInverseLength, 2)")
    else:
        CreateEdgeModel(device, region, "cos2c", "0")
    r_n = mat.get("aniso_n", "dimensionless")
    CreateEdgeModel(device, region, "mu_n_e", f"0.5*(mu_n_node@n0 + mu_n_node@n1)*(1 + {r_n - 1}*cos2c)")
    CreateEdgeModel(device, region, "mu_p_e", "0.5*(mu_p_node@n0 + mu_p_node@n1)")

    CreateEdgeModel(device, region, "vdiff", "(Potential@n0 - Potential@n1)/V_t")
    CreateEdgeModel(device, region, "vdiff:Potential@n0", "1/V_t")
    CreateEdgeModel(device, region, "vdiff:Potential@n1", "-1/V_t")
    CreateEdgeModel(device, region, "Bern01", "B(vdiff)")
    CreateEdgeModel(device, region, "Bern01:Potential@n0", "dBdx(vdiff)/V_t")
    CreateEdgeModel(device, region, "Bern01:Potential@n1", "-dBdx(vdiff)/V_t")
    Jn = "ElectronCharge*mu_n_e*EdgeInverseLength*V_t*kahan3(Electrons@n1*Bern01, Electrons@n1*vdiff, -Electrons@n0*Bern01)"
    Jp = "-ElectronCharge*mu_p_e*EdgeInverseLength*V_t*kahan3(Holes@n1*Bern01, -Holes@n0*Bern01, -Holes@n0*vdiff)"
    for name, expr in (("ElectronCurrent", Jn), ("HoleCurrent", Jp)):
        CreateEdgeModel(device, region, name, expr)
        for v in ("Potential", "Electrons", "Holes"):
            CreateEdgeModelDerivatives(device, region, name, expr, v)

    U = f"(Electrons*Holes - n_i^2)/({phys.tau_p}*(Electrons + n_i) + {phys.tau_n}*(Holes + n_i))"
    CreateNodeModel(device, region, "URecomb", U)
    gen = f" - ({extra_generation})" if extra_generation else ""
    for name, sign in (("ElectronGeneration", "-"), ("HoleGeneration", "+")):
        expr = f"{sign}ElectronCharge*(URecomb{gen})"
        CreateNodeModel(device, region, name, expr)
        for v in ("Electrons", "Holes"):
            CreateNodeModelDerivative(device, region, name, expr, v)
    for name, expr, var in (("NCharge", "-ElectronCharge*Electrons", "Electrons"), ("PCharge", "ElectronCharge*Holes", "Holes")):
        CreateNodeModel(device, region, name, expr)
        CreateNodeModelDerivative(device, region, name, expr, var)
    pne = f"-ElectronCharge*(Holes - Electrons + {_ionized_expr('Electrons', 'Holes', phys.incomplete_ionization)})"
    CreateNodeModel(device, region, "PotentialNodeCharge", pne)
    for v in ("Electrons", "Holes"):
        CreateNodeModelDerivative(device, region, "PotentialNodeCharge", pne, v)
    equation(device=device, region=region, name="PotentialEquation", variable_name="Potential",
             node_model="PotentialNodeCharge", edge_model="PotentialEdgeFlux", variable_update="log_damp")
    equation(device=device, region=region, name="ElectronContinuityEquation", variable_name="Electrons",
             time_node_model="NCharge", edge_model="ElectronCurrent", node_model="ElectronGeneration",
             variable_update="positive")
    equation(device=device, region=region, name="HoleContinuityEquation", variable_name="Holes",
             time_node_model="PCharge", edge_model="HoleCurrent", node_model="HoleGeneration",
             variable_update="positive")
    if phys.impact:
        impact_ionization_sic(device, region, mat)
    for cname, (circ, cND, cNA) in contacts.items():
        e = eq[cname]
        for eqn, var, edge, val in (("ElectronContinuityEquation", "Electrons", "ElectronCurrent", e["n"]),
                                    ("HoleContinuityEquation", "Holes", "HoleCurrent", e["p"])):
            nm = f"{cname}node{var.lower()}"
            CreateContactNodeModel(device, cname, nm, f"{var} - {val}")
            CreateContactNodeModel(device, cname, f"{nm}:{var}", "1")
            kw = dict(device=device, contact=cname, name=eqn, node_model=nm, edge_current_model=edge)
            if circ:
                kw["circuit_node"] = f"{cname}_bias"
            contact_equation(**kw)
    if solve_equilibrium:
        solve(type="dc", absolute_error=1e10, relative_error=1e-8, maximum_iterations=80)
    return {"T": T, **{k: v for k, v in c.items()}, "contacts": eq}


def alpha_sic(mat, E, cos2c=1.0):
    """Hatakeyama coefficients, interpolated between <11-20> (cos2c = 0) and <0001> (cos2c = 1). E in V/cm."""
    E = np.sqrt(np.asarray(E, float) ** 2 + 1.0)
    g = lambda k: mat.get(k, "1/cm" if k.startswith("ii_a") else "V/cm")
    out = []
    for cr in ("n", "p"):
        a = g(f"ii_a_{cr}_perp") + (g(f"ii_a_{cr}_par") - g(f"ii_a_{cr}_perp")) * cos2c
        b = g(f"ii_b_{cr}_perp") + (g(f"ii_b_{cr}_par") - g(f"ii_b_{cr}_perp")) * cos2c
        out.append(a * np.exp(-b / E))
    return tuple(out)


def impact_ionization_sic(device, region, mat):
    from devsim import equation
    from devsim.python_packages.model_create import CreateEdgeModel, CreateEdgeModelDerivatives
    Eabs = "pow(ElectricField^2 + 1, 0.5)"
    g = lambda k: mat.get(k, "1/cm" if k.startswith("ii_a") else "V/cm")
    al = {}
    for cr in ("n", "p"):
        a = f"({g(f'ii_a_{cr}_perp')} + {g(f'ii_a_{cr}_par') - g(f'ii_a_{cr}_perp')}*cos2c)"
        b = f"({g(f'ii_b_{cr}_perp')} + {g(f'ii_b_{cr}_par') - g(f'ii_b_{cr}_perp')}*cos2c)"
        al[cr] = f"({a}*exp(-{b}/{Eabs}))"
    gii = (f"({al['n']}*pow(ElectronCurrent^2 + 1e-24, 0.5) + {al['p']}*pow(HoleCurrent^2 + 1e-24, 0.5))/ElectronCharge")
    CreateEdgeModel(device, region, "ImpactGeneration", gii)
    for v in ("Potential", "Electrons", "Holes"):
        CreateEdgeModelDerivatives(device, region, "ImpactGeneration", gii, v)
    for name, expr in (("ImpactSourceElectron", "+ElectronCharge*ImpactGeneration"),
                       ("ImpactSourceHole", "-ElectronCharge*ImpactGeneration")):
        CreateEdgeModel(device, region, name, expr)
        for v in ("Potential", "Electrons", "Holes"):
            CreateEdgeModelDerivatives(device, region, name, expr, v)
    equation(device=device, region=region, name="ElectronContinuityEquation", variable_name="Electrons",
             time_node_model="NCharge", edge_model="ElectronCurrent", variable_update="positive",
             node_model="ElectronGeneration", edge_volume_model="ImpactSourceElectron")
    equation(device=device, region=region, name="HoleContinuityEquation", variable_name="Holes",
             time_node_model="PCharge", edge_model="HoleCurrent", variable_update="positive",
             node_model="HoleGeneration", edge_volume_model="ImpactSourceHole")
