"""Full drift-diffusion physics for a DEVSIM silicon region, beyond DEVSIM's simple_physics.

    from simws.tcad.physics import Physics, setup_region
    phys = Physics(T=350.0)                                   # every model evaluated at the lattice temperature T
    setup_region(dev, "si", mat, net_doping="...", total_doping="...", phys=phys,
                 contacts={"anode": False, "cathode": False})  # name -> is_circuit

Models (each switchable in `Physics`; parameters and references in simws/materials/data/silicon.yaml):
  doping_mobility  Masetti (1983): mu(N, T), lattice part mu_max (T/300)^-zeta
  field_mobility   Caughey-Thomas velocity saturation with v_sat(T) (Canali 1975), on top of mu(N, T)
  srh              Shockley-Read-Hall with Scharfetter lifetime tau(N) (T/300)^a, trap level E_t - E_i (trap_level_eV)
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
    trap_level_eV: float = 0.0           # SRH trap energy relative to the intrinsic level, E_t - E_i (deep-level traps)
    ge: Optional[str] = None             # Ge fraction x of Si(1-x)Ge(x) as a node expression (e.g. a graded HBT base)
    ge_mode: str = "strained"            # "strained" (pseudomorphic on Si) or "relaxed" (bulk alloy)
    # --- 0.9a cryogenic core (simws.tcad.cryo); all off by default, so 0.8 results are unchanged
    incomplete_ionization: bool = False  # Altermatt (2006) freeze-out of donors and acceptors
    fermi_dirac: bool = False            # Fermi-Dirac statistics (degeneracy factor in charge, currents, contacts)
    mobility: str = "masetti"            # "masetti" | "klaassen" (Klaassen 1992 unified, carrier-dependent)
    donor_species: str = "P"             # P | As (ionization parameters; Klaassen electron parameters)
    acceptor_species: str = "B"
    ii_model: str = "altermatt"          # "altermatt" | "charon" (constant energy, full ionization above ii_N_crit)
    ii_N_crit: Optional[float] = None    # charon model only (cm^-3)
    ii_E_const: Optional[float] = None   # charon model only (eV)
    initial_guess: Optional[Dict[str, list]] = None   # node values of Potential/Electrons/Holes (T continuation)
    # DEVSIM extended (80-bit long double) model evaluation, assembly and solve; None = on for cryo devices below
    # 200 K, where minority densities fall below 1e-300 cm^-3 and double-precision Newton updates stall
    extended_precision: Optional[bool] = None
    extra: Dict[str, float] = field(default_factory=dict)
    # A23b: contacts with a finite hole recombination velocity (contact name -> S in cm/s); the others ohmic
    hole_recombination_velocity: Dict[str, float] = field(default_factory=dict)

    @property
    def cryo(self) -> bool:
        return self.incomplete_ionization or self.fermi_dirac or self.mobility == "klaassen"

    @property
    def extended(self) -> bool:
        return self.extended_precision if self.extended_precision is not None else (self.cryo and self.T < 200.0)


def EnsureEdgeFromNodeModelExists(device, region, nodemodel):
    """Create <model>@n0 / <model>@n1 edge models. Own version: in older DEVSIM releases the helper of the same
    name tests `"m@n0" and "m@n1"` (always true) and never creates them."""
    from devsim import edge_from_node_model, get_edge_model_list
    have = get_edge_model_list(device=device, region=region)
    if f"{nodemodel}@n0" not in have or f"{nodemodel}@n1" not in have:
        edge_from_node_model(device=device, region=region, node_model=nodemodel)


def _sige_band_terms(device, region, phys):
    """Node-expression strings (eV) for the SiGe conduction and valence band shifts, incl. DOS-ratio terms."""
    if not phys.ge:
        return "0", "0"
    from devsim.python_packages.model_create import CreateNodeModel
    from simws.materials import load
    sg = load("sige")
    CreateNodeModel(device, region, "GeX", f"max({phys.ge}, 0)")
    if phys.ge_mode == "strained":
        k = sg.get("strained_dEg_per_x", "eV")
        dEc = f"({sg.get('strained_dEc_max', 'eV')}*GeX)"
        dEg = f"({k}*GeX)"
        # DOS: Nc ratio 2/3 and Nc*Nv ratio exp(-sqrt(k x)) above 1 % Ge, blended linearly from Si below it
        w = "min(GeX/0.01, 1)"
        ncr = f"(1 + ({sg.get('dos_Nc_ratio', 'dimensionless')} - 1)*{w})"
        gam = f"(1 + (exp(-pow({sg.get('dos_gamma_k', 'dimensionless')}*max(GeX, 0.01), 0.5)) - 1)*{w})"
        nvr = f"({gam}/{ncr})"
        return f"({dEc} + V_t*log({ncr}))", f"(({dEg} - {dEc}) + V_t*log({nvr}))"
    a, b, c = (sg.get(k, "eV") for k in ("relaxed_Eg_a", "relaxed_Eg_b", "relaxed_Eg_c"))
    va, vb = sg.get("relaxed_dEv_a", "eV"), sg.get("relaxed_dEv_b", "eV")
    dEg = f"(-({b})*GeX - ({c})*GeX^2)"
    dEv = f"(({va} + ({vb})*GeX)*GeX)"
    return f"({dEg} - {dEv})", dEv


def _permittivity_expr(mat, phys):
    eps_si = mat.get("eps_r", "dimensionless")
    if not phys.ge:
        return f"{eps_si * EPS0_F_PER_CM}"
    from simws.materials import load
    eps_ge = load("sige").get("eps_Ge", "dimensionless")
    return f"({eps_si} + {eps_ge - eps_si}*GeX)*{EPS0_F_PER_CM}"


from simws.materials.semiconductor import intrinsic_density  # noqa: E402,F401  (moved; DEVSIM-free)


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
                 solve_equilibrium=True, extra_charge=None, pre_solve=None, carriers=True):
    """Create every model and equation on `region`, the contacts in `contacts` (name -> is_circuit), and
    (optionally) solve the zero-bias equilibrium. `extra_generation` is a node expression in pairs/cm^3/s added
    to both continuity equations (e.g. an ion track). Bias parameters are '<contact>_bias' (simple_physics
    naming); set them with set_parameter or simws.tcad.solve.ramp_bias. Returns a dict of the values used.

    extra_charge: callable n_expr -> node expression of additional POSITIVE charge (elementary charges / cm^3),
                  given the electron-density expression to use (IntrinsicElectrons in the potential-only start,
                  Electrons afterwards); used for fixed oxide/interface charge and interface traps (simws.tcad.mos).
    pre_solve:    callable run after this region's Potential exists and before the first solve (e.g. to add an
                  oxide region's interface and gate contact).
    carriers:     False stops after the equilibrium (Poisson-Boltzmann) stage: carriers follow the potential with a
                  flat Fermi level. Exact for structures that carry no DC current (MOS quasi-static C-V)."""
    from devsim import equation, set_node_values, set_parameter, solve
    from devsim.python_packages.model_create import (CreateEdgeModel, CreateEdgeModelDerivatives, CreateNodeModel,
                                                      CreateNodeModelDerivative, CreateSolution)

    contacts = contacts or {}
    T = phys.T
    V_t = K_B_EV * T
    n_i = intrinsic_density(mat, T)
    used = {"T_K": T, "n_i_cm3": n_i, "V_t": V_t, "extended_precision": phys.extended}
    for k in ("extended_solver", "extended_model", "extended_equation"):    # global; reset_devsim clears them
        set_parameter(name=k, value=phys.extended)
    for name, val in {"Permittivity": mat.get("eps_r", "dimensionless") * EPS0_F_PER_CM, "ElectronCharge": Q_E,
                      "V_t": V_t, "n_i": n_i, "T": T}.items():
        set_parameter(device=device, region=region, name=name, value=val)

    # doping, band-gap narrowing, effective intrinsic density
    CreateNodeModel(device, region, "NetDoping", net_doping)
    CreateNodeModel(device, region, "Ntot", f"max({total_doping}, 1.0)")
    if phys.bgn:
        E0, Nr, C = mat.get("bgn_E0", "eV"), mat.get("bgn_N_ref", "cm**-3"), mat.get("bgn_C", "dimensionless")
        slot = f"{E0}*(log(Ntot/{Nr}) + pow(log(Ntot/{Nr})^2 + {C}, 0.5))"
        if phys.fermi_dirac:
            # Slotboom's narrowing is APPARENT (fitted at 300 K with Boltzmann statistics); with Fermi-Dirac the
            # physical narrowing is dEg_app - kT300 ln(gamma_300) of the majority carriers (degeneracy at 300 K),
            # assumed temperature independent. Using the apparent value with FD would count degeneracy twice.
            slot = f"({slot}) + {_fd_bgn_correction(mat, n_i_300=intrinsic_density(mat, 300.0))}"
        CreateNodeModel(device, region, "DeltaEg", slot)
    else:
        CreateNodeModel(device, region, "DeltaEg", "0")
    # band-edge shifts relative to the silicon reference (eV, positive = conduction band down / valence band up):
    # half of the doping BGN on each band, plus SiGe heterojunction offsets and density-of-states ratios
    # (simws.materials sige.yaml). Without Ge both reduce to 0.5*DeltaEg, i.e. the silicon model unchanged.
    dEc_ge, dEv_ge = _sige_band_terms(device, region, phys)
    CreateNodeModel(device, region, "dEc", f"0.5*DeltaEg + {dEc_ge}")
    CreateNodeModel(device, region, "dEv", f"0.5*DeltaEg + {dEv_ge}")
    CreateNodeModel(device, region, "n_ie", "n_i*exp((dEc + dEv)/(2*V_t))")
    eps_expr = _permittivity_expr(mat, phys)
    CreateNodeModel(device, region, "Eps", eps_expr)
    ionnet = _cryo_setup(device, region, mat, phys, n_i, used) if phys.cryo else (lambda nprime, pprime: "NetDoping")

    # potential-only start: n = n_i exp((psi + dEg/2)/Vt), p = n_i exp(-(psi - dEg/2)/Vt)
    CreateSolution(device, region, "Potential")
    for name, expr in (("IntrinsicElectrons", "n_i*exp((Potential + dEc)/V_t)"),
                       ("IntrinsicHoles", "n_i*exp(-(Potential - dEv)/V_t)"),
                       ("PotentialIntrinsicCharge", "-ElectronCharge*(kahan3(IntrinsicHoles, -IntrinsicElectrons, "
                        + ionnet("IntrinsicElectrons", "IntrinsicHoles") + ")"
                        + (f" + {extra_charge('IntrinsicElectrons')}" if extra_charge else "") + ")")):
        CreateNodeModel(device, region, name, expr)
        CreateNodeModelDerivative(device, region, name, expr, "Potential")
    EnsureEdgeFromNodeModelExists(device, region, "Eps")
    CreateEdgeModel(device, region, "EpsEdge", "0.5*(Eps@n0 + Eps@n1)")
    for name, expr in (("ElectricField", "(Potential@n0 - Potential@n1)*EdgeInverseLength"),
                       ("PotentialEdgeFlux", "EpsEdge*ElectricField")):
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
        _potential_contact(device, region, c, circ, cryo=phys.cryo)
    if phys.initial_guess and "Potential" in phys.initial_guess:
        set_node_values(device=device, region=region, name="Potential", values=list(phys.initial_guess["Potential"]))
    elif phys.cryo:
        set_node_values(device=device, region=region, name="Potential", values=_neutral_potential(device, region))
    if pre_solve:
        pre_solve()
    if solve_equilibrium:
        solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=60)
    if not carriers:
        return used

    # carriers
    CreateSolution(device, region, "Electrons"); CreateSolution(device, region, "Holes")
    set_node_values(device=device, region=region, name="Electrons", init_from="IntrinsicElectrons")
    set_node_values(device=device, region=region, name="Holes", init_from="IntrinsicHoles")
    if phys.fermi_dirac:
        _init_fd_carriers(device, region)
    if phys.initial_guess:
        for v in ("Electrons", "Holes"):
            if v in phys.initial_guess:
                set_node_values(device=device, region=region, name=v, values=list(phys.initial_guess[v]))
    for v in ("Potential", "Electrons", "Holes", "dEc", "dEv"):
        EnsureEdgeFromNodeModelExists(device, region, v)
    if phys.fermi_dirac:
        _fd_degeneracy_models(device, region)

    # mobility: Masetti per node -> edge average; optional velocity saturation on top
    if phys.mobility == "klaassen" and phys.mu_n is None:
        from devsim import node_solution
        for c in ("n", "p"):
            node_solution(device=device, region=region, name=f"mu_{c}_node_si")
            CreateNodeModel(device, region, f"mu_{c}_node", f"mu_{c}_node_si")
        update_mobility(device, region)
    elif phys.doping_mobility and phys.mu_n is None:
        for c in ("n", "p"):
            CreateNodeModel(device, region, f"mu_{c}_node", _masetti_expr(mat, T, c))
            CreateNodeModel(device, region, f"mu_{c}_node_si", _masetti_expr(mat, T, c))
    else:
        zn, zp = mat.get("mu_n_T_exponent", "dimensionless"), mat.get("mu_p_T_exponent", "dimensionless")
        for c, val in (("n", phys.mu_n or mat.get("mu_n", "cm**2/V/s") * (T / 300) ** -zn),
                       ("p", phys.mu_p or mat.get("mu_p", "cm**2/V/s") * (T / 300) ** -zp)):
            CreateNodeModel(device, region, f"mu_{c}_node", str(val))
            CreateNodeModel(device, region, f"mu_{c}_node_si", str(val))
    if phys.ge:
        from simws.materials import load as _load
        sg = _load("sige")
        for c in ("n", "p"):
            f = sg.get(f"mu_{c}_factor", "dimensionless")
            if f != 1.0:
                CreateNodeModel(device, region, f"mu_{c}_node", f"mu_{c}_node_si*(1 + {f - 1}*min(GeX/0.01, 1))")
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
    for c, sgn, band in (("n", "+", "dEc"), ("p", "-", "dEv")):
        vd = f"vdiff_{c}"
        # Fermi-Dirac: generalised Einstein relation through the effective potential
        # psi + dEc + V_t ln(gamma_n) (electrons) and psi - dEv - V_t ln(gamma_p) (holes)
        g, var = ("LnGn", "Electrons") if c == "n" else ("LnGp", "Holes")
        fd = f" {sgn} ({g}@n0 - {g}@n1)" if phys.fermi_dirac else ""
        CreateEdgeModel(device, region, vd, f"(Potential@n0 - Potential@n1 {sgn} ({band}@n0 - {band}@n1))/V_t{fd}")
        CreateEdgeModel(device, region, f"{vd}:Potential@n0", "1/V_t")
        CreateEdgeModel(device, region, f"{vd}:Potential@n1", "-1/V_t")
        CreateEdgeModel(device, region, f"Bern_{c}", f"B({vd})")
        CreateEdgeModel(device, region, f"Bern_{c}:Potential@n0", f"dBdx({vd})/V_t")
        CreateEdgeModel(device, region, f"Bern_{c}:Potential@n1", f"-dBdx({vd})/V_t")
        if phys.fermi_dirac:
            CreateEdgeModel(device, region, f"{vd}:{var}@n0", f"{sgn}{g}:{var}@n0")
            CreateEdgeModel(device, region, f"{vd}:{var}@n1", f"{'-' if sgn == '+' else '+'}{g}:{var}@n1")
            CreateEdgeModel(device, region, f"Bern_{c}:{var}@n0", f"dBdx({vd})*{vd}:{var}@n0")
            CreateEdgeModel(device, region, f"Bern_{c}:{var}@n1", f"dBdx({vd})*{vd}:{var}@n1")
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
    NP_EQ = "exp(LnGn + LnGp)*n_ie^2" if phys.fermi_dirac else "n_ie^2"
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
        et = phys.trap_level_eV
        n1, p1 = (f"n_ie*exp({et}/V_t)", f"n_ie*exp({-et}/V_t)") if et else ("n_ie", "n_ie")
        if phys.fermi_dirac:
            # Fermi-Dirac: equilibrium is n p = gamma_n gamma_p n_ie^2, so the driving term and the trap densities
            # carry the degeneracy factors (Sentaurus form); without them a degenerate emitter would generate
            n1, p1 = f"exp(LnGn)*{n1}", f"exp(LnGp)*{p1}"
        terms.append(f"(Electrons*Holes - {NP_EQ})/(tau_p*(Electrons + {n1}) + tau_n*(Holes + {p1}))")
    if phys.auger:
        terms.append(f"({mat.get('auger_Cn', 'cm**6/s')}*Electrons + {mat.get('auger_Cp', 'cm**6/s')}*Holes)"
                     f"*(Electrons*Holes - {NP_EQ})")
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
    nprime, pprime = ("(Electrons*exp(-LnGn))", "(Holes*exp(-LnGp))") if phys.fermi_dirac else ("Electrons", "Holes")
    pne = ("-ElectronCharge*(kahan3(Holes, -Electrons, " + ionnet(nprime, pprime) + ")"
           + (f" + {extra_charge('Electrons')}" if extra_charge else "") + ")")
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
        _carrier_contact(device, region, c, circ, cryo=phys.cryo, S_p=phys.hole_recombination_velocity.get(c))
    if solve_equilibrium:
        tol = 1e-8
        try:
            solve(type="dc", absolute_error=1e10, relative_error=tol, maximum_iterations=60)
        except Exception:
            if not phys.cryo:
                raise
            # cold devices: minority densities span ~80 decades and the Newton update of the smallest ones stalls
            # at ~1e-6 relative while the absolute residual is already negligible: accept 1e-5 and record it
            tol = 1e-5
            solve(type="dc", absolute_error=1e10, relative_error=tol, maximum_iterations=60)
        used["equilibrium_rel_tol"] = tol
        if phys.mobility == "klaassen":
            relax_mobility(device, region, dict(absolute_error=1e10, relative_error=tol, maximum_iterations=60))
    return used


# ohmic contacts: equilibrium densities with n p = n_ie^2, in a form that never cancels (DEVSIM evaluates both
# branches of ifelse, so 0.5*(-N + sqrt(N^2 + 4 n_ie^2)) would underflow to 0 at N = 1e20 and break the log)
_MAJ = "(0.5*(abs(NetDoping) + pow(NetDoping^2 + 4*n_ie^2, 0.5)))"
_N0 = f"ifelse(NetDoping > 0, {_MAJ}, n_ie^2/{_MAJ})"
_P0 = f"ifelse(NetDoping > 0, n_ie^2/{_MAJ}, {_MAJ})"


def _bias(contact):
    from devsim.python_packages.simple_physics import GetContactBiasName
    return GetContactBiasName(contact)


def _potential_contact(device, region, contact, is_circuit, cryo=False):
    from devsim import contact_equation
    from devsim.python_packages.model_create import (CreateContactNodeModel, CreateEdgeModel,
                                                      CreateEdgeModelDerivatives, InEdgeModelList)
    if not InEdgeModelList(device, region, "contactcharge_edge"):
        CreateEdgeModel(device, region, "contactcharge_edge", "EpsEdge*ElectricField")
        CreateEdgeModelDerivatives(device, region, "contactcharge_edge", "EpsEdge*ElectricField", "Potential")
    b = _bias(contact)
    # psi = bias + Vt ln(n0/n_i) - dEg/2  (equivalently bias - Vt ln(p0/n_i) + dEg/2, since n0 p0 = n_ie^2)
    expr = (f"Potential - {b} - V_t*log(N0eq/n_i) + V_t*LnGn0eq + dEc" if cryo
            else f"Potential - {b} - V_t*log({_N0}/n_i) + dEc")
    name = f"{contact}nodemodel"
    CreateContactNodeModel(device, contact, name, expr)
    CreateContactNodeModel(device, contact, f"{name}:Potential", "1")
    kw = dict(device=device, contact=contact, name="PotentialEquation", node_model=name,
              edge_charge_model="contactcharge_edge")
    if is_circuit:
        CreateContactNodeModel(device, contact, f"{name}:{b}", "-1")
        kw["circuit_node"] = b
    contact_equation(**kw)


def _carrier_contact(device, region, contact, is_circuit, cryo=False, S_p=None):
    """Ohmic carrier boundary conditions (n = n_eq, p = p_eq). A23b: with S_p (cm/s) the hole equation at the contact
    keeps the bulk flux and adds the surface loss q S_p (p - p_eq) per unit contact area (a Robin condition)."""
    from devsim import contact_equation
    from devsim.python_packages.model_create import CreateContactNodeModel
    for eq, var, edge, expr in (
            ("ElectronContinuityEquation", "Electrons", "ElectronCurrent",
             "Electrons - N0eq" if cryo else f"Electrons - {_N0}"),
            ("HoleContinuityEquation", "Holes", "HoleCurrent",
             "Holes - P0eq" if cryo else f"Holes - {_P0}")):
        name = f"{contact}node{var.lower()}"
        robin = S_p is not None and var == "Holes"
        if robin:
            peq = "P0eq" if cryo else f"({_P0})"
            # DEVSIM integrates a contact node_model over the node's volume: the surface flux is divided by it
            CreateContactNodeModel(device, contact, name,
                                   f"ElectronCharge * {float(S_p)!r} * (Holes - {peq}) * ContactSurfaceArea"
                                   " / NodeVolume")
            CreateContactNodeModel(device, contact, f"{name}:{var}",
                                   f"ElectronCharge * {float(S_p)!r} * ContactSurfaceArea / NodeVolume")
            kw = dict(device=device, contact=contact, name=eq, node_model=name, edge_model=edge,
                      edge_current_model=edge)
        else:
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


# ---------------------------------------------------------------------------------------------- 0.9a cryogenic core
_CRYO: Dict[tuple, dict] = {}


def _node(device, region, name):
    import numpy as np
    from devsim import get_node_model_values
    return np.array(get_node_model_values(device=device, region=region, name=name))


def _cryo_setup(device, region, mat, phys, n_i, used):
    """Band parameters, total donor/acceptor densities, contact equilibria; returns ionnet(n', p') -> expression of
    the ionized net doping N_D+ - N_A- (or NetDoping when incomplete ionization is off)."""
    from devsim import node_solution, set_node_values, set_parameter
    from devsim.python_packages.model_create import CreateNodeModel
    from simws.tcad import cryo as C
    T = phys.T
    Ncv, Nvv, Eg = C.band_params(mat, T, n_i)
    for k, v in (("Nc_eff", Ncv), ("Nv_eff", Nvv)):
        set_parameter(device=device, region=region, name=k, value=v)
    CreateNodeModel(device, region, "NDtot", "max(0.5*(Ntot + NetDoping), 0)")
    CreateNodeModel(device, region, "NAtot", "max(0.5*(Ntot - NetDoping), 0)")
    pd, pa = C.ii_params(mat, phys.donor_species), C.ii_params(mat, phys.acceptor_species)
    kw = dict(model=phys.ii_model, N_crit=phys.ii_N_crit, E_const=phys.ii_E_const)
    st = dict(mat=mat, T=T, Nc=Ncv, Nv=Nvv, pd=pd, pa=pa, phys=phys, kw=kw)
    _CRYO[(device, region)] = st
    used.update(Nc_eff_cm3=Ncv, Nv_eff_cm3=Nvv, Eg_eV=Eg, cryo=dict(incomplete_ionization=phys.incomplete_ionization,
                fermi_dirac=phys.fermi_dirac, mobility=phys.mobility, ii_model=phys.ii_model))
    # equilibrium (zero bias, local neutrality) densities for contacts and the initial guess
    ND, NA, nie = _node(device, region, "NDtot"), _node(device, region, "NAtot"), _node(device, region, "n_ie")
    n, p, npr, ppr = C.device_equilibrium(mat, ND, NA, nie, T, Ncv, Nvv, phys.donor_species, phys.acceptor_species,
                                          phys.incomplete_ionization, phys.fermi_dirac, **kw)
    import numpy as np
    for name, vals in (("N0eq", n), ("P0eq", p), ("N0prime", npr), ("LnGn0eq", np.log(n / npr))):
        node_solution(device=device, region=region, name=name)
        set_node_values(device=device, region=region, name=name, values=vals.tolist())

    def ionnet(nprime, pprime):
        if not phys.incomplete_ionization:
            return "NetDoping"
        d = C.ionized_expr(pd, "NDtot", nprime, Ncv, T, **kw)
        a = C.ionized_expr(pa, "NAtot", pprime, Nvv, T, **kw)
        return f"({d} - {a})"
    return ionnet


def _neutral_potential(device, region):
    """Initial potential = local charge-neutral value V_t ln(n'_eq/n_i) - dEc (Charon's 'equilibrium potential')."""
    import numpy as np
    from devsim import get_parameter
    Vt = get_parameter(device=device, region=region, name="V_t")
    ni = get_parameter(device=device, region=region, name="n_i")
    return (Vt * np.log(_node(device, region, "N0prime") / ni) - _node(device, region, "dEc")).tolist()


def _init_fd_carriers(device, region):
    """Electrons/holes consistent with FD at the solved potential: n = n'(psi) mapped through the statistics."""
    from devsim import set_node_values
    from simws.tcad import cryo as C
    st = _CRYO[(device, region)]
    set_node_values(device=device, region=region, name="Electrons",
                    values=C.density_from_prime(_node(device, region, "IntrinsicElectrons"), st["Nc"]).tolist())
    set_node_values(device=device, region=region, name="Holes",
                    values=C.density_from_prime(_node(device, region, "IntrinsicHoles"), st["Nv"]).tolist())


def _fd_degeneracy_models(device, region):
    from devsim.python_packages.model_create import CreateNodeModel, CreateNodeModelDerivative
    from simws.tcad import cryo as C
    st = _CRYO[(device, region)]
    for g, var, Nb in (("LnGn", "Electrons", st["Nc"]), ("LnGp", "Holes", st["Nv"])):
        expr = C.ln_gamma_expr(f"{var}/{Nb!r}")
        CreateNodeModel(device, region, g, expr)
        CreateNodeModelDerivative(device, region, g, expr, var)
        EnsureEdgeFromNodeModelExists(device, region, g)
        EnsureEdgeFromNodeModelExists(device, region, f"{g}:{var}")


def update_mobility(device, region, relax: float = 1.0) -> float:
    """Klaassen mobility from the current carrier densities (and ionized dopants when freeze-out is on), written
    into the node solutions mu_n_node_si / mu_p_node_si (log-space under-relaxation: mu_old^(1-r) mu_new^r).
    Returns the largest relative change of the full (unrelaxed) update."""
    import numpy as np
    from devsim import get_node_model_list, set_node_values
    from simws.tcad import cryo as C
    st = _CRYO[(device, region)]
    phys, mat, T = st["phys"], st["mat"], st["T"]
    ND, NA = _node(device, region, "NDtot"), _node(device, region, "NAtot")
    have = set(get_node_model_list(device=device, region=region))
    n = _node(device, region, "Electrons") if "Electrons" in have else _node(device, region, "N0eq")
    p = _node(device, region, "Holes") if "Holes" in have else _node(device, region, "P0eq")
    if phys.incomplete_ionization:
        npr = C.prime_from_density(n, st["Nc"], phys.fermi_dirac)
        ppr = C.prime_from_density(p, st["Nv"], phys.fermi_dirac)
        ND = ND * C.ionized_fraction(st["pd"], ND, npr, st["Nc"], T, **st["kw"])
        NA = NA * C.ionized_fraction(st["pa"], NA, ppr, st["Nv"], T, **st["kw"])
    change = 0.0
    for c in ("n", "p"):
        new = C.klaassen_mobility(mat, c, NA, ND, n, p, T, phys.donor_species)
        name = f"mu_{c}_node_si"
        if name in have:
            old = _node(device, region, name)
            if old.size and np.all(old > 0):
                change = max(change, float(np.max(np.abs(new / old - 1))))
                if relax < 1.0:
                    new = old * (new / old) ** relax
            else:
                change = 1.0
        set_node_values(device=device, region=region, name=name, values=new.tolist())
    return change


def relax_mobility(device, region, opts, tol=1e-9, max_iter=60) -> int:
    """Outer (Picard) iteration of the carrier-dependent Klaassen mobility to self-consistency after a solve.
    The coupling registry lists this as a two-way coupling inside the adapter process. A Newton failure after a
    mobility update restores the previous mobility and solution and retries with the update halved (log-space
    under-relaxation); the relaxation grows back after successes. The converged state does not depend on it."""
    from devsim import solve
    names = [f"mu_{c}_node_si" for c in ("n", "p")]
    r = 1.0
    for k in range(max_iter):
        saved = {v: _node(device, region, v) for v in names + ["Potential", "Electrons", "Holes"]}
        if update_mobility(device, region, relax=r) < tol:
            _restore(device, region, {v: saved[v] for v in names})   # keep the converged pair unchanged
            return k
        try:
            solve(type="dc", **opts)
            r = min(1.0, 2 * r)
        except Exception:
            _restore(device, region, saved)
            r *= 0.25
            if r < 1e-3:
                raise RuntimeError("Klaassen mobility update did not converge (convergence failure: Newton fails "
                                   "even for a 0.1 % mobility step)")
    raise RuntimeError(f"Klaassen mobility did not converge (convergence failure) in {max_iter} outer iterations")


def _restore(device, region, values):
    from devsim import set_node_values
    for k, v in values.items():
        set_node_values(device=device, region=region, name=k, values=v.tolist())


def solution_arrays(device, region) -> Dict[str, list]:
    """Node values of Potential, Electrons, Holes (for temperature continuation)."""
    return {v: _node(device, region, v).tolist() for v in ("Potential", "Electrons", "Holes")}


def _fd_bgn_correction(mat, n_i_300) -> str:
    """-kT ln(gamma) at 300 K for the majority carriers of the local doping (node expression, eV)."""
    from simws.tcad import cryo as C
    Nc3, Nv3, _ = C.band_params(mat, 300.0, n_i_300)
    kT = K_B_EV * 300.0
    nd, na = "max(0.5*(Ntot + NetDoping), 1)", "max(0.5*(Ntot - NetDoping), 1)"
    gn, gp = C.ln_gamma_expr(f"({nd})/{Nc3!r}"), C.ln_gamma_expr(f"({na})/{Nv3!r}")
    return f"(-{kT!r}*ifelse(NetDoping > 0, {gn}, {gp}))"
