"""Device library: the existing simws devices written as engine-neutral descriptions.

Each function returns a Problem that describes the same structure, models and solver settings as a legacy
engine-bound builder, so a translator can be checked against the legacy result:

    pn_diode_1d()   <->  simws.validation.engine_pairs._devsim_diode   (the DEVSIM/Solcore pair problem)
    npn_hbt_2d()    <->  simws.tcad.hbt.build / gummel                  (flagship A, non-cryogenic settings)

implant_si() is the first entry of the second problem family (contract A7): an ion implant into silicon.
see_si_slab() is the first entry of the third (contract A8): a particle into a silicon block with a sensitive volume.
rc_lowpass(), divider() and diode_iv() are circuits (contract A9).
"""
from __future__ import annotations

from typing import Optional, Sequence

from simws.spec.circuit import (AcSweep, CircuitProblem, DcSweep, Element, Model, Op, Pulse, Transient,
                                 VerilogA)
from simws.spec.see import (BeolLayer, EdepHistogram, SeeGeometry, SeePhysics, SeeProblem, SeeSource,
                             SeeStatistics)
from simws.spec.implant import Atom, Beam, ImplantModels, ImplantProblem, Layer, Statistics, Target
from simws.spec.schema import (Box, Contact, DCSweep, DeviceSpec, DopingTerm, Equilibrium, GradeTerm, MeshHints,
                               ModelSpec, Numerics, Problem, Region, Track)


def pn_diode_1d(NA_cm3: float = 1e17, ND_cm3: float = 1e16, junction_um: float = 1.0, length_um: float = 4.0,
                mu_n: float = 1000.0, mu_p: float = 400.0, tau_s: float = 1e-7, T: float = 300.0,
                nodes: int = 801, V: Optional[Sequence[float]] = None) -> Problem:
    """Abrupt silicon pn diode, p side at x < junction_um with the anode at x = 0. Constant mobilities and
    lifetimes, Boltzmann statistics, no band-gap narrowing, no Auger: the reference problem of the engine pairs."""
    h = length_um / (nodes - 1)
    dev = DeviceSpec(
        name="pn_diode_1d", dimension=1,
        regions=[Region(name="si", material="silicon", box=Box(x=(0.0, length_um)))],
        contacts=[Contact(name="anode", region="si", side="x_min"),
                  Contact(name="cathode", region="si", side="x_max")],
        doping=[DopingTerm(species="acceptor", concentration_cm3=NA_cm3, box=Box(x=(None, junction_um)), label="p side"),
                DopingTerm(species="donor", concentration_cm3=ND_cm3, box=Box(x=(junction_um, None)), label="n side")],
        mesh=MeshHints(x_lines=[(0.0, h), (length_um, h)]),
        description="abrupt Si pn diode, anode on the p side")
    models = ModelSpec(temperature_K=T, mobility="constant", mu_n_cm2_Vs=mu_n, mu_p_cm2_Vs=mu_p, srh="constant",
                       tau_n_s=tau_s, tau_p_s=tau_s, auger=False, band_gap_narrowing="off")
    analysis = Equilibrium() if V is None else DCSweep(contact="anode", values_V=[float(v) for v in V], step_V=0.05)
    return Problem(device=dev, models=models, analysis=analysis,
                   numerics=Numerics(relative_error=1e-12, absolute_error=1e10, max_iterations=80))


def n_bar_1d(N_D_cm3: float = 1e16, length_um: float = 2.0, mu_n: float = 1000.0, mu_p: float = 400.0,
             nodes: int = 2001, T: float = 300.0, V: Optional[Sequence[float]] = None) -> Problem:
    """Uniform n-type silicon bar (contract A11.3): contacts "left" (x_min) and "right" (x_max), constant mobilities,
    no SRH, Auger or band-gap narrowing, Boltzmann statistics, uniform mesh; a dc sweep of "right" or equilibrium."""
    h = length_um / (nodes - 1)
    dev = DeviceSpec(
        name="n_bar_1d", dimension=1,
        regions=[Region(name="si", material="silicon", box=Box(x=(0.0, length_um)))],
        contacts=[Contact(name="left", region="si", side="x_min"), Contact(name="right", region="si", side="x_max")],
        doping=[DopingTerm(species="donor", concentration_cm3=N_D_cm3, box=Box(), label="uniform")],
        mesh=MeshHints(x_lines=[(0.0, h), (length_um, h)]),
        description="uniform n-type silicon bar")
    models = ModelSpec(temperature_K=T, mobility="constant", mu_n_cm2_Vs=mu_n, mu_p_cm2_Vs=mu_p, srh="off",
                       auger=False, band_gap_narrowing="off")
    analysis = Equilibrium() if V is None else DCSweep(contact="right", values_V=[float(v) for v in V], step_V=0.05)
    return Problem(device=dev, models=models, analysis=analysis,
                   numerics=Numerics(relative_error=1e-12, absolute_error=1e10, max_iterations=80))


def npn_hbt_2d(width_um: float = 1.5, depth_um: float = 1.5, w_e: float = 0.5, t_e: float = 0.03, w_b: float = 0.05,
               N_E: float = 1e20, N_B: float = 2e18, N_C: float = 5e16, N_SC: float = 1e19, y_sc: float = 1.0,
               x_xb: float = 0.7, y_xb: float = 0.1, x_bc: float = 0.9, ge_peak: float = 0.15, ge_start: float = 0.0,
               tau_s: float = 1e-6, T: float = 300.0, mesh_scale: float = 1.0, eb_refine: bool = False,
               cryo: bool = False,
               VBE: Optional[Sequence[float]] = None, VCB: float = 0.0) -> Problem:
    """2D npn SiGe HBT (ge_peak = 0: the Si BJT twin); the geometry, doping and mesh of simws.tcad.hbt.HBTCase with
    its default (non-cryogenic, eb_refine off) settings. y is depth from the top surface; the emitter and base
    contacts are on top (y_min), the collector contact covers the bottom (y_max).

    eb_refine=True uses the legacy refined junction mesh (what the legacy adapter switches on at paper fidelity).
    cryo=True uses the legacy cold-device physics (HBTCase(cryo=True)), needed below about 250 K.

    The Gummel analysis follows the legacy sweep: at each VBE the collector is ramped to VBE + VCB (0.1 V steps),
    then the base to VBE (0.05 V steps); the emitter stays at 0 V."""
    yb = t_e + w_b
    doping = [
        DopingTerm(species="donor", concentration_cm3=N_E, box=Box(x=(None, w_e), y=(None, t_e)), include_upper=True,
                   label="emitter"),
        DopingTerm(species="donor", concentration_cm3=N_C, box=Box(y=(yb, None)), include_upper=True, label="collector"),
        DopingTerm(species="donor", concentration_cm3=N_SC, box=Box(y=(y_sc, None)), include_upper=True,
                   label="subcollector"),
        DopingTerm(species="acceptor", concentration_cm3=N_B, box=Box(y=(None, yb)), include_upper=True, label="base"),
        DopingTerm(species="acceptor", concentration_cm3=1e20, box=Box(x=(x_xb, None), y=(None, y_xb)),
                   include_upper=True, label="extrinsic base"),
    ]
    grading = []
    if ge_peak > 0 or ge_start > 0:
        grading = [GradeTerm(element="Ge", axis="y", start_um=t_e, stop_um=yb, x_start=ge_start, x_stop=ge_peak)]
    x_lines = sorted({(0.0, 0.05), (w_e - 0.05, 0.02), (w_e, 0.01), (w_e + 0.05, 0.02), (x_xb, 0.02), (x_bc, 0.05),
                      (width_um, 0.1)})
    y_lines = [(0.0, 0.005), (t_e, 0.002), (yb, 0.002), (yb + 0.1, 0.01), (y_sc, 0.05), (depth_um, 0.1)]
    if eb_refine:
        # contract A5: the legacy HBTCase.eb_refine mesh. The emitter-base doping step is abrupt and the default
        # mesh (2 nm there) is outside the asymptotic range; this grades the spacing down to 0.05 nm at the
        # emitter-base junction and 0.2 nm at the base-collector junction.
        x_lines = sorted({(x, h) for x, h in x_lines if abs(x - w_e) > 0.009} |
                         {(w_e - 0.008, 0.001), (w_e - 0.002, 0.0002), (w_e, 0.00005), (w_e + 0.002, 0.0002),
                          (w_e + 0.008, 0.001)})
        y_lines = sorted([(0.0, 0.005), (t_e - 0.008, 0.001), (t_e - 0.002, 0.0002), (t_e, 0.00005),
                          (t_e + 0.002, 0.0002), (t_e + 0.008, 0.001), (yb - 0.003, 0.0005), (yb, 0.0002),
                          (yb + 0.003, 0.0005), (yb + 0.1, 0.01), (y_sc, 0.05), (depth_um, 0.1)])
    dev = DeviceSpec(
        name="npn_hbt_2d", dimension=2,
        regions=[Region(name="si", material="silicon", box=Box(x=(0.0, width_um), y=(0.0, depth_um)))],
        contacts=[Contact(name="emitter", region="si", side="y_min", span=(0.0, w_e)),
                  Contact(name="base", region="si", side="y_min", span=(x_bc, width_um)),
                  Contact(name="collector", region="si", side="y_max")],
        doping=doping, grading=grading,
        mesh=MeshHints(x_lines=x_lines, y_lines=y_lines, scale=mesh_scale),
        description="2D npn SiGe HBT (legacy HBTCase layout)")
    # cryo=True is the legacy HBTCase(cryo=True) physics (contract A5): Fermi-Dirac statistics, Klaassen mobility,
    # Altermatt incomplete ionization; the engine switches to extended precision below 200 K on its own
    models = ModelSpec(temperature_K=T, mobility="klaassen" if cryo else "masetti", srh="constant", tau_n_s=tau_s,
                       tau_p_s=tau_s, auger=True, band_gap_narrowing="slotboom",
                       statistics="fermi_dirac" if cryo else "boltzmann",
                       incomplete_ionization="altermatt" if cryo else "off")
    if VBE is None:
        analysis = Equilibrium()
    else:
        analysis = DCSweep(contact="base", values_V=[float(v) for v in VBE], step_V=0.05,
                           tracking=[Track(contact="collector", offset_V=float(VCB), step_V=0.1)])
    return Problem(device=dev, models=models, analysis=analysis,
                   numerics=Numerics(relative_error=1e-8, absolute_error=1e10, max_iterations=40))


# ---------------------------------------------------------------------------------------------- implantation
# Contract A7.2: every value with its source (project rule 1).
#   isotope masses    AME2020 atomic mass evaluation (Wang et al., Chinese Physics C 45, 030003, 2021), rounded
#   silicon mass      IUPAC standard atomic weight (abridged, 2021)
#   silicon density   8 atoms per cubic cell of edge 0.5431020511 nm (CODATA 2018 lattice parameter of silicon)
#   silicon surface binding, cutoff, bulk binding
#                     RustBCA 3.0.0 example table scripts/materials.py ("example input values only")
#   the ion's cutoff (the silicon cutoff) and surface binding (none): a modelling choice, not data; acceptance
#                     check R14 shows the range does not depend on it
IMPLANT_IONS = {"B": (5, 11.009305), "P": (15, 30.973762), "As": (33, 74.921595)}       # symbol: (Z, isotope mass)
SI_MASS_AMU = 28.085
SI_DENSITY_CM3 = 4.994e22
SI_SURFACE_BINDING_EV, SI_CUTOFF_EV, SI_BULK_BINDING_EV = 4.72, 1.5, 0.0


def implant_si(ion: str = "P", energy_keV: float = 100.0, tilt_deg: float = 0.0, dose_cm2: Optional[float] = None,
               thickness_um: float = 2.0, n_ions: int = 20000, seed: int = 1,
               displacement_eV: Optional[float] = None, crystal: bool = False,
               rotation_deg: Optional[float] = None) -> ImplantProblem:
    """An ion implant into one layer ("si") of amorphous silicon. ion: "B" (11B), "P" (31P) or "As" (75As).
    Amorphous means no channeling: a crystalline wafer shows a deeper tail than this description can give.
    displacement_eV (contract A14): given, the silicon atom gets it and the recoils are followed (damage). There
    is no default: 12.5 eV is silicon's minimum and 36 eV its direction average (Holmstrom, Kuronen and Nordlund,
    Phys. Rev. B 78, 045202, 2008); 15 eV is a value common in BCA work.
    crystal (contract A15): a (100) silicon crystal, for the channeling engine; it needs tilt_deg 7 and
    rotation_deg 30 (boron) or 22 (arsenic), and a dose."""
    if ion not in IMPLANT_IONS:
        raise ValueError(f"unknown ion '{ion}'; known ions: {sorted(IMPLANT_IONS)}")
    Z, m = IMPLANT_IONS[ion]
    beam = Beam(ion=Atom(symbol=ion, Z=Z, mass_amu=m, cutoff_eV=SI_CUTOFF_EV, surface_binding_eV=0.0),
                energy_keV=float(energy_keV), tilt_deg=float(tilt_deg), dose_cm2=dose_cm2,
                rotation_deg=None if rotation_deg is None else float(rotation_deg))
    si = Atom(symbol="Si", Z=14, mass_amu=SI_MASS_AMU, cutoff_eV=SI_CUTOFF_EV,
              surface_binding_eV=SI_SURFACE_BINDING_EV, bulk_binding_eV=SI_BULK_BINDING_EV,
              displacement_eV=None if displacement_eV is None else float(displacement_eV))
    target = Target(atoms=[si], layers=[Layer(name="si", thickness_um=float(thickness_um),
                                              density_cm3={"Si": SI_DENSITY_CM3})],
                    structure="crystalline_si_100" if crystal else "amorphous")
    models = ImplantModels(recoils=displacement_eV is not None)
    return ImplantProblem(beam=beam, target=target, models=models,
                          statistics=Statistics(n_ions=int(n_ions), seed=int(seed)))


# ---------------------------------------------------------------------------------------- single-event deposition
def see_si_slab(particle: str = "alpha", energy_MeV: float = 5.0, sv_thickness_um: float = 10.0,
                sv_depth_um: float = 0.0, bulk_thickness_um: float = 100.0, width_um: float = 1000.0,
                tilt_deg: float = 0.0, beol=None, ion_Z: Optional[int] = None, ion_A: Optional[int] = None,
                hadronic: str = "none", n_events: int = 10000, seed: int = 1, bins: int = 2000,
                high_MeV: Optional[float] = None) -> SeeProblem:
    """A particle into a silicon block (Geant4 "G4_Si") with a sensitive volume as wide as the block (contract A8.2).
    beol: list of (name, material, thickness_um), the beam side first. The histogram is linear from 0 to high_MeV
    (default 1.05 x energy_MeV). Only Geant4's own material data are used: the library adds no numbers of its own."""
    geo = SeeGeometry(bulk_material="G4_Si", bulk_width_um=float(width_um), bulk_thickness_um=float(bulk_thickness_um),
                      sv_depth_um=float(sv_depth_um), sv_width_um=(float(width_um), float(width_um)),
                      sv_thickness_um=float(sv_thickness_um),
                      beol=[BeolLayer(name=n, material=m, thickness_um=float(t)) for n, m, t in (beol or [])])
    # rounded to 12 significant digits as the hub rounds job inputs, so the stored problem id is this problem's id
    hi = float(high_MeV) if high_MeV is not None else float(f"{1.05 * float(energy_MeV):.12g}")
    return SeeProblem(source=SeeSource(particle=particle, ion_Z=ion_Z, ion_A=ion_A, energy_MeV=float(energy_MeV),
                                       tilt_deg=float(tilt_deg)),
                      geometry=geo, physics=SeePhysics(hadronic=hadronic),
                      histogram=EdepHistogram(scale="lin", bins=int(bins), low_MeV=0.0, high_MeV=hi),
                      statistics=SeeStatistics(n_events=int(n_events), seed=int(seed)))


# ------------------------------------------------------------------------------------------------- circuits (A9)
# The Shockley equation (W. Shockley, Bell Syst. Tech. J. 28, 435, 1949) as a Verilog-A model, compiled by OpenVAF.
SHOCKLEY_VA = """// simws (contract A9.2): ideal diode, I = is * (exp(V / (n Vt)) - 1)
`include "disciplines.vams"
module shockley_diode(a, c);
  inout a, c;
  electrical a, c;
  parameter real is = 1e-14 from (0:inf);
  parameter real n = 1.0 from (0:inf);
  analog begin
    I(a, c) <+ is * (limexp(V(a, c) / (n * $vt)) - 1.0);
  end
endmodule
"""


def rc_lowpass(R_ohm: float = 1e3, C_F: float = 1e-9, analysis: str = "ac", points_per_decade: int = 10,
               f_start_Hz: float = 10.0, f_stop_Hz: float = 1e8, step_s: float = 1e-8,
               stop_s: float = 1e-5) -> CircuitProblem:
    """First-order RC low-pass: "vin" (in-0), "r1" (in-out), "c1" (out-0); probe v(out). analysis "ac" or "tran"."""
    if analysis == "ac":
        src = Element(name="vin", kind="vsource", nodes=["in", "0"], value=0.0, ac=1.0)
        an = AcSweep(points_per_decade=points_per_decade, f_start_Hz=f_start_Hz, f_stop_Hz=f_stop_Hz)
    elif analysis == "tran":
        src = Element(name="vin", kind="vsource", nodes=["in", "0"], value=0.0,
                      pulse=Pulse(v1=0.0, v2=1.0, delay_s=0.0, rise_s=1e-12, fall_s=1e-12, width_s=1.0,
                                  period_s=2.0))
        an = Transient(step_s=step_s, stop_s=stop_s, max_step_s=step_s)
    else:
        raise ValueError("analysis: 'ac' or 'tran'")
    return CircuitProblem(title="RC low-pass", elements=[
        src, Element(name="r1", kind="resistor", nodes=["in", "out"], value=R_ohm),
        Element(name="c1", kind="capacitor", nodes=["out", "0"], value=C_F)], analysis=an, probes=["v(out)"])


def divider(R1_ohm: float = 1e3, R2_ohm: float = 3e3, V: float = 1.0) -> CircuitProblem:
    """Resistive divider: "vin" on "in", "r1" (in-mid), "r2" (mid-0); operating point; probes v(mid), i(vin)."""
    return CircuitProblem(title="divider", elements=[
        Element(name="vin", kind="vsource", nodes=["in", "0"], value=V),
        Element(name="r1", kind="resistor", nodes=["in", "mid"], value=R1_ohm),
        Element(name="r2", kind="resistor", nodes=["mid", "0"], value=R2_ohm)],
        analysis=Op(), probes=["v(mid)", "i(vin)"])


def diode_iv(IS_A: float = 1e-14, N: float = 1.0, start: float = 0.30, stop: float = 0.65, step: float = 0.05,
             model: str = "builtin", temperature_C: float = 27.0, tnom_C: float = 27.0) -> CircuitProblem:
    """A diode "d1" (a-0) driven by "vin"; dc sweep of "vin"; probe i(vin) (= minus the diode current).
    model "builtin": the SPICE diode with is, n only; "va": SHOCKLEY_VA through OpenVAF."""
    if model == "builtin":
        m = Model(name="dmod", kind="diode", params={"is": IS_A, "n": N})
    elif model == "va":
        m = Model(name="dmod", kind="va", params={"is": IS_A, "n": N},
                  va=VerilogA(module="shockley_diode", source=SHOCKLEY_VA))
    else:
        raise ValueError("model: 'builtin' or 'va'")
    kind = "diode" if model == "builtin" else "va"
    return CircuitProblem(title=f"diode I-V ({model})", elements=[
        Element(name="vin", kind="vsource", nodes=["a", "0"], value=start),
        Element(name="d1", kind=kind, nodes=["a", "0"], model="dmod")], models=[m],
        analysis=DcSweep(source="vin", start=start, stop=stop, step=step), probes=["i(vin)"],
        temperature_C=temperature_C, tnom_C=tnom_C)
