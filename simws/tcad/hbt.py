"""2D npn SiGe heterojunction bipolar transistor (and its Si BJT twin) in DEVSIM.

    from simws.tcad.hbt import HBTCase, gummel, early_voltage
    g = gummel(HBTCase(ge_peak=0.15), VBE=np.linspace(0.4, 0.85, 19))
    g["IC_A"], g["IB_A"], g["beta"]

Geometry (x lateral, y depth, um; per um of emitter length out of plane):
    emitter n+ 1e20, y < t_e, x < w_e                    emitter contact on top over x < w_e
    intrinsic base p N_B, t_e < y < t_e + w_b             Ge graded 0 -> ge_peak across it (strained on Si)
    extrinsic base p+ 1e20, x > x_xb, y < y_xb            base contact on top over x > x_bc
    collector n N_C, y > t_e + w_b; subcollector n+ below y_sc; collector contact on the bottom
Physics: simws.tcad.physics with SiGe band offsets and DOS ratios (sige.yaml), Masetti mobility, SRH, Auger,
band-gap narrowing. Currents are in A per um of emitter length.

The collector-current gain of SiGe over Si at equal doping is checked against the generalized Moll-Ross /
Kroemer integral evaluated with the same n_ie(y) profile (simws.validation.sige_checks).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import simws.tcad  # noqa: F401
from simws.tcad.physics import Physics, contact_current, setup_region

UM = 1e-4


@dataclass
class HBTCase:
    width_um: float = 1.5
    depth_um: float = 1.5
    w_e: float = 0.5           # emitter window
    t_e: float = 0.03          # emitter depth
    w_b: float = 0.05          # intrinsic base width
    N_E: float = 1e20
    N_B: float = 2e18
    N_C: float = 5e16
    N_SC: float = 1e19
    y_sc: float = 1.0          # subcollector top
    x_xb: float = 0.7          # extrinsic base starts (lateral)
    y_xb: float = 0.1          # extrinsic base depth
    x_bc: float = 0.9          # base contact starts
    ge_peak: float = 0.15      # Ge fraction at the collector side of the base (0 = Si BJT); linear grade from 0
    ge_start: float = 0.0      # Ge fraction at the emitter side
    tau: float = 1e-6
    T: float = 300.0
    extra_recombination: Optional[str] = None   # node expression, pairs/cm^3/s added to recombination (TID)
    extra_generation: Optional[str] = None
    cryo: bool = False         # 0.9a: freeze-out (Altermatt), Fermi-Dirac statistics, Klaassen mobility
    initial_guess: Optional[dict] = None   # node values from a previous temperature (continuation)
    mesh_scale: float = 1.0    # multiplies every mesh spacing (0.5 = twice as fine in x and y; refinement studies)
    extended_precision: Optional[bool] = None   # None: automatic (cryo below 200 K), see simws.tcad.physics.Physics


def doping(case: HBTCase):
    U = UM
    em = f"step({case.w_e * U} - x)*step({case.t_e * U} - y)"
    nd = f"({case.N_E}*{em} + {case.N_C}*step(y - {(case.t_e + case.w_b) * U}) + {case.N_SC}*step(y - {case.y_sc * U}))"
    na = f"({case.N_B}*step({(case.t_e + case.w_b) * U} - y) + 1e20*step(x - {case.x_xb * U})*step({case.y_xb * U} - y))"
    return f"{nd} - {na}", f"{nd} + {na}"


def ge_profile(case: HBTCase):
    if case.ge_peak <= 0 and case.ge_start <= 0:
        return None
    U = UM
    y0, y1 = case.t_e * U, (case.t_e + case.w_b) * U
    return (f"step(y - {y0})*step({y1} - y)*({case.ge_start} + {case.ge_peak - case.ge_start}*(y - {y0})/{y1 - y0})")


def build(case: HBTCase, material=None, pre_setup=None, extra_charge=None, x_lines=()):
    """pre_setup(): runs after the device exists and before the physics (create solutions such as an ion-track
    footprint used by case.extra_generation). extra_charge: as in simws.tcad.physics.setup_region.
    x_lines: extra (position_um, spacing_um) mesh lines, e.g. to resolve an ion track."""
    from devsim import add_2d_contact, finalize_mesh, create_device, reset_devsim
    from simws.materials import load
    from simws.tcad.mixed_mode import _mesh_2d
    mat = material or load("silicon")
    reset_devsim()
    yb = case.t_e + case.w_b
    xl = sorted({(0, 0.05), (case.w_e - 0.05, 0.02), (case.w_e, 0.01), (case.w_e + 0.05, 0.02), (case.x_xb, 0.02),
                 (case.x_bc, 0.05), (case.width_um, 0.1)} | set(x_lines))
    yl = [(0, 0.005), (case.t_e, 0.002), (yb, 0.002), (yb + 0.1, 0.01), (case.y_sc, 0.05), (case.depth_um, 0.1)]
    s = case.mesh_scale
    _mesh_2d(case.width_um, case.depth_um, [(x, h * s) for x, h in xl], [(y, h * s) for y, h in yl])
    add_2d_contact(mesh="m", name="emitter", material="metal", region="si", yl=0, yh=0, xl=0, xh=case.w_e * UM, bloat=1e-10)
    add_2d_contact(mesh="m", name="base", material="metal", region="si", yl=0, yh=0, xl=case.x_bc * UM,
                   xh=case.width_um * UM, bloat=1e-10)
    add_2d_contact(mesh="m", name="collector", material="metal", region="si", yl=case.depth_um * UM,
                   yh=case.depth_um * UM, xl=0, xh=case.width_um * UM, bloat=1e-10)
    finalize_mesh(mesh="m")
    create_device(mesh="m", device="d")
    if pre_setup:
        pre_setup()
    net, tot = doping(case)
    gen = None
    if case.extra_recombination or case.extra_generation:
        parts = []
        if case.extra_generation:
            parts.append(f"({case.extra_generation})")
        if case.extra_recombination:
            parts.append(f"-({case.extra_recombination})")
        gen = " + ".join(parts)
    phys = Physics(T=case.T, ge=ge_profile(case), tau_n=case.tau, tau_p=case.tau,
                   incomplete_ionization=case.cryo, fermi_dirac=case.cryo,
                   mobility="klaassen" if case.cryo else "masetti", initial_guess=case.initial_guess,
                   extended_precision=case.extended_precision)
    return setup_region("d", "si", mat, net, tot, phys, {"emitter": False, "base": False, "collector": False},
                        extra_generation=gen, extra_charge=extra_charge)


def _currents():
    return contact_current("d", "collector"), contact_current("d", "base"), contact_current("d", "emitter")


def gummel(case: HBTCase, VBE, VCB=0.0, material=None, **build_kw):
    """Forward Gummel plot: emitter grounded, base at VBE, collector at VBE + VCB. Currents in A/um.
    build_kw go to build() (pre_setup, extra_charge, x_lines)."""
    from simws.tcad.solve import ramp_bias
    build(case, material, **build_kw)
    opts = dict(absolute_error=1e10, relative_error=1e-8, maximum_iterations=40)
    if VCB:
        ramp_bias("d", "collector", VCB, step=0.25, **opts)
    IC, IB = [], []
    for v in VBE:
        ramp_bias("d", "collector", v + VCB, step=0.1, **opts)
        ramp_bias("d", "base", v, step=0.05, **opts)
        ic, ib, _ = _currents()
        IC.append(abs(ic) * UM); IB.append(abs(ib) * UM)
    IC, IB = np.array(IC), np.array(IB)
    return {"VBE": np.asarray(VBE, float), "IC_A": IC, "IB_A": IB, "beta": IC / np.where(IB > 0, IB, np.nan)}


def early_voltage(case: HBTCase, VBE=0.7, VCB=(0.0, 0.5, 1.0, 1.5, 2.0), material=None):
    """Forward Early voltage at fixed VBE: V_A = IC / (dIC/dVCB) - VBE, from a linear fit of IC(VCB)."""
    from simws.tcad.solve import ramp_bias
    build(case, material)
    opts = dict(absolute_error=1e10, relative_error=1e-8, maximum_iterations=40)
    ramp_bias("d", "base", VBE, step=0.05, **opts)
    IC = []
    for vcb in VCB:
        ramp_bias("d", "collector", VBE + vcb, step=0.1, **opts)
        IC.append(abs(_currents()[0]) * UM)
    VCB, IC = np.asarray(VCB, float), np.asarray(IC)
    slope, ic0 = np.polyfit(VCB, IC, 1)
    return {"VCB": VCB, "IC_A": IC, "V_A": float(ic0 / slope - VBE) if slope > 0 else float("inf")}
