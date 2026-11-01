"""Mixed-mode device + circuit simulation in DEVSIM (single process, fully coupled Newton).

DEVSIM solves circuit elements (V, R, C) together with the device equations. A 2D device's contact current is
per unit depth (A/cm); a real device is W um deep, so circuit elements are scaled per unit depth:
    R_eff = R * W_cm   (Ohm cm),   C_eff = C / W_cm   (F/cm),   I_actual = I_device * W_cm.
`CircuitScale` does this so callers specify physical R (Ohm), C (F) and W (um).

seu_diode()  heavy-ion strike on a reverse-biased n+/p junction whose n+ node hangs on a load (R to V_DD,
             C to ground): returns the node-voltage transient and collected charge. Physics: drift-diffusion with
             SRH, Caughey-Thomas velocity saturation, Van Overstraeten-de Man impact ionization, ion track from
             simws.radiation.track (constant LET, SRIM or Geant4 profile).
rc_check()   verification: a resistor device in an RC circuit vs the analytic exponential.
"""
from __future__ import annotations

import simws.tcad  # noqa: F401  (safe devsim.reset_devsim)

import math
from dataclasses import dataclass, field

import numpy as np

UM = 1e-4


@dataclass
class CircuitScale:
    depth_um: float

    @property
    def W_cm(self):
        return self.depth_um * UM

    def R(self, ohm):
        return ohm * self.W_cm

    def C(self, farad):
        return farad / self.W_cm

    def I(self, amp_per_cm):
        return amp_per_cm * self.W_cm


def _contacts_2d(mesh, region, width_cm, height_cm, top=("drain", None), bottom=("sub", None)):
    from devsim import add_2d_contact
    name_t, xr_t = top
    xl, xh = xr_t if xr_t else (0.0, width_cm)
    add_2d_contact(mesh=mesh, name=name_t, material="metal", region=region, yl=0, yh=0, xl=xl, xh=xh, bloat=1e-10)
    add_2d_contact(mesh=mesh, name=bottom[0], material="metal", region=region, yl=height_cm, yh=height_cm,
                   xl=0, xh=width_cm, bloat=1e-10)


def _mesh_2d(width_um, height_um, x_lines, y_lines):
    """x_lines/y_lines: lists of (position_um, spacing_um). Thin pad regions above/below carry the contacts."""
    from devsim import add_2d_mesh_line, add_2d_region, create_2d_mesh
    create_2d_mesh(mesh="m")
    for pos, ps in x_lines:
        add_2d_mesh_line(mesh="m", dir="x", pos=pos * UM, ps=ps * UM)
    for pos, ps in y_lines:
        add_2d_mesh_line(mesh="m", dir="y", pos=pos * UM, ps=ps * UM)
    add_2d_mesh_line(mesh="m", dir="y", pos=-1e-8, ps=1e-8)
    add_2d_mesh_line(mesh="m", dir="y", pos=height_um * UM + 1e-8, ps=1e-8)
    add_2d_region(mesh="m", material="Si", region="si", yl=0, yh=height_um * UM)
    add_2d_region(mesh="m", material="Si", region="pad_t", yl=-1e-8, yh=0)
    add_2d_region(mesh="m", material="Si", region="pad_b", yl=height_um * UM, yh=height_um * UM + 1e-8)


def _physics(dev, reg, mat, netdoping, tau, circuit_contacts, field_mobility=True, impact=True, extra_gen=None,
             build_circuit=None):
    """Build the circuit first and create circuit contacts in circuit mode from the start (DEVSIM's recipe).
    Never set a parameter named like a circuit node (e.g. drain_bias): it would shadow the circuit node."""
    from devsim import set_node_values, set_parameter, solve
    from devsim.python_packages import simple_physics as sp
    from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
    from simws.tcad import devsim_materials, models

    if build_circuit:
        build_circuit()
    devsim_materials.apply(dev, reg, mat, tau_n=tau, tau_p=tau)
    CreateNodeModel(dev, reg, "NetDoping", netdoping)
    sp.CreateSiliconPotentialOnly(dev, reg)
    for c in ("drain", "sub"):
        if c not in circuit_contacts:
            set_parameter(name=sp.GetContactBiasName(c), value=0.0)
        sp.CreateSiliconPotentialOnlyContact(dev, reg, c, c in circuit_contacts)
    solve(type="dc", absolute_error=1.0, relative_error=1e-8, maximum_iterations=100)
    CreateSolution(dev, reg, "Electrons"); CreateSolution(dev, reg, "Holes")
    set_node_values(device=dev, region=reg, name="Electrons", init_from="IntrinsicElectrons")
    set_node_values(device=dev, region=reg, name="Holes", init_from="IntrinsicHoles")
    mun, mup = models.field_dependent_mobility(dev, reg, mat) if field_mobility else ("mu_n", "mu_p")
    sp.CreateSiliconDriftDiffusion(dev, reg, mu_n=mun, mu_p=mup)
    for c in ("drain", "sub"):
        sp.CreateSiliconDriftDiffusionAtContact(dev, reg, c, c in circuit_contacts)
    if extra_gen:
        CreateNodeModel(dev, reg, "ElectronGeneration", f"-ElectronCharge * (USRH - {extra_gen})")
        CreateNodeModel(dev, reg, "HoleGeneration", f"+ElectronCharge * (USRH - {extra_gen})")
    if impact:
        models.impact_ionization(dev, reg, mat)
    # 1e-6: with 1e20 cm^-3 doping the relative updates stall at the 1e-7 round-off floor
    solve(type="dc", absolute_error=1e10, relative_error=1e-6, maximum_iterations=60)


def _time_grid(t_end, t_fine_end, dt_fine, n_log=40):
    t = list(np.arange(dt_fine, t_fine_end + 0.5 * dt_fine, dt_fine))
    t += list(np.geomspace(t_fine_end * 1.05, t_end, n_log))
    return t


def rc_check(R_ext=2e4, C=5e-15, V=1.0, depth_um=1.0, donors=1e16, length_um=1.0, height_um=1.0, material=None):
    """Uniform n-type resistor (drain on top, grounded substrate contact at the bottom) as the device. Circuit:
    step source V -- R_ext -- node -- (C || device) -- ground. Analytic node voltage:
    v(t) = V Rd/(R_ext + Rd) (1 - exp(-t / tau)), tau = (C + C_geom) R_ext Rd / (R_ext + Rd), with Rd from the
    device's own DC current and C_geom = eps A / L its geometric capacitance. Returns the worst relative
    deviation over 0 < t < 5 tau (BDF1 with dt = tau/400)."""
    from devsim import (circuit_element, get_circuit_node_value, reset_devsim, solve, set_parameter, create_device,
                        finalize_mesh)
    from devsim.python_packages.simple_physics import GetContactBiasName
    from simws.materials import load

    mat = material or load("silicon")
    reset_devsim()
    sc = CircuitScale(depth_um)
    _mesh_2d(length_um, height_um, [(0, 0.1), (length_um, 0.1)], [(0, 0.02), (height_um, 0.02)])
    _contacts_2d("m", "si", length_um * UM, height_um * UM)
    finalize_mesh(mesh="m"); create_device(mesh="m", device="d")
    node = GetContactBiasName("drain")

    def circuit():
        circuit_element(name="V1", n1="vin", n2="0", value=0.0)
        circuit_element(name="R1", n1="vin", n2=node, value=sc.R(R_ext))
        circuit_element(name="C1", n1=node, n2="0", value=sc.C(C))

    _physics("d", "si", mat, f"{donors}", 1e30, circuit_contacts=("drain",), field_mobility=False, impact=False,
             build_circuit=circuit)
    # device resistance from a DC point (per-depth -> physical)
    circuit_alter = __import__("devsim").circuit_alter
    for v in np.linspace(0, V, 11)[1:]:
        circuit_alter(name="V1", value=float(v))
        solve(type="dc", absolute_error=1e10, relative_error=1e-9, maximum_iterations=60)
    v_inf = get_circuit_node_value(node=node, solution="dcop")
    Rd = R_ext * v_inf / (V - v_inf)
    # the device itself is a parallel-plate capacitor (eps A / L) in parallel with the load capacitor
    C_geom = mat.get("eps_r", "dimensionless") * 8.8541878128e-14 * (length_um * UM) * (depth_um * UM) / (height_um * UM)
    tau = (C + C_geom) * R_ext * Rd / (R_ext + Rd)
    for v in np.linspace(V, 0, 11)[1:]:
        circuit_alter(name="V1", value=float(v))
        solve(type="dc", absolute_error=1e10, relative_error=1e-9, maximum_iterations=60)
    solve(type="transient_dc", absolute_error=1e10, relative_error=1e-9, maximum_iterations=60)
    circuit_alter(name="V1", value=V)
    t_prev, worst, rows = 0.0, 0.0, []
    for t in np.linspace(tau / 400, 5 * tau, 2000):           # dt = tau/400 -> BDF1 error ~ 0.1 %
        # BDF1: DEVSIM's BDF2 factorization fails with circuit elements attached (singular matrix)
        solve(type="transient_bdf1", absolute_error=1e10, relative_error=1e-9,
              maximum_iterations=60, tdelta=t - t_prev, charge_error=1e3)
        v = get_circuit_node_value(node=node, solution="dcop")
        v_an = v_inf * (1 - math.exp(-t / tau))
        worst = max(worst, abs(v - v_an) / v_inf)
        rows.append((t, v, v_an))
        t_prev = t
    return {"tau_s": tau, "C_geom_F": C_geom, "R_device_ohm": Rd, "v_inf_V": v_inf, "worst_abs_dev_frac_of_vinf": worst,
            "samples": rows[:: max(1, len(rows) // 20)]}


@dataclass
class SEUCase:
    width_um: float = 4.0
    height_um: float = 5.0
    drain_halfwidth_um: float = 0.5     # n+ drain contact/implant centred at x = width/2
    nplus_depth_um: float = 0.15
    ND: float = 1e19                    # 1e20 on the fine track mesh stalls Newton (minority-carrier round-off
                                        # in the n+ layer); 1e19 keeps the junction abrupt enough for SEU studies
    NA: float = 1e16                    # p-substrate. With a circuit node attached, NA = 1e17 makes Newton cycle
                                        # on minority electrons (~ni^2/NA, relative-error noise) near 0.6 V
    tau_s: float = 1e-7
    depth_um: float = 1.0               # out-of-plane device width W
    vdd: float = 1.0
    R_load: float = 2e4                 # Ohm, pull-up to V_DD (models the restoring transistor)
    C_node: float = 2e-15               # F
    let: object = 10.0                  # MeV cm2/mg (constant) or callable of depth (um)
    track_length_um: float = 5.0
    track_radius_um: float = 0.05
    t_peak_s: float = 4e-12
    t_sigma_s: float = 1e-12
    t_end_s: float = 1e-9
    dt_fine_s: float = 2.5e-13
    field_mobility: bool = True
    max_step_splits: int = 8            # adaptive halving of a failed time step (down to dt / 256)
    impact_ionization: bool = False     # irrelevant at logic-level bias (alpha ~ 0 below ~2e5 V/cm) and, switched
                                        # on, it makes Newton cycle on minority holes in the n+ layer. Use it for
                                        # high-voltage devices (validated in simws.validation.avalanche).


def seu_diode(case: SEUCase = SEUCase(), material=None):
    from devsim import (circuit_element, create_device, finalize_mesh, get_circuit_node_value, get_contact_current,
                        get_node_model_values, reset_devsim, set_node_values, set_parameter, solve, add_2d_mesh_line)
    from devsim.python_packages.simple_physics import GetContactBiasName
    from devsim.python_packages.model_create import CreateSolution
    from simws.materials import load
    from simws.radiation.track import IonTrack

    mat = material or load("silicon")
    reset_devsim()
    sc = CircuitScale(case.depth_um)
    xc = case.width_um / 2
    fine = max(case.track_radius_um / 2.5, 0.01)
    xl = [(0, 0.2), (xc - 1.0, 0.05), (xc - 3 * case.track_radius_um, fine), (xc, fine),
          (xc + 3 * case.track_radius_um, fine), (xc + 1.0, 0.05), (case.width_um, 0.2)]
    yl = [(0, 0.01), (case.nplus_depth_um, 0.01), (1.0, 0.05), (case.height_um, 0.25)]
    _mesh_2d(case.width_um, case.height_um, sorted(set(xl)), yl)
    _contacts_2d("m", "si", case.width_um * UM, case.height_um * UM,
                 top=("drain", ((xc - case.drain_halfwidth_um) * UM, (xc + case.drain_halfwidth_um) * UM)))
    finalize_mesh(mesh="m"); create_device(mesh="m", device="d")
    dev, reg = "d", "si"
    node = GetContactBiasName("drain")

    def circuit():
        circuit_element(name="VDD", n1="vdd", n2="0", value=0.0)
        circuit_element(name="RL", n1="vdd", n2=node, value=sc.R(case.R_load))
        circuit_element(name="CN", n1=node, n2="0", value=sc.C(case.C_node))

    nplus = (f"{case.ND}*step(y - 0)*step({case.nplus_depth_um * UM} - y)"
             f"*step(x - {(xc - case.drain_halfwidth_um) * UM})*step({(xc + case.drain_halfwidth_um) * UM} - x)")
    netdop = f"{nplus} - {case.NA}"
    set_parameter(device=dev, region=reg, name="Gscale", value=0.0)
    CreateSolution(dev, reg, "TrackFootprint")
    _physics(dev, reg, mat, netdop, case.tau_s, circuit_contacts=("drain",), field_mobility=case.field_mobility,
             impact=False, extra_gen="Gscale * TrackFootprint", build_circuit=circuit)
    x = np.array(get_node_model_values(device=dev, region=reg, name="x")) / UM
    y = np.array(get_node_model_values(device=dev, region=reg, name="y")) / UM
    vol = np.array(get_node_model_values(device=dev, region=reg, name="NodeVolume"))
    track = IonTrack(let=case.let, x0_um=xc, length_um=case.track_length_um, radius_um=case.track_radius_um,
                     t_peak_s=case.t_peak_s, t_sigma_s=case.t_sigma_s, depth_um=case.depth_um)
    foot, mesh_factor = track.footprint(mat, x, y, vol)
    set_node_values(device=dev, region=reg, name="TrackFootprint", values=list(foot))
    from simws.tcad.solve import ramp_source
    opts = dict(absolute_error=1e10, relative_error=1e-6, maximum_iterations=60)
    ramp_source("VDD", case.vdd, step=0.1, **opts)                 # power up through the load (adaptive steps)
    if case.impact_ionization:                                     # switched on at the operating point
        from simws.tcad import models
        models.impact_ionization(dev, reg, mat)
        try:
            solve(type="dc", **opts)
        except Exception as exc:
            raise RuntimeError("SEU: DC solve failed after enabling impact ionization (minority-carrier noise in "
                               "the n+ layer); run with impact_ionization=False at logic-level bias") from exc
    v0 = get_circuit_node_value(node=node, solution="dcop")

    def i_sub():
        return sc.I(get_contact_current(device=dev, contact="sub", equation="ElectronContinuityEquation")
                    + get_contact_current(device=dev, contact="sub", equation="HoleContinuityEquation"))

    i0 = i_sub()
    solve(type="transient_dc", **opts)
    rows = [(0.0, v0, i0, 0.0)]
    t_prev = 0.0
    fine_end = case.t_peak_s + 6 * case.t_sigma_s
    n_split = 0

    def advance(t0, t1, level=0):
        # BDF1 step t0 -> t1; on Newton failure DEVSIM keeps the last converged state, so split the step in two
        nonlocal n_split
        set_parameter(device=dev, region=reg, name="Gscale", value=track.temporal(t1))
        try:
            solve(type="transient_bdf1", tdelta=t1 - t0, charge_error=1e3, **opts)
        except Exception:
            if level >= case.max_step_splits:
                raise RuntimeError(f"SEU transient: no convergence at t = {t1:.3e} s after {level} step halvings")
            n_split += 1
            tm = 0.5 * (t0 + t1)
            advance(t0, tm, level + 1)
            advance(tm, t1, level + 1)
            return
        rows.append((t1, get_circuit_node_value(node=node, solution="dcop"), i_sub(), track.temporal(t1)))

    for t in _time_grid(case.t_end_s, fine_end, case.dt_fine_s):
        advance(t_prev, t)
        t_prev = t
    r = np.array(rows)
    q_gen_fC = track.total_charge_fC(mat)
    q_coll_fC = float(np.trapezoid(np.abs(r[:, 2] - i0), r[:, 0])) * 1e15
    return {
        "case": {k: (v if not callable(v) else "callable") for k, v in case.__dict__.items()},
        "track_mesh_correction": mesh_factor, "time_step_splits": n_split,
        "deposited_charge_fC": q_gen_fC,
        "collected_charge_substrate_fC": q_coll_fC,
        "v_node_initial_V": float(v0), "v_node_min_V": float(r[:, 1].min()),
        "t_vmin_s": float(r[np.argmin(r[:, 1]), 0]),
        "time_s": r[:, 0], "v_node_V": r[:, 1], "i_sub_A": r[:, 2],
    }
