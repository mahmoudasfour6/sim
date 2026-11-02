"""Axisymmetric (2D cylindrical r-z) device simulation in DEVSIM, for ion strikes along the symmetry axis.

A heavy-ion track is a cylinder; a planar 2D simulation turns it into a sheet repeated along the depth, which
over-states the deposited charge per unit volume far from the track and makes collected charge and critical
LET depend on an arbitrary depth W. In r-z geometry the track is exactly round and every quantity is absolute:
currents in A, charges in fC, no W.

Coordinates: x = r (radius, axis at x = 0), y = z (depth, surface at y = 0). DEVSIM's cylindrical node volumes
and edge couplings include the full 2 pi (checked by `resistor_check` against I = q n mu V pi R^2 / L).

    from simws.tcad.axisym import AxiSEUCase, seu_axisymmetric
    r = seu_axisymmetric(AxiSEUCase(let=10.0))
    r["collected_charge_drain_fC"], r["deposited_charge_fC"]

Physics: simws.tcad.physics (Masetti mobility, Scharfetter SRH, Auger, band-gap narrowing, optional velocity
saturation and impact ionization, all at temperature T). The drain is held at a fixed bias (the standard
collected-charge / current-pulse simulation); couple the pulse to a circuit afterwards, or use
`load_ohm` / `load_farad` for a mixed-mode load (tested at low substrate doping, see mixed_mode notes).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import simws.tcad  # noqa: F401
from simws.tcad.physics import Physics, contact_current, setup_region

UM = 1e-4
_CYL = {"node_volume_model": "CylindricalNodeVolume", "edge_couple_model": "CylindricalEdgeCouple",
        "edge_node0_volume_model": "CylindricalEdgeNodeVolume@n0",
        "edge_node1_volume_model": "CylindricalEdgeNodeVolume@n1",
        "element_edge_couple_model": "ElementCylindricalEdgeCouple",
        "element_node0_volume_model": "ElementCylindricalNodeVolume@en0",
        "element_node1_volume_model": "ElementCylindricalNodeVolume@en1",
        "surface_area_model": "CylindricalSurfaceArea"}


def enable_cylindrical(device, regions, raxis="x", raxis_zero=0.0):
    """Switch DEVSIM's geometric models to r-z cylindrical. Call after create_device, before any physics.
    (simws's reset_devsim restores the Cartesian defaults.)"""
    from devsim import cylindrical_edge_couple, cylindrical_node_volume, cylindrical_surface_area, set_parameter
    set_parameter(name="raxis_variable", value=raxis)
    set_parameter(name="raxis_zero", value=raxis_zero)
    for r in regions:
        cylindrical_node_volume(device=device, region=r)
        cylindrical_edge_couple(device=device, region=r)
        cylindrical_surface_area(device=device, region=r)
    for k, v in _CYL.items():
        set_parameter(name=k, value=v)


def _mesh(radius_um, height_um, r_lines, z_lines, drain_radius_um=None):
    """r-z mesh with thin contact pads; drain contact on the top surface for r < drain_radius (whole top if None),
    substrate contact over the whole bottom."""
    from devsim import (add_2d_contact, add_2d_mesh_line, add_2d_region, create_2d_mesh, create_device,
                        finalize_mesh)
    create_2d_mesh(mesh="m")
    for pos, ps in r_lines:
        add_2d_mesh_line(mesh="m", dir="x", pos=pos * UM, ps=ps * UM)
    for pos, ps in z_lines:
        add_2d_mesh_line(mesh="m", dir="y", pos=pos * UM, ps=ps * UM)
    add_2d_mesh_line(mesh="m", dir="y", pos=-1e-8, ps=1e-8)
    add_2d_mesh_line(mesh="m", dir="y", pos=height_um * UM + 1e-8, ps=1e-8)
    add_2d_region(mesh="m", material="Si", region="si", yl=0, yh=height_um * UM)
    add_2d_region(mesh="m", material="Si", region="pad_t", yl=-1e-8, yh=0)
    add_2d_region(mesh="m", material="Si", region="pad_b", yl=height_um * UM, yh=height_um * UM + 1e-8)
    xh = (drain_radius_um if drain_radius_um else radius_um) * UM
    add_2d_contact(mesh="m", name="drain", material="metal", region="si", yl=0, yh=0, xl=0, xh=xh, bloat=1e-10)
    add_2d_contact(mesh="m", name="sub", material="metal", region="si", yl=height_um * UM, yh=height_um * UM,
                   xl=0, xh=radius_um * UM, bloat=1e-10)
    finalize_mesh(mesh="m")
    create_device(mesh="m", device="d")
    enable_cylindrical("d", ("si", "pad_t", "pad_b"))


def resistor_check(radius_um=5.0, length_um=10.0, donors=1e16, V=0.01, material=None):
    """Uniform n-type cylinder, contacts on both flat ends: I = q n mu V pi R^2 / L exactly (Masetti mu)."""
    from devsim import reset_devsim, set_parameter, solve
    from simws.materials import load
    from simws.tcad.physics import masetti_numpy
    from simws.units import Q_E
    mat = material or load("silicon")
    reset_devsim()
    _mesh(radius_um, length_um, [(0, 0.25), (radius_um, 0.25)], [(0, 0.25), (length_um, 0.25)])
    setup_region("d", "si", mat, f"{donors}", f"{donors}", Physics(srh=False, auger=False, bgn=False),
                 {"drain": False, "sub": False})
    set_parameter(device="d", name="drain_bias", value=V)
    solve(type="dc", absolute_error=1e10, relative_error=1e-12, maximum_iterations=40)
    I = abs(contact_current("d", "drain"))
    I_an = Q_E * donors * float(masetti_numpy(mat, donors)) * V * math.pi * (radius_um * UM) ** 2 / (length_um * UM)
    return {"I_A": I, "I_analytic_A": I_an, "rel_diff": I / I_an - 1, "pass": bool(abs(I / I_an - 1) < 1e-6)}


@dataclass
class AxiSEUCase:
    radius_um: float = 10.0
    height_um: float = 10.0
    drain_radius_um: float = 1.0          # n+ drain implant and contact, centred on the track
    nplus_depth_um: float = 0.2
    ND: float = 1e20
    NA: float = 1e17
    V_drain: float = 1.0
    let: object = 10.0                    # MeV cm2/mg, constant or callable of depth (um)
    track_length_um: float = 10.0
    track_radius_um: float = 0.05
    t_peak_s: float = 4e-12
    t_sigma_s: float = 1e-12
    t_end_s: float = 1e-9
    dt_fine_s: float = 2.5e-13
    max_step_splits: int = 10
    physics: Physics = field(default_factory=lambda: Physics(field_mobility=True))


def seu_axisymmetric(case: AxiSEUCase = AxiSEUCase(), material=None, progress=None):
    """Ion strike down the axis of a cylindrical n+/p diode held at V_drain. Returns current transients (A),
    collected charges (fC) at drain and substrate, deposited charge, and the mesh check of the track."""
    from devsim import get_node_model_values, reset_devsim, set_node_values, set_parameter, solve
    from devsim.python_packages.model_create import CreateSolution
    from simws.materials import load
    from simws.radiation.track import IonTrack
    from simws.tcad.mixed_mode import _time_grid
    from simws.tcad.solve import ramp_bias

    mat = material or load("silicon")
    reset_devsim()
    a = case.track_radius_um
    fine = max(a / 6, 0.004)        # track core: lumped-volume correction 0.94 here (0.84 at a/2.5); charge is
                                    # conserved exactly either way, the factor measures shape resolution
    rl = sorted({(0, fine), (3 * a, 2 * fine), (case.drain_radius_um, 0.05), (case.radius_um, 1.0)})
    zl = [(0, 0.01), (case.nplus_depth_um, 0.01), (1.0, 0.05), (case.height_um, 0.5)]
    _mesh(case.radius_um, case.height_um, rl, zl, case.drain_radius_um)
    Rd, d = case.drain_radius_um * UM, case.nplus_depth_um * UM
    nplus = f"{case.ND}*step({Rd} - x)*step({d} - y)"
    set_parameter(device="d", region="si", name="Gscale", value=0.0)
    CreateSolution("d", "si", "TrackFootprint")
    used = setup_region("d", "si", mat, f"{nplus} - {case.NA}", f"{nplus} + {case.NA}", case.physics,
                        {"drain": False, "sub": False}, extra_generation="Gscale*TrackFootprint")
    track = IonTrack(let=case.let, x0_um=0.0, length_um=case.track_length_um, radius_um=a,
                     t_peak_s=case.t_peak_s, t_sigma_s=case.t_sigma_s)
    r = np.array(get_node_model_values(device="d", region="si", name="x")) / UM
    z = np.array(get_node_model_values(device="d", region="si", name="y")) / UM
    vol = np.array(get_node_model_values(device="d", region="si", name="CylindricalNodeVolume"))  # incl. 2 pi
    foot, mesh_factor = track.footprint_axisymmetric(mat, r, z, vol)
    set_node_values(device="d", region="si", name="TrackFootprint", values=list(foot))

    opts = dict(absolute_error=1e10, relative_error=1e-6, maximum_iterations=40)
    ramp_bias("d", "drain", case.V_drain, step=0.1, **opts)

    def currents():
        return contact_current("d", "drain"), contact_current("d", "sub")

    i0 = currents()
    solve(type="transient_dc", **opts)
    rows = [(0.0, *i0, 0.0)]
    n_split = 0

    def advance(t0, t1, level=0):
        nonlocal n_split
        set_parameter(device="d", region="si", name="Gscale", value=track.temporal(t1))
        try:
            solve(type="transient_bdf1", tdelta=t1 - t0, charge_error=1e3, **opts)
        except Exception:
            if level >= case.max_step_splits:
                raise RuntimeError(f"axisymmetric SEU: no convergence at t = {t1:.3e} s after {level} halvings")
            n_split += 1
            tm = 0.5 * (t0 + t1)
            advance(t0, tm, level + 1)
            advance(tm, t1, level + 1)
            return
        rows.append((t1, *currents(), track.temporal(t1)))
        if progress:
            progress(t1)

    t_prev = 0.0
    for t in _time_grid(case.t_end_s, case.t_peak_s + 6 * case.t_sigma_s, case.dt_fine_s):
        advance(t_prev, t)
        t_prev = t
    a_ = np.array(rows)
    t, i_d, i_s = a_[:, 0], a_[:, 1] - i0[0], a_[:, 2] - i0[1]
    q_d = float(np.trapezoid(i_d, t)) * 1e15
    q_s = float(np.trapezoid(i_s, t)) * 1e15
    return {
        "case": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in case.__dict__.items()},
        "physics": used, "track_mesh_correction": mesh_factor, "time_step_splits": n_split,
        "deposited_charge_fC": track.total_charge_fC(mat),
        "collected_charge_drain_fC": abs(q_d), "collected_charge_sub_fC": abs(q_s),
        "peak_drain_current_A": float(np.max(np.abs(i_d))), "t_peak_current_s": float(t[np.argmax(np.abs(i_d))]),
        "dc_leakage_drain_A": i0[0],
        "time_s": t, "i_drain_A": i_d, "i_sub_A": i_s,
    }
