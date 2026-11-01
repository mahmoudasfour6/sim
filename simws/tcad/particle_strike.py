"""DEVSIM replica of Sandia Charon's `particleStrike` nightly test, validated code-to-code (sim_workspace
commit a73b95f: max deviation 0.9 % of peak over the strike pulse, no fitted parameters).

Charon's strike model (Charon_Ionization_ParticleStrike_impl.hpp): a cell is lit if any of its nodes lies
within `radius` of the track (here x = 0.5 um; 0.5 - 0.4 = 0.0999... < 0.1 in floating point, so the band is
0.39-0.61 um), rate = G0 * f(t), f a unit-area Gaussian with sigma = (t_off - t_on)/6, zero outside the window.
"""
from __future__ import annotations

import simws.tcad  # noqa: F401  (safe devsim.reset_devsim)

import math
from dataclasses import dataclass

import numpy as np

UM = 1e-4  # cm


@dataclass
class StrikeCase:
    bias_V: float = 1.0
    doping: float = 1.0e16           # cm^-3 each side, p for x < 0.5 um
    mu_n: float = 1000.0             # cm^2/Vs (deck values, not the material defaults)
    mu_p: float = 500.0
    G0: float = 1.0e26               # cm^-3 (Gaussian waveform: G0 * unit-area pulse)
    t_on: float = 1.0e-4
    t_off: float = 1.0e-3
    lit_half_width_um: float = 0.11
    nx: int = 100
    ny: int = 10
    dt_pulse: float = 2.5e-6


def _gauss(t, c: StrikeCase):
    if t < c.t_on or t > c.t_off:
        return 0.0
    s, mid = (c.t_off - c.t_on) / 6.0, 0.5 * (c.t_on + c.t_off)
    return math.exp(-(t - mid) ** 2 / (2 * s * s)) / (s * math.sqrt(2 * math.pi))


def run(case: StrikeCase = StrikeCase(), material=None):
    """Returns dict of numpy arrays: time_s, cathode_A_per_cm (Charon sign convention), G_peak."""
    from devsim import (add_2d_contact, add_2d_mesh_line, add_2d_region, create_2d_mesh, create_device,
                        finalize_mesh, get_contact_current, get_node_model_values, set_node_values,
                        set_parameter, solve)
    from devsim.python_packages import simple_physics as sp
    from devsim.python_packages.model_create import CreateNodeModel, CreateSolution

    from simws.materials import load
    from simws.tcad import devsim_materials

    from devsim import reset_devsim
    reset_devsim()                      # each run starts from a clean DEVSIM state
    mat = material or load("silicon", profile="charon")
    dev, reg = "strike", "si"
    create_2d_mesh(mesh="m")
    for i in range(case.nx + 1):
        add_2d_mesh_line(mesh="m", dir="x", pos=i * UM / case.nx, ps=UM / case.nx)
    for j in range(case.ny + 1):
        add_2d_mesh_line(mesh="m", dir="y", pos=j * 0.5 * UM / case.ny, ps=0.5 * UM / case.ny)
    add_2d_mesh_line(mesh="m", dir="x", pos=-1e-8, ps=1e-8)          # thin pads: DEVSIM puts 2D contacts
    add_2d_mesh_line(mesh="m", dir="x", pos=UM + 1e-8, ps=1e-8)      # on region boundaries
    add_2d_region(mesh="m", material="Si", region=reg, xl=0, xh=UM)
    add_2d_region(mesh="m", material="Si", region="pad_l", xl=-1e-8, xh=0)
    add_2d_region(mesh="m", material="Si", region="pad_r", xl=UM, xh=UM + 1e-8)
    for name, xc in (("anode", 0.0), ("cathode", UM)):
        add_2d_contact(mesh="m", name=name, material="metal", region=reg, xl=xc, xh=xc, yl=0, yh=0.5 * UM, bloat=1e-10)
    finalize_mesh(mesh="m")
    create_device(mesh="m", device=dev)

    devsim_materials.apply(dev, reg, mat, mu_n=case.mu_n, mu_p=case.mu_p)
    set_parameter(device=dev, region=reg, name="Gscale", value=0.0)
    CreateNodeModel(dev, reg, "Acceptors", f"{case.doping}*step(0.5e-4 - x)")
    CreateNodeModel(dev, reg, "Donors", f"{case.doping}*step(x - 0.5e-4)")
    CreateNodeModel(dev, reg, "NetDoping", "Donors - Acceptors")

    x = np.array(get_node_model_values(device=dev, region=reg, name="x")) / UM
    vol = np.array(get_node_model_values(device=dev, region=reg, name="NodeVolume"))
    h, dx = case.lit_half_width_um, np.abs(x - 0.5)
    lit = np.where(dx < h - 1e-9, 1.0, np.where(np.isclose(dx, h), 0.5, 0.0))
    lit *= (2 * h * UM * 0.5 * UM) / float((lit * vol).sum())      # integrated footprint = band area exactly
    CreateSolution(dev, reg, "StrikeFootprint")
    set_node_values(device=dev, region=reg, name="StrikeFootprint", values=list(lit))

    sp.CreateSiliconPotentialOnly(dev, reg)
    for c in ("anode", "cathode"):
        set_parameter(device=dev, name=sp.GetContactBiasName(c), value=0.0)
        sp.CreateSiliconPotentialOnlyContact(dev, reg, c)
    solve(type="dc", absolute_error=1.0, relative_error=1e-6, maximum_iterations=100)
    CreateSolution(dev, reg, "Electrons"); CreateSolution(dev, reg, "Holes")
    set_node_values(device=dev, region=reg, name="Electrons", init_from="IntrinsicElectrons")
    set_node_values(device=dev, region=reg, name="Holes", init_from="IntrinsicHoles")
    sp.CreateSiliconDriftDiffusion(dev, reg)
    for c in ("anode", "cathode"):
        sp.CreateSiliconDriftDiffusionAtContact(dev, reg, c)
    CreateNodeModel(dev, reg, "Gstrike", "Gscale * StrikeFootprint")
    CreateNodeModel(dev, reg, "ElectronGeneration", "-ElectronCharge * (USRH - Gstrike)")
    CreateNodeModel(dev, reg, "HoleGeneration", "+ElectronCharge * (USRH - Gstrike)")
    opts = dict(absolute_error=1e10, relative_error=1e-6, maximum_iterations=60)
    solve(type="dc", **opts)
    v = 0.0
    while v < case.bias_V - 1e-12:
        v = min(case.bias_V, v + 0.05)
        set_parameter(device=dev, name=sp.GetContactBiasName("anode"), value=v)
        solve(type="dc", **opts)

    def i_cathode():
        return -(get_contact_current(device=dev, contact="cathode", equation="ElectronContinuityEquation")
                 + get_contact_current(device=dev, contact="cathode", equation="HoleContinuityEquation"))

    times = list(np.geomspace(1e-6, case.t_on, 20))
    times += list(np.arange(case.t_on + case.dt_pulse, case.t_off + 0.5 * case.dt_pulse, case.dt_pulse))
    times += list(np.geomspace(case.t_off + 1e-6, 1e-2, 25))
    t_out, i_out, g_out = [0.0], [i_cathode()], [0.0]
    solve(type="transient_dc", **opts)
    t_prev = 0.0
    for t in times:
        g = case.G0 * _gauss(t, case)
        set_parameter(device=dev, region=reg, name="Gscale", value=g)   # BDF1: source at t_{n+1}
        solve(type="transient_bdf1", tdelta=t - t_prev, charge_error=1e3, **opts)
        t_out.append(t); i_out.append(i_cathode()); g_out.append(g)
        t_prev = t
    return {"time_s": np.array(t_out), "cathode_A_per_cm": np.array(i_out), "G_peak": np.array(g_out),
            "n_i": mat.get("n_i", "cm**-3"), "eps_r": mat.get("eps_r", "dimensionless")}


def compare(sim: dict, gold_time, gold_cathode) -> dict:
    tg, Ig = np.asarray(gold_time), np.asarray(gold_cathode)
    td, Id = sim["time_s"], sim["cathode_A_per_cm"]
    pulse = (tg > 1e-4) & (tg < 1e-3)
    dev = np.interp(tg, td, Id) - Ig
    q_c, q_d = np.trapezoid(Ig - Ig[1], tg), np.trapezoid(Id - Id[1], td)
    return {
        "dc_rel_diff": float(Id[1] / Ig[1] - 1),
        "peak_rel_diff": float(Id.max() / Ig.max() - 1),
        "extra_charge_rel_diff": float(q_d / q_c - 1),
        "max_abs_dev_in_pulse_frac_of_peak": float(np.max(np.abs(dev[pulse])) / Ig.max()),
        "charon": {"dc": float(Ig[1]), "peak": float(Ig.max()), "extra_charge": float(q_c)},
        "devsim": {"dc": float(Id[1]), "peak": float(Id.max()), "extra_charge": float(q_d)},
    }
