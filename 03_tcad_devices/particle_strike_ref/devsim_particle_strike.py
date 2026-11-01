#!/usr/bin/env python3
"""
DEVSIM replica of Sandia Charon's `particleStrike` nightly test (diode-particleStrike.sg.inp).

Device:   2D Si pn diode, x in [0,1] um, y in [0,0.5] um, uniform 0.01 x 0.05 um grid (same nodes as Charon).
          p-side (NA = 1e16) for x < 0.5 um, n-side (ND = 1e16) for x > 0.5 um. Anode at x = 0 (+1 V,
          forward bias), cathode at x = 1 um. Constant mobilities 1000 / 500 cm^2/Vs, eps_r = 11.9,
          no recombination (SRH, radiative, Auger all off).
Strike:   track x = 0.5 um, y: 0.5 -> 0 um, radius 0.1 um. Charon lights a whole cell if any of its nodes
          is within the radius (here: the 22 cell columns 0.39 < x < 0.61 um, see --lit-half-width). Rate = G0 * f(t) with
          G0 = 1e26 cm^-3 and f(t) a unit-area Gaussian, sigma = (t_end - t_start)/6, centred mid-window,
          and zero outside [t_start, t_end] (Charon_Ionization_ParticleStrike_impl.hpp).
Output:   CSV of time, anode/cathode current (A per cm of depth, same convention as Charon's 2D output).
"""
import argparse
import csv
import math

import numpy as np
from devsim import (add_2d_contact, add_2d_mesh_line, add_2d_region, create_2d_mesh, create_device,
                    finalize_mesh, get_contact_current, get_node_model_values, set_node_values,
                    set_parameter, solve)
from devsim.python_packages import simple_physics as sp
from devsim.python_packages.model_create import CreateNodeModel, CreateSolution

UM = 1e-4  # cm per um


def build(device="diode", region="si", nx=100, ny=10):
    create_2d_mesh(mesh="m")
    for i in range(nx + 1):
        add_2d_mesh_line(mesh="m", dir="x", pos=i * UM / nx, ps=UM / nx)
    for j in range(ny + 1):
        add_2d_mesh_line(mesh="m", dir="y", pos=j * 0.5 * UM / ny, ps=0.5 * UM / ny)
    # thin dummy regions outside x = 0 and x = 1 um: DEVSIM places 2D contacts on region boundaries
    add_2d_mesh_line(mesh="m", dir="x", pos=-1e-8, ps=1e-8)
    add_2d_mesh_line(mesh="m", dir="x", pos=UM + 1e-8, ps=1e-8)
    add_2d_region(mesh="m", material="Si", region=region, xl=0, xh=UM)
    add_2d_region(mesh="m", material="Si", region="pad_l", xl=-1e-8, xh=0)
    add_2d_region(mesh="m", material="Si", region="pad_r", xl=UM, xh=UM + 1e-8)
    add_2d_contact(mesh="m", name="anode", material="metal", region=region, xl=0, xh=0, yl=0, yh=0.5 * UM, bloat=1e-10)
    add_2d_contact(mesh="m", name="cathode", material="metal", region=region, xl=UM, xh=UM, yl=0, yh=0.5 * UM, bloat=1e-10)
    finalize_mesh(mesh="m")
    create_device(mesh="m", device=device)
    return device, region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="devsim_particle_strike.csv")
    ap.add_argument("--n-i", type=float, default=1.0e10, help="intrinsic density [cm^-3]")
    ap.add_argument("--nx", type=int, default=100)
    ap.add_argument("--ny", type=int, default=10)
    ap.add_argument("--lit-half-width", type=float, default=0.11,
                    help="half-width of the lit cell band [um]; 0.11 because Charon's test |x-0.5| < 0.1 "
                         "also catches the nodes at x = 0.40 and 0.60 (0.5-0.4 = 0.0999... in floating point)")
    ap.add_argument("--dc-only", action="store_true", help="stop after the 1 V DC point")
    ap.add_argument("--dt-pulse", type=float, default=2.5e-6, help="time step inside the strike window [s]")
    a = ap.parse_args()

    dev, reg = build(nx=a.nx, ny=a.ny)
    sp.SetSiliconParameters(dev, reg, 300)
    for name, val in (("Permittivity", 11.9 * 8.85418781e-14), ("n_i", a.n_i), ("n1", a.n_i), ("p1", a.n_i),
                      ("mu_n", 1000.0), ("mu_p", 500.0), ("taun", 1e30), ("taup", 1e30), ("Gscale", 0.0)):
        set_parameter(device=dev, region=reg, name=name, value=val)
    CreateNodeModel(dev, reg, "Acceptors", "1.0e16*step(0.5e-4 - x)")
    CreateNodeModel(dev, reg, "Donors", "1.0e16*step(x - 0.5e-4)")
    CreateNodeModel(dev, reg, "NetDoping", "Donors - Acceptors")

    # Strike footprint, weighted by the fraction of each node's control volume inside lit cells, then
    # normalised so the integrated footprint equals the lit band area exactly (2*h x 0.50 um).
    x = np.array(get_node_model_values(device=dev, region=reg, name="x")) / UM
    h = a.lit_half_width
    dx = np.abs(x - 0.5)
    lit = np.where(dx < h - 1e-9, 1.0, np.where(np.isclose(dx, h), 0.5, 0.0))
    vol = np.array(get_node_model_values(device=dev, region=reg, name="NodeVolume"))
    target_area = 2 * h * UM * 0.50 * UM
    lit *= target_area / float((lit * vol).sum())
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
    # Add the strike generation to both continuity equations (sign convention of simple_physics.CreateSRH).
    CreateNodeModel(dev, reg, "Gstrike", "Gscale * StrikeFootprint")
    CreateNodeModel(dev, reg, "ElectronGeneration", "-ElectronCharge * (USRH - Gstrike)")
    CreateNodeModel(dev, reg, "HoleGeneration", "+ElectronCharge * (USRH - Gstrike)")
    solve(type="dc", absolute_error=1e10, relative_error=1e-6, maximum_iterations=60)

    v = 0.0  # ramp the anode to +1 V
    while v < 1.0 - 1e-12:
        v = min(1.0, v + 0.05)
        set_parameter(device=dev, name=sp.GetContactBiasName("anode"), value=v)
        solve(type="dc", absolute_error=1e10, relative_error=1e-6, maximum_iterations=60)

    def current(c):
        return (get_contact_current(device=dev, contact=c, equation="ElectronContinuityEquation")
                + get_contact_current(device=dev, contact=c, equation="HoleContinuityEquation"))

    if a.dc_only:
        print(f"DC at 1 V: cathode {current('cathode'):.6f} A/cm (n_i={a.n_i:.4g})")
        return
    G0, t_on, t_off = 1.0e26, 1.0e-4, 1.0e-3
    sigma, mid = (t_off - t_on) / 6.0, 0.5 * (t_on + t_off)

    def f(t):
        return 0.0 if (t < t_on or t > t_off) else math.exp(-(t - mid) ** 2 / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))

    times = list(np.geomspace(1e-6, t_on, 20))
    times += list(np.arange(t_on + a.dt_pulse, t_off + 0.5 * a.dt_pulse, a.dt_pulse))
    times += list(np.geomspace(t_off + 1e-6, 1e-2, 25))
    rows = [(0.0, current("anode"), current("cathode"), 0.0)]
    solve(type="transient_dc", absolute_error=1e10, relative_error=1e-6, maximum_iterations=60)
    t_prev = 0.0
    for t in times:
        set_parameter(device=dev, region=reg, name="Gscale", value=G0 * f(t))  # BDF1: source at t_{n+1}
        solve(type="transient_bdf1", absolute_error=1e10, relative_error=1e-6, maximum_iterations=60,
              tdelta=t - t_prev, charge_error=1e3)
        rows.append((t, current("anode"), current("cathode"), G0 * f(t)))
        t_prev = t
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "anode_A_per_cm", "cathode_A_per_cm", "G_peak_cm3_s"])
        w.writerows(rows)
    Ic = np.array([r[2] for r in rows]); tt = np.array([r[0] for r in rows])
    print(f"DEVSIM particleStrike: DC cathode current {Ic[1]:.6f} A/cm, peak {Ic.max():.4f} A/cm at "
          f"t={tt[Ic.argmax()]:.4e} s, collected charge {np.trapezoid(Ic - Ic[1], tt):.5e} C/cm -> {a.out}")


if __name__ == "__main__":
    main()
