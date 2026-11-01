"""Validation of simws.tcad.models.

velocity_saturation()   uniform n-resistor current vs the analytic Caughey-Thomas drift current
                        (validated 2026-11-01: worst relative difference 6.7e-8 over 5e2-2e5 V/cm)
avalanche()             electron-injected multiplication in a p+n diode vs McIntyre's formula on the same
                        field, and breakdown voltage vs the ionization-integral and Sze's empirical rule
                        (validated: M within 0.38 % for 1.05 < M < 20; BV 51.30 V vs 51.30 V integral;
                        Sze's empirical 61.6 V is a rough fit, -17 % with Van Overstraeten-de Man coefficients)
"""
import math
import sys

import numpy as np

import simws.tcad  # noqa: F401,E402  (safe devsim.reset_devsim, must precede the devsim imports below)
from devsim import (add_1d_contact, add_1d_mesh_line, add_1d_region, create_1d_mesh, create_device,
                    finalize_mesh, get_contact_current, get_edge_model_values, get_node_model_values,
                    reset_devsim, set_node_values, set_parameter, solve)
from devsim.python_packages import simple_physics as sp
from devsim.python_packages.model_create import CreateNodeModel, CreateSolution

from simws.materials import load
from simws.tcad import devsim_materials, models
from simws.tcad.solve import ramp_bias

si = load("silicon")
Q = 1.602176634e-19
OPTS = dict(absolute_error=1e10, relative_error=1e-8, maximum_iterations=60)


def mesh_1d(lines, dev="d", reg="r"):
    reset_devsim()
    create_1d_mesh(mesh="m")
    for pos, ps, tag in lines:
        add_1d_mesh_line(mesh="m", pos=pos, ps=ps, tag=tag)
    add_1d_contact(mesh="m", name="a", tag="top", material="metal")
    add_1d_contact(mesh="m", name="c", tag="bot", material="metal")
    add_1d_region(mesh="m", material="Si", region=reg, tag1="top", tag2="bot")
    finalize_mesh(mesh="m"); create_device(mesh="m", device=dev)
    return dev, reg


def setup(dev, reg, netdoping, mu_fd=False, ii=False, tau=1e30):
    devsim_materials.apply(dev, reg, si, tau_n=tau, tau_p=tau)
    CreateNodeModel(dev, reg, "NetDoping", netdoping)
    sp.CreateSiliconPotentialOnly(dev, reg)
    for c in ("a", "c"):
        set_parameter(device=dev, name=sp.GetContactBiasName(c), value=0.0)
        sp.CreateSiliconPotentialOnlyContact(dev, reg, c)
    solve(type="dc", absolute_error=1.0, relative_error=1e-8, maximum_iterations=100)
    CreateSolution(dev, reg, "Electrons"); CreateSolution(dev, reg, "Holes")
    set_node_values(device=dev, region=reg, name="Electrons", init_from="IntrinsicElectrons")
    set_node_values(device=dev, region=reg, name="Holes", init_from="IntrinsicHoles")
    mun, mup = models.field_dependent_mobility(dev, reg, si) if mu_fd else ("mu_n", "mu_p")
    sp.CreateSiliconDriftDiffusion(dev, reg, mu_n=mun, mu_p=mup)
    for c in ("a", "c"):
        sp.CreateSiliconDriftDiffusionAtContact(dev, reg, c)
    if ii:
        models.impact_ionization(dev, reg, si)
    solve(type="dc", **OPTS)


def current(dev, c="a"):
    return (get_contact_current(device=dev, contact=c, equation="ElectronContinuityEquation")
            + get_contact_current(device=dev, contact=c, equation="HoleContinuityEquation"))


def velocity_saturation():
    L = 2e-4; ND = 1e17
    dev, reg = mesh_1d([(0, 1e-6, "top"), (L, 1e-6, "bot")])
    setup(dev, reg, f"{ND}", mu_fd=True)
    mu0, vsat, beta = si.get("mu_n", "cm**2/V/s"), si.get("v_sat", "cm/s"), si.get("ct_beta_n", "dimensionless")
    worst = 0.0
    print(" V [V]   E [V/cm]   J_DEVSIM [A/cm2]   J_analytic   rel.diff")
    for V in (0.1, 1, 2, 5, 10, 20, 40):
        ramp_bias(dev, "a", V, step=0.25)
        E = V / L
        v = mu0 * E / (1 + (mu0 * E / vsat) ** beta) ** (1 / beta)
        J_an = Q * ND * v
        J = abs(current(dev))
        worst = max(worst, abs(J / J_an - 1))
        print(f"{V:6.1f}  {E:9.3e}   {J:14.6e}   {J_an:12.6e}   {J / J_an - 1:+.2e}")
    return worst


def avalanche():
    """Electron-injected avalanche multiplication in a p+n diode. A uniform optical-like generation in the
    neutral p+ side injects a primary electron current far above numerical leakage noise. At each reverse
    bias, DEVSIM's multiplication M_DD = I(V)/I(V0) is compared with McIntyre's formula evaluated on the
    same DEVSIM field profile: M = 1 / (1 - int_0^W a_n exp(-int_0^x (a_n - a_p) dx') dx)."""
    from devsim.python_packages.model_create import CreateNodeModel as CNM
    NA, ND, xj, L = 1e19, 1e16, 0.5e-4, 8e-4
    lines = [(0, 2e-6, "top"), (xj - 0.05e-4, 2e-8, "j1"), (xj, 2e-8, "j"), (xj + 0.3e-4, 5e-8, "j2"), (L, 2e-6, "bot")]
    dop = f"{ND}*step(x-{xj}) - {NA}*step({xj}-x)"
    dev, reg = mesh_1d(lines)
    setup(dev, reg, dop, tau=1e-6)
    CNM(dev, reg, "Gopt", f"1e22*step(x-0.30e-4)*step(0.45e-4-x)")
    CNM(dev, reg, "ElectronGeneration", "-ElectronCharge * (USRH - Gopt)")
    CNM(dev, reg, "HoleGeneration", "+ElectronCharge * (USRH - Gopt)")
    models.impact_ionization(dev, reg, si)
    solve(type="dc", **OPTS)
    rows = []

    def record(v):
        x = np.array(get_node_model_values(device=dev, region=reg, name="x"))
        E = np.array(get_edge_model_values(device=dev, region=reg, name="ElectricField"))
        dx = np.diff(x)
        an, ap = models.alpha_numpy(si, E)
        inner = np.cumsum((an - ap) * dx) - 0.5 * (an - ap) * dx
        integral = float(np.sum(an * np.exp(-inner) * dx))
        rows.append((-v, abs(current(dev)), integral))

    ramp_bias(dev, "a", -2.0, step=0.5, callback=record)
    try:
        ramp_bias(dev, "a", -150.0, step=0.5, min_step=1e-4, callback=record)
    except RuntimeError:
        pass
    r = np.array(rows)
    I0 = r[r[:, 0] <= 2.0 + 1e-9][-1, 1]
    M_dd, M_int = r[:, 1] / I0, 1.0 / (1.0 - np.minimum(r[:, 2], 0.999999))
    sel = (M_int > 1.05) & (M_int < 20)
    worst = float(np.max(np.abs(M_dd[sel] / M_int[sel] - 1))) if sel.any() else float("nan")
    # Beyond breakdown the avalanche space charge pins the field, so the integral creeps toward 1 instead of
    # crossing it. Extrapolate (1 - integral) -> 0 from the pre-breakdown range instead.
    isel = (r[:, 2] > 0.5) & (r[:, 2] < 0.98) & (r[:, 0] < r[np.argmax(M_dd > 50), 0])
    ifit = np.polyfit(r[isel, 0], 1.0 - r[isel, 2], 1)
    v_int = -ifit[1] / ifit[0]
    fit_sel = (M_dd > 2) & (M_dd < 50)           # 1/M is close to linear in V just below breakdown
    fit = np.polyfit(r[fit_sel, 0], 1.0 / M_dd[fit_sel], 1)
    v_dd = -fit[1] / fit[0]
    print(" V [V]   M_DEVSIM    M_McIntyre   rel.diff")
    for V, md, mi in zip(r[sel, 0][::max(1, sel.sum() // 10)], M_dd[sel][::max(1, sel.sum() // 10)], M_int[sel][::max(1, sel.sum() // 10)]):
        print(f"{V:6.2f}   {md:9.4f}   {mi:9.4f}   {md / mi - 1:+.2e}")
    eg = si.get("Eg", "eV")
    v_sze = 60 * (eg / 1.1) ** 1.5 * (ND / 1e16) ** -0.75
    print(f"last converged reverse bias {r[-1, 0]:.3f} V (M_DEVSIM = {M_dd[-1]:.1f})")
    return worst, v_dd, v_int, v_sze


def run_all():
    worst_mu = velocity_saturation()
    worst_m, v_dd, v_int, v_sze = avalanche()
    return {
        "velocity_saturation_worst_rel_diff": worst_mu,
        "avalanche_M_worst_rel_diff": worst_m,
        "breakdown_V_devsim": v_dd, "breakdown_V_ionization_integral": v_int, "breakdown_V_sze_empirical": v_sze,
        "pass": bool(worst_mu < 1e-5 and worst_m < 1e-2 and abs(v_dd / v_int - 1) < 1e-2),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), indent=2))
