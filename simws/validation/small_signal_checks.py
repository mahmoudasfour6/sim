"""0.9b checks of the small-signal and noise analysis (simws.tcad.small_signal).

depletion_capacitance  reverse-biased 1D pn junction: AC capacitance vs sqrt(q eps NA ND / 2 (NA+ND) (Vbi - V - 2 kT/q))
                       (gate 1e-3)
shot_noise             forward 1D pn diode, short (diffusion noise only) and SRH-dominated (diffusion + g-r): current
                       noise = 2 q I (van Vliet: full shot noise of an ideal junction; gate 2e-3)
nyquist_2d             2D resistor at equilibrium: current noise = 4 kT G (gate 1e-6; checks the edge form of the
                       diffusion source and the contact-node Green's function in 2D)
hbt_ft                 HBT f_T from AC h21 (single-pole fit) vs 1/(2 pi tau), tau = dQ_B/dI_C from charge control
                       (0.9b gate: 5 %), SiGe and Si, low and high current
hbt_noise              HBT collector/base noise at low frequency, reported against 2 q I: the excess is the thermal
                       noise of the intrinsic base resistance reaching the collector through g_m (not a defect)
"""
from __future__ import annotations

import math

import numpy as np

from simws.units import Q_E

KB = 1.380649e-23


def _diode_1d(tau=1e-7, srh=True, NA=1e16, ND=1e16, L_um=2.0):
    import simws.tcad  # noqa: F401
    from devsim import (add_1d_contact, add_1d_mesh_line, add_1d_region, circuit_element, create_1d_mesh,
                        create_device, finalize_mesh, reset_devsim)
    from simws.materials import load
    from simws.tcad.physics import Physics, setup_region
    reset_devsim()
    L = L_um * 1e-4
    create_1d_mesh(mesh="m")
    add_1d_mesh_line(mesh="m", pos=0, ps=5e-7, tag="a")
    add_1d_mesh_line(mesh="m", pos=L / 2, ps=2e-7)
    add_1d_mesh_line(mesh="m", pos=L, ps=5e-7, tag="b")
    add_1d_contact(mesh="m", name="anode", tag="a", material="metal")
    add_1d_contact(mesh="m", name="cathode", tag="b", material="metal")
    add_1d_region(mesh="m", material="Si", region="si", tag1="a", tag2="b")
    finalize_mesh(mesh="m"); create_device(mesh="m", device="d")
    circuit_element(name="VA", n1="anode_bias", n2="0", value=0.0)
    used = setup_region("d", "si", load("silicon"), f"ifelse(x < {L / 2!r}, {-NA!r}, {ND!r})", f"{NA + ND!r}",
                        Physics(bgn=False, srh=srh, tau_n=tau, tau_p=tau, auger=False),
                        {"anode": True, "cathode": False})
    return used


def _to(V, opts):
    from devsim import get_circuit_node_value
    from simws.tcad.solve import ramp_source
    ramp_source("VA", V, start=get_circuit_node_value(node="anode_bias", solution="dcop"), step=0.1, **opts)


def depletion_capacitance(Vs=(-2.0, -1.0, 0.0), NA=1e16, ND=1e16) -> dict:
    from devsim import circuit_alter, get_circuit_node_value, solve
    used = _diode_1d(NA=NA, ND=ND)
    opts = dict(absolute_error=1e10, relative_error=1e-10, maximum_iterations=60)
    eps = 11.7 * 8.8541878128e-14
    Vt, ni = used["V_t"], used["n_i_cm3"]
    Vbi = Vt * math.log(NA * ND / ni ** 2)
    rows, worst = [], 0.0
    for V in Vs:
        _to(V, opts)
        circuit_alter(name="VA", param="acreal", value=1.0)
        f = 1e6
        solve(type="ac", frequency=f)
        C = abs(get_circuit_node_value(node="VA.I", solution="ssac_imag")) / (2 * math.pi * f)
        circuit_alter(name="VA", param="acreal", value=0.0)
        Cref = math.sqrt(Q_E * eps * NA * ND / (2 * (NA + ND) * (Vbi - V - 2 * Vt)))
        worst = max(worst, abs(C / Cref - 1))
        rows.append({"V": V, "C_F_cm2": C, "C_depletion": Cref})
    return {"rows": rows, "worst_rel_diff": worst, "pass": bool(worst < 1e-3)}


def shot_noise() -> dict:
    from devsim import get_circuit_node_value
    from simws.tcad.small_signal import current_noise
    rows, worst = [], 0.0
    for label, kw, Vs in (("short, diffusion only", dict(srh=False), (0.3, 0.4)),
                          ("SRH-dominated, tau 1 ns", dict(tau=1e-9), (0.2, 0.3, 0.4))):
        _diode_1d(**kw)
        opts = dict(absolute_error=1e10, relative_error=1e-12, maximum_iterations=60)
        for V in Vs:
            _to(V, opts)
            I = abs(get_circuit_node_value(node="VA.I", solution="dcop"))
            r = current_noise("VA.I", 1.0)
            ratio = r["S_A2_per_Hz"] / (2 * Q_E * I)
            worst = max(worst, abs(ratio - 1))
            rows.append({"case": label, "V": V, "I_A_cm2": I, "ratio_to_2qI": ratio,
                         "diffusion_share": r["S_diffusion"] / r["S_A2_per_Hz"]})
    return {"rows": rows, "worst_rel_diff": worst, "pass": bool(worst < 2e-3)}


def nyquist_2d(T=300.0) -> dict:
    import simws.tcad  # noqa: F401
    from devsim import (add_2d_contact, circuit_alter, circuit_element, create_device, finalize_mesh,
                        get_circuit_node_value, reset_devsim, solve)
    from simws.materials import load
    from simws.tcad.mixed_mode import _mesh_2d
    from simws.tcad.physics import Physics, setup_region
    from simws.tcad.small_signal import current_noise
    UM = 1e-4
    reset_devsim()
    _mesh_2d(1.0, 0.5, [(0, 0.05), (0.5, 0.02), (1.0, 0.05)], [(0, 0.02), (0.5, 0.05)])
    add_2d_contact(mesh="m", name="a", material="metal", region="si", xl=0, xh=0, yl=0, yh=0.5 * UM, bloat=1e-10)
    add_2d_contact(mesh="m", name="b", material="metal", region="si", xl=UM, xh=UM, yl=0, yh=0.5 * UM, bloat=1e-10)
    add_2d_contact(mesh="m", name="c", material="metal", region="si", xl=0.4 * UM, xh=0.6 * UM, yl=0, yh=0,
                   bloat=1e-10)
    finalize_mesh(mesh="m"); create_device(mesh="m", device="d")
    circuit_element(name="VA", n1="a_bias", n2="0", value=0.0)
    setup_region("d", "si", load("silicon"), "1e16", "1e16", Physics(T=T, bgn=False, srh=False, auger=False),
                 {"a": True, "b": False, "c": False})
    solve(type="dc", absolute_error=1e10, relative_error=1e-12, maximum_iterations=60)
    circuit_alter(name="VA", param="acreal", value=1.0)
    solve(type="ac", frequency=1.0)
    G = abs(get_circuit_node_value(node="VA.I", solution="ssac_real"))
    circuit_alter(name="VA", param="acreal", value=0.0)
    S = current_noise("VA.I", 1.0)["S_A2_per_Hz"]
    rel = S / (4 * KB * T * G) - 1
    return {"G_S_per_cm": G, "S_A2_per_Hz": S, "rel_diff": rel, "pass": bool(abs(rel) < 1e-6)}


def hbt_ft(points=((0.15, 0.75), (0.15, 0.85), (0.0, 0.85))) -> dict:
    from simws.tcad.hbt import HBTCase
    from simws.tcad.small_signal import hbt_small_signal
    rows, worst = [], 0.0
    for ge, V in points:
        r = hbt_small_signal(HBTCase(ge_peak=ge), VBE=V, VCE=1.0)
        worst = max(worst, abs(r["fT_rel_diff"]))
        rows.append({"ge_peak": ge, "VBE": V, "beta": r["beta"], "fT_ac_GHz": r["fT_ac_Hz"] / 1e9,
                     "fT_charge_GHz": r["charge"]["fT_charge_Hz"] / 1e9, "tau_ps": r["charge"]["tau_s"] * 1e12,
                     "fmax_GHz": r["fmax_Hz"] / 1e9, "rel_diff": r["fT_rel_diff"]})
    return {"rows": rows, "worst_rel_diff": worst, "pass": bool(worst < 0.05)}


def hbt_noise(points=((0.15, 0.75),)) -> dict:
    from simws.tcad.hbt import HBTCase
    from simws.tcad.small_signal import hbt_noise as hn
    rows = []
    for ge, V in points:
        r = hn(HBTCase(ge_peak=ge), VBE=V)
        rows.append({"ge_peak": ge, "VBE": V, "S_IC": r["collector"]["S_A2_per_Hz"], "S_IB": r["base"]["S_A2_per_Hz"],
                     "S_IC_over_2qIC": r["collector"]["ratio_to_2qI"], "S_IB_over_2qIB": r["base"]["ratio_to_2qI"]})
    return {"rows": rows}


def run_all(with_hbt=True) -> dict:
    res = {"depletion_capacitance": depletion_capacitance(), "shot_noise": shot_noise(), "nyquist_2d": nyquist_2d()}
    if with_hbt:
        res["hbt_ft"] = hbt_ft()
        res["hbt_noise"] = hbt_noise()
    res["pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict) and "pass" in v)
    return res
