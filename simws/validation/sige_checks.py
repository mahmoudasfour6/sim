"""Checks of the SiGe heterojunction models.

equilibrium_sige        (physics_checks) graded SiGe at zero bias: no current, n p = n_ie^2
collector_enhancement   SiGe HBT vs identical Si BJT at the same V_BE: the intrinsic-base electron current ratio
                        equals the generalized Moll-Ross (Kroemer) prediction G_Si / G_SiGe, where the base
                        Gummel integral G = int p / (D_n n_ie^2) dy is evaluated on each device's own solution
                        along a vertical cut under the emitter (so the true neutral-base edges are used)
base_current_unchanged  the base current (set by the Si emitter) is the same in both devices
ideality                intrinsic collector-current slope vs the Moll-Ross slope (graded base: > kT/q ln10)
buried_channel          strained-SiGe buried pMOS channel: holes appear in the well first and the onset moves by
                        about the valence-band offset relative to the same stack without Ge
tid_surface             interface traps over the EB periphery: delta-I_B equals the surface recombination integral
                        from the same solution, is linear in N_it, has ideality between 1 and 2; I_C unchanged
strike_conservation     ion strike through the emitter at low LET: the terminal charges sum to zero after relaxation,
                        the track footprint integrates to the deposited charge, collected <= deposited with the
                        emitter taking a small share (no bipolar gain), and the response is linear in LET.
                        (At high LET the track shunts emitter to collector and Q_C exceeds the deposited charge:
                        physical, the ion-shunt effect, but exaggerated by the planar 2D track.)
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from simws.tcad.hbt import HBTCase, UM, gummel
from simws.units import K_B_EV


def _gummel_integral(case, x_cut_um=0.1):
    from devsim import get_node_model_values
    g = lambda k: np.array(get_node_model_values(device="d", region="si", name=k))
    x, y, p, nie, mu = g("x") / UM, g("y") / UM, g("Holes"), g("n_ie"), g("mu_n_node")
    xs = np.unique(np.round(x, 6))
    xc = xs[np.argmin(np.abs(xs - x_cut_um))]
    sel = np.abs(x - xc) < 1e-6
    o = np.argsort(y[sel])
    ys, ps, ns, ms = y[sel][o] * UM, p[sel][o], nie[sel][o], mu[sel][o]
    Dn = K_B_EV * 300.0 * ms
    f = ps / (Dn * ns ** 2)
    return float(np.trapezoid(f, ys))


def _jn_intrinsic(case, x_cut_um=0.1):
    """Electron current density (A/cm^2) through the neutral intrinsic base along the vertical cut."""
    from devsim import edge_from_node_model, get_edge_model_values
    from simws.tcad.physics import EnsureEdgeFromNodeModelExists
    from devsim.python_packages.model_create import CreateEdgeModel
    for v in ("x", "y"):
        EnsureEdgeFromNodeModelExists("d", "si", v)
    CreateEdgeModel("d", "si", "xmid", "0.5*(x@n0 + x@n1)")
    CreateEdgeModel("d", "si", "ymid", "0.5*(y@n0 + y@n1)")
    CreateEdgeModel("d", "si", "dxe", "abs(x@n1 - x@n0)")
    e = lambda k: np.array(get_edge_model_values(device="d", region="si", name=k))
    xm, ym, dx, J = e("xmid") / UM, e("ymid") / UM, e("dxe"), e("ElectronCurrent")
    xs = np.unique(np.round(xm, 6))
    xc = xs[np.argmin(np.abs(xs - x_cut_um))]
    yc = case.t_e + 0.7 * case.w_b                                    # inside the neutral base at forward bias
    sel = (np.abs(xm - xc) < 1e-6) & (dx < 1e-12) & (np.abs(ym - yc) < 0.004)
    return float(np.mean(np.abs(J[sel])))


def _neutral_base_edge(case, x_cut_um=0.1):
    """Emitter-side edge of the neutral base on the intrinsic vertical cut at the present bias: the first node
    (from the emitter) where the hole density reaches half its base maximum. Returns (y_edge_um, Ge fraction,
    Ge-induced effective gap reduction dEg_eff = dEc + dEv - DeltaEg in eV, i.e. n_ie^2 grows as exp(dEg_eff/kT)
    there, DOS-ratio terms included; then the change of each across the bracketing mesh cell)."""
    from devsim import get_node_model_values, get_node_model_list
    g = lambda k: np.array(get_node_model_values(device="d", region="si", name=k))
    x, y = g("x") / UM, g("y") / UM
    xs = np.unique(np.round(x, 6))
    xc = xs[np.argmin(np.abs(xs - x_cut_um))]
    sel = (np.abs(x - xc) < 1e-6) & (y > case.t_e) & (y < case.t_e + case.w_b)
    o = np.argsort(y[sel])
    yy, p = y[sel][o], g("Holes")[sel][o]
    gap = (g("dEc") + g("dEv") - g("DeltaEg"))[sel][o]
    ge = g("GeX")[sel][o] if "GeX" in get_node_model_list(device="d", region="si") else np.zeros_like(yy)
    k = int(np.argmax(p >= 0.5 * p.max()))
    if k == 0:
        return float(yy[0]), float(ge[0]), float(gap[0]), float(yy[1] - yy[0]), float(ge[1] - ge[0]), float(gap[1] - gap[0])
    f = (0.5 * p.max() - p[k - 1]) / (p[k] - p[k - 1])        # linear between the bracketing nodes
    lerp = lambda a: float(a[k - 1] + f * (a[k] - a[k - 1]))
    step = lambda a: float(a[k] - a[k - 1])                   # change across the bracketing cell (uncertainty)
    return lerp(yy), lerp(ge), lerp(gap), step(yy), step(ge), step(gap)


def collector_enhancement(VBE=(0.5, 0.6, 0.65), ge_peak=0.15):
    """Ratio of the intrinsic-base electron current density (SiGe / Si) vs Moll-Ross on the same cut; the terminal
    current ratio is reported too (it is a few % lower: the emitter periphery injects into Ge-free silicon)."""
    out = {}
    res = {}
    for ge in (0.0, ge_peak):
        c = HBTCase(ge_peak=ge)
        rows = []
        for v in VBE:
            r = gummel(c, [v])
            rows.append((r["IC_A"][0], r["IB_A"][0], _gummel_integral(c), _jn_intrinsic(c)))
        res[ge] = rows
    worst, worst_ib = 0.0, 0.0
    for i, v in enumerate(VBE):
        (ic_si, ib_si, g_si, j_si), (ic_sg, ib_sg, g_sg, j_sg) = res[0.0][i], res[ge_peak][i]
        ratio_dd, ratio_mr = j_sg / j_si, g_si / g_sg
        out[f"VBE_{v:g}"] = {"Jn_ratio_devsim": ratio_dd, "ratio_moll_ross": ratio_mr, "rel_diff": ratio_dd / ratio_mr - 1,
                             "IC_terminal_ratio": ic_sg / ic_si, "IB_rel_diff": ib_sg / ib_si - 1}
        worst = max(worst, abs(ratio_dd / ratio_mr - 1)); worst_ib = max(worst_ib, abs(ib_sg / ib_si - 1))
    return {"cases": out, "worst_rel_diff": worst, "worst_IB_rel_diff": worst_ib,
            "pass": bool(worst < 0.005 and worst_ib < 0.01)}


def ideality(ge_peak=0.15):
    """Intrinsic collector-current slope vs the Moll-Ross slope from the same solutions. In a graded base the
    slope exceeds kT/q ln10: forward bias shrinks the E-B depletion, so the neutral base starts at less Ge."""
    c = HBTCase(ge_peak=ge_peak)
    J, G = [], []
    for v in (0.5, 0.6):
        gummel(c, [v]); J.append(_jn_intrinsic(c)); G.append(_gummel_integral(c))
    Vt = K_B_EV * 300.0
    sim = 1e3 * 0.1 / math.log10(J[1] / J[0])
    mr = 1e3 * 0.1 / math.log10(G[0] / G[1] * math.exp(0.1 / Vt))
    return {"mV_per_decade_devsim": sim, "mV_per_decade_moll_ross": mr, "ideal_kT_ln10": 1e3 * Vt * math.log(10),
            "pass": bool(abs(sim / mr - 1) < 0.01)}


def buried_channel(ge_x=0.25):
    """Strained-SiGe buried pMOS channel (5 nm oxide / 4 nm Si cap / 10 nm SiGe / n-Si 1e17): holes must form in the
    well first (well >> cap at onset), and the onset (1e11 cm^-2 of holes) must move toward zero gate voltage by
    roughly the effective valence-band offset dEv_eff = 0.75 x + kT ln(Nv ratio) (0.5-1.5x: the cap and the
    surface potential share the shift, so it is not exact)."""
    import math as _m
    from simws.materials import load
    from simws.tcad.mos import MOSCase, hole_sheets
    V = np.linspace(0, -2.0, 41)
    on = {}
    for x in (0.0, ge_x):
        h = hole_sheets(MOSCase(t_ox_nm=5, NA=-1e17, cap_nm=4, well_nm=10, ge_x=x), V)
        tot = h["cap"] + h["well"]
        on[x] = float(np.interp(_m.log(1e11), np.log(tot), V)) if tot.max() > 1e11 else None
        if x == ge_x:
            i = int(np.argmin(np.abs(tot - 1e11)))
            ratio = float(h["well"][i] / max(h["cap"][i], 1.0))
    sg = load("sige")
    Vt = K_B_EV * 300.0
    gam = _m.exp(-_m.sqrt(sg.get("dos_gamma_k", "dimensionless") * ge_x))
    dEv = sg.get("strained_dEg_per_x", "eV") * ge_x + Vt * _m.log(gam / sg.get("dos_Nc_ratio", "dimensionless"))
    shift = on[ge_x] - on[0.0]
    return {"V_onset_Si": on[0.0], "V_onset_SiGe": on[ge_x], "shift_V": shift, "dEv_eff_eV": dEv,
            "well_to_cap_at_onset": ratio, "pass": bool(ratio > 10 and 0.5 < shift / dEv < 1.5)}


def tid_surface():
    from simws.radiation.hbt_rad import excess_base_current
    VBE = [0.5, 0.6, 0.7]
    r = excess_base_current(HBTCase(), [0.0, 1e11, 1e12], VBE)
    dIB, Is = r["dIB_A"][1:], r["I_surf_A"][1:]
    acc = float(np.max(np.abs(dIB / Is - 1)))
    lin = float(np.max(np.abs(r["dIB_A"][2] / r["dIB_A"][1] / 10.0 - 1)))
    ic = float(np.max(np.abs(r["IC_A"][2] / r["IC_A"][0] - 1)))
    n = (0.1 / math.log(r["dIB_A"][2][1] / r["dIB_A"][2][0])) / (K_B_EV * 300.0)
    return {"dIB_vs_surface_integral_worst": acc, "linearity_dev": lin, "IC_change": ic, "ideality_dIB_0.5_0.6V": n,
            "dIB_over_IB_at_0.5V_1e12": float(r["dIB_A"][2][0] / r["IB_A"][0][0]),
            "pass": bool(acc < 0.01 and lin < 0.02 and ic < 1e-3 and 1.0 < n < 2.0)}


def strike_conservation():
    from simws.radiation.hbt_rad import StrikeSetup, strike
    rs = [strike(HBTCase(), StrikeSetup(let=L)) for L in (0.1, 0.2)]
    a, b = rs
    cons = max(abs(r["imbalance_fC"]) / r["Q_deposited_fC"] for r in rs)
    em = max(r["Q_emitter_fC"] / r["Q_collector_fC"] for r in rs)
    mesh = max(abs(r["track_mesh_correction"] - 1) for r in rs)
    lin = abs(b["Q_collector_fC"] / a["Q_collector_fC"] / 2.0 - 1)
    frac = a["Q_collector_fC"] / a["Q_deposited_fC"]
    return {"Q_dep_fC": [r["Q_deposited_fC"] for r in rs], "Q_C_fC": [r["Q_collector_fC"] for r in rs],
            "collected_fraction": frac, "imbalance_rel": cons, "emitter_share": em, "mesh_correction_dev": mesh,
            "linearity_dev": lin,
            "pass": bool(cons < 1e-3 and mesh < 0.03 and lin < 0.02 and 0 < frac <= 1.0 and em < 0.1)}


def run_all():
    from simws.validation.physics_checks import equilibrium_sige
    res = {"equilibrium_sige": equilibrium_sige(), "collector_enhancement": collector_enhancement(), "ideality": ideality(),
           "buried_channel": buried_channel(), "tid_surface": tid_surface(), "strike_conservation": strike_conservation()}
    res["pass"] = all(v["pass"] for v in res.values())
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), indent=1, default=float))
