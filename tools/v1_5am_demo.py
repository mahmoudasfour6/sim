"""Contracts A23 and A24: the IBM 5AM device against its traced measurements; prints the R160 to R175 lines and
writes the numbers to a JSON file. Slow (tens of 2D runs; the cold ones slower).

    PYTHONPATH=. python tools/v1_5am_demo.py --part a23 [--out results_a23.json]
    PYTHONPATH=. python tools/v1_5am_demo.py --part a24 [--out results_a24.json]
"""
import argparse
import json
import sys
import time

import numpy as np

from simws.validation import catalog as C
from simws.validation import ibm5am as M


def say(*a):
    print(*a, flush=True)


def ratio_line(model, meas):
    r = np.asarray(model, float) / np.asarray(meas, float)
    return r, float(np.max(np.abs(r - 1)))


def a23(out):
    k = M.knobs()
    say(f"[A23] knobs (stored calibration): s {k['s_um']:.4f} um, k_B {k['k_B']:.4f}, k_E {k['k_E']:.4f}"
        + (f", k_P {k['k_P']:.4f}, S_p {k['S_p']:.4g} cm/s (A23b)" if "S_p" in k else ""))
    cal = json.load(open(M.CAL_FILE))
    ric, rib, rrs = (10 ** np.asarray(cal["residuals_log10"][s]) for s in ("ic", "ib", "rs"))
    out["R161"] = {"IC_ratio": ric.tolist(), "IB_ratio": rib.tolist(), "RsBi_ratio": float(rrs[0])}
    say(f"R161 IC model/measured {np.round(ric, 3).tolist()} (limit 15 %); IB {np.round(rib, 3).tolist()} "
        f"(20 %); RsBi {rrs[0]:.3f} (10 %): model {cal['model']['RsBi_ohm_sq']:.0f}, "
        f"measured {cal['measured']['RsBi_ohm_sq']:.0f} ohm/sq")
    if "BVCBO" in cal:
        bv = cal["BVCBO"]
        out["R183"] = bv
        say(f"R183 BVCBO model {bv['model_V']:.2f} V, measured {bv['measured_V']:.1f} V, ratio "
            f"{bv['model_V'] / bv['measured_V']:.3f} (limit 10 %)")
    # R162: Pellish's chip
    v = [round(x, 2) for x in np.arange(0.60, 0.781, 0.02)]
    g = M.gummel(k, v)
    pc = C.curve("pellish2008.fig3")
    ic_m = M._interp_log(g["VBE_V"], pc["VBE_V"], pc["IC_A"])
    ib_m = M._interp_log(g["VBE_V"], pc["VBE_V"], pc["IB_A"])
    r, worst = ratio_line(g["IC_A"], ic_m)
    out["R162"] = {"VBE_V": v, "IC_model_A": g["IC_A"].tolist(), "IC_meas_A": ic_m.tolist(),
                   "IB_model_A": g["IB_A"].tolist(), "IB_meas_A": ib_m.tolist(), "worst": worst}
    say(f"R162 Pellish Fig. 3 IC model/measured {np.round(r, 3).tolist()} worst {100 * worst:.1f} % (limit 25 %); "
        f"reported IB {np.round(g['IB_A'] / ib_m, 3).tolist()}, beta model {np.round(g['IC_A'] / g['IB_A'], 1).tolist()}")
    # R163: Early voltage
    out["R163"] = {}
    for ib in (6.0, 12.0):
        e = M.early_voltage(k, ib * 1e-6)
        vm = M.early_voltage_measured(ib)
        out["R163"][f"{ib}"] = {"model": e, "V_A_meas": vm, "ratio": e["V_A"] / vm}
        say(f"R163 IB {ib} uA: V_A model {e['V_A']:.1f} V (VBE {e['VBE_V']:.4f}), measured {vm:.1f} V, "
            f"ratio {e['V_A'] / vm:.3f} (limit factor 1.5)")
    # R164: avalanche
    a = M.avalanche(k)
    c = C.curve("luo2011.fig4_2", 300.0)
    mm = [float(np.interp(x, c["VCB_V"], c["M_minus_1"])) for x in (2, 3, 4, 5)]
    mo = [a["M_minus_1"][a["VCB_V"].index(x)] for x in (2.0, 3.0, 4.0, 5.0)]
    out["R164"] = {"model": a, "M_minus_1_meas": mm}
    say(f"R164 M-1 at VCB 2, 3, 4, 5 V: model {['%.3g' % x for x in mo]}, measured {['%.3g' % x for x in mm]}, "
        f"ratios {[round(x / y, 3) for x, y in zip(mo, mm)]} (limit factor 2)")
    # R165: fT
    f = M.ft_curve(k, [round(x, 2) for x in np.arange(0.80, 0.981, 0.02)])
    c = C.curve("luo2011.fig3_7a")
    i = int(np.argmax(f["fT_GHz"]))
    j = int(np.argmax(c["fT_GHz"]))
    out["R165"] = {"model": f, "peak_model_GHz": f["fT_GHz"][i], "JE_peak_model": f["JE_mA_um2"][i],
                   "peak_meas_GHz": float(c["fT_GHz"][j]), "JE_peak_meas": float(c["JE_mA_um2"][j])}
    say(f"R165 peak fT model {f['fT_GHz'][i]:.1f} GHz at JE {f['JE_mA_um2'][i]:.3g} mA/um^2; measured "
        f"{c['fT_GHz'][j]:.1f} GHz at {c['JE_mA_um2'][j]:.3g} (limits 25 %, factor 2 in JE)")
    # R166: mesh
    g1 = M.gummel(k, [0.70])
    g2 = M.gummel(k, [0.70], mesh_scale=0.5)
    dc, db = g2["IC_A"][0] / g1["IC_A"][0] - 1, g2["IB_A"][0] / g1["IB_A"][0] - 1
    out["R166"] = {"dIC": float(dc), "dIB": float(db)}
    say(f"R166 mesh scale 0.5 at 0.70 V: IC {100 * dc:+.2f} %, IB {100 * db:+.2f} % (limit 3 %)")
    # R167: Moll-Ross, and the SiGe film
    mr = M.moll_ross(k, [0.60, 0.65])
    gg = M.gummel(k, [0.60, 0.65])
    f12 = C.curve("luo2011.fig1_2")
    film = (f12["read_while_tracing"]["Ge_above_1pct_u"][1] - f12["read_while_tracing"]["Ge_above_1pct_u"][0]) * k["s_um"]
    out["R167"] = {"IC_device_A": gg["IC_A"].tolist(), "IC_moll_ross_A": mr, "SiGe_film_nm": 1e3 * film}
    say(f"R167 IC device / Moll-Ross at 0.60, 0.65 V: {np.round(gg['IC_A'] / np.asarray(mr), 3).tolist()} "
        f"(reported); SiGe film (Ge > 1 %) {1e3 * film:.0f} nm (Sutton 90 to 150 nm)")


def a24(out):
    k = M.knobs()
    say(f"[A24] knobs unchanged: s {k['s_um']:.4f} um, k_B {k['k_B']:.4f}, k_E {k['k_E']:.4f}"
        + (f", k_P {k['k_P']:.4f}, S_p {k['S_p']:.4g} cm/s (A23b)" if "S_p" in k else ""))
    for T in M.A24_TEMPERATURES + (43.0,):
        t0 = time.time()
        tg = M.luo_targets(T)
        try:
            g = M.gummel(k, M.gummel_window(T), T=T)
        except Exception as e:                       # noqa: BLE001
            out[f"T{T:g}"] = {"error": f"{type(e).__name__}: {e}"[:300]}
            say(f"R170/R175 {T:g} K: no result ({type(e).__name__}: {str(e)[:120]})")
            continue
        v1 = M.vbe_at_current(g["VBE_V"], g["IC_A"], 1e-6)
        v100 = M.vbe_at_current(g["VBE_V"], g["IC_A"], 1e-4)
        beta = g["IC_A"] / g["IB_A"]
        ib = int(np.argmax(beta))
        rec = {"targets": tg, "VBE_1uA": v1, "VBE_100uA": v100, "beta_peak": float(beta[ib]),
               "VBE_beta_peak": float(g["VBE_V"][ib]), "RsBi_ohm_sq": g["RsBi_ohm_sq"],
               "VBE_V": g["VBE_V"].tolist(), "IC_A": g["IC_A"].tolist(), "IB_A": g["IB_A"].tolist(),
               "seconds": time.time() - t0}
        out[f"T{T:g}"] = rec
        tag = "R175" if T == 43.0 else "R170"
        say(f"{tag} {T:g} K: VBE at 1 uA {v1:.4f} V (measured {tg['VBE_1uA']:.4f}, {1e3 * (v1 - tg['VBE_1uA']):+.1f} mV); "
            f"at 100 uA {v100:.4f} V (measured {tg['VBE_100uA']:.4f}, {1e3 * (v100 - tg['VBE_100uA']):+.1f} mV) "
            f"(limit 20 mV) [{rec['seconds']:.0f} s]")
        if "beta_peak" in tg:
            say(f"R171 {T:g} K: peak beta {beta[ib]:.1f} at {g['VBE_V'][ib]:.3f} V, measured {tg['beta_peak']:.1f} at "
                f"{tg['VBE_beta_peak']:.3f} V, ratio {beta[ib] / tg['beta_peak']:.3f}")
    rs = C.curve("luo2011.fig4_1")
    out["R172"] = {}
    for T in (223.0, 162.0, 100.0, 75.0):
        try:
            r = M.rsbi(k, T)
        except Exception as e:                       # noqa: BLE001
            say(f"R172 {T:g} K: no result ({type(e).__name__})")
            continue
        m = float(10 ** np.interp(T, rs["T_K"], np.log10(rs["RsBi_ohm_sq"])))
        out["R172"][f"{T:g}"] = {"model": r, "measured": m, "ratio": r / m}
        say(f"R172 {T:g} K: RsBi model {r:.4g}, measured {m:.4g} ohm/sq, ratio {r / m:.3f}")
    out["R173"] = {}
    for T in (162.0, 93.0):
        tg = M.luo_targets(T)
        try:
            a = M.avalanche(k, T=T, VCB=(0.0, 2.0, 3.0, 4.0),
                            VBE_guess=[round(x, 3) for x in np.arange(tg["VBE_1uA"] - 0.03, tg["VBE_100uA"], 0.01)])
        except Exception as e:                       # noqa: BLE001
            say(f"R173 {T:g} K: no result ({type(e).__name__})")
            continue
        c = C.curve("luo2011.fig4_2", T)
        mm = [float(np.interp(x, c["VCB_V"], c["M_minus_1"])) for x in (2, 3, 4)]
        out["R173"][f"{T:g}"] = {"model": a, "measured": mm}
        say(f"R173 {T:g} K M-1 at 2, 3, 4 V: model {['%.3g' % x for x in a['M_minus_1'][1:]]}, measured "
            f"{['%.3g' % x for x in mm]}")
    s = C.curve("sutton2009.fig72")
    out["R174"] = {}
    for cur in s["curves"]:
        T = float(cur["T_K"])
        vm = M.vbe_at_current(cur["VBE_V_C"], np.maximum.accumulate(np.asarray(cur["JC_A_um2"])), 1e-8)
        try:
            g = M.gummel(k, [round(x, 3) for x in np.arange(vm - 0.06, vm + 0.061, 0.02)], T=T, L_um=1.0)
        except Exception as e:                       # noqa: BLE001
            say(f"R174 {T:g} K: no result ({type(e).__name__})")
            continue
        vmod = M.vbe_at_current(g["VBE_V"], g["IC_A"] / (0.5 * 1.0), 1e-8)
        out["R174"][f"{T:g}"] = {"model": vmod, "measured": vm}
        say(f"R174 Sutton {T:g} K (0.5 x 1.0 um^2): VBE at JC 10 nA/um^2 model {vmod:.4f}, measured {vm:.4f} "
            f"({1e3 * (vmod - vm):+.1f} mV; limit 30 mV)")


def robin(out):
    """A23b R180 and R181: the hole recombination velocity contact against its closed form, 1D and 2D."""
    from devsim import get_contact_current
    from simws.engines import devsim_engine as E
    from simws.materials import load
    from simws.materials.semiconductor import intrinsic_density
    from simws.spec.schema import (Box, Contact, DCSweep, DeviceSpec, DopingTerm, MeshHints, ModelSpec, Numerics,
                                   Problem, Region)
    q, Vt, V = 1.602176634e-19, 0.025852, 0.5
    Wp, Wn, NA, ND, mup = 0.1, 2.0, 1e20, 1e17, 400.0

    def problem(S, dim):
        box = Box(x=(0.0, Wp + Wn)) if dim == 1 else Box(x=(0.0, 0.5), y=(0.0, Wp + Wn))
        lo, hi = ("x_min", "x_max") if dim == 1 else ("y_min", "y_max")
        ax = "x" if dim == 1 else "y"
        dop = [DopingTerm(species="acceptor", concentration_cm3=NA, box=Box(**{ax: (None, Wp)})),
               DopingTerm(species="donor", concentration_cm3=ND, box=Box(**{ax: (Wp, None)}), include_upper=True)]
        lines = [(0.0, 0.01), (Wp, 0.0005), (Wp + Wn, 0.02)]
        mesh = MeshHints(x_lines=lines) if dim == 1 else MeshHints(x_lines=[(0.0, 0.1), (0.5, 0.1)], y_lines=lines)
        dev = DeviceSpec(name="robin", dimension=dim, regions=[Region(name="si", material="silicon", box=box)],
                         contacts=[Contact(name="anode", region="si", side=lo),
                                   Contact(name="cathode", region="si", side=hi, hole_recombination_velocity_cm_s=S)],
                         doping=dop, mesh=mesh)
        m = ModelSpec(temperature_K=300.0, mobility="constant", mu_n_cm2_Vs=1000.0, mu_p_cm2_Vs=mup, srh="off",
                      auger=False, band_gap_narrowing="off")
        return Problem(device=dev, models=m, analysis=DCSweep(contact="anode", values_V=[V], step_V=0.1),
                       numerics=Numerics(relative_error=1e-10, absolute_error=1e10, max_iterations=60))

    def jp(S, dim):
        st = E._build(problem(S, dim))
        st.ramp("anode", V, 0.05)
        j = abs(get_contact_current(device=E.DEVICE, contact="cathode", equation="HoleContinuityEquation"))
        return j if dim == 1 else j / 0.5e-4

    ni = intrinsic_density(load("silicon"), 300.0)
    xn = np.sqrt(2 * 11.7 * 8.854e-14 * (Vt * np.log(NA * ND / ni ** 2) - V) / (q * ND))
    Wneu, Dp, pn0 = Wn * 1e-4 - xn, mup * Vt, ni ** 2 / ND
    out["R180"] = {}
    for S in (1e2, 1e3, 1e4, 1e5):
        cf = q * pn0 * (np.exp(V / Vt) - 1) / (Wneu / Dp + 1 / S)
        r = jp(S, 1) / cf
        out["R180"][f"{S:g}"] = r
        say(f"R180 S {S:g} cm/s: model / closed form {r:.4f} (limit 1 %)")
    big, ohm = jp(1e8, 1), jp(None, 1)
    out["R180"]["1e8_vs_ohmic"] = big / ohm
    say(f"R180 S 1e8 / ohmic {big / ohm:.5f} (limit 0.5 %)")
    out["R181"] = {}
    for S in (1e3, 1e5):
        r = jp(S, 2) / jp(S, 1)
        out["R181"][f"{S:g}"] = r
        say(f"R181 S {S:g}: 2D / 1D {r:.6f} (limit 0.1 %)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a23", "a24", "robin"], required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {}
    try:
        {"a23": a23, "a24": a24, "robin": robin}[a.part](res)
    finally:
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    sys.exit(0)
