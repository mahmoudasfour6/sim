"""Contract A25: the 5AM total-dose acceptance runs (R189, R193 to R199), printed as [Rnnn] lines and saved as JSON.

    python tools/v1_tid_demo.py --part fit --out fit.json        # A25.4 (writes simws/data/validation/ibm5am_tid.json)
    python tools/v1_tid_demo.py --part holdout --out res.json    # R189, R193 to R199 with the stored knobs

The gates are the contract's (A25.5); this script reports, the test suite (tests/v1/test_tid_5am.py) decides."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simws.validation import catalog as C  # noqa: E402
from simws.validation import ibm5am as M  # noqa: E402
from simws.validation import ibm5am_tid as T  # noqa: E402

VBE = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
DOSE_GRID = [0.18, 0.315, 0.63, 1.05, 1.8, 3.15, 5.4, 6.3]       # Mrad(SiO2): every traced dose within 3 %


def factor(a, b):
    return float(max(a / b, b / a)) if a > 0 and b > 0 else float("inf")


def loglog(x, xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return float(np.exp(np.interp(np.log(x), np.log(xs), np.log(np.maximum(ys, 1e-300)))))


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def holdout():
    k, t = M.knobs(), T.tid_knobs()
    res = {"knobs": k, "tid_knobs": t}
    # R193: the fit
    d = json.load(open(T.TID_FILE))
    for m in ("forward", "inverse"):
        for D, mod, meas in zip(d["measured"][m]["dose_Mrad"], d["model"][m], d["measured"][m]["value"]):
            print(f"[R193] {m} {D:g} Mrad: model {mod:.4g} against {meas:.4g}: factor {factor(mod, meas):.3f} (limit 1.5)")
    # the 0.5 x 1.0 chip, 300 K, both modes, on the dose grid
    curves = {m: T.dose_curve(t, DOSE_GRID, m, L_um=1.0, VBE=VBE) for m in ("forward", "inverse")}
    res["chip_1um"] = {m: {kk: v for kk, v in c.items() if kk not in ("post",)} for m, c in curves.items()}
    # R189: the inverse-mode Gummel against Fig. 22
    f22 = C.load("sutton2004.fig22")
    a = f22["assignment"]
    pre = curves["inverse"]["pre"]
    for v in (0.5, 0.6, 0.7, 0.8):
        jb_m = float(np.exp(np.interp(v, pre["VBE_V"], np.log(pre["IB_A"] / (T.W_E_UM * 1.0)))))
        jc_m = float(np.exp(np.interp(v, pre["VBE_V"], np.log(pre["JC_A_um2"]))))
        jb = float(10 ** np.interp(v, f22[a["IB"]]["VBE_V"], np.log10(f22[a["IB"]]["J_A_um2"])))
        jc = float(10 ** np.interp(v, f22[a["IC"]]["VBE_V"], np.log10(f22[a["IC"]]["J_A_um2"])))
        print(f"[R189] inverse VBE {v} V: JB model {jb_m:.3e} against {jb:.3e} (factor {factor(jb_m, jb):.2f}, "
              f"limit 3); JC model {jc_m:.3e} against {jc:.3e} (reported)")
    # R194, R195: Figs. 24 and 26
    for m, ds in (("forward", "sutton2004.fig24"), ("inverse", "sutton2004.fig26")):
        c = curves[m]
        for ser, key in (("VBE_0.6V", "at_VBE_0.6"), ("JC_1uA_um2", "at_JC_1uA_um2")):
            s = C.series(ds, ser)
            roles = C.point_roles(ds, ser)
            for D, meas, r in zip(s["dose_rad_SiO2"] / 1e6, s["value"], roles):
                mod = loglog(D, c["dose_Mrad"], c[key])
                tag = "R194" if ser == "VBE_0.6V" else "R195"
                print(f"[{tag}] {m} {ser} {D:.3g} Mrad ({r}): model {mod:.4g} against {meas:.4g}: factor "
                      f"{factor(mod, meas):.2f} (limit 2)")
        lo, hi = loglog(0.315, c["dose_Mrad"], c["at_VBE_0.6"]), loglog(3.15, c["dose_Mrad"], c["at_VBE_0.6"])
        print(f"[R194] {m}: model slope d ln / d ln D from 0.315 to 3.15 Mrad {np.log(hi / lo) / np.log(10):.2f} "
              f"(Sutton: D^{'1' if m == 'forward' else '0.3'})")
    # R196: Fig. 33, the 0.5 x 2.5 chip
    c25 = {m: T.dose_curve(t, DOSE_GRID, m, L_um=2.5, VBE=VBE) for m in ("forward", "inverse")}
    res["chip_2p5um"] = {m: {kk: v for kk, v in c.items() if kk not in ("post",)} for m, c in c25.items()}
    for src in ("proton_63MeV", "Co60", "Xray_10keV"):
        for m in ("forward", "inverse"):
            s = C.series("sutton2007.fig33", f"{src}/{m}")
            for D, meas in zip(s["dose_rad_SiO2"] / 1e6, s["value"]):
                mod = loglog(D, c25[m]["dose_Mrad"], c25[m]["at_VBE_0.6"])
                print(f"[R196] {src} {m} {D:.3g} Mrad: model {mod:.4g} against {meas:.4g}: factor "
                      f"{factor(mod, meas):.2f} (limit 2)")
    # R197: ideality at 3 Mrad (the grid's 3.15), 0.5 x 2.5
    f39 = C.load("sutton2007.fig39")["series"]
    i3 = DOSE_GRID.index(3.15)
    for m in ("forward", "inverse"):
        post, pre = c25[m]["post"][i3], c25[m]["pre"]
        for v in (0.55, 0.6, 0.7):
            n_m = T.ideality(post, pre, v)
            for src in ("Co60", "Xray_10keV", "proton_63MeV"):
                vv, nn = np.asarray(f39[src][m]["VBE_V"]), np.asarray(f39[src][m]["n"])
                j = int(np.argmin(np.abs(vv - v)))
                print(f"[R197] {m} {src} VBE {v} V (traced {vv[j]:.3f}): model n {n_m:.2f} against {nn[j]:.2f}: "
                      f"|diff| {abs(n_m - nn[j]):.2f} (limit 0.3)")
    # R198: Fig. 72 post-irradiation 300 K IB
    f72 = C.load("sutton2009.fig72")
    pre300 = [c for c in f72["curves"] if c["T_K"] == 300.0][0]
    p1 = curves["forward"]["pre"]
    for v in (0.5, 0.6, 0.7):
        jb_m = float(np.exp(np.interp(v, p1["VBE_V"], np.log(p1["IB_A"] / T.W_E_UM))))
        jb = float(10 ** np.interp(v, pre300["VBE_V_B"], np.log10(pre300["JB_A_um2"])))
        print(f"[R198] pre-irradiation JB at {v} V: model {jb_m:.3e} against {jb:.3e} (reported)")
    for c in f72["post_irradiation"]:
        if c["T_K"] != 300.0:
            continue
        D = c["dose_rad_SiO2"] / 1e6
        post = T.sweep(k, VBE, "forward", t, D, L_um=1.0)
        # the device's IB after dose: IB0 (bulk, x 2L) + the excess (x P_E)
        ib = (p1["IB_cm"] + (post["IB_cm"] - p1["IB_cm"]) * T.perimeter_factor(1.0)) * M.ibm5am_scale(1.0)
        vv, jj = np.asarray(c["VBE_V"]), np.asarray(c["J_A_um2"])
        m = (vv >= 0.45) & (vv <= 0.75)
        fs = [factor(float(np.exp(np.interp(x, post["VBE_V"], np.log(ib / T.W_E_UM)))), y) for x, y in zip(vv[m], jj[m])]
        print(f"[R198] {D:g} Mrad, 300 K: JB model / Fig. 72 at {m.sum()} traced VBE in 0.45-0.75 V: worst factor "
              f"{max(fs):.2f}, median {np.median(fs):.2f} (limit 2)")
    # R199: Fig. 73, JC = 10 nA/um^2, 300 K and 77 K
    f73 = C.load("prakash2006.fig73")["series"]
    for TK in (300.0, 77.0):
        win = [round(v, 3) for v in (np.arange(0.45, 0.70, 0.025) if TK == 300.0 else np.arange(0.90, 1.08, 0.02))]
        try:
            pre = T.sweep(k, win, "forward", None, 0.0, T=TK, L_um=1.0)
            v10 = T.vbe_at_jc(pre, 1e-8)
            s = f73[f"{TK:.1f}"]
            for D, meas in zip(np.asarray(s["dose_rad_SiO2"]) / 1e6, s["value"]):
                post = T.sweep(k, win, "forward", t, float(D), T=TK, L_um=1.0)
                mod = T.at(pre, post, v10)
                print(f"[R199] {TK:g} K, {D:.3g} Mrad, JC 10 nA/um^2 (VBE {v10:.3f} V): model {mod:.4g} against "
                      f"{meas:.4g}: factor {factor(mod, meas):.2f} (limit 2)")
        except Exception as e:  # noqa: BLE001
            print(f"[R199] {TK:g} K: no result ({type(e).__name__}: {str(e)[:160]})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("fit", "holdout"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = T.fit(log=lambda s: print(s, flush=True)) if a.part == "fit" else holdout()
    with open(a.out, "w") as f:
        json.dump(jsonable(res), f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
