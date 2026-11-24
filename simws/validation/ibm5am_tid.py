"""Contract A25: total dose on the IBM 5AM (the frozen A23b device) with interface traps and oxide charge on the EB
spacer and the STI sidewall, fitted to Sutton's 63 MeV proton data at 300 K.

    from simws.validation import ibm5am_tid as T
    t = T.tid_knobs()                                   # {'k_F', 'k_I', 'k_Q'} from ibm5am_tid.json
    c = T.dose_curve(t, [1.05], "forward", L_um=1.0)   # dIB/IB0 at VBE = 0.6 V and at JC = 1 uA/um^2
    T.fit()                                             # A25.4: writes ibm5am_tid.json

Scaling to one device (A25.3): the bulk currents x 2L (both halves of the cell, as A23.2); the excess base current
that the interfaces add x the emitter perimeter P_E = 2 (L + W_E), W_E = 0.5 um drawn. So
    dIB/IB0 (device) = (dI'_B / I'_B0)(half cell, per cm) x P_E / (2 L).
JC is the collector current (forward) or the physical emitter's current (inverse) over the drawn AE = W_E x L.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional, Sequence

import numpy as np

from simws.spec.library import ibm5am_hbt, ibm5am_scale
from simws.validation import catalog as C
from simws.validation import ibm5am as M

TID_FILE = os.path.join(C.DIR, "ibm5am_tid.json")
TID_RANGES = {"k_F": (1e9, 1e13), "k_I": (1e9, 1e13), "k_Q": (0.0, 1e13)}      # cm^-2 per Mrad(SiO2), A25.4
K_Q_OFFSET = 1e8                     # k_Q is fitted as log10(1e8 + k_Q)
W_E_UM = 0.5                         # drawn emitter width (Sutton Table 3)
VBE_FIX = 0.6                        # Sutton's fixed-VBE extraction
JC_FIX_A_UM2 = 1e-6                  # Sutton's fixed-JC extraction (Figs. 24, 26)
CAL_DOSES_MRAD = (1.05, 6.3)         # A25.1: the traced doses of the calibration points
CAL_L_UM = 1.0                       # the calibration chip, 0.5 x 1.0 um^2


def tid_knobs(path: str = TID_FILE) -> Dict[str, float]:
    with open(path) as f:
        d = json.load(f)
    return {k: float(d["knobs"][k]) for k in ("k_F", "k_I", "k_Q")}


def check_knobs(t: Dict[str, float]) -> None:
    for name, (lo, hi) in TID_RANGES.items():
        if not lo <= t[name] <= hi:
            raise ValueError(f"tid: {name} = {t[name]:g} outside its range [{lo:g}, {hi:g}] (A25.4)")


def traps(t: Dict[str, float], dose_Mrad: float) -> dict:
    """A25.4: (N_it, N_ot) in cm^-2 on the spacer and the STI after dose_Mrad Mrad(SiO2)."""
    check_knobs(t)
    D = float(dose_Mrad)
    if D < 0:
        raise ValueError("tid: the dose must be >= 0")
    return {"spacer": (t["k_F"] * D, t["k_Q"] * D), "sti": (t["k_I"] * D, t["k_Q"] * D)}


def perimeter_factor(L_um: float) -> float:
    """P_E / (2 L): the interface excess per cm of edge -> one device, over the bulk per cm -> one device."""
    return 2.0 * (L_um + W_E_UM) / (2.0 * L_um)


def sweep(k: Dict[str, float], VBE: Sequence[float], mode: str = "forward", t: Optional[Dict[str, float]] = None,
          dose_Mrad: float = 0.0, T: float = 300.0, L_um: float = 1.0, step_V: float = 0.05) -> dict:
    """One Gummel sweep (forward or inverse, VCB = 0) of the 5AM with the interfaces of dose_Mrad (none when t is
    None or the dose is 0). Per-cm currents of the half cell (A/cm) and the device's bulk-scaled currents (A)."""
    from simws.engines import devsim_engine as E
    tr = traps(t, dose_Mrad) if (t is not None and dose_Mrad > 0) else {"spacer": None, "sti": None}
    p = ibm5am_hbt(k["s_um"], k["k_B"], k["k_E"], k_P=k.get("k_P", 1.0), S_p=k.get("S_p"), L_um=L_um, T=T,
                   VBE=[float(v) for v in VBE], mode=mode, spacer=tr["spacer"], sti=tr["sti"])
    p = p.model_copy(update={"analysis": p.analysis.model_copy(update={"step_V": step_V})})
    t0 = time.time()
    r = E.run(p)
    out_c = "collector" if mode == "forward" else "emitter"
    ib = np.asarray(r.currents["base"], float)
    ic = np.abs(np.asarray(r.currents[out_c], float))
    sc = ibm5am_scale(L_um)
    return {"VBE_V": np.asarray(r.sweep_V, float), "IB_cm": ib, "IC_cm": ic, "IB_A": ib * sc, "IC_A": ic * sc,
            "JC_A_um2": ic * sc / (W_E_UM * L_um), "interface_A_cm": r.diagnostics.get("interface_current_A_per_cm",
                                                                                          {}),
            "mode": mode, "dose_Mrad": float(dose_Mrad), "T_K": T, "L_um": L_um, "seconds": time.time() - t0,
            "problem_id": p.problem_id(), "flags": r.flags}


def ratio(pre: dict, post: dict) -> np.ndarray:
    """dIB/IB0 of one device at the sweep's VBE points (A25.3 scaling)."""
    assert np.allclose(pre["VBE_V"], post["VBE_V"]) and pre["L_um"] == post["L_um"]
    return (post["IB_cm"] - pre["IB_cm"]) / pre["IB_cm"] * perimeter_factor(pre["L_um"])


def _interp_ln(x, xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return float(np.exp(np.interp(x, xs, np.log(np.maximum(ys, 1e-300)))))


def vbe_at_jc(pre: dict, J_A_um2: float) -> float:
    """The VBE where the pre-irradiation JC reaches J (interpolated in ln JC; dose does not move JC). A JC that does
    not cross J inside the sweep raises ValueError (no extrapolation)."""
    jc = np.asarray(pre["JC_A_um2"], float)
    if not (jc.min() <= J_A_um2 <= jc.max()):
        raise ValueError(f"tid: JC {J_A_um2:g} A/um^2 is outside the sweep's {jc.min():.3g} to {jc.max():.3g}")
    return float(np.interp(np.log(J_A_um2), np.log(jc), pre["VBE_V"]))


def at(pre: dict, post: dict, vbe: float) -> float:
    """dIB/IB0 at one VBE: the excess and IB0 interpolated in ln between the sweep's points."""
    d = (post["IB_cm"] - pre["IB_cm"]) * perimeter_factor(pre["L_um"])
    return _interp_ln(vbe, pre["VBE_V"], np.maximum(d, 1e-300)) / _interp_ln(vbe, pre["VBE_V"], pre["IB_cm"])


# ------------------------------------------------------------------------------------------------ calibration
def measured_calibration() -> dict:
    """A25.1: the four calibration points, dIB/IB0 at VBE = 0.6 V, 1.05 and 6.3 Mrad, forward (Fig. 24) and
    inverse (Fig. 26)."""
    out = {}
    for mode, ds in (("forward", "sutton2004.fig24"), ("inverse", "sutton2004.fig26")):
        s = C.series(ds, "VBE_0.6V")
        roles = C.point_roles(ds, "VBE_0.6V")
        m = roles == "calibration"
        out[mode] = {"dose_Mrad": [float(v) / 1e6 for v in s["dose_rad_SiO2"][m]], "value": [float(v) for v in
                                                                                              s["value"][m]]}
    return out


def _eval_point(args):
    k, t, mode, dose = args
    return sweep(k, [VBE_FIX], mode, t, dose, L_um=CAL_L_UM)


def fit(start: Optional[Dict[str, float]] = None, log=print, path: str = TID_FILE, workers: int = 2) -> dict:
    """A25.4: least squares in log10 of dIB/IB0 at VBE = 0.6 V, forward and inverse, at the two calibration doses;
    x = (log10 k_F, log10 k_I, log10(1e8 + k_Q)). Each evaluation runs the four sweeps on `workers` processes; the
    pre-irradiation sweeps once."""
    import multiprocessing as mp
    from scipy.optimize import least_squares
    k = M.knobs()
    meas = measured_calibration()
    st = start or {"k_F": 1e11, "k_I": 1e11, "k_Q": 1e11}
    x0 = np.array([np.log10(st["k_F"]), np.log10(st["k_I"]), np.log10(K_Q_OFFSET + st["k_Q"])])
    lo = np.array([np.log10(TID_RANGES["k_F"][0]), np.log10(TID_RANGES["k_I"][0]), np.log10(K_Q_OFFSET)])
    hi = np.array([np.log10(TID_RANGES["k_F"][1]), np.log10(TID_RANGES["k_I"][1]),
                   np.log10(K_Q_OFFSET + TID_RANGES["k_Q"][1])])
    pool = mp.get_context("spawn").Pool(workers, maxtasksperchild=1)
    pre = {m: sweep(k, [VBE_FIX], m, None, 0.0, L_um=CAL_L_UM) for m in ("forward", "inverse")}
    history = []

    def knobs_of(x):
        return {"k_F": float(10 ** x[0]), "k_I": float(10 ** x[1]), "k_Q": float(max(10 ** x[2] - K_Q_OFFSET, 0.0))}

    def model(t):
        jobs = [(k, t, m, d) for m in ("forward", "inverse") for d in meas[m]["dose_Mrad"]]
        res = pool.map(_eval_point, jobs)
        out, i = {}, 0
        for m in ("forward", "inverse"):
            out[m] = []
            for _ in meas[m]["dose_Mrad"]:
                out[m].append(float(ratio(pre[m], res[i])[0]))
                i += 1
        return out

    def f(x):
        t = knobs_of(x)
        mdl = model(t)
        r = np.concatenate([np.log10(np.maximum(np.asarray(mdl[m]), 1e-12) / np.asarray(meas[m]["value"]))
                            for m in ("forward", "inverse")])
        history.append({"knobs": t, "model": mdl, "residuals_log10": [float(v) for v in r]})
        log(f"  A25 fit eval {len(history):2d}: k_F {t['k_F']:.4g} k_I {t['k_I']:.4g} k_Q {t['k_Q']:.4g}  "
            f"fwd {mdl['forward']} inv {mdl['inverse']}  residuals {np.round(r, 4).tolist()}")
        return r

    try:
        sol = least_squares(f, x0, bounds=(lo, hi), diff_step=0.02, xtol=1e-3, ftol=1e-4, max_nfev=40)
        t = knobs_of(sol.x)
        mdl = model(t)
    finally:
        pool.close()
        pool.join()
    out = {"contract": "A25.4", "knobs": t, "ranges": TID_RANGES, "device": k, "VBE_V": VBE_FIX, "L_um": CAL_L_UM,
           "measured": meas, "model": mdl,
           "residuals_log10": {m: [float(np.log10(a / b)) for a, b in zip(mdl[m], meas[m]["value"])]
                               for m in ("forward", "inverse")},
           "pre_irradiation_IB_A": {m: float(pre[m]["IB_A"][0]) for m in pre},
           "evaluations": history,
           "optimizer": {"status": int(sol.status), "message": str(sol.message), "nfev": int(sol.nfev)}}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


# ------------------------------------------------------------------------------------------------ hold-outs
def _sweep_job(args):
    k, VBE, mode, t, D, T, L_um = args
    return sweep(k, VBE, mode, t, D, T=T, L_um=L_um)


def dose_curve(t: Dict[str, float], doses_Mrad: Sequence[float], mode: str, L_um: float = 1.0,
               VBE: Sequence[float] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8), T: float = 300.0,
               k: Optional[Dict[str, float]] = None, workers: int = 1) -> dict:
    """dIB/IB0 at VBE = 0.6 V and at JC = 1 uA/um^2 (and the full sweeps) for each dose. The sweeps are
    independent: workers > 1 runs them on that many processes (the results are the same). A value whose VBE is
    outside the sweep is NaN (no extrapolation)."""
    k = k or M.knobs()
    jobs = [(k, list(VBE), mode, None, 0.0, T, L_um)] + [(k, list(VBE), mode, t, float(D), T, L_um)
                                                         for D in doses_Mrad]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
            res = pool.map(_sweep_job, jobs, chunksize=1)
    else:
        res = [_sweep_job(j) for j in jobs]
    pre, posts = res[0], res[1:]
    try:
        v_jc = vbe_at_jc(pre, JC_FIX_A_UM2)
    except ValueError:
        v_jc = None
    inside = lambda v: v is not None and pre["VBE_V"].min() <= v <= pre["VBE_V"].max()  # noqa: E731
    out = {"pre": pre, "VBE_at_JC_1uA_um2": v_jc, "dose_Mrad": [float(D) for D in doses_Mrad], "at_VBE_0.6": [],
           "at_JC_1uA_um2": [], "post": posts}
    for post in posts:
        out["at_VBE_0.6"].append(at(pre, post, VBE_FIX) if inside(VBE_FIX) else float("nan"))
        out["at_JC_1uA_um2"].append(at(pre, post, v_jc) if inside(v_jc) else float("nan"))
    return out


def ideality(post: dict, pre: dict, vbe: float, T: float = 300.0) -> float:
    """n of the excess base current, (q / kT) / (d ln dIB / dVBE), from the sweep's neighbouring points."""
    kT = 8.617333262e-5 * T
    d = post["IB_cm"] - pre["IB_cm"]
    v = pre["VBE_V"]
    i = int(np.argmin(np.abs(v - vbe)))
    j0, j1 = max(i - 1, 0), min(i + 1, len(v) - 1)
    slope = (np.log(d[j1]) - np.log(d[j0])) / (v[j1] - v[j0])
    return float(1.0 / (kT * slope))
