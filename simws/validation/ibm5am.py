"""The IBM 5AM device of contracts A23 and A24: runs, the base sheet resistance, the three-knob fit, and the
comparisons against the traced data (simws.validation.catalog).

    from simws.validation import ibm5am as M
    k = M.knobs()                                   # the stored calibration (simws/data/validation/ibm5am_calibration.json)
    g = M.gummel(k, [0.6, 0.7, 0.78], T=300.0)      # dict: VBE_V, IC_A, IB_A (one device, L = 2.5 um), RsBi_ohm_sq
    M.fit()                                         # the A23.3 calibration (slow: tens of 2D runs)
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional, Sequence

import numpy as np

from simws.spec.library import IBM5AM_RANGES, ibm5am_hbt, ibm5am_scale
from simws.validation import catalog as C

CAL_FILE = os.path.join(C.DIR, "ibm5am_calibration.json")
Q = 1.602176634e-19
FIT_IC_WINDOW = (0.60, 0.78)        # A23.3
FIT_IB_WINDOW = (0.70, 0.78)
FIT_VBE = (0.60, 0.66, 0.70, 0.74, 0.78)


def knobs(path: str = CAL_FILE) -> Dict[str, float]:
    with open(path) as f:
        d = json.load(f)
    out = {k: float(d["knobs"][k]) for k in ("s_um", "k_B", "k_E")}
    for k in ("k_P", "S_p"):                     # A23b
        if d["knobs"].get(k) is not None:
            out[k] = float(d["knobs"][k])
    return out


def _rsbi_hook(out: dict):
    """A23.3: RsBi = 1 / (q integral mu_p p dy) on the vertical cut x = 0, between the two metallurgical junctions,
    from the equilibrium solution (the engine's state right after its equilibrium solve)."""
    def hook(_state):
        from devsim import get_node_model_values
        from simws.engines.devsim_engine import DEVICE
        g = {n: np.asarray(get_node_model_values(device=DEVICE, region="si", name=n), float)
             for n in ("x", "y", "Holes", "mu_p_node", "NetDoping")}
        cut = g["x"] < 1e-9
        order = np.argsort(g["y"][cut])
        y, p, mu, nd = (g[k][cut][order] for k in ("y", "Holes", "mu_p_node", "NetDoping"))
        ptype = nd < 0
        i = int(np.argmin(nd))                       # the base: the most p-type point
        lo = i
        while lo > 0 and ptype[lo - 1]:
            lo -= 1
        hi = i
        while hi < len(y) - 1 and ptype[hi + 1]:
            hi += 1
        sig = Q * np.trapezoid(mu[lo:hi + 1] * p[lo:hi + 1], y[lo:hi + 1])      # S per square
        out["RsBi_ohm_sq"] = float(1.0 / sig) if sig > 0 else float("inf")
        out["base_cut_um"] = [float(y[lo] * 1e4), float(y[hi] * 1e4)]
    return hook


def gummel(k: Dict[str, float], VBE: Sequence[float], T: float = 300.0, L_um: float = 2.5, VCB: float = 0.0,
           mesh_scale: float = 1.0, step_V: float = 0.1) -> dict:
    """One Gummel sweep of the 5AM device through the DEVSIM engine; currents of one device (A), with RsBi."""
    from simws.engines import devsim_engine as E
    from simws.spec.schema import DCSweep, Track
    an = DCSweep(contact="base", values_V=[float(v) for v in VBE], step_V=step_V,
                 tracking=[Track(contact="collector", offset_V=float(VCB), step_V=0.1)])
    p = ibm5am_hbt(k["s_um"], k["k_B"], k["k_E"], k_P=k.get("k_P", 1.0), S_p=k.get("S_p"), L_um=L_um, T=T, mesh_scale=mesh_scale, analysis=an)
    extra: dict = {}
    t0 = time.time()
    r = E.run(p, _post_build=_rsbi_hook(extra))
    sc = ibm5am_scale(L_um)
    return {"VBE_V": np.asarray(r.sweep_V, float), "IC_A": np.abs(np.asarray(r.currents["collector"])) * sc,
            "IB_A": np.asarray(r.currents["base"], float) * sc, "seconds": time.time() - t0, "flags": r.flags,
            "problem_id": p.problem_id(), **extra}


def _interp_log(x, xs, ys):
    ys = np.asarray(ys, float)
    m = ys > 0
    return 10 ** np.interp(x, np.asarray(xs, float)[m], np.log10(ys[m]))


def measured_300k() -> dict:
    """The calibration data (A23.3): Luo Fig. 5.7a, Fig. 4.7 and Fig. 4.1 at 300 K."""
    ic = C.curve("luo2011.fig5_7a", 300.0)
    be = C.curve("luo2011.fig4_7", 300.0)
    rs = C.curve("luo2011.fig4_1")
    i300 = int(np.argmin(np.abs(rs["T_K"] - 300.0)))
    return {"ic": (ic["VBE_V"], ic["IC_mA"] * 1e-3), "beta": (be["VBE_V"], be["beta"]),
            "RsBi_ohm_sq": float(rs["RsBi_ohm_sq"][i300])}


def residuals(g: dict, meas: dict) -> dict:
    """log10(model / measured) for IC (0.60-0.78 V), IB = IC / beta (0.70-0.78 V) and RsBi (A23.3)."""
    v = g["VBE_V"]
    ic_m = _interp_log(v, *meas["ic"])
    beta_m = np.interp(v, *meas["beta"])
    ib_m = ic_m / beta_m
    s_ic = (v >= FIT_IC_WINDOW[0] - 1e-9) & (v <= FIT_IC_WINDOW[1] + 1e-9)
    s_ib = (v >= FIT_IB_WINDOW[0] - 1e-9) & (v <= FIT_IB_WINDOW[1] + 1e-9)
    return {"ic": np.log10(g["IC_A"][s_ic] / ic_m[s_ic]), "ib": np.log10(g["IB_A"][s_ib] / ib_m[s_ib]),
            "rs": np.array([np.log10(g["RsBi_ohm_sq"] / meas["RsBi_ohm_sq"])]),
            "VBE_ic": v[s_ic], "VBE_ib": v[s_ib], "IC_meas_A": ic_m[s_ic], "IB_meas_A": ib_m[s_ib]}


def _vector(res: dict) -> np.ndarray:
    """Each set weighted equally (A23.3): residuals divided by the square root of the set's size."""
    return np.concatenate([res[k] / np.sqrt(len(res[k])) for k in ("ic", "ib", "rs")])


def fit(start: Optional[Dict[str, float]] = None, log=print, path: str = CAL_FILE) -> dict:
    """A23.3: least squares over (s_um, k_B, k_E) within their ranges; writes the calibration file."""
    from scipy.optimize import least_squares
    meas = measured_300k()
    names = ("s_um", "k_B", "k_E")
    x0 = np.array([(start or {"s_um": 0.8, "k_B": 1.0, "k_E": 1.0})[n] for n in names])
    lo = np.array([IBM5AM_RANGES[n][0] for n in names])
    hi = np.array([IBM5AM_RANGES[n][1] for n in names])
    history = []

    def f(x):
        k = dict(zip(names, (float(v) for v in x)))
        g = gummel(k, FIT_VBE)
        r = residuals(g, meas)
        vec = _vector(r)
        history.append({"knobs": k, "cost": float(0.5 * np.sum(vec ** 2)), "seconds": round(g["seconds"], 1),
                        "rms_log10": {s: float(np.sqrt(np.mean(r[s] ** 2))) for s in ("ic", "ib", "rs")}})
        log(f"  fit eval {len(history):2d}: s {k['s_um']:.4f} k_B {k['k_B']:.4f} k_E {k['k_E']:.4f}  "
            f"rms log10 IC {history[-1]['rms_log10']['ic']:.4f} IB {history[-1]['rms_log10']['ib']:.4f} "
            f"RsBi {history[-1]['rms_log10']['rs']:.4f}  ({g['seconds']:.0f} s)")
        return vec

    sol = least_squares(f, x0, bounds=(lo, hi), x_scale=np.array([0.1, 0.1, 0.3]), diff_step=0.02,
                        xtol=1e-3, ftol=1e-4, max_nfev=40)
    k = dict(zip(names, (float(v) for v in sol.x)))
    g = gummel(k, FIT_VBE)
    r = residuals(g, meas)
    out = {"contract": "A23.3", "knobs": k, "ranges": IBM5AM_RANGES, "fit_VBE_V": list(FIT_VBE),
           "windows": {"IC": FIT_IC_WINDOW, "IB": FIT_IB_WINDOW},
           "residuals_log10": {s: [float(v) for v in r[s]] for s in ("ic", "ib", "rs")},
           "model": {"VBE_V": [float(v) for v in g["VBE_V"]], "IC_A": [float(v) for v in g["IC_A"]],
                     "IB_A": [float(v) for v in g["IB_A"]], "RsBi_ohm_sq": g["RsBi_ohm_sq"]},
           "measured": {"RsBi_ohm_sq": meas["RsBi_ohm_sq"]}, "evaluations": history,
           "optimizer": {"status": int(sol.status), "message": str(sol.message), "nfev": int(sol.nfev)}}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


# ------------------------------------------------------------------------------------------ hold-outs (A23.4)
def _run(k, analysis, T=300.0, L_um=2.5, impact=False, mesh_scale=1.0, hook=None):
    from simws.engines import devsim_engine as E
    p = ibm5am_hbt(k["s_um"], k["k_B"], k["k_E"], k_P=k.get("k_P", 1.0), S_p=k.get("S_p"), L_um=L_um, T=T, mesh_scale=mesh_scale,
                   impact_ionization=impact, analysis=analysis)
    return E.run(p, _post_build=hook), p


def vbe_at_current(V, I, target):
    """The VBE where a rising current reaches `target` (log interpolation)."""
    V, I = np.asarray(V, float), np.asarray(I, float)
    m = I > 0
    return float(np.interp(np.log10(target), np.log10(I[m]), V[m]))


def early_voltage(k, IB_target_A: float, L_um: float = 2.5, vce=(0.6, 1.0, 1.3, 1.6, 2.0)) -> dict:
    """A23 R163: the VBE that gives IB_target at VCE = 1 V, then IC over VCE 0.6 to 2.0 V at that VBE;
    V_A = IC(1.3 V) / slope of the straight line through IC(VCE)."""
    from simws.spec.schema import DCSweep, Track
    vg = np.round(np.arange(0.74, 0.92, 0.02), 4)
    g, _ = _run(k, DCSweep(contact="base", values_V=list(vg), step_V=0.1,
                           tracking=[Track(contact="collector", offset_V=0.0, step_V=0.1)]), L_um=L_um)
    sc = ibm5am_scale(L_um)
    # at VCB = 0 the VCE equals VBE (about 0.8 V); IB hardly depends on VCB here, the VBE is refined below
    v0 = vbe_at_current(vg, np.asarray(g.currents["base"]) * sc, IB_target_A)
    r, _ = _run(k, DCSweep(contact="collector", values_V=[float(v) for v in vce], step_V=0.1,
                           fixed_V={"base": float(v0)}), L_um=L_um)
    ic = np.abs(np.asarray(r.currents["collector"], float)) * sc
    ib = np.asarray(r.currents["base"], float) * sc
    b, a = np.polyfit(np.asarray(vce, float), ic, 1)
    return {"VBE_V": v0, "VCE_V": list(vce), "IC_A": ic.tolist(), "IB_A": ib.tolist(),
            "V_A": float((a + b * 1.3) / b) if b > 0 else float("inf")}


def early_voltage_measured(IB_uA: float) -> float:
    c = C.curve("pellish2008.fig4")["curves"][f"{IB_uA}"]
    v, i = np.asarray(c["VCE_V"], float), np.asarray(c["IC_mA"], float) * 1e-3
    m = (v >= 0.6) & (v <= 2.0)
    b, a = np.polyfit(v[m], i[m], 1)
    return float((a + b * 1.3) / b)


def ionization_integral(y_cm, E_Vcm, T: float):
    """Electron-initiated multiplication along a 1D cut (A23.7): 1 - 1/M = integral alpha_n exp(-integral
    (alpha_n - alpha_p)) dy, the van Overstraeten-de Man coefficients of simws.tcad.models at T. Returns M - 1."""
    from scipy.integrate import cumulative_trapezoid
    from simws.materials import load
    from simws.tcad.models import alpha_numpy
    from simws.tcad.physics import impact_gamma
    mat = load("silicon")
    an, ap = alpha_numpy(mat, E_Vcm, gamma=impact_gamma(mat, T))
    inner = cumulative_trapezoid(an - ap, y_cm, initial=0.0)
    s = float(np.trapezoid(an * np.exp(-inner), y_cm))
    return float(1.0 / (1.0 - s) - 1.0) if s < 1.0 else float("inf")


def _ionization_s(y_cm, E_Vcm, T: float) -> float:
    """1 - 1/M of `ionization_integral` (reaches 1 at breakdown)."""
    from scipy.integrate import cumulative_trapezoid
    from simws.materials import load
    from simws.tcad.models import alpha_numpy
    from simws.tcad.physics import impact_gamma
    mat = load("silicon")
    an, ap = alpha_numpy(mat, E_Vcm, gamma=impact_gamma(mat, T))
    inner = cumulative_trapezoid(an - ap, y_cm, initial=0.0)
    return float(np.trapezoid(an * np.exp(-inner), y_cm))


def _cut_field():
    from devsim import get_node_model_values
    from simws.engines import devsim_engine as E
    x, y, psi, nd = (np.asarray(get_node_model_values(device=E.DEVICE, region="si", name=n), float)
                     for n in ("x", "y", "Potential", "NetDoping"))
    cut = x < 1e-9
    o = np.argsort(y[cut])
    yc, pc, ndc = y[cut][o], psi[cut][o], nd[cut][o]
    b = int(np.argmin(ndc))
    return yc[b:], -np.gradient(pc, yc)[b:]


def bvcbo(k, T: float = 300.0, V_max: float = 20.0, step: float = 0.5) -> dict:
    """A23b.3: emitter and base at 0 V, the collector ramped; BVCBO where 1 - 1/M on the cut x = 0 (from the
    base's most p-type point) reaches 1, interpolated linearly between the last two steps."""
    from simws.engines import devsim_engine as E
    from simws.spec.schema import Equilibrium
    p = ibm5am_hbt(k["s_um"], k["k_B"], k["k_E"], k_P=k.get("k_P", 1.0), S_p=k.get("S_p"), T=T,
                   analysis=Equilibrium())
    st = E._build(p)
    V, S = [0.0], [_ionization_s(*_cut_field(), T)]
    v = 0.0
    while v < V_max - 1e-9:
        v += step
        st.ramp("collector", v, step)
        V.append(v)
        S.append(_ionization_s(*_cut_field(), T))
        if S[-1] >= 1.0:
            bv = V[-2] + (1.0 - S[-2]) * (V[-1] - V[-2]) / (S[-1] - S[-2])
            return {"BVCBO_V": float(bv), "V": V, "one_minus_1_over_M": S}
    return {"BVCBO_V": float("inf"), "V": V, "one_minus_1_over_M": S}


def avalanche(k, T: float = 300.0, VCB=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0), IC_target_A: float = 1e-5,
              VBE_guess: Optional[Sequence[float]] = None, L_um: float = 2.5) -> dict:
    """A23 R164 / A24 R173 (as clarified in A23.7): VBE where IC = 10 uA at VCB = 0; at each VCB the DC solution
    without impact ionization and M - 1 from the electron ionization integral on the cut x = 0 (below the emitter,
    through the base-collector junction)."""
    from devsim import get_node_model_values
    from simws.engines import devsim_engine as E
    from simws.spec.schema import DCSweep, Track
    vg = list(VBE_guess) if VBE_guess is not None else list(np.round(np.arange(0.60, 0.80, 0.02), 4))
    g, _ = _run(k, DCSweep(contact="base", values_V=vg, step_V=0.1,
                           tracking=[Track(contact="collector", offset_V=0.0, step_V=0.1)]), T=T, L_um=L_um)
    sc = ibm5am_scale(L_um)
    v0 = vbe_at_current(vg, np.abs(np.asarray(g.currents["collector"])) * sc, IC_target_A)
    p = ibm5am_hbt(k["s_um"], k["k_B"], k["k_E"], k_P=k.get("k_P", 1.0), S_p=k.get("S_p"), L_um=L_um, T=T, VBE=[v0])
    st = E._build(p)
    st.ramp("base", float(v0), 0.1)
    out = {"VBE_V": v0, "VCB_V": list(VCB), "IC_A": [], "M_minus_1": []}
    for vcb in VCB:
        st.ramp("collector", float(v0 + vcb), 0.1)
        x, y, psi, nd = (np.asarray(get_node_model_values(device=E.DEVICE, region="si", name=n), float)
                         for n in ("x", "y", "Potential", "NetDoping"))
        cut = x < 1e-9
        o = np.argsort(y[cut])
        yc, pc, ndc = y[cut][o], psi[cut][o], nd[cut][o]
        Ey = -np.gradient(pc, yc)
        b = int(np.argmin(ndc))              # from the base (its most p-type point) through the collector junction
        out["M_minus_1"].append(ionization_integral(yc[b:], Ey[b:], T))
        out["IC_A"].append(abs(float(st.currents()["collector"])) * sc)
    return out


def ft_curve(k, VBE: Sequence[float], VCB: float = 0.0, f_Hz: float = 2e9, L_um: float = 2.5, T: float = 300.0):
    """A23 R165: small signal (base = port 1, collector = port 2, emitter AC ground) at each VBE; fT = f |h21| at
    2 GHz; JE in mA/um^2 over the drawn emitter area 0.5 x L."""
    from simws.analysis import two_port as TP
    from simws.spec.schema import Bias, SmallSignal
    out = {"VBE_V": [], "JE_mA_um2": [], "fT_GHz": []}
    sc = ibm5am_scale(L_um)
    for v in VBE:
        an = SmallSignal(bias=[Bias(contact="collector", value_V=float(v + VCB), step_V=0.1),
                                Bias(contact="base", value_V=float(v), step_V=0.1)],
                         ports=("base", "collector"), frequencies_Hz=[f_Hz])
        r, _ = _run(k, an, T=T, L_um=L_um)
        y = {kk: complex(r.y_real[kk][0], r.y_imag[kk][0]) for kk in ("11", "12", "21", "22")}
        h = abs(TP.h21(y["11"], y["12"], y["21"], y["22"]))
        ie = abs(float(r.currents["emitter"][0])) * sc
        out["VBE_V"].append(float(v))
        out["JE_mA_um2"].append(ie * 1e3 / (0.5 * L_um))
        out["fT_GHz"].append(f_Hz * h / 1e9)
    return out


def rsbi(k, T: float) -> float:
    """RsBi at temperature T from the equilibrium solve (A23.3's definition)."""
    from simws.spec.schema import Equilibrium
    extra: dict = {}
    _run(k, Equilibrium(), T=T, hook=_rsbi_hook(extra))
    return extra["RsBi_ohm_sq"]


# ------------------------------------------------------------------------------------------ A24: cold and hot
A24_TEMPERATURES = (393.0, 223.0, 162.0, 93.0)          # Luo's (A24.1); 77 K is Sutton's (R174); 43 K reported


def luo_targets(T: float) -> dict:
    """From the traces at T: VBE at IC = 1 uA and 100 uA (Fig. 5.7a), the beta peak and its VBE (Fig. 4.7, where
    traced), RsBi (Fig. 4.1, nearest point)."""
    out = {"T_K": T}
    c = C.curve("luo2011.fig5_7a", T)
    out["VBE_1uA"] = vbe_at_current(c["VBE_V"], np.maximum.accumulate(c["IC_mA"] * 1e-3), 1e-6)
    out["VBE_100uA"] = vbe_at_current(c["VBE_V"], np.maximum.accumulate(c["IC_mA"] * 1e-3), 1e-4)
    try:
        b = C.curve("luo2011.fig4_7", T)
        i = int(np.argmax(b["beta"]))
        out["beta_peak"], out["VBE_beta_peak"] = float(b["beta"][i]), float(b["VBE_V"][i])
    except KeyError:
        pass
    return out


def gummel_window(T: float) -> list:
    """The VBE grid A24 runs at T: from 50 mV below the lower of (the beta peak, IC = 1 uA) to 30 mV above
    IC = 100 uA, every 10 mV."""
    t = luo_targets(T)
    lo = min(t["VBE_1uA"], t.get("VBE_beta_peak", t["VBE_1uA"])) - 0.05
    hi = t["VBE_100uA"] + 0.03
    return [round(float(v), 3) for v in np.arange(round(lo, 2), hi + 1e-9, 0.01)]


def moll_ross(k, VBE: Sequence[float], T: float = 300.0, L_um: float = 2.5) -> list:
    """A23 R167 (reported): the generalized Moll-Ross / Kroemer collector current of the vertical cut x = 0,
    IC = q A_E V_t exp(VBE / V_t) / integral over the base of p / (mu_n n_ie^2) dy (equilibrium p, mu_n, n_ie of
    the device's own models), with A_E the effective emitter (2 x 0.21 um x L)."""
    from simws.spec.schema import Equilibrium
    got: dict = {}

    def hook(_s):
        from devsim import get_node_model_values
        from simws.engines.devsim_engine import DEVICE
        g = {n: np.asarray(get_node_model_values(device=DEVICE, region="si", name=n), float)
             for n in ("x", "y", "Holes", "mu_n_node", "n_ie", "NetDoping")}
        cut = g["x"] < 1e-9
        o = np.argsort(g["y"][cut])
        y, p, mu, ni, nd = (g[n][cut][o] for n in ("y", "Holes", "mu_n_node", "n_ie", "NetDoping"))
        ptype = nd < 0
        i = int(np.argmin(nd))
        lo, hi = i, i
        while lo > 0 and ptype[lo - 1]:
            lo -= 1
        while hi < len(y) - 1 and ptype[hi + 1]:
            hi += 1
        got["G"] = float(np.trapezoid(p[lo:hi + 1] / (mu[lo:hi + 1] * ni[lo:hi + 1] ** 2), y[lo:hi + 1]))
    _run(k, Equilibrium(), T=T, hook=hook)
    vt = 1.380649e-23 * T / Q
    A = 2 * 0.21e-4 * L_um * 1e-4
    return [float(Q * A * vt * np.exp(v / vt) / got["G"]) for v in VBE]


# ------------------------------------------------------------------------------------------ A23b: the revised fit
A23B_FIXED = {"s_um": 0.80, "k_E": 1.0}
BVCBO_MEASURED_V = 10.5                    # Lourenco Table 6.1 (from Ahlgren et al. 1997)


def _eval_gummel(k):
    return gummel(k, FIT_VBE)


def _eval_bv(k):
    return bvcbo(k, step=1.0)


def fit_b(start: Optional[Dict[str, float]] = None, log=print, path: str = CAL_FILE) -> dict:
    """A23b.3: least squares over (k_B, k_P, log10 S_p), s_um = 0.80 and k_E = 1 fixed; the data of A23.3 plus
    BVCBO = 10.5 V, each set weighted equally. Each evaluation runs the Gummel sweep and the BVCBO ramp in two
    processes."""
    import multiprocessing as mp
    from scipy.optimize import least_squares
    meas = measured_300k()
    st = start or {"k_B": 0.63, "k_P": 1.0, "S_p": 1e5}
    x0 = np.array([st["k_B"], st["k_P"], np.log10(st["S_p"])])
    lo = np.array([IBM5AM_RANGES["k_B"][0], IBM5AM_RANGES["k_P"][0], np.log10(IBM5AM_RANGES["S_p"][0])])
    hi = np.array([IBM5AM_RANGES["k_B"][1], IBM5AM_RANGES["k_P"][1], np.log10(IBM5AM_RANGES["S_p"][1])])
    history = []
    pool = mp.get_context("spawn").Pool(2, maxtasksperchild=1)

    def knobs_of(x):
        return {**A23B_FIXED, "k_B": float(x[0]), "k_P": float(x[1]), "S_p": float(10 ** x[2])}

    def evaluate(k):
        a = pool.apply_async(_eval_gummel, (k,))
        b = pool.apply_async(_eval_bv, (k,))
        return a.get(), b.get()

    def f(x):
        k = knobs_of(x)
        g, bv = evaluate(k)
        r = residuals(g, meas)
        r["bv"] = np.array([np.log10(bv["BVCBO_V"] / BVCBO_MEASURED_V)])
        vec = np.concatenate([r[s] / np.sqrt(len(r[s])) for s in ("ic", "ib", "rs", "bv")])
        history.append({"knobs": k, "cost": float(0.5 * np.sum(vec ** 2)), "BVCBO_V": bv["BVCBO_V"],
                        "rms_log10": {s: float(np.sqrt(np.mean(r[s] ** 2))) for s in ("ic", "ib", "rs", "bv")}})
        h = history[-1]["rms_log10"]
        log(f"  fit_b eval {len(history):2d}: k_B {k['k_B']:.4f} k_P {k['k_P']:.4f} S_p {k['S_p']:.4g}  rms log10 "
            f"IC {h['ic']:.4f} IB {h['ib']:.4f} RsBi {h['rs']:.4f} BV {h['bv']:.4f} (BVCBO {bv['BVCBO_V']:.2f} V)")
        return vec

    try:
        sol = least_squares(f, x0, bounds=(lo, hi), x_scale=np.array([0.1, 0.2, 0.3]), diff_step=0.02,
                            xtol=1e-3, ftol=1e-4, max_nfev=30)
        k = knobs_of(sol.x)
        g, bv = evaluate(k)
    finally:
        pool.close()
        pool.join()
    r = residuals(g, meas)
    out = {"contract": "A23b.3", "knobs": k, "fixed": A23B_FIXED, "ranges": IBM5AM_RANGES,
           "fit_VBE_V": list(FIT_VBE), "windows": {"IC": FIT_IC_WINDOW, "IB": FIT_IB_WINDOW},
           "residuals_log10": {s: [float(v) for v in r[s]] for s in ("ic", "ib", "rs")},
           "BVCBO": {"model_V": bv["BVCBO_V"], "measured_V": BVCBO_MEASURED_V},
           "model": {"VBE_V": [float(v) for v in g["VBE_V"]], "IC_A": [float(v) for v in g["IC_A"]],
                     "IB_A": [float(v) for v in g["IB_A"]], "RsBi_ohm_sq": g["RsBi_ohm_sq"]},
           "measured": {"RsBi_ohm_sq": meas["RsBi_ohm_sq"]}, "evaluations": history,
           "optimizer": {"status": int(sol.status), "message": str(sol.message), "nfev": int(sol.nfev)}}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out
