"""SiGe device study: HBT vs Si BJT (Gummel, gain, Early voltage), the buried strained-SiGe pMOS channel, and
the HBT radiation response (excess base current vs dose, collected charge vs LET). Each task is an independent
DEVSIM run, so they go to a process pool.

    from simws.studies.sige_devices import run
    res = run(workers=4)
"""
from __future__ import annotations

import dataclasses
from concurrent.futures import ProcessPoolExecutor

import numpy as np

GE_PEAK = 0.15
VBE = np.round(np.arange(0.40, 0.851, 0.025), 4)
LETS = (0.3, 1.0, 3.0, 10.0)
DOSES_KRAD = (0.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
TID_VBE = (0.4, 0.5, 0.6, 0.7, 0.8)


def _task(spec):
    kind, arg = spec
    from simws.tcad.hbt import HBTCase, early_voltage, gummel
    if kind == "gummel":
        r = gummel(HBTCase(ge_peak=arg), VBE)
        return {k: np.asarray(v).tolist() for k, v in r.items()}
    if kind == "early":
        r = early_voltage(HBTCase(ge_peak=arg))
        return {"VCB": r["VCB"].tolist(), "IC_A": r["IC_A"].tolist(), "V_A": r["V_A"]}
    if kind == "mos":
        from simws.tcad.mos import MOSCase, cv, hole_sheets
        c = MOSCase(t_ox_nm=5, NA=-1e17, cap_nm=4, well_nm=10, ge_x=arg)
        h = hole_sheets(c, np.linspace(0.0, -3.0, 31))
        q = cv(c, np.linspace(-3.0, 1.5, 46))
        return {"V_sheets": np.asarray(h["V"]).tolist(), "cap_cm2": np.asarray(h["cap"]).tolist(),
                "well_cm2": np.asarray(h["well"]).tolist(), "V_cv": np.asarray(q["V"]).tolist(),
                "C_over_Cox": (np.asarray(q["C_F_cm2"]) / q["C_ox_F_cm2"]).tolist()}
    if kind == "tid":
        from simws.radiation.hbt_rad import dose_response
        r = dose_response(HBTCase(ge_peak=GE_PEAK), DOSES_KRAD, TID_VBE)
        return {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
    if kind == "strike":
        from simws.radiation.hbt_rad import StrikeSetup, strike
        ge, let = arg
        r = strike(HBTCase(ge_peak=ge), StrikeSetup(let=let))
        return {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
    raise ValueError(kind)


def run(workers=4):
    specs = ([("gummel", g) for g in (0.0, GE_PEAK)] + [("early", g) for g in (0.0, GE_PEAK)]
             + [("mos", x) for x in (0.0, 0.25)] + [("tid", None)]
             + [("strike", (g, L)) for g in (0.0, GE_PEAK) for L in LETS])
    with ProcessPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(_task, specs))
    res = {"ge_peak": GE_PEAK, "gummel": {}, "early": {}, "mos": {}, "strike": {}, "tid": None}
    for (kind, arg), o in zip(specs, outs):
        if kind in ("gummel", "early"):
            res[kind]["SiGe" if arg else "Si"] = o
        elif kind == "mos":
            res["mos"][f"x_{arg:g}"] = o
        elif kind == "tid":
            res["tid"] = o
        else:
            g, L = arg
            res["strike"].setdefault("SiGe" if g else "Si", []).append(o)
    gs, gg = res["gummel"]["Si"], res["gummel"]["SiGe"]
    i6 = int(np.argmin(np.abs(VBE - 0.6)))
    res["summary"] = {
        "IC_ratio_at_0.6V": gg["IC_A"][i6] / gs["IC_A"][i6],
        "beta_peak": {"Si": float(np.nanmax(gs["beta"])), "SiGe": float(np.nanmax(gg["beta"]))},
        "V_A": {k: v["V_A"] for k, v in res["early"].items()},
        "Q_collected_fC": {k: {f"LET_{r['let']:g}": r["Q_collector_fC"] for r in v} for k, v in res["strike"].items()},
        "beta_at_0.5V_vs_dose": dict(zip([f"{d:g}krad" for d in DOSES_KRAD],
                                         [row[1] for row in res["tid"]["beta"]])),
    }
    return res
