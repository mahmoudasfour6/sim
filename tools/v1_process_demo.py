"""Implant -> anneal -> device -> circuit (contract A16.6, R99).

    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/v1_process_demo.py [--store DIR] [--inprocess]

Boron 35 keV, 1e14 cm^-2, tilt 7 deg into amorphous silicon (RustBCA, "empirical" stopping) on an n-type 1e16
background; 1000 C for 30 min with the intrinsic and the Fermi diffusivity on both anneal engines (DEVSIM and FiPy);
the two engines compared and the comparison recorded (trust); a DEVSIM diode from the Fermi profile; its Shockley
fit; the SPICE diode of the fit on ngspice against the device at 0.6 V.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

AREA_M2 = 1e-10           # 10 um x 10 um diode for the circuit
JUNCTION_TOL = 0.01       # A16.8 / R94


def main(argv) -> int:
    from simws.chain import diode as D
    from simws.chain import process as P
    from simws.compare import compare_results, record
    from simws.hub.jobs import Hub
    from simws.hub.trust import assess_stored
    from simws.spec.library import implant_si
    ap_ = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap_.add_argument("--store", default=None)
    ap_.add_argument("--inprocess", action="store_true")
    a = ap_.parse_args(argv[1:])
    hub = Hub(a.store, mode="inprocess" if a.inprocess else "subprocess")
    print("store:", hub.store.root, "| mode:", hub.mode, flush=True)

    imp = implant_si("B", 35, tilt_deg=7.0, dose_cm2=1e14, thickness_um=1.0)
    imp = imp.with_changes({"models.electronic_stopping": "empirical"})
    r = hub.submit("implant.rustbca", params={"problem": imp})
    if r.status != "ok":
        print("implant:", r.status, r.message[-300:])
        return 1
    ri = hub.result(r.id)
    print(f"\nimplant B 35 keV 1e14 (RustBCA, empirical): Rp {ri.q('implant.Rp').value * 1e9:.1f} nm, "
          f"dRp {ri.q('implant.dRp').value * 1e9:.1f} nm")
    ini = P.initial_from_implant(ri)
    print(f"as-implanted dose in the profile: {ini.dose_cm2():.4e} cm^-2")

    bad = 0
    results = {}
    print(f"\n{'model':>9s} {'engine':>14s} {'xj nm':>9s} {'dose cm^-2':>12s} {'trust':>10s}")
    for model in ("intrinsic", "fermi"):
        ap = P.anneal_problem("B", ini, background_cm3=1e16, background_type="donor", steps=[(1000.0, 1800.0)],
                              model=model)
        ids = {}
        for eng in ("anneal.devsim_anneal", "anneal.fipy"):
            ra = hub.submit(eng, params={"problem": ap})
            if ra.status != "ok":
                bad += 1
                print(f"{model:>9s} {eng:>14s}: {ra.status}: {ra.message[-200:]}")
                continue
            ids[eng] = ra.id
        if len(ids) == 2:
            a_id, b_id = ids["anneal.devsim_anneal"], ids["anneal.fipy"]
            c = compare_results(hub, a_id, "anneal.junction_depth", b_id, tolerance=JUNCTION_TOL)   # A16.8
            record(hub, c)
            if not c.passed:
                bad += 1
                print(f"  {model}: junction depth differs by {c.worst:.2e} (limit {JUNCTION_TOL:g})")
        for eng, rid in ids.items():
            res = hub.result(rid)
            print(f"{model:>9s} {eng.split('.')[1]:>14s} {res.q('anneal.junction_depth').value * 1e9:9.1f} "
                  f"{res.q('anneal.dose').value * 1e-4:12.4e} {assess_stored(hub, rid).level:>10s}")
            results[(model, eng)] = (res, ap)

    res, ap = results[("fermi", "anneal.devsim_anneal")]
    V = [round(v, 3) for v in np.arange(0.30, 0.6501, 0.05)]
    dp = P.diode_from_anneal(res, ap, V=V)
    rd = hub.submit("device.devsim", params={"problem": dp})
    if rd.status != "ok":
        print("device:", rd.status, rd.message[-300:])
        return 1
    J = P.forward_J(hub.result(rd.id), dp)
    fit = D.fit_shockley(P.forward_V(dp), J, dp.models.temperature_K)
    print(f"\nDEVSIM diode from the Fermi profile (top p+, n 1e16): J(0.6 V) = {J[V.index(0.6)]:.4e} A/m^2")
    print(f"Shockley fit: Js = {fit['Js_A_m2']:.4e} A/m^2, N = {fit['N']:.4f}, "
          f"largest residual {100 * fit['max_rel_residual']:.2f} %")
    cp = D.circuit_from_fit(fit, area_m2=AREA_M2, start=0.30, stop=0.65, step=0.05)
    rc = hub.submit("circuit.ngspice", params={"problem": cp})
    if rc.status != "ok":
        print("ngspice:", rc.status, rc.message[-300:])
        return 1
    rcr = hub.result(rc.id)
    iq = next(q for q in rcr.quantities if q.name.startswith("circuit.i."))
    vs = np.asarray(iq.axes[0].values, float)
    i06 = abs(float(np.interp(0.6, vs, np.asarray(iq.array(), float))))
    dev06 = J[V.index(0.6)] * AREA_M2
    print(f"ngspice diode ({AREA_M2 * 1e12:g} um^2) at 0.6 V: {i06:.4e} A; the device: {dev06:.4e} A "
          f"({100 * (i06 / dev06 - 1):+.2f} %)")
    print("\nNot modelled: transient enhanced diffusion from the implant damage (it would deepen the boron "
          "junction), clustering, solubility. Anneal results compared engine to engine are 'verified'.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
