"""Transient enhanced diffusion, the effective +1 model (contract A22): Stolk's Figure 1 and a boron junction.

    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/v1_ted_demo.py [--store DIR] [--inprocess]

R150: Stolk et al. 1997 Fig. 1 (B 30 keV 1.5e14, SIMS) annealed at 800 C for 35 min from the traced as-implanted
profile, with and without TED, against the traced diffused SIMS. R153: B 35 keV 1e14 crystal (A20) on n-type 1e16
at 900 C 10 s, 1000 C 10 s and 1000 C 30 min, junction with and without TED.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main(argv) -> int:
    from simws.chain import process as P
    from simws.compare import compare_results, record
    from simws.hub.jobs import Hub
    from simws.hub.trust import assess_stored
    from simws.process import ted
    from simws.process.diffusivity import kelvin
    from simws.spec.library import implant_si
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--inprocess", action="store_true")
    a = ap.parse_args(argv[1:])
    hub = Hub(a.store, mode="inprocess" if a.inprocess else "subprocess")
    print("store:", hub.store.root, "| mode:", hub.mode, flush=True)
    print(f"model: S(790 C) = {ted.S_REF:.0f}, E_S = {ted.E_S_EV:.3f} eV, E_tau = {ted.E_TAU_EV} eV, "
          f"tau_ref = {ted.TAU_REF_S:g} s at 815 C for 5e13 Si (Rp {ted.RP_REF_UM * 1e3:.2f} nm)")

    ini, (dx, dc), _ = ted.stolk_fig1()
    Q, rp = ted.damage(P.anneal_problem("B", ini, thickness_um=1.5))
    T = kelvin(800.0)
    print(f"\n== R150: Stolk Fig. 1, B 30 keV 1.5e14, 800 C 35 min (traced dose {Q:.3g} cm^-2, mean depth "
          f"{rp * 1e3:.1f} nm; tau {ted.tau_s(T, Q, rp):.0f} s, S {ted.supersaturation(T):.0f}, enhancement "
          f"{ted.enhancement('B', T):.0f})")
    meas = {lv: ted.depth_at(dx, dc, lv) for lv in (3e17, 1e17)}
    print(f"  SIMS after:   x(3e17) {meas[3e17] * 1e3:6.1f} nm, x(1e17) {meas[1e17] * 1e3:6.1f} nm")
    print(f"  SIMS before:  x(3e17) {ted.depth_at(ini.depth_um, ini.concentration_cm3, 3e17) * 1e3:6.1f} nm, "
          f"x(1e17) {ted.depth_at(ini.depth_um, ini.concentration_cm3, 1e17) * 1e3:6.1f} nm")
    bad = 0
    for mode in ("none", "plus_one"):
        p = P.anneal_problem("B", ini, background_cm3=1e15, background_type="donor", steps=[(800.0, 2100.0)],
                             model="fermi", thickness_um=1.5).with_changes({"models.ted": mode})
        ids = {}
        for eng in ("anneal.devsim_anneal", "anneal.fipy"):
            r = hub.submit(eng, params={"problem": p})
            if r.status != "ok":
                bad += 1
                print(f"  {mode} {eng}: {r.status}: {r.message[-200:]}")
                continue
            ids[eng] = r.id
            res = hub.result(r.id)
            q = res.q("anneal.concentration")
            x = np.asarray(q.axes[0].values) * 1e6
            c = np.asarray(q.array()) * 1e-6
            got = {lv: ted.depth_at(x, c, lv) for lv in (3e17, 1e17)}
            dev = {lv: 100 * (got[lv] / meas[lv] - 1) for lv in got}
            print(f"  {mode:8s} {eng.split('.')[1]:14s} x(3e17) {got[3e17] * 1e3:6.1f} nm ({dev[3e17]:+5.1f} %), "
                  f"x(1e17) {got[1e17] * 1e3:6.1f} nm ({dev[1e17]:+5.1f} %), peak {c.max():.3g} cm^-3")
        if len(ids) == 2 and mode == "plus_one":
            from simws.measured import ingest
            m = ingest(hub, "stolk1997_fig1_diffused", None,
                       {"anneal.depth_at_3e17": ([meas[3e17] * 1e-6], "m"),
                        "anneal.depth_at_1e17": ([meas[1e17] * 1e-6], "m")},
                       {"source": "Stolk et al., J. Appl. Phys. 81, 6031 (1997), Fig. 1 (traced, A22)",
                        "digitized": True}, label="validation")
            for name in ("anneal.depth_at_3e17", "anneal.depth_at_1e17"):
                for eng, rid in ids.items():
                    cm = compare_results(hub, rid, name, m.id, tolerance=0.15)          # A22 R150
                    record(hub, cm)
            ce = compare_results(hub, ids["anneal.devsim_anneal"], "anneal.depth_at_1e17", ids["anneal.fipy"],
                                 tolerance=0.01)                                            # A16.8
            record(hub, ce)
            print(f"  trust of the TED result: {assess_stored(hub, ids['anneal.devsim_anneal']).level}")
    print("  (gate: within 15 % at both levels with TED; the peak is not gated: boron clustering is not modelled)")

    print("\n== R153: B 35 keV 1e14 crystal (Tasch, A20) on n-type 1e16, Fermi")
    crys = implant_si("B", 35.0, tilt_deg=7.0, dose_cm2=1e14, crystal=True, rotation_deg=30.0,
                      thickness_um=1.5).with_changes({"depth_bin_um": 0.002})
    r = hub.submit("implant.channeling", params={"problem": crys})
    ini2 = P.initial_from_implant(hub.result(r.id))
    for T_C, t_s in ((900.0, 10.0), (1000.0, 10.0), (1000.0, 1800.0)):
        row = []
        for mode in ("none", "plus_one"):
            p = P.anneal_problem("B", ini2, background_cm3=1e16, background_type="donor", steps=[(T_C, t_s)],
                                 model="fermi", thickness_um=2.0).with_changes({"models.ted": mode})
            ra = hub.submit("anneal.devsim_anneal", params={"problem": p})
            row.append(hub.result(ra.id).q("anneal.junction_depth").value * 1e9 if ra.status == "ok" else float("nan"))
        Qb, rpb = ted.damage(P.anneal_problem("B", ini2, thickness_um=2.0))
        print(f"  {T_C:6.0f} C {t_s:6.0f} s: junction {row[0]:6.1f} nm without TED, {row[1]:6.1f} nm with "
              f"(tau {ted.tau_s(kelvin(T_C), Qb, rpb):.3g} s)")
    print("\nTED is an effective model (A22.2); trust: two engines agree -> verified; R150 compares against a "
          "measurement.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
