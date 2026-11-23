"""Mixed mode (A18) and the G4SEE depth profile in the device (A19): critical charge and upset probability.

    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/v1_mixed_mode_demo.py [--store DIR] [--inprocess] [--fast]

Part 1 (A18 R115): the reference n+/p diode of A17.6 (1 um^2) on a node at 1 V through R with 1 fF: the critical
charge solved in mixed mode, next to the fixed-bias TCAD pulse and Messenger's pulse (both through ngspice).
Part 2 (A19 R124): protons at normal incidence; G4SEE gives the mean deposit in 0.5 um slices over the device's
5 um; the device runs with that depth profile in mixed mode; the upset probability per proton follows from G4SEE's
spectrum in the whole 0-5 um column, with the mixed-mode q_crit and with Messenger's.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

AREA_M2 = 1e-12            # 1 um^2
C_F = 1e-15
Q_LO, Q_HI = 0.05e-15, 100e-15


def rc_node(R_ohm: float, C_farad: float = C_F):
    from simws.spec import CircuitProblem
    return CircuitProblem.model_validate({
        "title": f"node {C_farad * 1e15:g} fF, {R_ohm / 1e3:g} kOhm", "probes": ["v(n)"],
        "elements": [{"name": "vdd", "kind": "vsource", "nodes": ["vdd", "0"], "value": 1.0},
                     {"name": "r1", "kind": "resistor", "nodes": ["vdd", "n"], "value": R_ohm},
                     {"name": "c1", "kind": "capacitor", "nodes": ["n", "0"], "value": C_farad}],
        "analysis": {"kind": "tran", "step_s": 1e-13, "stop_s": 3e-10, "max_step_s": 1e-13}})


def _try(fn):
    try:
        return fn(), None
    except ValueError as e:
        if "does not upset" in str(e):
            return None, f"no upset up to {Q_HI * 1e15:g} fC"
        raise


def _fmt(out, err):
    return f"{out['qcrit_C'] * 1e15:9.3f}" if out else f"{'none':>9s}  ({err})"


def main(argv) -> int:
    from simws import engines
    from simws.chain import see as C
    from simws.hub.jobs import Hub
    from simws.spec.library import see_si_slab
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--inprocess", action="store_true")
    ap.add_argument("--fast", action="store_true", help="fewer events (5000) for a quick look")
    a = ap.parse_args(argv[1:])
    hub = Hub(a.store, mode="inprocess" if a.inprocess else "subprocess")
    print("store:", hub.store.root, "| mode:", hub.mode, flush=True)
    spice = engines.module("ngspice").available()
    dev = C.reference_strike_diode()

    print("\n== Part 1: critical charge, node at 1 V, 1 fF, threshold 0.5 V, 1 um^2 (A18 R115)")
    print(f"{'R':>8s} {'mixed mode fC':>14s} {'fixed-bias TCAD fC':>19s} {'Messenger fC':>13s}")
    qmixed = {}
    for R in (10e3, 100e3):
        devl = C.with_load(dev, "top", 1.0, R, C_F, AREA_M2)
        mm, e1 = _try(lambda: C.critical_charge_mixed(hub, devl, 0.5, AREA_M2, Q_LO, Q_HI))
        qmixed[R] = mm
        if spice:
            fb, e2 = _try(lambda: C.critical_charge_tcad(hub, rc_node(R), "n", 0.5, dev, "top", AREA_M2, Q_LO, Q_HI))
            me, e3 = _try(lambda: C.critical_charge(hub, rc_node(R), "n", 0.5, 1e-12, 10e-12, q_low_C=Q_LO,
                                                    q_high_C=Q_HI))
            print(f"{R / 1e3:6g} k {_fmt(mm, e1):>14s} {_fmt(fb, e2):>19s} {_fmt(me, e3):>13s}")
        else:
            print(f"{R / 1e3:6g} k {_fmt(mm, e1):>14s}   (ngspice not installed: no circuit comparisons)")

    print("\n== Part 2: protons into the 5 um column, R 100 kOhm, 1 fF (A19 R124)")
    n = 5000 if a.fast else 20000
    R = 100e3
    for E, hi in ((5.0, 0.3), (2.0, 0.6)):
        sp = see_si_slab("proton", E, sv_thickness_um=5.0, bulk_thickness_um=50.0, width_um=20.0, n_events=n,
                         high_MeV=hi, bins=3000).with_changes({"geometry.sv_width_um": [1.0, 1.0]})
        dp = C.deposit_profile(hub, sp, np.round(np.arange(0.0, 5.0001, 0.5), 6))
        prof = C.profile_from_deposit(dp)
        r_full = hub.submit("see.g4see", params={"problem": sp})
        full = hub.result(r_full.id)
        devp = C.with_load(C.strike_from_see(dev, prof), "top", 1.0, R, C_F, AREA_M2)
        mm, e1 = _try(lambda: C.critical_charge_mixed(hub, devp, 0.5, AREA_M2, Q_LO, Q_HI))
        shape = np.asarray(dp["MeV_per_um"])
        print(f"\nproton {E:g} MeV: mean deposit in 0-5 um {full.q('see.edep_mean').value / 1e6 * 1e3:.2f} keV "
              f"(slices sum {dp['total_MeV'] * 1e3:.2f} keV); MeV/um from {shape.min():.4f} to {shape.max():.4f}")
        if mm:
            up = C.upset_probability(full, mm)
            print(f"  mixed mode, G4SEE profile: q_crit {mm['qcrit_C'] * 1e15:.3f} fC, P(upset) per proton "
                  f"{up['P']:.4g} +- {up['sigma']:.2g}")
        else:
            print(f"  mixed mode, G4SEE profile: {e1}")
        if qmixed.get(R):
            up = C.upset_probability(full, qmixed[R])
            print(f"  mixed mode, uniform 0.25-0.60 um (as if all were deposited in the depletion region): q_crit {qmixed[R]['qcrit_C'] * 1e15:.3f} fC, P(upset) "
                  f"{up['P']:.4g}")
        if spice:
            me, e3 = _try(lambda: C.critical_charge(hub, rc_node(R), "n", 0.5, 1e-12, 10e-12, q_low_C=Q_LO,
                                                    q_high_C=Q_HI))
            if me:
                up = C.upset_probability(full, me)
                print(f"  Messenger (1 ps, 10 ps):   q_crit {me['qcrit_C'] * 1e15:.3f} fC, P(upset) {up['P']:.4g}")
    print("\nFlags: strike:mixed_mode, strike:1d_column, strike:mean_profile. One device engine: unverified.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
