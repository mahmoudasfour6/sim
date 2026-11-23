"""TCAD strike current as the circuit pulse (contract A17, R106; needs the circuit engine).

    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/v1_strike_pulse_demo.py [--store DIR] [--inprocess]

The reference n+/p diode of A17.6 (1 um^2) on a circuit node held at 1 V through 10 kOhm with 1 fF to ground. First
the device alone at four deposits (collected share, peak current, width); then the critical charge three ways: the
device's own pulse rerun at every trial charge, one device pulse scaled linearly, and Messenger's double
exponential (1 ps, 10 ps) of A10.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

AREA_M2 = 1e-12            # 1 um^2
R_OHM, C_F = 10e3, 1e-15


def rc_node(R_ohm: float = R_OHM, C_farad: float = C_F):
    from simws.spec import CircuitProblem
    return CircuitProblem.model_validate({
        "title": f"node {C_farad * 1e15:g} fF, {R_ohm / 1e3:g} kOhm", "probes": ["v(n)"],
        "elements": [{"name": "vdd", "kind": "vsource", "nodes": ["vdd", "0"], "value": 1.0},
                     {"name": "r1", "kind": "resistor", "nodes": ["vdd", "n"], "value": R_ohm},
                     {"name": "c1", "kind": "capacitor", "nodes": ["n", "0"], "value": C_farad}],
        "analysis": {"kind": "tran", "step_s": 1e-13, "stop_s": 3e-10, "max_step_s": 1e-13}})


def fwhm(t, i):
    k = int(np.argmax(i))
    half = 0.5 * i[k]
    above = np.nonzero(i >= half)[0]
    return float(t[above[-1]] - t[above[0]]) if above.size else float("nan")


def main(argv) -> int:
    from simws import engines
    from simws.chain import see as C
    from simws.hub.jobs import Hub
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--inprocess", action="store_true")
    a = ap.parse_args(argv[1:])
    if not engines.module("ngspice").available():
        print("ngspice is not installed: bash tools/install_spice.sh")
        return 2
    hub = Hub(a.store, mode="inprocess" if a.inprocess else "subprocess")
    print("store:", hub.store.root, "| mode:", hub.mode, flush=True)
    dev = C.reference_strike_diode()
    print(f"\nreference diode (A17.6), {AREA_M2 * 1e12:g} um^2, top at +1 V, pairs on 0.25-0.60 um, "
          f"pulse 20 ps +- 5 ps")
    print(f"{'charge fC':>10s} {'collected':>10s} {'peak uA':>9s} {'FWHM ps':>8s} {'kirchhoff':>10s}")
    for q in (0.1e-15, 1e-15, 10e-15, 100e-15):
        r = hub.submit("device.devsim", params={"problem": C.strike_problem(dev, q, AREA_M2)})
        if r.status != "ok":
            print(f"{q * 1e15:10.1f}: {r.status}: {r.message[-200:]}")
            return 1
        res = hub.result(r.id)
        p = np.asarray(C.tcad_pulse(res, "top", AREA_M2).points, float)
        frac = res.q("tr.Q.top").value / res.q("tr.Q_generated").value
        print(f"{q * 1e15:10.1f} {100 * frac:9.2f}% {p[:, 1].max() * 1e6:9.2f} {fwhm(p[:, 0], p[:, 1]) * 1e12:8.2f} "
              f"{res.q('dc.kirchhoff_residual').value:10.1e}")
    Q_LO, Q_HI = 0.05e-15, 100e-15
    for R in (R_OHM, 100e3):                 # 10 kOhm is R106; 100 kOhm (the A10 demo's R) is extra context
        node = rc_node(R, C_F)
        print(f"\nnode at 1 V, {R / 1e3:g} kOhm, {C_F * 1e15:g} fF, threshold 0.5 V"
              + ("  (R106)" if R == R_OHM else "  (extra: a weaker restoring current)"))
        print(f"{'pulse':>20s} {'q_crit fC':>10s} {'collected fC':>13s} {'runs':>5s}")
        for label, fn in (
                ("TCAD pulse, rerun", lambda: C.critical_charge_tcad(hub, node, "n", 0.5, dev, "top", AREA_M2,
                                                                     q_low_C=Q_LO, q_high_C=Q_HI, rerun=True)),
                ("TCAD pulse, scaled", lambda: C.critical_charge_tcad(hub, node, "n", 0.5, dev, "top", AREA_M2,
                                                                      q_low_C=Q_LO, q_high_C=Q_HI, rerun=False)),
                ("Messenger 1/10 ps", lambda: C.critical_charge(hub, node, "n", 0.5, 1e-12, 10e-12,
                                                                q_low_C=Q_LO, q_high_C=Q_HI))):
            try:
                out = fn()
            except ValueError as e:
                if "does not upset" not in str(e):
                    raise
                print(f"{label:>20s} {'none':>10s}   no upset up to {Q_HI * 1e15:g} fC deposited")
                continue
            col = out.get("collected_at_qcrit_C")
            print(f"{label:>20s} {out['qcrit_C'] * 1e15:10.3f} {'-' if col is None else f'{col * 1e15:.3f}':>13s} "
                  f"{out['runs']:5d}")
    print("\nThe device is held at its bias (strike:fixed_bias): the node's fall does not slow the collection, as "
          "it would in mixed mode. One engine: unverified.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
