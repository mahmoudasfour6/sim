"""Single event to circuit (contract A10; needs G4SEE and the circuit engine).

Alphas of 5 MeV through a 10 um silicon sensitive volume (G4SEE), then three circuit nodes held at 1 V through
100 kOhm with different capacitances (ngspice): the critical charge of each node by bisection, and the probability
that one alpha upsets it.

    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/v1_upset_demo.py [--store DIR] [--inprocess]
"""
from __future__ import annotations

import argparse
import sys

TAU_R, TAU_F = 1e-12, 10e-12        # pulse time constants: illustrative inputs, not data for any real technology


def rc_node(C_F: float, R_ohm: float = 100e3):
    from simws.spec import CircuitProblem
    return CircuitProblem.model_validate({
        "title": f"node {C_F * 1e15:g} fF", "probes": ["v(n)"],
        "elements": [{"name": "vdd", "kind": "vsource", "nodes": ["vdd", "0"], "value": 1.0},
                     {"name": "r1", "kind": "resistor", "nodes": ["vdd", "n"], "value": R_ohm},
                     {"name": "c1", "kind": "capacitor", "nodes": ["n", "0"], "value": C_F}],
        "analysis": {"kind": "tran", "step_s": 1e-13, "stop_s": 3e-10, "max_step_s": 1e-13}})


def main(argv) -> int:
    from simws import engines
    from simws.chain import see as C
    from simws.hub.jobs import Hub
    from simws.spec.library import see_si_slab

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--inprocess", action="store_true")
    args = ap.parse_args(argv[1:])
    for e, how in (("g4see", "tools/install_g4see.sh"), ("ngspice", "tools/install_spice.sh")):
        if not engines.module(e).available():
            print(f"{e} is not installed: bash {how}")
            return 2
    hub = Hub(args.store, mode="inprocess" if args.inprocess else "subprocess")
    r = hub.submit("see.g4see", params={"problem": see_si_slab("alpha", 5.0, sv_thickness_um=10.0)})
    if r.status != "ok":
        print(f"G4SEE: {r.status}: {r.message}")
        return 1
    e = r.q("see.edep_mean").value
    print(f"5 MeV alphas, 10 um SV: mean deposit {e / 1e6:.4f} MeV = {C.charge_C(e) * 1e15:.2f} fC "
          f"(pair energy {C.PAIR_ENERGY_SI_EV} eV)\n")
    print(f"{'node C':>7s}  {'critical charge':>16s}  {'E_crit MeV':>10s}  {'P(upset) per alpha':>20s}  runs")
    for c_f in (60e-15, 150e-15, 200e-15):
        qc = C.critical_charge(hub, rc_node(c_f), "n", 0.5, TAU_R, TAU_F, q_low_C=5e-15, q_high_C=3e-13)
        up = C.upset_probability(hub.result(r.id), qc)
        print(f"{c_f * 1e15:5.0f} fF  {qc['qcrit_C'] * 1e15:12.2f} fC  {up['E_crit_eV'] / 1e6:10.4f}  "
              f"{up['P']:8.4f} [{up['P_low']:.4f}, {up['P_high']:.4f}]  {qc['runs']}")
    print("\nP is per alpha that crosses the SV as the G4SEE source does (through its centre). The pulse time "
          "constants are illustrative inputs. Trust: unverified (contract A10.2).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
