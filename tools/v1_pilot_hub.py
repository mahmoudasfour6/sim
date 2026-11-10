"""First end-to-end use of architecture v1 through the hub (laptop: needs DEVSIM and Charon).

One description of a pn diode is run on two engines by the hub, the two stored results are compared, the
comparison is recorded, and each result gets its trust label.

    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/v1_pilot_hub.py [--store DIR] [--inprocess]

Gate (contract amendment A3.2, fixed before the run): dc.I.anode, DEVSIM against Charon as reference, relative
difference <= 0.02 where the reference is above 1e-2 A/m^2 (= 1e-6 A/cm^2): the 0.9a cross-check gate, unchanged.
Results go to the hub store of this checkout (printed below), like any other hub job.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOLERANCE, FLOOR_A_M2 = 0.02, 1e-2          # contract A3.2
QUANTITY = "dc.I.anode"
ENGINES = ("devsim", "charon")


def main(argv) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from v1_charon_check import _quiet, problem

    from simws import catalogue
    from simws.engines.planner import plan
    from simws.hub import trust
    from simws.hub.jobs import Hub
    from simws.study import Study

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None, help="hub store folder (default: the store of this checkout)")
    ap.add_argument("--inprocess", action="store_true", help="run the adapters in this Python instead of the env runner")
    args = ap.parse_args(argv[1:])

    p = problem()
    print("problem:", p.device.name, "| id", p.problem_id()[:12], "| T =", p.models.temperature_K, "K")
    print(plan(p, engines=list(ENGINES)).summary())
    hub = Hub(args.store, mode="inprocess" if args.inprocess else "subprocess")
    print("store:", hub.store.root, "| mode:", hub.mode, flush=True)

    print("running both engines through the hub (a few minutes) ...", flush=True)
    with _quiet():
        sr = Study(hub, p, engines=list(ENGINES)).sweep({})
    bad = 0
    for pt in sr.points:
        rid = pt.get("result_id")
        print(f"  {pt['engine']:7s} status {pt['status']:14s} result {(rid or '-')[:12]}")
        if pt["status"] != "ok":
            bad += 1
            for r in pt.get("reasons") or []:
                print("     ", str(r)[-1500:])
    if bad:
        print("PILOT STOPPED: an engine did not return an ok result (see above).")
        return 2

    cs = sr.compare_engines(QUANTITY, ENGINES[0], ENGINES[1], tolerance=TOLERANCE, floor=FLOOR_A_M2)
    if not cs:
        print("PILOT STOPPED: no comparison could be made.")
        return 2
    c = cs[0]
    print(f"compare {QUANTITY}: {c.engine_a} against {c.engine_b} (reference)")
    print(f"  worst relative difference {c.worst} at {c.worst_at} V | points compared {c.n_points}, "
          f"below the floor {c.n_below_floor} | interpolation {c.interpolation}")
    print(f"  tolerance {c.tolerance} | passed {c.passed} | independent engines {c.independent}")

    levels = []
    for pt in sr.points:
        t = trust.assess_stored(hub, pt["result_id"])
        levels.append(t.level)
        print(f"  trust {pt['engine']:7s} {t.level:10s} in_range {t.in_range} | {'; '.join(t.reasons)}")
    print(catalogue.table(catalogue.rows(hub, adapter="device.*")))
    ok = bool(c.passed) and all(lv == "verified" for lv in levels)
    print("PILOT", "PASS: both results are stored and labelled verified by a different engine." if ok
          else "FAIL: see the comparison and the trust lines above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
