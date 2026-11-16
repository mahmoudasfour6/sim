"""First paper-fidelity result on the new path (laptop or build machine: needs DEVSIM).

Runs the library SiGe HBT at two biases through the hub at paper fidelity: three meshes, an error bar on every
terminal current (contract amendment A4), accepted by the gatekeeper.

    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/v1_paper_hbt.py [--store DIR] [--inprocess] [--default-mesh]

With the refined emitter-base mesh (the default here, contract A5) the three meshes have about 25, 49 and 98
thousand nodes: expect tens of minutes to a few hours depending on the machine. Use the overnight queue.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

VBE = [0.6, 0.7]


def main(argv) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from v1_charon_check import _quiet

    from simws.hub import trust
    from simws.hub.jobs import Hub
    from simws.spec.library import npn_hbt_2d

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--inprocess", action="store_true")
    ap.add_argument("--default-mesh", action="store_true",
                    help="use the default mesh (2 nm at the emitter-base junction) instead of the refined one")
    args = ap.parse_args(argv[1:])

    p = npn_hbt_2d(VBE=VBE, eb_refine=not args.default_mesh)        # contract A5: refined junction mesh by default
    print("emitter-base mesh:", "default (2 nm)" if args.default_mesh else "refined (0.05 nm at the junction)")
    hub = Hub(args.store, mode="inprocess" if args.inprocess else "subprocess")
    print("problem:", p.device.name, "| id", p.problem_id()[:12], "| store:", hub.store.root, flush=True)
    print("running three meshes at paper fidelity (refined mesh: long) ...", flush=True)
    with _quiet():
        res = hub.submit("device.devsim", params={"problem": p.model_dump(mode="json")}, fidelity="paper")
    print("status:", res.status, "| result", res.id[:12], "|", (res.message or "")[-600:])
    if res.status != "ok":
        return 2
    rec = res.verification["mesh"]
    print("mesh scales:", [round(s, 4) for s in rec["mesh_scales"]], "| all points in the asymptotic range:", rec["passed"])
    print(f"{'contact':10s} {'VBE (V)':>8s} {'I (A/m)':>14s} {'sigma (A/m)':>12s} {'rel':>9s} {'order':>7s}  bound")
    for name, per in rec["quantities"].items():
        q = res.q(name)
        for j, pt in enumerate(per):
            order = "-" if pt["order"] is None else f"{pt['order']:.2f}"
            kind = "noise floor" if pt["below_floor"] else ("conservative" if pt["conservative_bound"] else "GCI")
            rel = "-" if pt["rel_sigma"] is None else f"{pt['rel_sigma']:.2e}"
            print(f"{name[5:]:10s} {pt['axis_value']:8.2f} {q.value[j]:14.6e} {q.uncertainty.sigma[j]:12.3e} "
                  f"{rel:>9s} {order:>7s}  {kind}")
    t = trust.assess_stored(hub, res.id)
    print("flags:", "; ".join(res.flags))
    print(f"trust: {t.level} | in_range {t.in_range} | {'; '.join(t.reasons)}")
    print("PAPER RESULT STORED: the hub accepted it at paper fidelity with mesh error bars.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
