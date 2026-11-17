"""Ion implantation through the hub (contract amendment A7; needs RustBCA: tools/install_rustbca.sh).

Boron, phosphorus and arsenic into amorphous silicon at 50 and 100 keV, dose 1e15 ions/cm^2. Each run is a hub
job like any other: stored, error bars attached, trust label assigned.

    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/v1_implant_demo.py [--store DIR] [--inprocess]

The reference columns are the published range tables of contract A7.7 R16 (Gibbons, Johnson and Mylroie, Projected
Range Statistics, 2nd ed. 1975, as reproduced in G. Chapman, SFU ENSC 495 lecture 7). They are a 1975 transport
theory, not a measurement: the comparison shows the translation is sound, it does not validate the physics.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

REFERENCE_UM = {("B", 50): (0.1608, 0.0504), ("B", 100): (0.2994, 0.0710),          # (Rp, dRp) in micrometres
                ("P", 50): (0.0607, 0.0256), ("P", 100): (0.1238, 0.0456)}
CASES = [(ion, e) for ion in ("B", "P", "As") for e in (50, 100)]
DOSE_CM2 = 1e15


def main(argv) -> int:
    from simws import catalogue, engines
    from simws.engines.planner import plan
    from simws.hub import trust
    from simws.hub.jobs import Hub
    from simws.spec.library import implant_si

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None, help="hub store folder (default: the store of this checkout)")
    ap.add_argument("--inprocess", action="store_true", help="run the adapter in this Python instead of the env runner")
    args = ap.parse_args(argv[1:])

    if not args.inprocess:
        pass                                    # the env runner reports a missing engine itself
    elif not engines.module("rustbca").available():
        print("RustBCA is not installed for this Python: bash tools/install_rustbca.sh")
        return 2
    print(plan(implant_si("P", 100)).summary())
    hub = Hub(args.store, mode="inprocess" if args.inprocess else "subprocess")
    print("store:", hub.store.root, "| mode:", hub.mode, flush=True)
    print(f"\n{'ion':3s} {'keV':>4s}  {'Rp nm':>16s}  {'dRp nm':>15s}  {'lateral nm':>14s}  {'table Rp':>8s} {'diff':>7s}  "
          f"{'table dRp':>9s} {'diff':>7s}  {'peak cm^-3':>10s}  trust")
    bad = 0
    for ion, e in CASES:
        r = hub.submit("implant.rustbca", params={"problem": implant_si(ion, e, dose_cm2=DOSE_CM2)})
        if r.status != "ok":
            bad += 1
            print(f"{ion:3s} {e:4d}  status {r.status}: {r.message[-300:]}")
            continue
        nm = lambda name: (r.q(name).value * 1e9, r.q(name).uncertainty.sigma * 1e9)      # noqa: E731
        (rp, rps), (dr, drs), (la, las) = nm("implant.Rp"), nm("implant.dRp"), nm("implant.lateral_straggle")
        ref = REFERENCE_UM.get((ion, e))
        cols = (f"{ref[0] * 1e3:8.1f} {100 * (rp / (ref[0] * 1e3) - 1):+6.1f}%  {ref[1] * 1e3:9.1f} "
                f"{100 * (dr / (ref[1] * 1e3) - 1):+6.1f}%") if ref else f"{'-':>8s} {'-':>7s}  {'-':>9s} {'-':>7s}"
        peak = float(np.max(r.q("implant.concentration").array())) * 1e-6
        t = trust.assess_stored(hub, r.id)
        print(f"{ion:3s} {e:4d}  {rp:8.2f} +- {rps:4.2f}  {dr:7.2f} +- {drs:4.2f}  {la:6.2f} +- {las:4.2f}  {cols}  "
              f"{peak:10.3e}  {t.level}" + ("" if t.in_range else " (outside tested range)"), flush=True)
    print(f"\npeak concentration for a dose of {DOSE_CM2:g} ions/cm^2 (as implanted, before any anneal).")
    print("error bars: counting statistics only. Amorphous target: no channeling tail.")
    print("trust is 'unverified' until a second, different implant engine is connected and compared (contract A7.6).\n")
    print(catalogue.table(catalogue.rows(hub, adapter="implant.*")))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
