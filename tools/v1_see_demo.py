"""Single-event energy deposition through the hub (contract amendment A8; needs G4SEE: tools/install_g4see.sh).

Alphas and protons into a silicon block with a sensitive volume (SV) at the surface. Each run is a hub job like
any other: stored, error bars attached, trust label assigned. The reference column is the mean energy loss from
the NIST ASTAR / PSTAR stopping powers (simws.radiation.nist_star): a continuous-slowing-down estimate built on
ICRU 49, which Geant4 is expected to share for these particles, so agreement checks the set-up, not the physics.

    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/v1_see_demo.py [--store DIR] [--inprocess]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

# (particle, energy MeV, SV thickness um): an alpha stopping in a thick SV, partial deposits in thin ones
CASES = [("alpha", 5.0, 10.0), ("alpha", 5.0, 50.0), ("proton", 10.0, 100.0), ("proton", 5.0, 20.0)]
THRESHOLDS_MEV = (0.1, 0.5, 1.0)
PAIR_ENERGY_NOTE = "divide by the electron-hole pair energy of silicon to get collected charge (not done here)"


def main(argv) -> int:
    from simws import catalogue, engines
    from simws.hub import trust
    from simws.hub.jobs import Hub
    from simws.radiation import nist_star
    from simws.spec.library import see_si_slab

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None, help="hub store folder (default: the store of this checkout)")
    ap.add_argument("--inprocess", action="store_true", help="run the adapter in this Python instead of the env runner")
    args = ap.parse_args(argv[1:])
    if not engines.module("g4see").available():
        print("G4SEE is not installed: bash tools/install_g4see.sh")
        return 2
    hub = Hub(args.store, mode="inprocess" if args.inprocess else "subprocess")
    print("store:", hub.store.root, "| mode:", hub.mode, flush=True)
    head = "".join(f"  P(>={t:g} MeV)" for t in THRESHOLDS_MEV)
    print(f"\n{'particle':8s} {'MeV':>5s} {'SV um':>6s}  {'mean deposit MeV':>18s}  {'NIST MeV':>8s} {'diff':>7s}{head}  trust")
    bad = 0
    for part, E, t in CASES:
        bulk = max(100.0, 3 * t)
        r = hub.submit("see.g4see", params={"problem": see_si_slab(part, E, sv_thickness_um=t, bulk_thickness_um=bulk)})
        if r.status != "ok":
            bad += 1
            print(f"{part:8s} {E:5g} {t:6g}  status {r.status}: {(r.message or '')[-300:]}")
            continue
        m = r.q("see.edep_mean")
        mean, sig = m.value / 1e6, m.uncertainty.sigma / 1e6
        try:
            ref = nist_star.deposit(part, E, 0.0, t)
            refs = f"{ref:8.4f} {100 * (mean / ref - 1):+6.2f}%"
        except ValueError:                       # the particle stops inside the SV: everything is deposited
            ref = E
            refs = f"{ref:8.4f} {100 * (mean / ref - 1):+6.2f}%"
        pa = r.q("see.p_above")
        thr = np.asarray(pa.axes[0].values) / 1e6
        cols = "".join(f"  {pa.array()[int(np.argmin(abs(thr - x)))]:11.4f}" for x in THRESHOLDS_MEV)
        tr = trust.assess_stored(hub, r.id)
        print(f"{part:8s} {E:5g} {t:6g}  {mean:9.4f} +- {sig:6.4f}  {refs}{cols}  {tr.level}"
              + ("" if tr.in_range else " (outside tested range)"), flush=True)
    print("\nerror bars: counting statistics and histogram binning only. Electromagnetic physics only (no nuclear")
    print(f"reactions). Energy deposited, not charge: {PAIR_ENERGY_NOTE}.")
    print("trust is 'unverified' until a second, independent single-event code is connected (contract A8.6).\n")
    print(catalogue.table(catalogue.rows(hub, adapter="see.*")))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
