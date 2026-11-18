"""Drift-diffusion against the electron energy balance (contract A11): how much do hot electrons matter?

1. A uniform n-type silicon bar (2 um, 1e16 cm^-3): current and electron temperature against bias, both models.
2. The pn diode of the engine pairs: the forward current with both models (a model-form check, not verification:
   both run on DEVSIM).

    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/v1_energy_balance_demo.py [--store DIR] [--inprocess]
"""
from __future__ import annotations

import argparse
import sys


def main(argv) -> int:
    from simws.compare import compare_results
    from simws.hub import trust
    from simws.hub.jobs import Hub
    from simws.spec.library import n_bar_1d, pn_diode_1d

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--inprocess", action="store_true")
    args = ap.parse_args(argv[1:])
    hub = Hub(args.store, mode="inprocess" if args.inprocess else "subprocess")

    def sub(adapter, p):
        r = hub.submit(adapter, params={"problem": p})
        if r.status != "ok":
            raise SystemExit(f"{adapter}: {r.status}: {r.message}")
        return r

    print("1. n-type bar, 2 um, 1e16 cm^-3, mu_0 = 1000 cm^2/Vs")
    print(f"   {'V':>5s}  {'field kV/cm':>11s}  {'J drift-diff A/m^2':>19s}  {'J energy-bal A/m^2':>19s}  {'ratio':>6s}  {'Tn max K':>8s}")
    for v in (0.2, 2.0, 6.0, 20.0):
        p = n_bar_1d(V=[v])
        a, b = sub("device.devsim", p), sub("device.devsim_eb", p)
        ja, jb = abs(a.q("dc.I.right").array()[0]), abs(b.q("dc.I.right").array()[0])
        print(f"   {v:5.1f}  {v / 2e-4 / 1e3:11.1f}  {ja:19.5e}  {jb:19.5e}  {jb / ja:6.3f}  "
              f"{b.diagnostics['eb_Tn_max_K']:8.1f}")
    print("   drift-diffusion here has no velocity saturation (constant mobility); the energy balance saturates"
          " the electron velocity by itself.\n")

    print("2. pn diode (engine-pair problem), forward current")
    V = [round(0.30 + 0.05 * k, 2) for k in range(7)]
    p = pn_diode_1d(V=V)
    a, b = sub("device.devsim", p), sub("device.devsim_eb", p)
    before = trust.assess_stored(hub, a.id).level
    c = compare_results(hub, a.id, "dc.I.anode", b.id, tolerance=0.01)
    for v, ia, ib in zip(V, a.q("dc.I.anode").array(), b.q("dc.I.anode").array()):
        print(f"   {v:4.2f} V  drift-diffusion {ia:.5e}  energy balance {ib:.5e}  A/m^2  ({100 * (ib / ia - 1):+.3f} %)")
    print(f"   worst difference {100 * c.worst:.3f} %; independent: {c.independent} (same code, different physics);"
          f" trust of the drift-diffusion result: {before}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
