"""Channeling in crystalline silicon from the published fits (contract A15), next to the amorphous BCA result.

    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/v1_channeling_demo.py [--store DIR] [--inprocess]

Boron 35 keV (Tasch et al. 1989, tilt 7, rotation 30) and arsenic 120 keV (Suzuki et al. 2000, tilt 7, rotation 22)
at several doses: mean depth, straggle, channeled share and the depth where the concentration falls to 1e17 and
1e16 cm^-3; then the same beam into amorphous silicon with RustBCA (boron in "empirical" stopping).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def depth_at(r, level_cm3):
    """Deepest depth (nm) where the concentration is still above level_cm3."""
    c = np.asarray(r.q("implant.concentration").array(), float) * 1e-6           # 1/m^3 -> 1/cm^3
    x = np.asarray(r.q("implant.concentration").axes[0].values, float) * 1e9
    above = np.nonzero(c >= level_cm3)[0]
    return float(x[above[-1]]) if above.size else float("nan")


def main(argv) -> int:
    from simws.hub.jobs import Hub
    from simws.spec.library import implant_si
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--inprocess", action="store_true")
    a = ap.parse_args(argv[1:])
    hub = Hub(a.store, mode="inprocess" if a.inprocess else "subprocess")
    print("store:", hub.store.root, "| mode:", hub.mode, flush=True)
    bad = 0
    for ion, E, rot, doses in (("B", 35, 30.0, (2e13, 1e14, 5e14, 2e15)), ("As", 120, 22.0, (1e12, 1e13, 1e14, 1e15))):
        print(f"\n{ion} {E} keV into crystalline (100) Si, tilt 7, rotation {rot:g}"
              f"  ({'Tasch 1989' if ion == 'B' else 'Suzuki 2000'})")
        print(f"{'dose cm^-2':>11s} {'Rp nm':>8s} {'dRp nm':>8s} {'channeled':>10s} {'x(1e17) nm':>11s} {'x(1e16) nm':>11s}")
        for d in doses:
            p = implant_si(ion, E, tilt_deg=7.0, dose_cm2=d, crystal=True, rotation_deg=rot,
                           thickness_um=1.5 if ion == "B" else 1.0)
            r = hub.submit("implant.channeling", params={"problem": p})
            if r.status != "ok":
                bad += 1
                print(f"  {d:.0e}: {r.status}: {r.message[-200:]}")
                continue
            print(f"{d:11.0e} {r.q('implant.Rp').value * 1e9:8.1f} {r.q('implant.dRp').value * 1e9:8.1f} "
                  f"{100 * r.q('implant.channeled_fraction').value:9.1f}% {depth_at(r, 1e17):11.0f} "
                  f"{depth_at(r, 1e16):11.0f}")
        amorph = implant_si(ion, E, tilt_deg=7.0, dose_cm2=doses[1], thickness_um=1.5 if ion == "B" else 1.0)
        if ion == "B":
            amorph = amorph.with_changes({"models.electronic_stopping": "empirical"})
        r = hub.submit("implant.rustbca", params={"problem": amorph})
        if r.status == "ok":
            print(f"{'amorphous':>11s} {r.q('implant.Rp').value * 1e9:8.1f} {r.q('implant.dRp').value * 1e9:8.1f} "
                  f"{'-':>10s} {depth_at(r, 1e17):11.0f} {depth_at(r, 1e16):11.0f}   RustBCA, dose {doses[1]:.0e}")
        else:
            print(f"  amorphous RustBCA: {r.status}: {r.message[-160:]}")
    print("\nChanneling makes the deep tail: compare the depths at 1e16 cm^-3. The channeled share falls with dose as "
          "the crystal is damaged. Unverified by rule (one source per ion).")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
