"""Two implant engines on the same problems, and damage profiles (contract amendments A12, A13, A14).

Needs RustBCA (tools/install_rustbca.sh, which also builds its command-line program) and the Geant4 implant program
(tools/install_g4implant.sh). Each run is a hub job; each RustBCA-against-Geant4 comparison is recorded, so a
RustBCA result whose comparison passes becomes "verified".

    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/v1_implant_verify_demo.py [--store DIR] [--inprocess]
        [--ed 36]

Part 1, ranges (20 000 ions): boron and phosphorus at 50 keV and boron at 100 keV, in Lindhard-Scharff mode (both
engines use the same electronic stopping formula, so the comparison tests the collision codes: A13.5 R65,
Rp within 10 %, dRp within 15 %) and in "empirical" mode (boron only on RustBCA: A12; Geant4 uses its own data).
Part 2, damage (2000 ions): vacancies per ion and their mean depth from both engines (A14.5 R73: within 20 % and
15 %) for the displacement energy --ed (default 36 eV, silicon's direction average from Holmstrom, Kuronen and
Nordlund, Phys. Rev. B 78, 045202, 2008; the acceptance checks used 15 eV, a common BCA value).
"""
from __future__ import annotations

import argparse
import sys

TABLE_UM = {("B", 50): (0.1608, 0.0504), ("B", 100): (0.2994, 0.0710), ("P", 50): (0.0607, 0.0256)}
TOL = {"implant.Rp": 0.10, "implant.dRp": 0.15, "implant.vacancies_per_ion": 0.20, "implant.vacancy_depth_mean": 0.15}


def main(argv) -> int:
    from simws import compare
    from simws.hub import trust
    from simws.hub.jobs import Hub
    from simws.spec.library import implant_si

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None, help="hub store folder (default: the store of this checkout)")
    ap.add_argument("--inprocess", action="store_true", help="run the adapters in this Python")
    ap.add_argument("--ed", type=float, default=36.0, help="silicon displacement energy for part 2 (eV)")
    args = ap.parse_args(argv[1:])
    hub = Hub(args.store, mode="inprocess" if args.inprocess else "subprocess")
    print("store:", hub.store.root, "| mode:", hub.mode, flush=True)
    bad = 0

    def run(engine, problem):
        nonlocal bad
        r = hub.submit(f"implant.{engine}", params={"problem": problem})
        if r.status != "ok":
            bad += 1
            print(f"  {engine}: status {r.status}: {r.message[-300:]}")
            return None
        return r

    def nm(r, name):
        return r.q(name).value * 1e9

    print("\nPart 1: ranges (nm). 'table' = Gibbons, Johnson and Mylroie 1975 (a theory, not a measurement).")
    print(f"{'ion':3s} {'keV':>4s} {'stopping':16s} {'RustBCA Rp':>10s} {'Geant4 Rp':>9s} {'diff':>7s}  "
          f"{'RustBCA dRp':>11s} {'Geant4 dRp':>10s} {'diff':>7s}  {'table Rp/dRp':>13s}  RustBCA trust")
    for stopping in ("lindhard_scharff", "empirical"):
        for ion, e in (("B", 50), ("B", 100), ("P", 50)):
            p = implant_si(ion, e).with_changes({"models.electronic_stopping": stopping})
            g = run("g4implant", p)
            rb = run("rustbca", p) if (stopping == "lindhard_scharff" or ion == "B") else None
            if g is None:
                continue
            tab = TABLE_UM[(ion, e)]
            line = f"{ion:3s} {e:4d} {stopping:16s} "
            if rb is not None:
                cells = []
                for name in ("implant.Rp", "implant.dRp"):
                    c = compare.compare_results(hub, g.id, name, rb.id, tolerance=TOL[name])   # as R65
                    if stopping == "lindhard_scharff":
                        compare.record(hub, c)           # R65: the verification (same stopping formula)
                    cells.append((nm(rb, name), nm(g, name), 100 * (nm(g, name) / nm(rb, name) - 1),
                                  "" if c.passed else "*"))
                (a1, b1, d1, f1), (a2, b2, d2, f2) = cells
                t = trust.assess_stored(hub, rb.id)
                line += (f"{a1:10.1f} {b1:9.1f} {d1:+6.1f}%{f1:1s} {a2:11.1f} {b2:10.1f} {d2:+6.1f}%{f2:1s} "
                         f"{tab[0] * 1e3:6.1f}/{tab[1] * 1e3:5.1f}  {t.level}")
            else:
                line += (f"{'-':>10s} {nm(g, 'implant.Rp'):9.1f} {'':7s}  {'-':>11s} {nm(g, 'implant.dRp'):10.1f} "
                         f"{'':7s}  {tab[0] * 1e3:6.1f}/{tab[1] * 1e3:5.1f}  (RustBCA has no empirical factor)")
            print(line, flush=True)
    print("diff = Geant4 / RustBCA - 1; * = outside the A13 tolerance (Rp 10 %, dRp 15 %). Lindhard-Scharff rows are "
          "recorded as comparisons (trust); empirical rows use different stopping data and are not.")

    print(f"\nPart 2: damage, Ed = {args.ed:g} eV, 2000 ions, Lindhard-Scharff, full cascade (no recombination).")
    print(f"{'ion':3s} {'keV':>4s} {'RustBCA vac/ion':>15s} {'Geant4 vac/ion':>14s} {'diff':>7s}  "
          f"{'RustBCA depth nm':>16s} {'Geant4 depth nm':>15s} {'diff':>7s}")
    for ion, e in (("B", 50), ("P", 50)):
        p = implant_si(ion, e, n_ions=2000, displacement_eV=args.ed)
        rb, g = run("rustbca", p), run("g4implant", p)
        if rb is None or g is None:
            continue
        v = [(rb.q(n).value, g.q(n).value) for n in ("implant.vacancies_per_ion", "implant.vacancy_depth_mean")]
        for n in ("implant.vacancies_per_ion", "implant.vacancy_depth_mean"):
            compare.record(hub, compare.compare_results(hub, g.id, n, rb.id, tolerance=TOL[n]))   # as R73
        print(f"{ion:3s} {e:4d} {v[0][0]:15.1f} {v[0][1]:14.1f} {100 * (v[0][1] / v[0][0] - 1):+6.1f}%  "
              f"{v[1][0] * 1e9:16.1f} {v[1][1] * 1e9:15.1f} {100 * (v[1][1] / v[1][0] - 1):+6.1f}%", flush=True)
    print("diff = Geant4 / RustBCA - 1. At 15 eV Geant4 counts 14 % fewer, at 36 eV about 33 % more (A14.7): the "
          "two codes do not agree on damage counts to better than about 30 %.")
    print("vacancies scale roughly as 1/Ed; the full-cascade count is an upper bound on what survives at room "
          "temperature.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
