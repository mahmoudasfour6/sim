"""Crystal (channeling) implant -> anneal -> device, next to the amorphous implant (A20 R130-R133, A21 R145).

    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/v1_crystal_chain_demo.py [--store DIR] [--inprocess]

Boron 35 keV 1e14 (tilt 7, rotation 30; Tasch 1989) and arsenic 120 keV 1e13 (tilt 7, rotation 22; Suzuki 2000)
into crystalline (100) Si, and the same beams into amorphous Si (RustBCA); each profile annealed with the Fermi
diffusivity on both anneal engines; the junction depths, the digitisation envelope of the crystal profile, and
the DEVSIM diode of the boron junctions.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

CASES = (("B", 35.0, 1e14, 7.0, 30.0, 1.5, "donor", 1800.0),
         ("As", 120.0, 1e13, 7.0, 22.0, 1.0, "acceptor", 600.0),
         ("P", 50.0, 1e14, 10.0, 0.0, 2.0, "acceptor", 600.0))          # A21 R145: condition B


def main(argv) -> int:
    from simws.chain import diode as D
    from simws.chain import process as P
    from simws.compare import compare_results, record
    from simws.hub.jobs import Hub
    from simws.hub.trust import assess_stored
    from simws.spec.library import implant_si
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--inprocess", action="store_true")
    a = ap.parse_args(argv[1:])
    hub = Hub(a.store, mode="inprocess" if a.inprocess else "subprocess")
    print("store:", hub.store.root, "| mode:", hub.mode, flush=True)
    bad = 0
    for ion, E, dose, tilt, rot, T_um, bg, t_s in CASES:
        print(f"\n== {ion} {E:g} keV {dose:.0e} cm^-2, tilt {tilt:g}, rotation {rot:g}, into Si, background {bg} 1e16, "
              f"1000 C {t_s:g} s, Fermi")
        crys = implant_si(ion, E, tilt_deg=tilt, dose_cm2=dose, crystal=True, rotation_deg=rot,
                          thickness_um=T_um).with_changes({"depth_bin_um": 0.002})
        amor = implant_si(ion, E, tilt_deg=tilt, dose_cm2=dose, thickness_um=T_um).with_changes({"depth_bin_um": 0.002})
        if ion == "B":
            amor = amor.with_changes({"models.electronic_stopping": "empirical"})
        rows = {}
        for label, prob, eng in (("crystal", crys, "implant.channeling"), ("amorphous", amor, "implant.rustbca")):
            r = hub.submit(eng, params={"problem": prob})
            if r.status != "ok":
                bad += 1
                print(f"  {label}: {r.status}: {r.message[-200:]}")
                continue
            res = hub.result(r.id)
            ini = P.initial_from_implant(res)
            print(f"  {label:9s} ({eng}): dose in the profile {ini.dose_cm2():.4e} cm^-2 "
                  f"({100 * (ini.dose_cm2() / dose - 1):+.2f} % of the beam), trust {assess_stored(hub, r.id).level}")
            out = {}
            for an in ("anneal.devsim_anneal", "anneal.fipy"):
                p = P.anneal_problem(ion, ini, background_cm3=1e16, background_type=bg, steps=[(1000.0, t_s)],
                                     model="fermi", thickness_um=max(2.0, T_um))
                ra = hub.submit(an, params={"problem": p})
                if ra.status != "ok":
                    bad += 1
                    print(f"    {an}: {ra.status}: {ra.message[-200:]}")
                    continue
                out[an] = (ra.id, hub.result(ra.id), p)
            if len(out) == 2:
                (ia, ra_, pa), (ib, rb_, _) = out["anneal.devsim_anneal"], out["anneal.fipy"]
                c = compare_results(hub, ia, "anneal.junction_depth", ib, tolerance=0.01)
                record(hub, c)
                xa, xb = ra_.q("anneal.junction_depth").value * 1e9, rb_.q("anneal.junction_depth").value * 1e9
                print(f"    junction: devsim {xa:.1f} nm, fipy {xb:.1f} nm ({100 * (xb / xa - 1):+.4f} %), "
                      f"anneal trust {assess_stored(hub, ia).level}")
                rows[label] = (xa, ra_, pa)
            if label == "crystal" and "anneal.devsim_anneal" in out:
                band = []
                for b in (-1, 1):
                    p = P.anneal_problem(ion, P.initial_from_implant(res, band=b), background_cm3=1e16,
                                         background_type=bg, steps=[(1000.0, t_s)], model="fermi",
                                         thickness_um=max(2.0, T_um))
                    rb = hub.submit("anneal.devsim_anneal", params={"problem": p})
                    band.append(hub.result(rb.id).q("anneal.junction_depth").value * 1e9 if rb.status == "ok"
                                else float("nan"))
                print(f"    envelope of the reported sigma (band -1 / +1): junction {band[0]:.1f} / {band[1]:.1f} nm")
        if len(rows) == 2:
            deeper = rows["crystal"][0] - rows["amorphous"][0]
            print(f"  crystal junction {'deeper' if deeper > 0 else 'NOT deeper'} by {deeper:.1f} nm "
                  f"({100 * deeper / rows['amorphous'][0]:+.1f} %)")
            if ion == "B":
                V = [round(v, 3) for v in np.arange(0.30, 0.6501, 0.05)]
                for label in ("crystal", "amorphous"):
                    _, res_a, p_a = rows[label]
                    dp = P.diode_from_anneal(res_a, p_a, V=V)
                    rd = hub.submit("device.devsim", params={"problem": dp})
                    if rd.status != "ok":
                        print(f"  diode ({label}): {rd.status}: {rd.message[-200:]}")
                        continue
                    J = P.forward_J(hub.result(rd.id), dp)
                    fit = D.fit_shockley(P.forward_V(dp), J, dp.models.temperature_K)
                    print(f"  diode ({label}): J(0.6 V) {J[V.index(0.6)]:.4e} A/m^2, Js {fit['Js_A_m2']:.4e} A/m^2, "
                          f"N {fit['N']:.4f}")
    print("\nNot modelled: implant damage (TED, A22), so both junctions are shallower than on a real wafer.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
