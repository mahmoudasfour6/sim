"""Geant4 ion track in Si -> radial/depth deposition and LET profile, checked against the G4EmCalculator LET."""
import argparse
from pathlib import Path
import numpy as np
from simws.materials import load
from simws.provenance import stamp, write_json
from simws.radiation import geant4

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--Z", type=int, default=8); ap.add_argument("--A", type=int, default=16)
ap.add_argument("--energy-MeV", type=float, default=100.0)
ap.add_argument("--thickness-um", type=float, default=20.0)
ap.add_argument("--events", type=int, default=200)
ap.add_argument("--prefix", default=None)
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
geant4.build(a.prefix)
csv = geant4.run(out / "deposition.csv", prefix=a.prefix, Z=a.Z, A=a.A, energy_MeV=a.energy_MeV, material="G4_Si",
                 thickness_um=a.thickness_um, rmax_um=2.0, nr=40, nz=40, events=a.events, cut_um=1.0, seed=1)
dep = geant4.load(csv)
mat = load("silicon")
z, let = dep.let_profile(mat)
r_frac = dep.radial_fraction()
r = dep.r_edges_um[1:]
inner = z < 0.5 * a.thickness_um                      # entrance half: delta electrons build up with depth
res = {"ion": f"Z={a.Z} A={a.A}", "energy_MeV": a.energy_MeV, "events": a.events,
       "meta": dep.meta,
       "LET_deposited_entrance_half": float(np.mean(let[inner])),
       "LET_deposited_mean": float(np.mean(let)),
       "radius_50pct_energy_um": float(np.interp(0.5, r_frac, r)),
       "radius_90pct_energy_um": float(np.interp(0.9, r_frac, r)),
       "LET_ratio_entrance_deposit_to_G4EmCalculator": float(np.mean(let[z < 2.0]) / dep.meta["entrance_LET_MeV_cm2_per_mg"]),
       "fraction_beyond_rmax": float(1 - r_frac[-1])}
write_json(out / "summary.json", res, stamp(parameters=vars(a)))
print(res)
