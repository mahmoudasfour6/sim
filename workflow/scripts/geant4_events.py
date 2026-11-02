"""Geant4 per-event depositions for the ion list (process_rad env). Thickness and radius match the
axisymmetric device (10 um deep); 100 radial bins of 20 nm resolve the delta-ray core."""
import argparse
from pathlib import Path
from simws.provenance import stamp, write_json
from simws.radiation import geant4

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--ions", default="17:35:300,36:84:1000")       # Z:A:E_MeV (Cl-35 LET ~9, Kr-84 LET ~24)
ap.add_argument("--events", type=int, default=20)
ap.add_argument("--prefix", default=None)
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
geant4.build(a.prefix)
made = []
for spec in a.ions.split(","):
    Z, A, E = spec.split(":")
    tag = f"Z{Z}_A{A}_E{E}"
    avg = geant4.run(out / f"{tag}_avg.csv", int(Z), int(A), float(E), thickness_um=10.0, rmax_um=2.0, nr=100, nz=20,
                     events=a.events, cut_um=0.1, seed=7, prefix=a.prefix, events_out=out / f"{tag}_events.csv")
    d = geant4.load(avg)
    made.append({"tag": tag, "Z": int(Z), "A": int(A), "E_MeV": float(E), "events": a.events,
                 "entrance_LET": d.meta["entrance_LET_MeV_cm2_per_mg"], "events_csv": str(out / f"{tag}_events.csv")})
    print(f"{tag}: entrance LET {d.meta['entrance_LET_MeV_cm2_per_mg']:.2f} MeV cm2/mg, {a.events} events")
write_json(out / "index.json", {"ions": made}, stamp(parameters=vars(a)))
