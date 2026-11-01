"""Solution verification: observed order, Richardson extrapolate and GCI for the workflow's key quantities."""
import argparse
from pathlib import Path
from simws.provenance import stamp, write_json
from simws.studies import convergence as cv
from simws.verify import refinement_study

STUDIES = {"electrothermal_mesh": (cv.electrothermal_mesh, 4, "T_max at 5 V [K]"),
           "particle_strike_dt": (cv.particle_strike_dt, 4, "strike charge [C/cm]")}

ap = argparse.ArgumentParser()
ap.add_argument("--study", choices=STUDIES, required=True)
ap.add_argument("--outdir", required=True)
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
fn, levels, label = STUDIES[a.study]
res = refinement_study(fn, levels=levels, ratio=2.0)
print(res.table(label))
write_json(out / f"{a.study}.json", res.as_dict(), stamp(parameters={"study": a.study, "levels": levels}))
