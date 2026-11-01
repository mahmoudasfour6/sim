"""Sobol sensitivity of the electro-thermal resistor (5 V) to material-parameter uncertainty, run locally on
`--workers` processes. Results go to an SQLite store, so an interrupted run resumes without recomputing."""
import argparse
from pathlib import Path
import numpy as np
from simws.provenance import stamp, write_json
from simws.studies import electrothermal_uq as uq
from simws.sweep import Store, run_design

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--n", type=int, default=64)           # base samples (power of 2): n*(d+2) runs
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
store = Store(out / "runs.sqlite")
design = uq.SPACE.sobol(a.n)
run_design(uq.stage, design, store, workers=a.workers)
fails = store.failures()
res = {"n_base": a.n, "runs": len(design), "failures": len(fails), "space": uq.SPACE.bounds}
for o in uq.OUTPUTS:
    y = np.array([store.lookup(p)[o] for p in design])
    res[o] = {"mean": float(y.mean()), "std": float(y.std(ddof=1)), "p05": float(np.percentile(y, 5)),
              "p95": float(np.percentile(y, 95)), "sobol": uq.SPACE.sobol_indices(store, o, a.n)}
write_json(out / "summary.json", res, stamp(parameters=vars(a)))
for o in uq.OUTPUTS:
    r = res[o]
    print(f"{o}: {r['mean']:.4g} +- {r['std']:.2g} (5-95 %: {r['p05']:.4g} .. {r['p95']:.4g})")
    for k, s in r["sobol"].items():
        print(f"   {k:22s} S1 = {s['S1']:6.3f} +- {s['S1_conf']:.3f}   ST = {s['ST']:6.3f} +- {s['ST_conf']:.3f}")
