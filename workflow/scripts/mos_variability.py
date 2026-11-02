"""Polynomial-chaos surrogate of MOS process variation: train on a Latin hypercube, check on held-out runs,
then Sobol indices and output distributions (incl. 3-sigma tails) from a million surrogate samples."""
import argparse
from pathlib import Path
import numpy as np
from simws.provenance import stamp, write_json
from simws.studies import mos_variability as mv
from simws.surrogate import PCE
from simws.sweep import Store, run_design

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--n-train", type=int, default=100)
ap.add_argument("--n-test", type=int, default=20)
ap.add_argument("--workers", type=int, default=1)
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
store = Store(out / "runs.sqlite")
train = PCE.design(mv.VARIABLES, a.n_train, seed=0)
test = PCE.design(mv.VARIABLES, a.n_test, seed=99)
run_design(mv.stage, train + test, store, workers=a.workers)
if store.failures():
    raise SystemExit(f"failed: {store.failures()[:2]}")
res = {"variables": [v.__dict__ for v in mv.VARIABLES], "n_train": a.n_train, "n_test": a.n_test}
for o in mv.OUTPUTS:
    y = [store.lookup(p)[o] for p in train]
    pce = PCE.fit(mv.VARIABLES, train, y, max_degree=4)
    yt = np.array([store.lookup(p)[o] for p in test]); yp = pce.predict(test)
    s = pce.summary(n=1_000_000)
    s["holdout_rmse"] = float(np.sqrt(np.mean((yt - yp) ** 2)))
    s["holdout_max_abs"] = float(np.max(np.abs(yt - yp)))
    res[o] = s
    print(f"{o}: mean {s['mean']:.4g}, std {s['std']:.3g}, Q2 {s['loo_q2']:.5f}, holdout rmse {s['holdout_rmse']:.2e}; "
          + ", ".join(f"{k} ST {v['ST']:.2f}" for k, v in s["sobol"].items()))
write_json(out / "summary.json", res, stamp(parameters=vars(a)))
