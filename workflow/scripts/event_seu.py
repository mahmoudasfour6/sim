"""Each Geant4 event drives its own axisymmetric DEVSIM strike (parallel workers), plus a constant-LET
Gaussian-track reference at the same mean deposited energy, per ion (tcad env)."""
import argparse, json
from pathlib import Path
import numpy as np
from simws.materials import load
from simws.provenance import stamp, write_json
from simws.radiation.geant4 import load_events
from simws.radiation.let import let_from_dedx
from simws.studies import geant4_events as ge
from simws.sweep import Store, run_design

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--n-events", type=int, default=12)
ap.add_argument("--workers", type=int, default=1)
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
store = Store(out / "runs.sqlite")
mat = load("silicon")
ions = json.loads(Path(a.index).read_text())["results"]["ions"]
design = []
for ion in ions:
    meta, evs = load_events(ion["events_csv"])
    n = min(a.n_events, len(evs))
    ion["n"] = n
    ion["mean_let"] = float(np.mean([let_from_dedx((e.edep_MeV.sum() + e.overflow_MeV.sum()) / meta["thickness_um"], mat)
                                     for e in evs[:n]]))
    ion["thickness_um"] = meta["thickness_um"]
    design += [{"kind": "event", "events_csv": ion["events_csv"], "event": i, "V_drain": 1.0} for i in range(n)]
    design.append({"kind": "gauss", "let": ion["mean_let"], "thickness_um": meta["thickness_um"], "V_drain": 1.0})


run_design(ge.dispatch_stage, design, store, workers=a.workers)
if store.failures():
    raise SystemExit(f"failed runs: {store.failures()[:2]}")
res = {}
for ion in ions:
    ev_rows = [store.lookup(p) for p in design if p.get("events_csv") == ion["events_csv"]]
    g = store.lookup({"kind": "gauss", "let": ion["mean_let"], "thickness_um": ion["thickness_um"], "V_drain": 1.0})
    s = ge.summarize(ev_rows, g)
    s.update(mean_let=ion["mean_let"], entrance_let=ion["entrance_LET"], ion=ion["tag"])
    res[ion["tag"]] = s
    print(f"{ion['tag']}: LET {ion['mean_let']:.2f}, collected {s['collected_fC_mean']:.1f} +- {s['collected_fC_std']:.1f} fC "
          f"({s['n_events']} events), Gaussian track {g['collected_charge_drain_fC']:.1f} fC, "
          f"efficiency bias of Gaussian {100 * s['gaussian_bias_in_efficiency']:+.1f} %")
write_json(out / "summary.json", res, stamp(parameters=vars(a)))
