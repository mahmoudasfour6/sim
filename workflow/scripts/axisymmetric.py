"""Axisymmetric heavy-ion strikes on a cylindrical n+/p diode at realistic doping (1e20 / 1e17), fixed bias,
over a LET list, run in parallel worker processes (each run single-threaded). Writes each transient as CSV."""
import argparse
from pathlib import Path
import numpy as np
from simws.provenance import stamp, write_json
from simws.studies.geant4_events import let_stage as stage   # importable, so any process start method works
from simws.sweep import Store, run_design


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--lets", default="0.03,0.1,0.3,1,10,40")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    store = Store(out / "runs.sqlite")
    design = [{"let": float(x)} for x in a.lets.split(",")]
    run_design(stage, design, store, workers=a.workers)
    if store.failures():
        raise SystemExit(f"failed runs: {store.failures()}")
    runs = []
    for p in design:
        r = store.lookup(p)
        np.savetxt(out / f"axi_let{p['let']:g}.csv", np.column_stack([r.pop("time_s"), r.pop("i_drain_A"), r.pop("i_sub_A")]),
                   delimiter=",", header="time_s,i_drain_A,i_sub_A", comments="")
        runs.append(r)
        print(f"LET {p['let']:6.3g}: deposited {r['deposited_charge_fC']:.2f} fC, collected drain "
              f"{r['collected_charge_drain_fC']:.2f} fC, peak {1e3 * r['peak_drain_current_A']:.3f} mA, "
              f"mesh factor {r['track_mesh_correction']:.4f}")
    ref = next((r for r in runs if r["case"]["let"] == 10.0), runs[-1])
    write_json(out / "summary.json", dict(ref, all_lets=runs), stamp(parameters=vars(a)))
