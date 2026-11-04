"""simws 0.9c gate, part 1: each engine pair agrees on its reference problems within the tolerances fixed in advance
(simws.validation.engine_pairs.TOLERANCES, fixed 2026-11-04), and the code-to-code spread is recorded as a
model-form component in the store's registry (simws.hub.model_form).

    python workflow/scripts/engine_pairs.py --outdir results/engine_pairs --parts dft     # in the DFT env (pw.x, gpaw)
    python workflow/scripts/engine_pairs.py --outdir results/engine_pairs --parts fem,dd --store store   # tcad env
The second call merges the DFT part written by the first (dft.json) and writes engine_pairs.json.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--parts", default="fem,dd")
ap.add_argument("--store", default=None, help="hub store root that receives the model-form registry")
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
parts = a.parts.split(",")

from simws.validation import engine_pairs as E

if parts == ["dft"]:
    import subprocess

    def fetch(target):
        subprocess.run(["bash", "tools/fetch_pseudodojo.sh", str(target)], check=False)
    rr = E.dft_pair_resolved(out / "dft_work", fetch_dojo=fetch)
    r = dict(rr["final"], resolution=rr)
    (out / "dft.json").write_text(json.dumps(r, indent=1, default=float))
    print("pre-registered (QE 40 Ry rrkj, GPAW 500 eV, k8):", "PASS" if rr["preregistered"]["pass"] else "FAIL",
          f"L gap diff {1e3 * rr['preregistered']['rows']['gap_L_eV']['difference']:.1f} meV")
    for name, steps in rr.get("convergence", {}).items():
        for st in steps:
            print(f"convergence {name:8s} {st['from']:>11s} -> {st['to']:<11s} worst change "
                  f"{100 * st['worst']:.1f} % of tolerance")
    if "converged" in rr:
        print("converged:", rr["converged"], "| top setting:", "PASS" if rr["top_setting"]["pass"] else "FAIL",
              f"L gap diff {1e3 * rr['top_setting']['rows']['gap_L_eV']['difference']:.1f} meV")
    if "pseudodojo" in rr:
        print(f"QE PseudoDojo vs GPAW: {'PASS' if rr['pseudodojo']['pass'] else 'FAIL'}; QE L gap rrkj - dojo "
              f"{1e3 * rr['rrkj_vs_dojo_L_gap_eV']:+.1f} meV")
    print("decision:", rr["decision"])
    for k, v in r["rows"].items():
        print(f"dft: {k:16s} QE {v['qe']:.5f}  GPAW {v['gpaw']:.5f}  diff {v['difference']:.2e} (tol {v['tolerance']})"
              f" {'ok' if v['pass'] else 'FAIL'}")
    print("DFT PAIR", "PASS" if r["pass"] else "FAIL")
    raise SystemExit(0)          # the record is the result: the engine_pairs rule merges it and reports the gate

res = E.run_all(out / "work", parts=tuple(p for p in parts if p != "dft"))
if (out / "dft.json").is_file():
    res["dft"] = json.loads((out / "dft.json").read_text())
    res["model_form"] = E.registry_entries(res)
gated = [k for k in ("dft", "fem", "dd") if isinstance(res.get(k), dict)]
res["pass"] = all(res[k].get("pass") is not False for k in gated) and all(
    res[k].get("pass") is not None for k in gated)
for k in gated:
    print(f"{k}: {json.dumps({kk: vv for kk, vv in res[k].items() if kk in ('pass', 'status', 'poisson_nodal_rel', 'dark_rel', 'Jsc', 'heat_Tmax')}, default=float)}")
print("model-form components (relative sigma):", json.dumps(res["model_form"]))
if a.store and res["model_form"]:
    from simws.hub import model_form
    model_form.record(a.store, res["model_form"], source=str(out / "engine_pairs.json"))
    print(f"recorded in {a.store}/model_form.json")
(out / "engine_pairs.json").write_text(json.dumps(res, indent=1, default=float))
print("ENGINE PAIRS", "PASS" if res["pass"] else "FAIL")
# a failed pair is a finding (reported, recorded, and asserted by the slow test), not a broken workflow
raise SystemExit(0)
