"""simws 0.9a gate (Build Plan): the cryogenic core and the paper-fidelity mesh error bars.

    python workflow/scripts/cryo_gate.py --outdir results/cryo_gate [--flagship results/flagship_a/flagship_a.json]

checks     closed-form device checks (simws.validation.cryo_checks): freeze-out equals the neutrality solution to
           1e-6, small-bias current equals q n mu V/L to 1e-4, zero-bias current vanishes on a degenerate graded
           bar (generalized Einstein relation), conductance matrix symmetric to 1e-6 (Onsager)
devices    flagship study A (cryogenic core on) converged for the SiGe HBT and the Si BJT at every temperature of
           its 77-400 K sweep, with no partial sweeps
mesh       one paper-fidelity devsim.hbt_gummel job: three meshes, GCI per bias point, accepted by the gatekeeper
           (every quantity carries a numerical uncertainty)
The Charon cross-check (cold diode within 2 %) runs separately on the laptop: tools/run_charon_xcheck.sh.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--flagship", default="results/flagship_a/flagship_a.json")
ap.add_argument("--parts", default="checks,devices,mesh")
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
parts = a.parts.split(",")
res = {}

if "checks" in parts:
    from simws.validation import cryo_checks as K
    r = K.run_all()
    res["checks"] = r
    print(f"checks: equilibrium {r['resistor_equilibrium']['worst_rel_diff']:.2e}, ohmic "
          f"{r['resistor_ohmic']['worst_rel_diff']:.2e}, Einstein {r['einstein_graded']['worst_rel']:.2e}, "
          f"Onsager {r['onsager']['worst_rel']:.2e}, FD statistics {r['statistics_error']['max_density_error']:.2e}"
          f" -> {'PASS' if r['pass'] else 'FAIL'}")

if "devices" in parts:
    f = json.loads(Path(a.flagship).read_text())
    rows, ok = [], f.get("version", 1) >= 2
    for row in f["rows"]:
        bad = {k: v for k, v in row["status"].items() if v != "ok"}
        partial = [x for x in row["flags"] if x.startswith("partial_sweep")]
        rows.append({"T_K": row["T_K"], "status": row["status"], "partial": partial})
        ok &= not bad and not partial
        print(f"devices: T={row['T_K']:>5} K {row['status']} {partial or ''}")
    res["devices"] = {"rows": rows, "spec_version": f.get("version"), "pass": bool(ok)}

if "mesh" in parts:
    from simws.hub.jobs import Hub
    hub = Hub(out / "store")
    r = hub.submit("devsim.hbt_gummel", params={"case": {"ge_peak": 0.15}, "VBE": (np.array([0.6, 0.7]), "V"),
                                                "T": (300.0, "K")}, fidelity="paper")
    rec = r.verification.get("mesh", {}) if r.status == "ok" else {}
    sig = {}
    if r.status == "ok":
        for q in r.quantities:
            if q.uncertainty.kind == "sigma":
                v, s = np.asarray(q.value, float), np.asarray(q.uncertainty.sigma, float)
                sig[q.name] = (s / np.abs(v)).tolist()
    res["mesh"] = {"status": r.status, "message": r.message, "flags": r.flags, "relative_sigma": sig,
                   "per_bias": rec.get("quantities"), "all_orders_found": rec.get("passed"),
                   "pass": bool(r.status == "ok" and {"hbt.IC", "hbt.IB", "hbt.beta"} <= set(sig))}
    print(f"mesh: {r.status} relative sigma {json.dumps(sig)}")

res["pass"] = all(v["pass"] for k, v in res.items() if isinstance(v, dict) and "pass" in v)
(out / "cryo_gate.json").write_text(json.dumps(res, indent=1, default=float))
print("CRYO GATE", "PASS" if res["pass"] else "FAIL")
raise SystemExit(0 if res["pass"] else 1)
