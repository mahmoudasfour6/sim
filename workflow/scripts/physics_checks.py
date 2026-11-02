"""Closed-form checks of the full drift-diffusion physics (BGN equilibrium, Masetti mobility, SRH+Auger)."""
import argparse, json
from pathlib import Path
from simws.axisym_checks import axisym_resistor  # noqa: F401  (imported for the combined report below)
from simws.provenance import stamp, write_json
from simws.validation import physics_checks as pc

ap = argparse.ArgumentParser(); ap.add_argument("--outdir", required=True); a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
res = pc.run_all()
res["axisymmetric_resistor"] = axisym_resistor()
res["pass"] = bool(res["pass"] and abs(res["axisymmetric_resistor"]["rel_diff"]) < 1e-6)
write_json(out / "physics_checks.json", res, stamp())
print(json.dumps({k: v.get("pass") if isinstance(v, dict) else v for k, v in res.items()}, indent=1))
if not res["pass"]:
    raise SystemExit("physics checks FAILED")
