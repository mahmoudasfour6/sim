"""Analytic checks of the device-physics models (velocity saturation, impact ionization)."""
import argparse, json
from pathlib import Path
from simws.provenance import stamp, write_json
from simws.validation import device_models as dm

ap = argparse.ArgumentParser(); ap.add_argument("--outdir", required=True); a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
res = dm.run_all()
write_json(out / "validation.json", res, stamp())
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, (list, dict))}, indent=1, default=str))
if not res["pass"]:
    raise SystemExit("device-model validation FAILED")
