"""SiGe: model checks (heterojunction equilibrium, Moll-Ross collector current, ideality, buried channel, HBT TID
and ion-strike conservation) and the device study (HBT vs Si BJT, buried pMOS channel, HBT radiation)."""
import argparse, json
from pathlib import Path
from simws.provenance import stamp, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--what", choices=("checks", "devices"), required=True)
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
if a.what == "checks":
    from simws.validation import sige_checks
    res = sige_checks.run_all()
    write_json(out / "sige_checks.json", res, stamp())
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in res.items()}, indent=1))
    if not res["pass"]:
        raise SystemExit("SiGe checks FAILED")
else:
    from simws.studies.sige_devices import run
    res = run(workers=a.workers)
    write_json(out / "devices.json", res, stamp(parameters=vars(a)))
    print(json.dumps(res["summary"], indent=1, default=float))
