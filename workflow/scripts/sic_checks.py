"""4H-SiC model checks: ionization (closed form and DEVSIM), anisotropic mobility, breakdown."""
import argparse, json
from pathlib import Path
from simws.provenance import stamp, write_json
from simws.validation import sic_checks

ap = argparse.ArgumentParser(); ap.add_argument("--outdir", required=True); a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
res = sic_checks.run_all()
write_json(out / "sic_checks.json", res, stamp())
print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in res.items()}, indent=1))
b = res["breakdown"]
print(f"breakdown N_D={b['ND']:.0e}: DEVSIM {b['BV_devsim']} V, ionization integral {b['BV_ionization_integral']} V, "
      f"eps Ec^2/2qN {b['BV_from_Ec']:.1f} V, critical field {b['E_crit_MV_cm']:.2f} MV/cm")
if not res["pass"]:
    raise SystemExit("SiC checks FAILED")
