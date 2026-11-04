"""simws 0.9c gate, part 2: 4H-SiC Schottky I-V(T) against a published dataset held out from fitting.

    python workflow/scripts/sic_heldout.py --outdir results/sic_schottky_heldout [--dataset NAME]
Runs every approved dataset under simws/validation/data/sic_schottky/ (or the one named). Exit code 3 = no approved
dataset yet (the gate is pending, not failed).
"""
import argparse
import json
from pathlib import Path

from simws.validation import sic_schottky_heldout as H

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--dataset", default=None)
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
names = [a.dataset] if a.dataset else sorted(
    d.name for d in H.DATA.glob("*") if (d / "meta.json").is_file()
    and json.loads((d / "meta.json").read_text()).get("approved")) if H.DATA.is_dir() else []
if not names:
    print("SIC HELD-OUT: no approved dataset yet (pending)")
    raise SystemExit(3)
res = {"self_test": H.self_test(), "datasets": {n: H.run(n) for n in names}}
res["pass"] = res["self_test"]["pass"] and all(r["pass"] for r in res["datasets"].values())
for n, r in res["datasets"].items():
    print(f"{n}: phi_B0 {r['fit']['phi_B0_eV']:.3f} eV, sigma_s {1e3 * r['fit']['sigma_s_eV']:.0f} meV; held-out worst "
          f"ratio {2.718281828 ** r['heldout']['worst_abs_ln_ratio']:.2f} -> {'PASS' if r['pass'] else 'FAIL'}")
(out / "heldout.json").write_text(json.dumps(res, indent=1, default=float))
print("SIC HELD-OUT", "PASS" if res["pass"] else "FAIL")
raise SystemExit(0 if res["pass"] else 1)
