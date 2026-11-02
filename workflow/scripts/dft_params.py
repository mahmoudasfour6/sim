"""First-principles silicon parameters (LDA/PBE/PBEsol ensemble + HSE gap + CHGNet benchmark), the 'dft' material
profile, and their checks against measurements."""
import argparse, json
from pathlib import Path
from simws.provenance import stamp, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--ranks", type=int, default=None)
ap.add_argument("--quick", action="store_true", help="PBE only, no convergence variants (smoke test)")
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
from simws.studies.dft_params import Config, run
cfg = Config.from_workspace(ranks=a.ranks)
if a.quick:
    import dataclasses
    cfg = dataclasses.replace(cfg, functionals=("PBE",), convergence=False, chgnet=False)
print(f"pseudopotentials: PBE {cfg.pbe}, PBE fully-relativistic {cfg.pbe_fr}, LDA {cfg.lda} (in {cfg.pseudo_dir})")
res = run(cfg, out, workdir=out / "runs")
write_json(out / "params.json", res, stamp(parameters=vars(a)))
from simws.validation import dft_checks
chk = dft_checks.check(out)
write_json(out / "checks.json", chk, stamp())
for k, v in chk["parameters"].items():
    print(f"  {k:10s} DFT {v['dft']:.4g} +- {v['sigma']:.2g}  measured {v['measured']:.4g}  ({100 * v['rel_dev']:+.1f} %)"
          f"  {'ok' if v['pass'] else 'FAIL'}")
print(f"  n_i(T) vs measured: worst {100 * chk['intrinsic']['worst_rel_dev']:.1f} %;  "
      f"gap {chk.get('gap', {}).get('Eg0_hybrid', float('nan')):.3f} eV;  checks pass={chk['pass']}")
if not chk["pass"] and not a.quick:
    raise SystemExit("DFT parameter checks FAILED (see checks.json)")
if a.quick:
    print("  (quick mode: single functional; the elastic-constant tolerances are meant for the ensemble mean)")
