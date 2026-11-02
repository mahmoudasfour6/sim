"""Promote a computed material profile (e.g. results/dft/silicon.dft.yaml) into the git-tracked database
(simws/materials/data/computed/), after showing each value next to the literature value it would sit beside.

    python tools/promote_computed.py results/dft/silicon.dft.yaml            # review only
    python tools/promote_computed.py results/dft/silicon.dft.yaml --yes      # copy it in (then git add / commit)
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simws.materials import load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()
src = Path(a.file)
doc = yaml.safe_load(src.read_text())
for mat, profiles in doc.items():
    base = load(mat)
    for prof, entries in profiles.items():
        print(f"{mat} / profile '{prof}': {len(entries)} parameters")
        for k, e in entries.items():
            lit = ""
            if k in base.parameters:
                p = base.parameters[k]
                lit = f"   literature {p.value:.5g} {p.unit} ({p.status})"
            sig = e.get("sigma")
            print(f"  {k:22s} {e['value']:.5g} +- {sig:.2g} {e['unit']}{lit}" if sig is not None
                  else f"  {k:22s} {e['value']:.5g} {e['unit']}{lit}")
        dst = ROOT / "simws" / "materials" / "data" / "computed" / f"{mat}.{prof}.yaml"
        if a.yes:
            dst.parent.mkdir(parents=True, exist_ok=True)
            out = {mat: {prof: entries}}
            dst.write_text(f"# promoted from {src}\n" + yaml.safe_dump(out, sort_keys=False))
            print(f"  -> {dst.relative_to(ROOT)} (now: git add {dst.relative_to(ROOT)} && git commit)")
        else:
            print(f"  (review only: add --yes to write {dst.relative_to(ROOT)})")
