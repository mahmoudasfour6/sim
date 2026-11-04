"""Paper package v1 (0.9c): one archive that holds everything needed to check or rerun a study.

    ~/miniforge3/envs/workflow/bin/python tools/paper_package.py --title "Cryogenic SiGe HBT" --version 1.0 \
        --author "Asfour, Mahmoud|Sabanci University|" [--results-glob "flagship_a/**"] [--allow-dirty]

On top of tools/release.py's bundle (source at the exact commit, conda + pip locks, results, MANIFEST.sha256,
CITATION.cff, PROVENANCE.json) it adds:
  validation/        every gate record found under results/ (cryo_gate, gate_0p9b, engine_pairs, Charon gate.json,
                     the SiC held-out gate) and VALIDATION.md: one table row per check with tolerance and achieved value
  budgets/           BUDGETS.md: uncertainty budgets (numerical, parametric, model-form, calibration) of the paper-
                     fidelity results in the hub store, and the model-form registry they used
  container/         the Apptainer and Docker recipes and tools/container_envs.py (rebuilds every env from the locks)
  REPRODUCE.md       the commands, in order, that regenerate the results (host or container)
Nothing is uploaded: the package stays local unless you run tools/release.py --upload yourself.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATES = {"cryo_gate/cryo_gate.json": "0.9a cryogenic core", "gate_0p9b/gate_0p9b.json": "0.9b contacts, AC/noise, "
         "geometry and coupling", "engine_pairs/engine_pairs.json": "0.9c engine pairs",
         "sic_schottky_heldout/heldout.json": "0.9c SiC Schottky held-out"}


def _release():
    spec = importlib.util.spec_from_file_location("release", ROOT / "tools" / "release.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _walk(prefix, obj, rows):
    """Flatten a gate record into (check, value, tolerance, pass) rows where the record states them."""
    if isinstance(obj, dict):
        tol = obj.get("tolerance") or obj.get("tolerances")
        val = next((obj[k] for k in ("worst_rel_diff", "worst_rel", "rel_diff", "worst", "difference", "rel",
                                     "worst_conservation", "worst_abs_ln_ratio", "poisson_nodal_rel") if k in obj), None)
        if "pass" in obj and (val is not None or tol is not None):
            rows.append((prefix, val, tol, obj["pass"]))
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and k not in ("rows", "points", "per_bias"):
                _walk(f"{prefix}.{k}" if prefix else k, v, rows)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, dict):
                _walk(f"{prefix}[{i}]", v, rows)


def validation(out: Path, res: Path):
    d = out / "validation"; d.mkdir(parents=True, exist_ok=True)
    lines = ["# Validation record", "", "| gate | check | achieved | tolerance | pass |", "|---|---|---|---|---|"]
    found = 0
    for rel, label in GATES.items():
        f = res / rel
        if not f.is_file():
            lines.append(f"| {label} | (no record: {rel}) | | | |")
            continue
        found += 1
        shutil.copy2(f, d / f.name)
        rec = json.loads(f.read_text())
        rows = []
        _walk("", rec, rows)
        lines.append(f"| {label} | overall | | | {rec.get('pass')} |")
        for name, val, tol, ok in rows:
            fmt = lambda x: f"{x:.3g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else json.dumps(x)
            lines.append(f"| | {name} | {fmt(val) if val is not None else ''} | {fmt(tol) if tol is not None else ''}"
                         f" | {ok} |")
    for f in sorted(res.glob("charon_xcheck/**/gate.json")):
        shutil.copy2(f, d / "charon_gate.json"); found += 1
        lines.append(f"| Charon cross-check | see charon_gate.json | | | {json.loads(f.read_text()).get('gate_pass')} |")
    (d / "VALIDATION.md").write_text("\n".join(lines) + "\n")
    return found


def budgets(out: Path, store: Path):
    d = out / "budgets"; d.mkdir(parents=True, exist_ok=True)
    lines = ["# Uncertainty budgets of paper-fidelity results", "",
             "Components are standard uncertainties added in quadrature (simws.hub.uncertainty).", "",
             "| result | quantity | value (first) | sigma (first) | components |", "|---|---|---|---|---|"]
    n = 0
    if (store / "model_form.json").is_file():
        shutil.copy2(store / "model_form.json", d / "model_form.json")
    try:
        from simws.hub.jobs import Hub
        hub = Hub(store)
        rids = [r[0] for r in hub.graph.db.execute(
            "SELECT DISTINCT result_id FROM jobs WHERE fidelity='paper' AND status='ok'")]
        for rid in rids:
            env = hub.store.envelope(rid)
            for q in env.quantities:
                if q.uncertainty.kind != "sigma":
                    continue
                first = lambda x: x[0] if isinstance(x, list) and x else x
                comps = ", ".join(f"{k}={first(v):.3g}" for k, v in q.uncertainty.components.items())
                lines.append(f"| {rid[:10]} | {q.name} | {first(q.value):.6g} | {first(q.uncertainty.sigma):.3g} |"
                             f" {comps} |")
                n += 1
    except Exception as e:  # noqa: BLE001
        lines.append(f"\n(store not readable here: {e})")
    (d / "BUDGETS.md").write_text("\n".join(lines) + "\n")
    return n


REPRODUCE = """# Reproduce

Host (the laptop layout: ~/miniforge3 with the envs in envs/*.explicit.txt):

    tar xzf source.tar.gz && cd sim_workspace-*-source
    python tools/container_envs.py --envs-dir ../envs --prefix ~/miniforge3/envs     # once
    snakemake --cores 8 && snakemake studies --cores 8 && snakemake paper --cores 8
    bash tools/run_charon_xcheck.sh                                                  # needs a Charon build

Container:

    apptainer build simws.sif container/simws.def
    apptainer run --bind $PWD/results:/opt/sim_workspace/results simws.sif snakemake studies --cores 8

Check the archive: `sha256sum -c MANIFEST.sha256`. Gate records and their tolerances: validation/VALIDATION.md.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--author", action="append", required=True)
    ap.add_argument("--slug", default="sim_workspace")
    ap.add_argument("--license", default="MIT")
    ap.add_argument("--description", default="")
    ap.add_argument("--keyword", action="append", default=[])
    ap.add_argument("--paper-doi", default="")
    ap.add_argument("--results-glob", action="append", default=[])
    ap.add_argument("--max-file-mb", type=float, default=200)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--store", default=str(ROOT / "store"))
    a = ap.parse_args()
    R = _release()
    out, tgz, _ = R.build(a)
    nv = validation(out, ROOT / "results")
    nb = budgets(out, Path(a.store))
    c = out / "container"; c.mkdir(exist_ok=True)
    for f in ("container/simws.def", "container/Dockerfile", "tools/container_envs.py"):
        if (ROOT / f).is_file():
            shutil.copy2(ROOT / f, c / Path(f).name)
    (out / "REPRODUCE.md").write_text(REPRODUCE)
    lines = [f"{R.sha256(f)}  {f.relative_to(out)}" for f in sorted(out.rglob("*")) if f.is_file()
             and f.name != "MANIFEST.sha256"]
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    tgz.unlink(missing_ok=True)
    with tarfile.open(tgz, "w:gz") as t:
        t.add(out, arcname=out.name)
    print(f"paper package: {tgz} ({tgz.stat().st_size / 1e6:.1f} MB): {nv} gate records, {nb} budgeted quantities")


if __name__ == "__main__":
    main()
