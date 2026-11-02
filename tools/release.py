"""Make a citable, self-contained release of one commit for a paper (Zenodo-ready).

    ~/miniforge3/envs/workflow/bin/python tools/release.py --title "SEU in n+/p diodes" --version 1.0 \
        --author "Asfour, Mahmoud|Sabanci University|0000-0000-0000-0000"         # build the bundle only
    ... --upload --sandbox            # then upload to sandbox.zenodo.org as a DRAFT (ZENODO_TOKEN env var)
    ... --upload                      # upload to zenodo.org as a DRAFT; you press Publish on the web page

The bundle (release/<name>/ and release/<name>.tar.gz) holds:
  source.tar.gz        `git archive` of the exact commit (refuses a dirty tree unless --allow-dirty)
  envs/                every conda lockfile (envs/*.yml, envs/*.explicit.txt) so the environments rebuild exactly
  results/             summaries, figures, CSV transients, sweep databases (no meshes or scratch); --results-glob adds
  MANIFEST.sha256      SHA-256 of every file, so readers can verify integrity
  CITATION.cff         citation metadata (GitHub and Zenodo read it)
  .zenodo.json         Zenodo deposition metadata (title, creators, license, related identifiers)
  PROVENANCE.json      commit, branch, dirty flag, date, machine, simws version, per-result provenance stamps

Publishing is never automatic: --upload creates a draft and prints its link; a DOI is minted only when you
publish it yourself, after checking the files.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_GLOBS = ["**/summary.json", "**/comparison.json", "**/selftest.json", "**/validation.json",
                "**/physics_checks.json", "**/*.csv", "figures/*", "verification/*.json", "**/runs.sqlite",
                "**/index.json", "materials/*"]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha256(p: Path):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_author(s):
    parts = [x.strip() for x in s.split("|")]
    name = parts[0]
    family, given = (name.split(",", 1) + [""])[:2] if "," in name else (name.split()[-1], " ".join(name.split()[:-1]))
    return {"name": name if "," in name else f"{family}, {given}", "family": family.strip(), "given": given.strip(),
            "affiliation": parts[1] if len(parts) > 1 else "", "orcid": parts[2] if len(parts) > 2 else ""}


def build(a):
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    if dirty and not a.allow_dirty:
        raise SystemExit("working tree has uncommitted changes: commit first (or --allow-dirty, recorded in PROVENANCE)")
    name = f"{a.slug}-v{a.version}"
    out = ROOT / "release" / name
    if out.exists():
        shutil.rmtree(out)
    (out / "envs").mkdir(parents=True)
    subprocess.run(["git", "archive", "--format=tar.gz", f"--prefix={name}-source/", "-o", str(out / "source.tar.gz"), commit],
                   cwd=ROOT, check=True)
    for f in sorted((ROOT / "envs").glob("*")):
        if f.suffix in (".yml", ".txt"):
            shutil.copy2(f, out / "envs" / f.name)
    res = ROOT / "results"
    copied, stamps = 0, {}
    for g in RESULT_GLOBS + a.results_glob:
        for f in res.glob(g):
            if f.is_file() and "nightly" not in f.parts and f.stat().st_size < a.max_file_mb * 1e6:
                dst = out / "results" / f.relative_to(res)
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
                    copied += 1
                    if f.suffix == ".json":
                        try:
                            pv = json.loads(f.read_text()).get("provenance", {})
                            stamps[str(f.relative_to(res))] = {k: pv.get(k) for k in ("git_commit", "git_dirty", "created_utc")}
                        except Exception:  # noqa: BLE001
                            pass
    authors = [parse_author(x) for x in a.author]
    today = dt.date.today().isoformat()
    try:
        from simws import __version__ as simws_version
    except Exception:  # noqa: BLE001
        simws_version = "unknown"
    prov = {"commit": commit, "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "dirty": dirty, "date": today,
            "machine": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
            "simws_version": simws_version, "result_files": copied, "result_provenance": stamps}
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=1))
    cff = ["cff-version: 1.2.0", f'message: "If you use this software or data, please cite it as below."',
           f'title: "{a.title}"', f'version: "{a.version}"', f"date-released: {today}", f'commit: "{commit}"',
           f"license: {a.license}", "type: software", "authors:"]
    for au in authors:
        cff += [f'  - family-names: "{au["family"]}"', f'    given-names: "{au["given"]}"']
        if au["affiliation"]:
            cff.append(f'    affiliation: "{au["affiliation"]}"')
        if au["orcid"]:
            cff.append(f'    orcid: "https://orcid.org/{au["orcid"]}"')
    if a.paper_doi:
        cff += ["references:", "  - type: article", f'    doi: "{a.paper_doi}"', f'    title: "{a.title}"']
    (out / "CITATION.cff").write_text("\n".join(cff) + "\n")
    zen = {"title": a.title, "upload_type": "software", "version": a.version, "license": a.license.lower(),
           "description": a.description or f"Simulation code, environments and results for: {a.title}. "
                          f"Exact commit {commit}. See PROVENANCE.json and MANIFEST.sha256.",
           "creators": [{k: v for k, v in {"name": au["name"], "affiliation": au["affiliation"],
                                           "orcid": au["orcid"]}.items() if v} for au in authors],
           "keywords": a.keyword or ["TCAD", "DEVSIM", "single-event effects", "Geant4", "reproducibility"]}
    if a.paper_doi:
        zen["related_identifiers"] = [{"identifier": a.paper_doi, "relation": "isSupplementTo", "scheme": "doi"}]
    (out / ".zenodo.json").write_text(json.dumps(zen, indent=1))
    lines = [f"{sha256(f)}  {f.relative_to(out)}" for f in sorted(out.rglob("*")) if f.is_file()]
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    tgz = out.parent / f"{name}.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        t.add(out, arcname=name)
    print(f"release bundle: {tgz} ({tgz.stat().st_size / 1e6:.1f} MB), commit {commit[:10]}, {copied} result files")
    return out, tgz, zen


def upload(tgz: Path, meta: dict, sandbox: bool):
    """Create a DRAFT deposition with the bundle and metadata. Never publishes."""
    import requests
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit("set ZENODO_TOKEN (Zenodo > Applications > Personal access token, scopes deposit:write)")
    base = "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"
    p = {"access_token": token}
    r = requests.post(f"{base}/deposit/depositions", params=p, json={}, timeout=60); r.raise_for_status()
    dep = r.json()
    with open(tgz, "rb") as f:
        requests.put(f"{dep['links']['bucket']}/{tgz.name}", data=f, params=p, timeout=3600).raise_for_status()
    requests.put(f"{base}/deposit/depositions/{dep['id']}", params=p, json={"metadata": meta}, timeout=60).raise_for_status()
    print(f"draft created: {dep['links']['html']}  (check the files, then press Publish to mint the DOI)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--author", action="append", required=True, help='"Family, Given|Affiliation|ORCID"')
    ap.add_argument("--slug", default="sim_workspace")
    ap.add_argument("--license", default="MIT")
    ap.add_argument("--description", default="")
    ap.add_argument("--keyword", action="append", default=[])
    ap.add_argument("--paper-doi", default="")
    ap.add_argument("--results-glob", action="append", default=[])
    ap.add_argument("--max-file-mb", type=float, default=200)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--sandbox", action="store_true")
    a = ap.parse_args()
    out, tgz, zen = build(a)
    if a.upload:
        upload(tgz, zen, a.sandbox)


if __name__ == "__main__":
    main()
