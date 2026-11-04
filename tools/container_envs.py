"""Rebuild every conda environment of the workspace from its locks (used by the container recipes).

    python tools/container_envs.py --envs-dir envs --prefix /opt/conda/envs [--only tcad,workflow]

For each <env>.explicit.txt: `conda create --prefix <prefix>/<env> --file <env>.explicit.txt` (exact builds with md5).
Then the pip-only packages from <env>.pip.txt: lines `name==version` whose distribution conda did not install are
installed with --no-deps (their dependencies are already in the explicit lock). Editable or URL lines are reported
and skipped (the repository itself is on PYTHONPATH, not installed).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs-dir", default="envs")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--conda", default=sys.executable.replace("/bin/python", "/bin/conda"))
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    for lock in sorted(Path(a.envs_dir).glob("*.explicit.txt")):
        env = lock.name[:-len(".explicit.txt")]
        if only and env not in only:
            continue
        pre = Path(a.prefix) / env
        print(f"== {env}", flush=True)
        subprocess.run([a.conda, "create", "-y", "--prefix", str(pre), "--file", str(lock)], check=True)
        pipf = Path(a.envs_dir) / f"{env}.pip.txt"
        if not pipf.is_file():
            continue
        listed = subprocess.run([a.conda, "list", "--prefix", str(pre), "--json"], check=True, capture_output=True,
                                text=True).stdout
        have = {norm(p["name"]) for p in json.loads(listed)}
        todo, skipped = [], []
        for line in pipf.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)$", line)
            if not m:
                skipped.append(line)
                continue
            if norm(m.group(1)) not in have:
                todo.append(line)
        if todo:
            subprocess.run([str(pre / "bin" / "python"), "-m", "pip", "install", "--no-deps", *todo], check=True)
        print(f"   pip-only: {len(todo)} installed, {len(skipped)} skipped (editable/URL): {skipped[:5]}")


if __name__ == "__main__":
    main()
