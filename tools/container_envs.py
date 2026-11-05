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
import shutil
import subprocess
import sys
from pathlib import Path


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def normalise_lock(lock: Path) -> Path:
    """Explicit locks written by newer conda use 'URL md5=HASH' and percent-encoded epochs ('1%21164'); older conda
    reads only 'URL#HASH'. Write that form next to the lock (the packages and hashes are unchanged)."""
    from urllib.parse import unquote
    out = []
    for line in lock.read_text().splitlines():
        t = line.strip()
        if t.startswith("http"):
            parts = t.split()
            url = parts[0]
            md5 = next((p.split("=", 1)[1] for p in parts[1:] if p.startswith("md5=")), None)
            if "#" in url:
                url, md5 = url.split("#", 1)
            head, name = url.rsplit("/", 1)
            line = f"{head}/{unquote(name)}" + (f"#{md5}" if md5 else "")
        out.append(line)
    fixed = lock.with_suffix(".normalised.txt")
    fixed.write_text("\n".join(out) + "\n")
    return fixed


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
        fixed = normalise_lock(lock)
        try:
            subprocess.run([a.conda, "create", "-y", "--prefix", str(pre), "--file", str(fixed)], check=True)
        except subprocess.CalledProcessError:
            free = shutil.disk_usage(a.prefix).free / 2 ** 30
            if free < 3.0:
                # errno 28: updating conda cannot help, and the retry only buries the real error
                raise SystemExit(f"   conda create for {env} failed with {free:.1f} GB free in the build: out of disk "
                                 f"space (see the free-space preflight in tools/container_check.sh)")
            # an older conda than the one that wrote the lock: update conda in base once, then retry
            print("   conda create failed: updating conda in the base env and retrying", flush=True)
            subprocess.run([a.conda, "install", "-y", "-n", "base", "-c", "conda-forge", "conda"], check=True)
            subprocess.run([a.conda, "create", "-y", "--prefix", str(pre), "--file", str(fixed)], check=True)
        pipf = Path(a.envs_dir) / f"{env}.pip.txt"
        if not pipf.is_file():
            _clean(a.conda, env)
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
        missing = []
        if todo:
            pip = [str(pre / "bin" / "python"), "-m", "pip", "install", "--no-deps", "--no-cache-dir"]
            # local-version wheels such as torch==2.15.0+cpu live on the PyTorch CPU index, not on PyPI
            if any("+cpu" in t for t in todo):
                pip += ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
            if subprocess.run(pip + todo).returncode != 0:
                # one bad pin must not sink the image: install one by one and record what could not be installed
                for t in todo:
                    if subprocess.run(pip + [t], capture_output=True).returncode != 0:
                        missing.append(t)
        if missing:
            with open(Path(a.envs_dir).parent / "CONTAINER_MISSING.txt", "a") as fh:
                fh.write(f"{env}: " + " ".join(missing) + "\n")
            print(f"   NOT INSTALLED in {env}: {missing}")
        print(f"   pip-only: {len(todo) - len(missing)} installed, {len(missing)} failed, {len(skipped)} skipped "
              f"(editable/URL): {skipped[:5]}")
        _clean(a.conda, env)


def _clean(conda, env):
    """Empty the package cache after each env: otherwise the downloaded archives and their extracted copies of every
    env pile up in /opt/conda/pkgs on top of the envs themselves (about doubling the build's disk use)."""
    subprocess.run([conda, "clean", "-afy"], stdout=subprocess.DEVNULL, check=False)
    print(f"   {env} done; {shutil.disk_usage('/').free / 2 ** 30:.1f} GB free in the build", flush=True)


if __name__ == "__main__":
    main()
