"""Metadata stamped onto every result: which code, which commit, which environment, which inputs."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from simws.config import ROOT


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp(inputs=(), parameters=None, **extra) -> dict:
    """Return a provenance record. `inputs` are file paths (hashed), `parameters` any JSON-able dict."""
    return {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "conda_env": os.environ.get("CONDA_PREFIX", sys.prefix),
        "python": platform.python_version(),
        "host": platform.node(),
        "inputs": {str(p): sha256(p) for p in inputs if Path(p).is_file()},
        "parameters": parameters or {},
        **extra,
    }


def write_json(path, data: dict, provenance: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"results": data, "provenance": provenance}, indent=2, default=float) + "\n")
