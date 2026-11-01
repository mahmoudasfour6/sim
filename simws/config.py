"""Workspace configuration. Paths come from config/config.yaml (or SIMWS_CONFIG), never from code."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load() -> dict:
    path = Path(os.environ.get("SIMWS_CONFIG", ROOT / "config" / "config.yaml"))
    cfg = yaml.safe_load(path.read_text()) if path.exists() else {}
    cfg.setdefault("workspace_root", str(ROOT))

    def expand(v):
        return os.path.expanduser(os.path.expandvars(v)) if isinstance(v, str) else v

    return {k: ({kk: expand(vv) for kk, vv in v.items()} if isinstance(v, dict) else expand(v)) for k, v in cfg.items()}


def path(key: str) -> Path:
    """Look up a dotted key such as 'toolchain.charon_exe' and return it as a Path."""
    node = load()
    for part in key.split("."):
        node = node[part]
    return Path(node)


def env_python(env: str) -> Path:
    return path("envs_root") / env / "bin" / "python"
