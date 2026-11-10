"""Engine registry (architecture v1, section 4.2). Engine modules are imported lazily, only on request, so
`import simws.engines` works without any engine (or devsim) installed."""
from __future__ import annotations

import importlib
from typing import Dict, List

from simws.engines.base import Capability

ENGINES: Dict[str, str] = {"devsim": "simws.engines.devsim_engine", "charon": "simws.engines.charon_engine"}


def names() -> List[str]:
    """Registered engine names, sorted."""
    return sorted(ENGINES)


def module(name: str):
    """Import and return the engine module. KeyError listing the known names when `name` is unknown."""
    if name not in ENGINES:
        raise KeyError(f"unknown engine '{name}'; known engines: {names()}")
    return importlib.import_module(ENGINES[name])


def capability(name: str) -> Capability:
    return module(name).CAPABILITY


def unsupported(name: str, problem) -> List[str]:
    """Reasons engine `name` cannot run `problem` (empty = it can)."""
    return module(name).unsupported(problem)
