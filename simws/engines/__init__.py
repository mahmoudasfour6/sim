"""Engine registry (architecture v1, section 4.2). Engine modules are imported lazily, only on request, so
`import simws.engines` works without any engine (or devsim) installed."""
from __future__ import annotations

import importlib
from typing import Dict, List

from simws.engines.base import Capability

ENGINES: Dict[str, str] = {"devsim": "simws.engines.devsim_engine", "charon": "simws.engines.charon_engine",
                           "rustbca": "simws.engines.rustbca_engine", "g4see": "simws.engines.g4see_engine"}


def names() -> List[str]:
    """Registered engine names, sorted."""
    return sorted(ENGINES)


def module(name: str):
    """Import and return the engine module. KeyError listing the known names when `name` is unknown."""
    if name not in ENGINES:
        raise KeyError(f"unknown engine '{name}'; known engines: {names()}")
    return importlib.import_module(ENGINES[name])


def capability(name: str):
    """The engine's capability sheet (a Capability for device engines, an ImplantCapability for implant engines)."""
    return module(name).CAPABILITY


def family(name: str) -> str:
    """The problem family the engine solves (contract A7.3): "device" unless the module sets FAMILY."""
    return getattr(module(name), "FAMILY", "device")


def adapter_name(name: str) -> str:
    """The hub adapter of an engine: "<family>.<engine>", e.g. "device.devsim", "implant.rustbca"."""
    return f"{family(name)}.{name}"


def unsupported(name: str, problem) -> List[str]:
    """Reasons engine `name` cannot run `problem` (empty = it can). An engine of another problem family gives
    exactly one reason, starting with "problem kind:"; its own check is not called."""
    from simws import spec
    fam, pf = family(name), spec.family(problem)
    if fam != pf:
        return [f"problem kind: {name} solves {fam} problems, not {pf} problems"]
    return module(name).unsupported(problem)
