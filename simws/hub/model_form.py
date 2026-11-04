"""Cross-engine model-form registry (0.9c).

The 0.9c engine pairs (simws.validation.engine_pairs) measure how far two independent open-source codes disagree on
the same reference problem. That disagreement is a model-form uncertainty (the choice of code is a modelling choice),
so it is kept here and added to the budgets of paper-fidelity results whose quantities belong to the same class.

    from simws.hub import model_form as MF
    MF.record(store_root, {"devsim.current": 1.7e-4, ...}, source="results/engine_pairs/engine_pairs.json")
    MF.attach(result_envelope, store_root)       # done by Hub._commit for paper fidelity

The registry is <store>/model_form.json: per class, a relative standard uncertainty (range / sqrt(12)), the source
file and date. The class of a quantity comes from CLASSES (quantity-name prefixes); a quantity in no class is left
alone. Components already present are never overwritten, and the total sigma is recombined in quadrature.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# quantity-name prefix -> registry class
CLASSES = {
    "hbt.IC": "devsim.current", "hbt.IB": "devsim.current", "mos.": "devsim.current", "strike.": "devsim.current",
    "dft.a0": "dft.a0", "dft.B0": "dft.B0", "dft.Eg": "dft.gap", "dft.gap": "dft.gap",
    "thermal.": "thermal.T", "electrothermal.T": "thermal.T",
}


def _path(root) -> Path:
    return Path(root) / "model_form.json"


def load(root) -> Dict[str, dict]:
    p = _path(root)
    return json.loads(p.read_text()) if p.is_file() else {}


def record(root, entries: Dict[str, float], source: str) -> Dict[str, dict]:
    reg = load(root)
    now = dt.datetime.now().isoformat(timespec="seconds")
    for k, v in entries.items():
        reg[k] = {"relative_sigma": float(v), "source": str(source), "recorded": now,
                  "rule": "range / sqrt(12) over two independent engines"}
    _path(root).parent.mkdir(parents=True, exist_ok=True)
    _path(root).write_text(json.dumps(reg, indent=1))
    return reg


def class_of(name: str) -> Optional[str]:
    for pre, cls in CLASSES.items():
        if name.startswith(pre):
            return cls
    return None


def attach(res, root) -> list:
    """Add model_form components to the sigma-kind quantities of a ResultEnvelope. Returns the classes used."""
    reg = load(root)
    used = []
    if not reg:
        return used
    for q in res.quantities:
        cls = class_of(q.name)
        if cls is None or cls not in reg or q.uncertainty.kind != "sigma" or "model_form" in q.uncertainty.components:
            continue
        val = np.asarray(q.value, float)
        mf = np.abs(val) * reg[cls]["relative_sigma"]
        comps = {k: np.asarray(v, float) for k, v in q.uncertainty.components.items()}
        if not comps:                                        # a bare sigma counts as one unnamed component
            comps = {"unspecified": np.asarray(q.uncertainty.sigma, float)}
        comps["model_form"] = mf
        tot = np.sqrt(sum(np.broadcast_to(c, val.shape) ** 2 for c in comps.values()))
        q.uncertainty.components = {k: (c.tolist() if np.ndim(c) else float(c)) for k, c in comps.items()}
        q.uncertainty.sigma = tot.tolist() if np.ndim(tot) else float(tot)
        used.append(cls)
    if used:
        res.provenance["model_form"] = {c: reg[c] for c in sorted(set(used))}
    return sorted(set(used))
