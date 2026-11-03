"""Canonical form and content-addressed identifiers (interface specification: identifiers section).

    from simws.hub.canonical import canonical_json, job_id, canonical_quantity
    jid = job_id(key_dict)                        # 32 hex characters

Canonical form: JSON with sorted keys and no whitespace; floats in job inputs rounded to 12 significant digits (so
floating-point noise from upstream arithmetic does not create new jobs) and printed with Python's shortest repr;
NaN and infinity refused; numpy scalars and arrays converted to Python numbers and lists; quantities converted to
the canonical unit of their dimension (SI base units, except energies in eV and LET in MeV cm2/mg) and written as
{"dims": {...}, "unit": <canonical unit>, "value": ...}, so the hash does not depend on how a unit was spelled or
on the pint version; paths never enter, files enter by the SHA-256 of their content.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from simws.units import Q_, ureg

ID_LEN = 32

# dimension exceptions to SI base units (spec: conventions registry, row "Units")
_ENERGY = ureg.parse_expression("eV").dimensionality
_LET = ureg.parse_expression("MeV*cm**2/mg").dimensionality
_EXCEPTIONS = {_ENERGY: "eV", _LET: "MeV*cm**2/mg"}


class CanonicalError(ValueError):
    pass


def canonical_unit(unit: str) -> str:
    """The canonical unit string for the dimension of `unit`."""
    q = Q_(1.0, unit)
    d = q.dimensionality
    if d in _EXCEPTIONS:
        return _EXCEPTIONS[d]
    if not d:
        return "dimensionless"
    return f"{q.to_base_units().units:~C}"


def to_canonical(value, unit: str):
    """(value in the canonical unit, canonical unit string, dimension dict)."""
    q = Q_(np.asarray(value, dtype=float), unit)
    cu = canonical_unit(unit)
    v = q.to(cu).magnitude if cu != "dimensionless" else q.to("dimensionless").magnitude
    dims = {str(k): float(v_) for k, v_ in sorted(q.dimensionality.items())}
    return v, cu, dims


SIG_DIGITS = 12          # job keys round floats to 12 significant digits (0.9a): 0.1 + 0.2 and 0.3 are the same job


def round_sig(x):
    """Round every float in a (nested) value to SIG_DIGITS significant digits; ints, strings, bools unchanged."""
    if isinstance(x, bool) or x is None or isinstance(x, (str, int)) and not isinstance(x, float):
        return x
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return float(f"{f:.{SIG_DIGITS}g}") if math.isfinite(f) else f
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return round_sig(x.tolist())
    if isinstance(x, (list, tuple)):
        return [round_sig(v) for v in x]
    if isinstance(x, dict):
        return {k: round_sig(v) for k, v in x.items()}
    return x


def canonical_quantity(value, unit: str) -> dict:
    v, cu, dims = to_canonical(value, unit)
    return {"dims": dims, "unit": cu, "value": round_sig(_plain(v))}


def _plain(x: Any):
    """Convert numpy and pydantic objects to plain JSON values; refuse NaN and infinity."""
    if hasattr(x, "model_dump"):
        return _plain(x.model_dump(mode="python"))
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(_plain(v) for v in x)
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        if not math.isfinite(f):
            raise CanonicalError("NaN or infinity cannot be hashed")
        return f
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        raise CanonicalError(f"paths never enter a hash; pass the file's sha256 instead: {x}")
    return x


def canonical_json(obj: Any) -> str:
    return json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def job_id(key: Any) -> str:
    return sha256_text(canonical_json(key))[:ID_LEN]


def result_id(jid: str, output_hashes: list) -> str:
    return sha256_text(jid + "".join(sorted(output_hashes)))[:ID_LEN]
