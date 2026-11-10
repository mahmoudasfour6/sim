"""Measured data stored like engine results (architecture v1, section 6.2).

    ingest(hub, "diode_iv", ("V", volts, "V"), {"I": (amps, "A")}, {"source": "Author 2020, Fig. 3"})

The data become an ordinary result (adapter 'measured.<name>', flag 'measured') in the store and the graph, so
compare / trust / catalogue treat them like any other result. A citation in meta["source"] is required.
"""
from __future__ import annotations

import csv
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from simws.hub import gatekeeper
from simws.hub.canonical import canonical_json, job_id, sha256_text
from simws.hub.schema import Axis, JobEnvelope, JobKey, Manifest, Quantity, ResultEnvelope, Uncertainty

ADAPTER_VERSION = "1.0"


def _vec(x, what: str) -> List[float]:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{what}: expected a 1-D sequence, got shape {a.shape}")
    return [float(v) for v in a]


def _scalar(x, what: str) -> float:
    a = np.asarray(x, dtype=float)
    if a.size != 1:
        raise ValueError(f"{what}: without an axis a column must hold exactly one value, got {a.size}")
    return float(a.reshape(-1)[0])


def ingest(hub, name: str, axis: Optional[Tuple[str, Sequence[float], str]],
           columns: Dict[str, Tuple[Sequence[float], str]], meta: Dict[str, Any],
           sigma: Optional[Dict[str, Sequence[float]]] = None, label: str = "validation") -> ResultEnvelope:
    """Store measured data as a result. Idempotent: the same name + data + meta returns the stored envelope."""
    if not name or not isinstance(name, str):
        raise ValueError("name must be a non-empty string")
    if not isinstance(meta, dict) or not str(meta.get("source", "")).strip():
        raise ValueError("meta['source'] (a citation) is required: no data without a source")
    if label not in ("none", "calibration", "validation"):
        raise ValueError(f"label must be 'none', 'calibration' or 'validation', got {label!r}")
    if not columns:
        raise ValueError("no columns given")
    try:
        meta_json = json.loads(json.dumps(meta))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"meta must be JSON-serialisable: {exc}") from exc
    sigma = dict(sigma or {})
    for k in sigma:
        if k not in columns:
            raise ValueError(f"sigma given for unknown column '{k}'")

    ax: Optional[Axis] = None
    n = None
    if axis is not None:
        an, av, au = axis
        ax = Axis(name=an, unit=au, values=_vec(av, f"axis {an}"))
        n = len(ax.values)
    data: Dict[str, Tuple[Any, str]] = {}
    for cn, (vals, unit) in columns.items():
        if ax is None:
            data[cn] = (_scalar(vals, f"column {cn}"), unit)
        else:
            v = _vec(vals, f"column {cn}")
            if len(v) != n:
                raise ValueError(f"column {cn} has {len(v)} values, axis {ax.name} has {n}")
            data[cn] = (v, unit)
    sig: Dict[str, Any] = {}
    for cn, s in sigma.items():
        if ax is None:
            sig[cn] = _scalar(s, f"sigma {cn}")
        else:
            s = _vec(s, f"sigma {cn}")
            if len(s) != n:
                raise ValueError(f"sigma {cn} has {len(s)} values, axis {ax.name} has {n}")
            sig[cn] = s

    payload = {"axis": None if ax is None else ax.model_dump(), "columns": {k: list(v) for k, v in data.items()},
               "sigma": sig, "label": label}
    digest = sha256_text(canonical_json(payload))
    key = JobKey(adapter=f"measured.{name}", adapter_version=ADAPTER_VERSION, engine="measurement",
                 engine_version="n/a", env_lock="measured", params={"meta": meta_json, "data_sha256": digest})
    jid = job_id({"key": key.model_dump(), "fidelity": "screening"})
    job = JobEnvelope(id=jid, key=key, env="local")

    cached = hub.graph.result_for_job(jid)
    if cached and hub.store.has(cached):
        r = hub.store.envelope(cached)
        r.diagnostics["cache"] = "hit"
        return r

    flags = ["measured"] + (["digitized"] if meta.get("digitized") else [])
    qs = []
    for cn, (val, unit) in data.items():
        unc = Uncertainty(kind="sigma", sigma=sig[cn]) if cn in sig else Uncertainty()
        qs.append(Quantity(name=cn, value=val, unit=unit, axes=[ax] if ax is not None else [], uncertainty=unc,
                           flags=["measured"], label=label, source=str(meta["source"])))
    res = ResultEnvelope(job_id=jid, status="ok", quantities=qs, flags=flags, provenance={"measured": meta_json})
    staging = hub.store.staging(jid)
    (staging / "job.json").write_text(job.model_dump_json(indent=1))
    man = Manifest(name=key.adapter, version=ADAPTER_VERSION, env="local", engine="measurement")
    res, _rep = gatekeeper.apply(job, res, man, staging)
    with hub.store.lock():
        res = hub.store.finalize(staging, res)
        hub.graph.commit_result(job, res)
        nid = f"dataset:{name}"
        hub.graph.add_node(nid, "Dataset", name, {"source": meta_json["source"]})
        hub.graph.link(nid, res.id, "produced", {"job": jid})
    return res


def load_csv(path, x_col: int = 0, y_col: int = 1, skip_header: int = 0,
             delimiter: str = ",") -> Tuple[List[float], List[float]]:
    """Two numeric columns of a delimited text file; skips `skip_header` lines, blank lines and '#' comments."""
    xs: List[float] = []
    ys: List[float] = []
    with open(path, newline="") as f:
        for i, row in enumerate(csv.reader(f, delimiter=delimiter)):
            if i < skip_header or not row or not "".join(row).strip() or row[0].lstrip().startswith("#"):
                continue
            try:
                xs.append(float(row[x_col]))
                ys.append(float(row[y_col]))
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{path}: line {i + 1}: cannot read columns {x_col},{y_col}: {exc}") from exc
    return xs, ys
