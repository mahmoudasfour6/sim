"""The 5AM validation datasets (contract A23.1): `simws/data/validation/catalog.json` names every dataset of the
validation plan with its paper, figure, device, conditions and role; the traced ones point at their data file.

    from simws.validation import catalog
    catalog.entry("luo2011.fig5_7a")            # the catalog row
    catalog.curve("luo2011.fig5_7a", T_K=300)  # one traced curve (dict of arrays and the source)
    catalog.role("luo2011.fig5_7a", T_K=300)   # 'calibration'
    catalog.ingest(hub, "luo2011.fig5_7a", T_K=393, axis=("VBE", "VBE_V", "V"),
                   columns={"IC": ("IC_mA", "mA")})   # simws.measured.ingest with the catalog's role as label
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np

DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "validation")


def _load(name: str) -> dict:
    with open(os.path.join(DIR, name)) as f:
        return json.load(f)


def catalog() -> dict:
    return _load("catalog.json")


def entry(dataset: str) -> dict:
    for d in catalog()["datasets"]:
        if d["id"] == dataset:
            return d
    raise KeyError(f"no dataset {dataset!r} in the catalog")


def role(dataset: str, T_K: Optional[float] = None) -> str:
    e = entry(dataset)
    if T_K is not None:
        r = e.get("curve_roles", {}).get(f"{float(T_K):.1f}")
        if r:
            return r
    return e["role"]


def figure(dataset: str) -> Tuple[dict, dict]:
    """(the figure's traced data, the file's source block)."""
    e = entry(dataset)
    if not e.get("file"):
        raise ValueError(f"dataset {dataset!r} is not traced yet (no data file)")
    d = _load(e["file"])
    return d[e["key"]], d["source"]


def curve(dataset: str, T_K: Optional[float] = None) -> dict:
    """One curve of a figure: the figure's own arrays, or with T_K the curve of that temperature."""
    fig, src = figure(dataset)
    if isinstance(fig.get("curves"), dict) and T_K is not None:
        raise ValueError(f"{dataset}: its curves are not per temperature; T_K does not apply")
    if "curves" in fig and T_K is not None:
        for c in fig["curves"]:
            if abs(float(c.get("T_K", -1)) - float(T_K)) < 0.05:
                out = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in c.items()}
                out["source"] = src
                return out
        raise KeyError(f"{dataset}: no curve at {T_K} K")
    out = {k: (np.asarray(v, float) if isinstance(v, list) and v and isinstance(v[0], (int, float)) else v)
           for k, v in fig.items()}
    out["source"] = src
    return out


def ingest(hub, dataset: str, axis: Optional[Tuple[str, str, str]], columns: Dict[str, Tuple[str, str]],
           T_K: Optional[float] = None, rows=None):
    """simws.measured.ingest of one traced curve: axis = (axis name, the curve's key, unit); columns = {quantity
    name: (the curve's key, unit)}; rows (optional) a boolean mask or index array on the curve's points. The label
    is the catalog's role; trend and input datasets are refused (A23.1)."""
    from simws.measured import ingest as _ingest
    r = role(dataset, T_K)
    if r not in ("calibration", "validation"):
        raise ValueError(f"{dataset}: role {r!r} is not ingested (only calibration and validation data are)")
    fig, _ = figure(dataset)
    if isinstance(fig.get("curves"), list) and T_K is None:
        raise ValueError(f"{dataset} has one curve per temperature: give T_K")
    pointwise_T = "curves" not in fig and isinstance(fig.get("T_K"), list)   # one series over temperature
    c = curve(dataset, None if pointwise_T else T_K)
    sel = slice(None) if rows is None else rows
    if pointwise_T and rows is None:
        # each point carries its own temperature: with T_K only that temperature's points, labelled by its role;
        # without, only the points of the default role (A23.1: a point is never ingested under another's role)
        Ts = np.asarray(c["T_K"], float)
        special = [float(t) for t in entry(dataset).get("curve_roles", {})]
        if T_K is not None:
            sel = np.abs(Ts - float(T_K)) < 0.5
        else:
            sel = ~np.any([np.abs(Ts - t) < 0.5 for t in special], axis=0) if special else slice(None)
        if isinstance(sel, np.ndarray) and not sel.any():
            raise ValueError(f"{dataset}: no traced point at {T_K} K")
    ax = None
    if axis is not None:
        ax = (axis[0], [float(v) for v in np.asarray(c[axis[1]], float)[sel]], axis[2])
    cols = {name: ([float(v) for v in np.atleast_1d(np.asarray(c[key], float))[sel]], unit)
            for name, (key, unit) in columns.items()}
    e = entry(dataset)
    meta = {"source": f"{e['paper']}; read from {e['read_from']} ({c['source']['cite']})",
            "pdf_sha256": c["source"]["pdf_sha256"], "dataset": dataset, "device": e["device"],
            "conditions": e["conditions"], "digitized": True}
    if T_K is not None:
        meta["T_K"] = float(T_K)
    name = dataset if T_K is None else f"{dataset}.T{float(T_K):g}"
    return _ingest(hub, name, ax, cols, meta, label=r)


def load(dataset: str, T_K: Optional[float] = None) -> dict:
    """A23.1's name for `curve`."""
    return curve(dataset, T_K)


# ---------------------------------------------------------------------------------------------- A25.1: dose series
def series(dataset: str, path: str) -> dict:
    """One dose series of a total-dose figure, by its path in the figure's `series` ('VBE_0.6V',
    'proton_63MeV/forward', '300.0'): arrays dose_rad_SiO2 and value (and min / max where error bars were drawn)."""
    fig, src = figure(dataset)
    s = fig.get("series")
    if not isinstance(s, dict):
        raise ValueError(f"{dataset} has no dose series")
    for k in path.split("/"):
        if k not in s:
            raise KeyError(f"{dataset}: no series {path!r}")
        s = s[k]
    out = {k: (np.asarray([np.nan if v is None else v for v in s[k]], float) if isinstance(s[k], list) else s[k])
           for k in s}
    out["source"] = src
    return out


def point_roles(dataset: str, path: str):
    """The role of every point of a dose series (A25.1): 'calibration' for the doses the catalog row names under
    calibration_points for that series, the row's role for the rest."""
    e = entry(dataset)
    s = series(dataset, path)
    roles = np.array([e["role"]] * len(s["dose_rad_SiO2"]), dtype=object)
    cp = e.get("calibration_points")
    if cp and cp.get("series") == path:
        for d in cp["dose_rad_SiO2"]:
            roles[np.abs(s["dose_rad_SiO2"] / float(d) - 1.0) < 0.02] = "calibration"
    return roles


def ingest_series(hub, dataset: str, path: str, quantity: str = "dIB_over_IB0"):
    """simws.measured.ingest of a dose series, one measured set per role (never a point under another's role):
    returns {role: record}. Trend and input datasets are refused, as `ingest`."""
    from simws.measured import ingest as _ingest
    e = entry(dataset)
    if e["role"] not in ("calibration", "validation"):
        raise ValueError(f"{dataset}: role {e['role']!r} is not ingested (only calibration and validation data are)")
    s = series(dataset, path)
    r = point_roles(dataset, path)
    out = {}
    for lab in ("calibration", "validation"):
        m = r == lab
        if not m.any():
            continue
        meta = {"source": f"{e['paper']}; read from {e['read_from']} ({s['source']['cite']})",
                "pdf_sha256": s["source"]["pdf_sha256"], "dataset": dataset, "series": path, "device": e["device"],
                "conditions": e["conditions"], "digitized": True}
        name = f"{dataset}.{path.replace('/', '.')}.{lab}"
        out[lab] = _ingest(hub, name, ("dose", [float(v) for v in s["dose_rad_SiO2"][m]], "rad"),
                           {quantity: ([float(v) for v in s["value"][m]], "dimensionless")}, meta, label=lab)
    return out
