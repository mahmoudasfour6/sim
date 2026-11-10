"""Compare two quantities (engine vs engine, engine vs measurement) and record it in the graph (section 6.3).

b is the reference. The comparison runs on the points of a's axis that lie inside b's axis range (never
extrapolating). A point with b == 0 under 'relative' counts as 0 when a == 0 too and as infinity otherwise.
"""
from __future__ import annotations

import math
from typing import List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from simws.hub.schema import Quantity
from simws.units import Q_

_EDGE = {"measurement": "validated_against", "engine_pair": "checked_by"}


class Comparison(BaseModel):
    kind: Literal["engine_pair", "measurement"]
    a: str = ""
    b: str = ""
    quantity_a: str
    quantity_b: str
    metric: Literal["relative", "ratio"]
    tolerance: float
    worst: Optional[float]
    worst_at: Optional[float]
    n_points: int
    window: Optional[Tuple[float, float]] = None
    passed: bool
    inconclusive: bool
    note: str = ""
    # amendment A2
    interpolation: Literal["linear", "log", "none"] = "none"       # the method actually used
    measurement_label: Optional[Literal["calibration", "validation"]] = None
    engine_a: str = ""
    engine_b: str = ""
    independent: bool = False


def _convert(vals: np.ndarray, src: str, dst: str, what: str) -> np.ndarray:
    try:
        return np.asarray(Q_(vals, src).to(dst).magnitude, dtype=float)
    except Exception as exc:  # noqa: BLE001  (pint DimensionalityError and friends)
        raise ValueError(f"{what}: cannot convert '{src}' to '{dst}': {exc}") from exc


def _metric(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    out = np.empty(len(a))
    for i, (x, y) in enumerate(zip(a, b)):
        if metric == "relative":
            if y == 0:
                out[i] = 0.0 if x == 0 else math.inf
            else:
                out[i] = abs(x / y - 1.0)
        else:
            if x == 0 or y == 0 or (x > 0) != (y > 0):
                out[i] = math.inf
            else:
                out[i] = max(x / y, y / x)
    out[np.isnan(out)] = math.inf
    return out


def compare(qa: Quantity, qb: Quantity, tolerance: float, metric: str = "relative",
            window: Optional[Tuple[float, float]] = None, min_points: int = 1,
            interpolation: str = "auto") -> Comparison:
    if metric not in ("relative", "ratio"):
        raise ValueError(f"metric must be 'relative' or 'ratio', got {metric!r}")
    if interpolation not in ("auto", "linear", "log"):
        raise ValueError(f"interpolation must be 'auto', 'linear' or 'log', got {interpolation!r}")
    kind = "measurement" if ("measured" in qa.flags or "measured" in qb.flags) else "engine_pair"
    label = None
    if kind == "measurement":                      # A2.2: the measured side's label (b's if b is measured)
        lab = qb.label if "measured" in qb.flags else qa.label
        label = lab if lab in ("calibration", "validation") else None
    used = "none"
    ya = qa.array()
    yb = _convert(qb.array(), qb.unit, qa.unit, f"{qb.name} -> {qa.unit}")   # also checks the dimension

    if not qa.axes and not qb.axes:
        va = np.atleast_1d(ya).astype(float)
        vb = np.atleast_1d(yb).astype(float)
        if va.size != 1 or vb.size != 1:
            raise ValueError("quantities without an axis must be scalars")
        w = _metric(va, vb, metric)
        pa, pb, xs = va, vb, np.array([math.nan])
        worst_at = None
        mask_w = w
    else:
        if len(qa.axes) != 1 or len(qb.axes) != 1:
            raise ValueError("both quantities must be scalars or have exactly one axis")
        axa, axb = qa.axes[0], qb.axes[0]
        xa = np.asarray(axa.values, dtype=float)
        xb = _convert(np.asarray(axb.values, dtype=float), axb.unit, axa.unit, f"axis {axb.name} -> {axa.unit}")
        if ya.shape != xa.shape or yb.shape != xb.shape:
            raise ValueError("quantity length does not match its axis")
        order = np.argsort(xb, kind="stable")
        xb, yb = xb[order], yb[order]
        if len(xb) == 0:
            sel = np.zeros(len(xa), dtype=bool)
        else:
            eps = 1e-12 * max(abs(xb[0]), abs(xb[-1]), xb[-1] - xb[0], 0.0)   # unit-conversion rounding only
            sel = (xa >= xb[0] - eps) & (xa <= xb[-1] + eps)
        if window is not None:
            sel &= (xa >= window[0]) & (xa <= window[1])
        xs, pa = xa[sel], ya[sel]
        log_ok = bool(len(yb)) and not (np.any(yb == 0) or np.any(np.sign(yb) != np.sign(yb[0])))
        if interpolation == "auto":                # A2.1: a straight line through an exponential is far above it
            interpolation = "log" if log_ok else "linear"
        used = interpolation
        if interpolation == "log":
            if len(yb) and not log_ok:
                raise ValueError("log interpolation needs every reference value non-zero and of one sign")
            pb = np.sign(yb[0]) * np.exp(np.interp(xs, xb, np.log(np.abs(yb)))) if len(xs) else np.array([])
        else:
            pb = np.interp(xs, xb, yb) if len(xs) else np.array([])
        mask_w = _metric(pa, pb, metric)
        worst_at = None

    n = len(pa)
    if n == 0:
        worst = None
    else:
        i = int(np.argmax(mask_w))
        worst = float(mask_w[i])
        if qa.axes:
            worst_at = float(xs[i])
    inconclusive = n < min_points
    passed = (not inconclusive) and worst is not None and worst <= tolerance
    note = f"{n} point(s) compared, fewer than min_points={min_points}" if inconclusive else ""
    return Comparison(kind=kind, quantity_a=qa.name, quantity_b=qb.name, metric=metric, tolerance=tolerance,
                      worst=worst, worst_at=worst_at, n_points=n,
                      window=None if window is None else (float(window[0]), float(window[1])),
                      passed=bool(passed), inconclusive=bool(inconclusive), note=note,
                      interpolation=used, measurement_label=label)      # independent: set by compare_results


def compare_results(hub, rid_a: str, name_a: str, rid_b: str, name_b: Optional[str] = None, **kw) -> Comparison:
    """compare(quantity name_a of result rid_a, quantity name_b (default name_a) of rid_b, **kw)."""
    ra, rb = hub.result(rid_a), hub.result(rid_b)
    c = compare(ra.q(name_a), rb.q(name_b or name_a), **kw)
    ea, eb = _engine_of(hub, rid_a), _engine_of(hub, rid_b)
    # A2.3: only a different engine is a second opinion; a sketch is nobody's opinion
    independent = c.kind == "measurement" or (bool(ea) and bool(eb) and ea != eb
                                              and "sketch" not in ra.flags and "sketch" not in rb.flags)
    return c.model_copy(update={"a": rid_a, "b": rid_b, "engine_a": ea, "engine_b": eb,
                                "independent": bool(independent)})


def _engine_of(hub, rid: str) -> str:
    try:
        return str(hub.store.job(rid).key.engine or "")
    except Exception:  # noqa: BLE001  (no stored job envelope: engine unknown, so never independent)
        return ""


def record(hub, cmp: Comparison) -> None:
    """Store the comparison on the edge a -> b ('validated_against' or 'checked_by')."""
    if not cmp.a or not cmp.b:
        raise ValueError("record needs both result ids (a and b)")
    etype = _EDGE[cmp.kind]
    d = cmp.model_dump()
    with hub.store.lock():
        old = [x for dst, _t, data in hub.graph.edges_from(cmp.a, etype) if dst == cmp.b
               for x in data.get("comparisons", [])]
        kept = [x for x in old if (x.get("quantity_a"), x.get("quantity_b")) != (cmp.quantity_a, cmp.quantity_b)]
        hub.graph.link(cmp.a, cmp.b, etype, {"comparisons": kept + [d]})
        hub.graph.event("compared", cmp.a, {"b": cmp.b, "kind": cmp.kind, "quantity_a": cmp.quantity_a,
                                            "quantity_b": cmp.quantity_b, "passed": cmp.passed})


def records_for(hub, result_id: str) -> List[dict]:
    """Every stored comparison dict in which the result is a or b."""
    import json
    rows = hub.graph.db.execute(
        "SELECT data FROM edges WHERE type IN ('validated_against','checked_by') AND (src=? OR dst=?) "
        "ORDER BY src, dst, type", (result_id, result_id)).fetchall()
    out = []
    for (data,) in rows:
        for c in json.loads(data or "{}").get("comparisons", []):
            if result_id in (c.get("a"), c.get("b")):
                out.append(c)
    return out
