"""Flat, searchable list of committed results (architecture v1, section 6.4)."""
from __future__ import annotations

import fnmatch
from typing import Any, Dict, List, Optional, Tuple


def _temperature(params: Dict[str, Any]) -> Optional[float]:
    try:
        t = params["problem"]["models"]["temperature_K"]
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            return float(t)
    except (KeyError, TypeError):
        pass
    t = params.get("T")
    if isinstance(t, dict):
        t = t.get("value")
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        return float(t)
    return None


def rows(hub, adapter: Optional[str] = None, status: Optional[str] = "ok", quantity: Optional[str] = None,
         flag: Optional[str] = None, fidelity: Optional[str] = None,
         temperature: Optional[Tuple[float, float]] = None, trust: Optional[str] = None) -> List[dict]:
    """One row per committed result, newest first. adapter: fnmatch pattern; quantity, flag: substring matches;
    temperature: inclusive (lo, hi) in K; trust: a level from simws.hub.trust.LEVELS."""
    from simws import compare
    from simws.hub import trust as T
    recs = hub.graph.db.execute(
        "SELECT job_id, result_id, status, adapter, fidelity, finished FROM jobs").fetchall()
    by_result: Dict[str, list] = {}
    for r in recs:
        by_result.setdefault(r[1], []).append(r)
    out: List[dict] = []
    for rid, group in by_result.items():
        if not rid or not hub.store.has(rid):
            continue
        res = hub.store.envelope(rid)
        row = next((g for g in group if g[0] == res.job_id), group[0])      # prefer the producing job
        jid, _rid, st, ad, fid, fin = row
        if status is not None and st != status:
            continue
        if adapter is not None and not fnmatch.fnmatchcase(ad or "", adapter):
            continue
        if fidelity is not None and fid != fidelity:
            continue
        names = [q.name for q in res.quantities]
        if quantity is not None and not any(quantity in n for n in names):
            continue
        if flag is not None and not any(flag in f for f in res.flags):
            continue
        key = hub.store.job(rid).key
        temp = _temperature(key.params)
        if temperature is not None and (temp is None or not (temperature[0] <= temp <= temperature[1])):
            continue
        t = T.assess(res, compare.records_for(hub, rid))
        if trust is not None and t.level != trust:
            continue
        out.append({"result_id": rid, "job_id": jid, "adapter": ad, "engine": key.engine, "status": st,
                    "fidelity": fid, "finished": fin, "flags": list(res.flags), "quantities": names,
                    "temperature_K": temp, "trust": t.level, "in_range": t.in_range})
    out.sort(key=lambda r: (r["finished"] or "", r["result_id"]), reverse=True)
    return out


def table(rows: List[dict]) -> str:
    """Plain-text table of catalogue rows."""
    cols = ["result_id", "adapter", "engine", "status", "fidelity", "finished", "temperature_K", "trust",
            "in_range", "flags", "quantities"]

    def cell(r, c):
        v = r.get(c)
        if c == "result_id" and isinstance(v, str):
            return v[:10]
        if isinstance(v, list):
            s = ",".join(map(str, v))
            return s if len(s) <= 40 else s[:37] + "..."
        return "-" if v is None else str(v)

    body = [[cell(r, c) for c in cols] for r in rows]
    w = [max(len(c), *(len(b[i]) for b in body)) if body else len(c) for i, c in enumerate(cols)]
    lines = ["  ".join(c.ljust(w[i]) for i, c in enumerate(cols)), "  ".join("-" * x for x in w)]
    lines += ["  ".join(b[i].ljust(w[i]) for i in range(len(cols))) for b in body]
    return "\n".join(lines)
