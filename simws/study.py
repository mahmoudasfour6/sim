"""Small parameter-study layer over the planner, the engine adapters (device.*, implant.*) and compare (architecture v1, section 6.5).

    st = Study(hub, problem)                                   # engines=None -> [plan(problem).primary]
    r = st.sweep({"models.temperature_K": [250, 300, 350]})   # full factorial, last key varies fastest
    r.table("dc.I.anode"); r.compare_engines("dc.I.anode", "devsim", "charon", tolerance=0.05)
"""
from __future__ import annotations

import itertools
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from simws.hub.schema import Quantity


class StudyResult:
    def __init__(self, hub, points: List[dict]):
        self.hub = hub
        self.points = points

    def quantity(self, name: str, engine: Optional[str] = None) -> List[Tuple[dict, Quantity]]:
        """(point, quantity) for every point with status ok (and the given engine) in sweep order."""
        out = []
        for p in self.points:
            if p["status"] != "ok" or not p["result_id"] or (engine is not None and p["engine"] != engine):
                continue
            out.append((p, self.hub.result(p["result_id"]).q(name)))
        return out

    def table(self, name: str) -> str:
        keys: List[str] = []
        for p in self.points:
            for k in p["changes"]:
                if k not in keys:
                    keys.append(k)
        head = keys + ["engine", "status", "trust", "in_range", name]
        body = []
        for p in self.points:
            val = ""
            if p["status"] == "ok" and p["result_id"]:
                try:
                    q = self.hub.result(p["result_id"]).q(name)
                    v = q.value
                    if isinstance(v, list):
                        a = q.array().ravel()
                        val = f"[{a[0]:.4g} .. {a[-1]:.4g}] n={a.size} {q.unit}" if a.size else f"[] {q.unit}"
                    else:
                        val = f"{float(v):.6g} {q.unit}"
                except KeyError:
                    val = "(no such quantity)"
            elif p["reasons"]:
                val = "; ".join(p["reasons"])[:80]
            body.append([str(p["changes"].get(k, "")) for k in keys]
                        + [p["engine"], p["status"], str(p["trust"] or "-"),
                           "-" if p["in_range"] is None else str(p["in_range"]), val])
        w = [max(len(h), *(len(b[i]) for b in body)) if body else len(h) for i, h in enumerate(head)]
        lines = ["  ".join(h.ljust(w[i]) for i, h in enumerate(head)), "  ".join("-" * x for x in w)]
        lines += ["  ".join(b[i].ljust(w[i]) for i in range(len(head))) for b in body]
        return "\n".join(lines)

    def compare_engines(self, name: str, a: str, b: str, tolerance: float, record: bool = True, **kw) -> list:
        """Compare quantity `name` of engine a against engine b (reference) for every combination where both ran ok."""
        from simws import compare as C
        by: Dict[str, Dict[str, dict]] = {}
        order: List[str] = []
        for p in self.points:
            k = json.dumps(p["changes"], sort_keys=True, default=str)
            if k not in by:
                by[k] = {}
                order.append(k)
            by[k][p["engine"]] = p
        out = []
        for k in order:
            pa, pb = by[k].get(a), by[k].get(b)
            if not pa or not pb or pa["status"] != "ok" or pb["status"] != "ok":
                continue
            c = C.compare_results(self.hub, pa["result_id"], name, pb["result_id"], name, tolerance=tolerance, **kw)
            if not c.note:
                c = c.model_copy(update={"note": f"changes={json.dumps(pa['changes'], default=str)}"})
            if record:
                C.record(self.hub, c)
            out.append(c)
        return out


class Study:
    def __init__(self, hub, problem, engines: Optional[Sequence[str]] = None, fidelity: str = "screening"):
        self.hub = hub
        self.problem = problem
        self.engines = list(engines) if engines is not None else None
        self.fidelity = fidelity

    def _engines(self) -> List[str]:
        if self.engines is not None:
            return self.engines
        from simws.engines import planner
        plan = planner.plan(self.problem)
        if plan.primary is None:
            raise ValueError("no engine supports this problem:\n" + plan.summary())
        return [plan.primary]

    def sweep(self, over: Dict[str, Sequence]) -> StudyResult:
        from simws import engines as E
        from simws.hub import trust as T
        engs = self._engines()
        keys = list(over)
        points: List[dict] = []
        for combo in itertools.product(*(list(over[k]) for k in keys)):
            changes = dict(zip(keys, combo))
            p = self.problem.with_changes(changes)
            for eng in engs:
                pt = {"changes": changes, "engine": eng, "status": "", "result_id": None, "reasons": [],
                      "trust": None, "in_range": None}
                reasons = E.unsupported(eng, p)
                if reasons:
                    pt.update(status="refused", reasons=list(reasons))
                else:
                    res = self.hub.submit(E.adapter_name(eng), params={"problem": p.model_dump(mode="json")},
                                          fidelity=self.fidelity)
                    pt.update(status=res.status, result_id=res.id or None,
                              reasons=[res.message] if (res.status != "ok" and res.message) else [])
                    if res.id:
                        t = T.assess_stored(self.hub, res.id)
                        pt.update(trust=t.level, in_range=t.in_range)
                points.append(pt)
        return StudyResult(self.hub, points)
