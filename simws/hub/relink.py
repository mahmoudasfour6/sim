"""Relink cached results after an environment or key-format upgrade (design review: canary relinking).

When an environment lock or the job-key recipe changes, every job gets a new ID and the cache would miss
everything. Instead, per adapter, a few committed jobs (canaries) are rerun under the new key; if every canary
reproduces its stored quantities (relative tolerance rtol), the remaining results of that adapter are linked to
their new IDs (graph edge 'equivalent_to', event 'relinked') and served from the cache. If any canary differs,
nothing of that adapter is linked: those results are recomputed when next requested.

    python -m simws.hub.relink                 # after the installer re-locks the environments
    python -m simws.hub.relink --canaries 3 --rtol 1e-9 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

import numpy as np

from simws.hub import adapter as A


def _close(a, b, rtol) -> bool:
    try:
        x, y = np.asarray(a, float), np.asarray(b, float)
    except (TypeError, ValueError):
        return a == b
    if x.shape != y.shape:
        return False
    return bool(np.all(np.abs(x - y) <= rtol * np.maximum(np.abs(x), np.abs(y)) + 1e-300))


def compare(old, new, rtol) -> List[str]:
    """Differences between two result envelopes' quantities (empty = reproduced)."""
    diffs = []
    ov = {(q.name, q.member): q for q in old.quantities}
    nv = {(q.name, q.member): q for q in new.quantities}
    if set(ov) != set(nv):
        diffs.append(f"quantity sets differ: {sorted(set(ov) ^ set(nv))[:5]}")
    for k in set(ov) & set(nv):
        a, b = ov[k], nv[k]
        if a.unit != b.unit:
            diffs.append(f"{k}: unit {a.unit} vs {b.unit}")
        elif not _close(a.value, b.value, rtol):
            diffs.append(f"{k}: values differ beyond rtol {rtol:g}")
    return diffs


def stale_jobs(hub) -> Dict[str, List[tuple]]:
    """{adapter: [(old job, old result id, new job envelope), ...]} for ok results whose key has changed."""
    out: Dict[str, List[tuple]] = {}
    known = set(A.names())
    rows = hub.graph.db.execute("SELECT job_id, result_id, adapter FROM jobs WHERE status='ok'").fetchall()
    for jid, rid, name in rows:
        if name not in known or not hub.store.has(rid):
            continue
        old = hub.store.job(rid)
        if old.id != jid:
            continue                                  # already an alias row
        new = hub.make_job(name, params=old.key.params, inputs=old.key.inputs, member=old.key.member,
                           seed=old.key.seed, fidelity=old.fidelity, budget=old.budget, priority=old.priority)
        if new.id != old.id and not hub.graph.result_for_job(new.id):
            out.setdefault(name, []).append((old, rid, new))
    return out


def relink(hub, canaries: int = 2, rtol: float = 1e-9, dry_run: bool = False, log=print) -> dict:
    report = {}
    for name, items in sorted(stale_jobs(hub).items()):
        cm = A.manifest(name).cost_model
        tol = float(cm.get("relink_rtol", rtol))
        k = int(min(canaries, cm.get("relink_canaries", canaries)))
        picks, rest = items[:k], items[k:]
        ok, notes = True, []
        for old, rid, new in picks:
            if dry_run:
                continue
            res = hub.submit(name, params=old.key.params, inputs=old.key.inputs, member=old.key.member,
                             seed=old.key.seed, fidelity=old.fidelity, budget=old.budget, force=True)
            d = ["canary failed: " + res.status] if res.status != "ok" else compare(hub.store.envelope(rid), res, tol)
            if d:
                ok = False
                notes.append({"old_result": rid, "differences": d[:5]})
        linked = 0
        if ok and not dry_run:
            for old, rid, new in rest:
                hub.graph.alias_job(new, rid, old.id)
                linked += 1
        report[name] = {"stale": len(items), "canaries": len(picks), "reproduced": ok, "linked": linked,
                        "notes": notes, "rtol": tol}
        log(f"relink {name}: {len(items)} changed keys, {len(picks)} canaries "
            f"{'reproduced' if ok else 'DIFFER'}; linked {linked}")
    return report


def main(argv=None):
    from simws.hub.jobs import Hub
    ap = argparse.ArgumentParser(prog="python -m simws.hub.relink")
    ap.add_argument("--store"); ap.add_argument("--canaries", type=int, default=2)
    ap.add_argument("--rtol", type=float, default=1e-9); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rep = relink(Hub(a.store), a.canaries, a.rtol, a.dry_run)
    print(json.dumps(rep, indent=1))
    return 0 if all(v["reproduced"] for v in rep.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
