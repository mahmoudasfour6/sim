"""Store maintenance: retention pruning and the disk report (run nightly by tools/nightly.sh).

    python -m simws.hub.maintain            # prune validated until_validated fields, write a disk report
    python -m simws.hub.maintain --dry-run

A result's until_validated fields (large meshes, transient fields) are deleted once it is validated: at least one
job consumed it and every consumer's result is ok and not stale. The envelope and all quantities stay; deleted
fields are regenerated from the stored job key if they are ever needed. The graph records a 'pruned' event.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from simws.hub.graph import Graph
from simws.hub.store import Store


def validated(g: Graph, rid: str) -> bool:
    consumers = [s for s, _, _ in g.edges_to(rid, "consumed")]
    consumers += [s for s, _, _ in g.db.execute(
        "SELECT src, type, data FROM edges WHERE type='consumed' AND dst LIKE ?", (f"{rid}:%",)).fetchall()]
    if not consumers:
        return False
    for job in set(consumers):
        r = g.db.execute("SELECT result_id, status FROM jobs WHERE job_id=?", (job,)).fetchone()
        if not r or r[1] != "ok" or g.is_superseded(r[0]):
            return False
    return True


def prune_all(store: Store, g: Graph, dry_run=False) -> dict:
    freed, pruned = 0, []
    for (rid,) in g.db.execute("SELECT id FROM nodes WHERE type='Result'").fetchall():
        if not store.has(rid):
            continue
        env = store.envelope(rid)
        if not any(f.retention == "until_validated" for f in env.fields):
            continue
        if not validated(g, rid):
            continue
        if dry_run:
            pruned.append(rid); continue
        with store.lock():
            b = store.prune(rid, validated=True)
            if b:
                g.event("pruned", rid, {"bytes": b})
        if b:
            freed += b; pruned.append(rid)
    return {"pruned": pruned, "bytes_freed": freed}


def report(store: Store, g: Graph) -> dict:
    d = store.disk_report()
    counts = dict(g.db.execute("SELECT type, COUNT(*) FROM nodes GROUP BY type").fetchall())
    status = dict(g.db.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall())
    return {"date": dt.datetime.now().isoformat(timespec="seconds"), **d, "nodes": counts, "jobs": status}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m simws.hub.maintain")
    ap.add_argument("--store"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    st = Store(a.store)
    g = Graph(st.root / "graph.sqlite")
    p = prune_all(st, g, a.dry_run)
    rep = {**report(st, g), "prune": p, "dry_run": a.dry_run}
    out = st.root / "reports"; out.mkdir(exist_ok=True)
    (out / f"disk_{dt.date.today().isoformat()}.json").write_text(json.dumps(rep, indent=1))
    gb = rep["store_bytes"] / 1e9
    print(f"store {gb:.2f} GB, disk {100 * rep['disk_used_fraction']:.0f}% used, pruned {len(p['pruned'])} result(s), "
          f"freed {p['bytes_freed'] / 1e6:.1f} MB" + ("  WARNING: disk above 80%" if rep["warning"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
