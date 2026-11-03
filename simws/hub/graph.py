"""Knowledge graph (interface specification: knowledge graph model).

SQLite with write-ahead logging; only the committer writes (inside Store.lock()). Node types: Job, Result,
Quantity, MaterialLaw, Dataset, Model, Study. Edge types: produced, consumed, depends_on, uses_model,
validated_against, calibrated_on, checked_by, supersedes. An append-only event ledger records committed,
rejected, stale and pruned events.

    g = Graph(store.root / "graph.sqlite")
    g.result_for_job(job_id)          # cached result ID or None
    g.quantity(result_id, "hbt.beta")
    g.is_stale(node_id); g.to_networkx()
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

from simws.hub.schema import JobEnvelope, ResultEnvelope, now_utc

NODE_TYPES = {"Job", "Result", "Quantity", "MaterialLaw", "Dataset", "Model", "Study"}
EDGE_TYPES = {"produced", "consumed", "depends_on", "uses_model", "validated_against", "calibrated_on",
              "checked_by", "supersedes", "equivalent_to"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT, data TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS edges (src TEXT NOT NULL, dst TEXT NOT NULL, type TEXT NOT NULL, data TEXT,
                                  PRIMARY KEY (src, dst, type));
CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, type TEXT, ref TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, result_id TEXT, status TEXT, adapter TEXT, member TEXT,
                                 fidelity TEXT, finished TEXT);
CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS nodes_name ON nodes(name);
"""


def qid(result_id: str, name: str, member: str = "base") -> str:
    return f"{result_id}:{name}" + ("" if member == "base" else f"@{member}")


class Graph:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(_SCHEMA)

    # ------------------------------------------------------------------------------------------- writes
    def _node(self, cur, nid, ntype, name=None, data=None):
        assert ntype in NODE_TYPES, ntype
        cur.execute("INSERT OR IGNORE INTO nodes VALUES (?,?,?,?,?)",
                    (nid, ntype, name, json.dumps(data or {}, default=str), now_utc()))

    def _edge(self, cur, src, dst, etype, data=None):
        assert etype in EDGE_TYPES, etype
        cur.execute("INSERT OR REPLACE INTO edges VALUES (?,?,?,?)", (src, dst, etype, json.dumps(data or {})))

    def event(self, etype: str, ref: str, data=None, cur=None):
        c = cur or self.db
        c.execute("INSERT INTO events (ts, type, ref, data) VALUES (?,?,?,?)",
                  (now_utc(), etype, ref, json.dumps(data or {}, default=str)))

    def commit_result(self, job: JobEnvelope, res: ResultEnvelope, input_nodes: Iterable[str] = (),
                      supersedes: Optional[str] = None):
        """Record one committed result in a single transaction (caller holds the store lock)."""
        cur = self.db.cursor()
        cur.execute("BEGIN")
        try:
            self._node(cur, job.id, "Job", job.key.adapter, {"key": job.key.model_dump(), "fidelity": job.fidelity})
            self._node(cur, res.id, "Result", job.key.adapter,
                       {"status": res.status, "flags": res.flags, "job": job.id, "message": res.message})
            cur.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?)",
                        (job.id, res.id, res.status, job.key.adapter, job.key.member, job.fidelity,
                         res.finished_utc))
            for src in input_nodes:
                self._edge(cur, job.id, src, "consumed")
            for q in res.quantities:
                nid = qid(res.id, q.name, q.member)
                self._node(cur, nid, "Quantity", q.name,
                           {"unit": q.unit, "member": q.member, "flags": q.flags,
                            "scalar": q.value if isinstance(q.value, (int, float)) else None,
                            "uncertainty": q.uncertainty.kind})
                self._edge(cur, res.id, nid, "produced")
            for e in res.physics_edges:
                src = qid(res.id, e.output)
                dst = e.input if ":" in e.input else qid(res.id, e.input)
                self._edge(cur, src, dst, "depends_on",
                           {"mechanism": e.mechanism, "sensitivity": e.sensitivity, "method": e.method})
            if supersedes:
                self._edge(cur, res.id, supersedes, "supersedes")
                self.event("stale", supersedes, {"superseded_by": res.id}, cur)
            self.event("committed" if res.status == "ok" else "failed", res.id,
                       {"job": job.id, "adapter": job.key.adapter, "status": res.status}, cur)
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def alias_job(self, new_job, result_id: str, old_job_id: str):
        """Serve an existing result for a new job key (canary-verified relinking after an upgrade)."""
        cur = self.db.cursor()
        cur.execute("BEGIN")
        try:
            self._node(cur, new_job.id, "Job", new_job.key.adapter,
                       {"key": new_job.key.model_dump(), "fidelity": new_job.fidelity, "alias_of": old_job_id})
            cur.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?)",
                        (new_job.id, result_id, "ok", new_job.key.adapter, new_job.key.member, new_job.fidelity,
                         now_utc()))
            self._edge(cur, new_job.id, old_job_id, "equivalent_to")
            self.event("relinked", result_id, {"new_job": new_job.id, "old_job": old_job_id}, cur)
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def link(self, src: str, dst: str, etype: str, data=None):
        cur = self.db.cursor()
        self._edge(cur, src, dst, etype, data)

    def add_node(self, nid: str, ntype: str, name=None, data=None):
        cur = self.db.cursor()
        self._node(cur, nid, ntype, name, data)

    # ------------------------------------------------------------------------------------------- reads
    def result_for_job(self, job_id: str) -> Optional[str]:
        r = self.db.execute("SELECT result_id, status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return r[0] if r and r[1] == "ok" else None

    def job_status(self, job_id: str) -> Optional[str]:
        r = self.db.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return r[0] if r else None

    def edges_from(self, src: str, etype: Optional[str] = None) -> List[tuple]:
        q = "SELECT dst, type, data FROM edges WHERE src=?" + (" AND type=?" if etype else "")
        return [(d, t, json.loads(x)) for d, t, x in self.db.execute(q, (src, etype) if etype else (src,))]

    def edges_to(self, dst: str, etype: Optional[str] = None) -> List[tuple]:
        q = "SELECT src, type, data FROM edges WHERE dst=?" + (" AND type=?" if etype else "")
        return [(s, t, json.loads(x)) for s, t, x in self.db.execute(q, (dst, etype) if etype else (dst,))]

    def events(self, since: int = 0) -> List[dict]:
        return [{"seq": s, "ts": ts, "type": t, "ref": r, "data": json.loads(d)}
                for s, ts, t, r, d in self.db.execute("SELECT * FROM events WHERE seq>? ORDER BY seq", (since,))]

    def is_superseded(self, rid: str) -> bool:
        return bool(self.db.execute("SELECT 1 FROM edges WHERE dst=? AND type='supersedes'", (rid,)).fetchone())

    def _result_of(self, nid: str) -> str:
        return nid.split(":", 1)[0]

    def ancestors(self, nid: str) -> List[str]:
        """Results upstream of a node along consumed (via its job) and depends_on edges."""
        seen, stack, out = set(), [nid], []
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            rid = self._result_of(n)
            row = self.db.execute("SELECT data FROM nodes WHERE id=?", (rid,)).fetchone()
            if row:
                job = json.loads(row[0]).get("job")
                if job:
                    for d, _, _ in self.edges_from(job, "consumed"):
                        out.append(self._result_of(d)); stack.append(d)
            for d, _, _ in self.edges_from(n, "depends_on"):
                if self._result_of(d) != rid:
                    out.append(self._result_of(d))
                stack.append(d)
        return sorted(set(out) - {self._result_of(nid)})

    def is_stale(self, nid: str) -> bool:
        return any(self.is_superseded(r) for r in self.ancestors(nid)) or self.is_superseded(self._result_of(nid))

    def to_networkx(self):
        import networkx as nx
        G = nx.MultiDiGraph()
        for nid, t, name, data, _ in self.db.execute("SELECT * FROM nodes"):
            G.add_node(nid, type=t, name=name, **json.loads(data or "{}"))
        for s, d, t, data in self.db.execute("SELECT * FROM edges"):
            G.add_edge(s, d, type=t, **json.loads(data or "{}"))
        return G
