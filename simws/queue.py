"""Overnight job queue (interface specification: scheduling).

    python -m simws.queue add devsim.hbt_gummel --params '{"VBE": [[0.5, 0.6, 0.7], "V"], "T": [350, "K"]}' --priority overnight
    python -m simws.queue run --until 07:00          # run queued jobs, start none after 07:00 (today or tomorrow)
    python -m simws.queue run --for 8h --max-jobs 50
    python -m simws.queue status

The queue lives next to the store (queue.sqlite). Each entry is one hub.submit call, so a job is atomic: its result
is committed in one transaction or not at all. A run interrupted by a crash or shutdown leaves entries 'running'
with a dead process id; the next `run` puts them back to 'queued' (job-level checkpoint) and the hub cache skips
everything already committed. Order: priority (interactive, batch, overnight), then submission order.
Quantities given as [value, unit] pairs in JSON are read as (value, unit) quantities.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

PRIORITY = {"interactive": 0, "batch": 1, "overnight": 2}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queue (
  id INTEGER PRIMARY KEY AUTOINCREMENT, adapter TEXT NOT NULL, params TEXT, inputs TEXT, member TEXT, seed INTEGER,
  fidelity TEXT, budget TEXT, priority TEXT, status TEXT, result_id TEXT, result_status TEXT, pid INTEGER,
  created TEXT, started TEXT, finished TEXT, error TEXT);
"""


def _now():
    return dt.datetime.now().isoformat(timespec="seconds")


def _tuples(x):
    """JSON [value, unit] pairs -> (value, unit) tuples (the hub's quantity form)."""
    if isinstance(x, dict):
        return {k: _tuples(v) for k, v in x.items()}
    if isinstance(x, list):
        if len(x) == 2 and isinstance(x[1], str) and not isinstance(x[0], str):
            try:
                from simws.units import Q_
                Q_(1.0, x[1])
                return (x[0], x[1])
            except Exception:  # noqa: BLE001
                pass
        return [_tuples(v) for v in x]
    return x


class Queue:
    def __init__(self, root=None):
        from simws.hub.store import default_root
        self.root = Path(root) if root else default_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "queue.sqlite", timeout=60, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(_SCHEMA)

    def add(self, adapter, params=None, inputs=None, member="base", seed=None, fidelity="screening", budget=None,
            priority="overnight") -> int:
        if priority not in PRIORITY:
            raise ValueError(f"priority must be one of {list(PRIORITY)}")
        from simws.hub import adapter as A
        A.manifest(adapter)                                        # fail now, not at 3 a.m.
        cur = self.db.execute(
            "INSERT INTO queue (adapter, params, inputs, member, seed, fidelity, budget, priority, status, created) "
            "VALUES (?,?,?,?,?,?,?,?, 'queued', ?)",
            (adapter, json.dumps(params or {}), json.dumps(inputs or {}), member, seed, fidelity,
             json.dumps(budget or {}), priority, _now()))
        return cur.lastrowid

    def recover(self) -> int:
        """Entries left 'running' by a dead process go back to 'queued'."""
        n = 0
        for qid, pid in self.db.execute("SELECT id, pid FROM queue WHERE status='running'").fetchall():
            alive = False
            if pid:
                try:
                    os.kill(pid, 0); alive = True
                except OSError:
                    alive = False
            if not alive:
                self.db.execute("UPDATE queue SET status='queued', pid=NULL, error='recovered after interruption' "
                                "WHERE id=?", (qid,))
                n += 1
        return n

    def next(self) -> Optional[tuple]:
        rows = self.db.execute("SELECT id, adapter, params, inputs, member, seed, fidelity, budget, priority FROM queue "
                               "WHERE status='queued'").fetchall()
        if not rows:
            return None
        return sorted(rows, key=lambda r: (PRIORITY[r[8]], r[0]))[0]

    def run(self, until: Optional[dt.datetime] = None, max_jobs: Optional[int] = None, hub=None, log=print) -> dict:
        from simws.hub.jobs import Hub
        from simws.hub.schema import Budget
        hub = hub or Hub(self.root)
        rec = self.recover()
        if rec:
            log(f"queue: {rec} interrupted job(s) re-queued")
        done = 0
        stats = {"ok": 0, "failed": 0, "recovered": rec}
        while True:
            if until and dt.datetime.now() >= until:
                log("queue: time limit reached"); break
            if max_jobs is not None and done >= max_jobs:
                break
            row = self.next()
            if row is None:
                break
            qid, adapter, params, inputs, member, seed, fidelity, budget, _ = row
            self.db.execute("UPDATE queue SET status='running', pid=?, started=? WHERE id=?", (os.getpid(), _now(), qid))
            try:
                res = hub.submit(adapter, params=_tuples(json.loads(params)), inputs=_tuples(json.loads(inputs)),
                                 member=member, seed=seed, fidelity=fidelity, budget=Budget(**json.loads(budget)))
                st = "done" if res.status == "ok" else "failed"
                self.db.execute("UPDATE queue SET status=?, result_id=?, result_status=?, finished=?, error=? WHERE id=?",
                                (st, res.id, res.status, _now(), res.message[:2000] or None, qid))
                stats["ok" if res.status == "ok" else "failed"] += 1
                log(f"queue: #{qid} {adapter} -> {res.status} {res.id}")
            except Exception as e:  # noqa: BLE001  the queue must survive any single job
                self.db.execute("UPDATE queue SET status='failed', finished=?, error=? WHERE id=?",
                                (_now(), f"{type(e).__name__}: {e}"[:2000], qid))
                stats["failed"] += 1
                log(f"queue: #{qid} {adapter} -> error {e}")
            done += 1
        stats["remaining"] = self.db.execute("SELECT COUNT(*) FROM queue WHERE status='queued'").fetchone()[0]
        return stats

    def status(self) -> list:
        return self.db.execute("SELECT id, adapter, priority, status, result_status, result_id, created, finished "
                               "FROM queue ORDER BY id").fetchall()


def parse_until(s: str, now: Optional[dt.datetime] = None) -> dt.datetime:
    """'07:00' -> next occurrence of 07:00; ISO date-times are taken as given."""
    now = now or dt.datetime.now()
    if len(s) <= 5 and ":" in s:
        h, m = (int(x) for x in s.split(":"))
        t = now.replace(hour=h, minute=m, second=0, microsecond=0)
        return t if t > now else t + dt.timedelta(days=1)
    return dt.datetime.fromisoformat(s)


def parse_for(s: str) -> dt.timedelta:
    unit = s[-1]
    v = float(s[:-1])
    return dt.timedelta(**{{"h": "hours", "m": "minutes", "s": "seconds", "d": "days"}[unit]: v})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m simws.queue")
    ap.add_argument("--store", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("adapter"); a.add_argument("--params", default="{}"); a.add_argument("--inputs", default="{}")
    a.add_argument("--member", default="base"); a.add_argument("--seed", type=int)
    a.add_argument("--fidelity", default="screening"); a.add_argument("--budget", default="{}")
    a.add_argument("--priority", default="overnight")
    r = sub.add_parser("run")
    r.add_argument("--until"); r.add_argument("--for", dest="for_"); r.add_argument("--max-jobs", type=int)
    sub.add_parser("status")
    args = ap.parse_args(argv)
    q = Queue(args.store)
    if args.cmd == "add":
        qid = q.add(args.adapter, json.loads(args.params), json.loads(args.inputs), args.member, args.seed,
                    args.fidelity, json.loads(args.budget), args.priority)
        print(f"queued #{qid}")
    elif args.cmd == "run":
        until = parse_until(args.until) if args.until else (dt.datetime.now() + parse_for(args.for_) if args.for_ else None)
        print(json.dumps(q.run(until, args.max_jobs)))
    else:
        for row in q.status():
            print(" ".join(str(x) for x in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
