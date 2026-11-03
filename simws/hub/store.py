"""Content-addressed store (interface specification: identifiers; data labels, storage and retention).

    store/
      staging/<job id>/          job.json, data/, logs/        written by the runner
      objects/<id[:2]>/<id>/     envelope.json, job.json, data/, logs/   immutable, moved here by the committer
      graph.sqlite               knowledge graph (simws.hub.graph)
      .commit.lock               single-writer lock

Retention classes: permanent (kept), until_validated (deleted by `prune(validated=True)` once dependents pass),
scratch (deleted at commit). Deleted files are regenerated from the stored job key.
"""
from __future__ import annotations

import fcntl
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from simws.hub.canonical import canonical_json, result_id, sha256_file, sha256_text
from simws.hub.schema import JobEnvelope, ResultEnvelope


def default_root() -> Path:
    env = os.environ.get("SIMWS_STORE")
    if env:
        return Path(env)
    try:
        from simws.config import load
        r = load().get("hub", {}).get("store")
        if r:
            return Path(os.path.expanduser(r))
    except Exception:  # noqa: BLE001
        pass
    from simws.config import ROOT
    return ROOT / "store"


class Store:
    def __init__(self, root=None):
        self.root = Path(root) if root else default_root()
        for d in ("staging", "objects"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- paths
    def staging(self, job_id: str) -> Path:
        p = self.root / "staging" / job_id
        (p / "data").mkdir(parents=True, exist_ok=True)
        (p / "logs").mkdir(parents=True, exist_ok=True)
        return p

    def object_dir(self, rid: str) -> Path:
        return self.root / "objects" / rid[:2] / rid

    def has(self, rid: str) -> bool:
        return (self.object_dir(rid) / "envelope.json").is_file()

    def envelope(self, rid: str) -> ResultEnvelope:
        return ResultEnvelope.model_validate_json((self.object_dir(rid) / "envelope.json").read_text())

    def job(self, rid: str) -> JobEnvelope:
        return JobEnvelope.model_validate_json((self.object_dir(rid) / "job.json").read_text())

    # ---------------------------------------------------------------- lock
    @contextmanager
    def lock(self):
        f = open(self.root / ".commit.lock", "a+")
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()

    # ---------------------------------------------------------------- staging -> objects
    def output_hashes(self, staging: Path):
        data = staging / "data"
        return sorted(f"{p.relative_to(data)}:{sha256_file(p)}" for p in data.rglob("*") if p.is_file())

    def finalize(self, staging: Path, env: ResultEnvelope) -> ResultEnvelope:
        """Delete scratch files, compute the result ID, move into objects/ (caller holds the lock)."""
        for f in env.fields:
            if f.retention == "scratch":
                (staging / "data" / f.path).unlink(missing_ok=True)
        env.fields = [f for f in env.fields if f.retention != "scratch"]
        # the ID covers the data files and the quantity values, so a non-deterministic rerun never collides
        qhash = sha256_text(canonical_json([q.model_dump() for q in env.quantities]))
        rid = result_id(env.job_id, self.output_hashes(staging) + [f"quantities:{qhash}", f"status:{env.status}"])
        env.id = rid
        (staging / "envelope.json").write_text(env.model_dump_json(indent=1))
        dst = self.object_dir(rid)
        if dst.exists():                               # identical result already committed (same job, same bytes)
            shutil.rmtree(staging)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging), dst)
        return env

    # ---------------------------------------------------------------- retention
    def prune(self, rid: str, validated: bool) -> int:
        """Delete the until_validated fields of a result once its dependents have passed. The envelope stays as
        committed (objects are immutable); the graph records a 'pruned' event. Returns bytes freed."""
        if not validated:
            return 0
        freed = 0
        for f in self.envelope(rid).fields:
            p = self.object_dir(rid) / "data" / f.path
            if f.retention == "until_validated" and p.exists():
                freed += p.stat().st_size
                p.unlink()
        return freed

    def disk_report(self, warn_fraction=0.8) -> dict:
        total = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        st = shutil.disk_usage(self.root)
        used_frac = (st.total - st.free) / st.total
        return {"store_bytes": total, "disk_total": st.total, "disk_free": st.free, "disk_used_fraction": used_frac,
                "warning": used_frac > warn_fraction}
