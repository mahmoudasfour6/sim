"""Job submission (interface specification: job lifecycle).

    from simws.hub.jobs import Hub
    hub = Hub()                                    # store + graph at the default root
    res = hub.submit("devsim.hbt_gummel", params={"T": (300, "K"), "VCE": (1.0, "V")})
    res.q("hbt.beta").value, res.flags, res.status

submit():
 1. builds the job key (adapter + version, engine + version, env lock hash, canonical params and inputs, member, seed)
 2. returns the cached result when the same key already has an ok result in the store
 3. writes staging/<job id>/job.json and runs the adapter in its env (`python -m simws.run`), or in this process's
    python when that env is absent (flagged 'runner:local' in provenance), or in-process (mode="inprocess", tests)
 4. on failure, retries with the manifest's fallbacks for that failure category (each fallback is a new job key;
    at most 3 attempts)
 5. runs the gatekeeper and commits under the store lock: store.finalize + graph.commit_result, one transaction
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from simws.hub import adapter as A
from simws.hub import gatekeeper
from simws.hub.canonical import canonical_quantity, job_id, sha256_file
from simws.hub.graph import Graph
from simws.hub.schema import Budget, JobEnvelope, JobKey, ResultEnvelope
from simws.hub.store import Store

MAX_ATTEMPTS = 3


def env_lock(env: str) -> str:
    """SHA-256 (first 16 hex) of envs/<env>.explicit.txt written by the installer; 'unlocked:<env>' otherwise."""
    from simws.config import ROOT, load
    root = Path(load().get("workspace_root", ROOT))
    for base in (root, ROOT):
        p = base / "envs" / f"{env}.explicit.txt"
        if p.is_file():
            return sha256_file(p)[:16]
    return f"unlocked:{env}"


def canonical_value(v: Any) -> Any:
    """(value, unit) tuples and {'value','unit'} dicts -> canonical quantity; numpy -> lists; others unchanged.
    Strings that are result/quantity references ('<rid>' or '<rid>:<name>') are kept as they are."""
    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], str):
        return canonical_quantity(v[0], v[1])
    if isinstance(v, dict) and set(v) >= {"value", "unit"} and "dims" not in v:
        return canonical_quantity(v["value"], v["unit"])
    if isinstance(v, dict):
        return {k: canonical_value(x) for k, x in v.items()}
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, list):
        return [canonical_value(x) for x in v]
    return v


def _refs(inputs: Dict[str, Any]) -> List[str]:
    out = []
    for v in inputs.values():
        vals = v if isinstance(v, list) else [v]
        for x in vals:
            if isinstance(x, str) and len(x.split(":", 1)[0]) == 32 and all(c in "0123456789abcdef" for c in x.split(":", 1)[0]):
                out.append(x)
    return out


class Hub:
    def __init__(self, root=None, mode: str = "subprocess"):
        self.store = Store(root)
        self.graph = Graph(self.store.root / "graph.sqlite")
        self.mode = mode                       # subprocess | inprocess

    # ------------------------------------------------------------------ keys
    def make_job(self, name: str, params=None, inputs=None, member="base", seed=None, fidelity="screening",
                 budget: Optional[Budget] = None, priority="interactive") -> JobEnvelope:
        ad = A.get(name)
        m = ad.manifest
        params = ad.resolve(dict(params or {}))
        key = JobKey(adapter=m.name, adapter_version=m.version, engine=m.engine, engine_version=m.engine_version,
                     env_lock=env_lock(m.env), inputs=canonical_value(dict(inputs or {})),
                     params=canonical_value(dict(params or {})), member=member, seed=seed)
        # fidelity is part of the key: paper and screening runs of the same inputs are different jobs
        jid = job_id({"key": key.model_dump(), "fidelity": fidelity})
        return JobEnvelope(id=jid, key=key, env=m.env, budget=budget or Budget(), fidelity=fidelity,
                           priority=priority)

    # ------------------------------------------------------------------ running
    def _python_for(self, env: str):
        try:
            from simws.config import env_python
            p = env_python(env)
            if p.exists():
                return str(p), False
        except Exception:  # noqa: BLE001
            pass
        return sys.executable, True

    def _run(self, job: JobEnvelope, staging: Path) -> ResultEnvelope:
        (staging / "job.json").write_text(job.model_dump_json(indent=1))
        if self.mode == "inprocess":
            res = A.get(job.key.adapter).execute(job, staging)
            res.provenance["runner"] = "inprocess"
            return res
        py, local = self._python_for(job.env)
        envv = dict(os.environ, OMP_NUM_THREADS=str(job.budget.threads))
        repo = str(Path(__file__).resolve().parents[2])
        envv["PYTHONPATH"] = repo + (os.pathsep + envv["PYTHONPATH"] if envv.get("PYTHONPATH") else "")
        try:
            with open(staging / "logs" / "runner.out", "w") as out:
                proc = subprocess.run([py, "-m", "simws.run", job.key.adapter, str(staging)], stdout=out,
                                      stderr=subprocess.STDOUT, env=envv, timeout=job.budget.wall_s + 60)
        except subprocess.TimeoutExpired:
            return ResultEnvelope(job_id=job.id, status="timeout", message="killed after wall-time budget + 60 s")
        rp = staging / "result.json"
        if not rp.is_file():
            tail = (staging / "logs" / "runner.out").read_text()[-2000:]
            return ResultEnvelope(job_id=job.id, status="engine_error",
                                  message=f"runner exited {proc.returncode} without a result: {tail}")
        res = ResultEnvelope.model_validate_json(rp.read_text())
        rp.unlink()
        res.provenance["runner"] = "local" if local else f"env:{job.env}"
        if local and "runner:local" not in res.flags:
            res.flags.append("runner:local")
        return res

    def _commit(self, job: JobEnvelope, res: ResultEnvelope, staging: Path, supersedes=None) -> ResultEnvelope:
        m = A.manifest(job.key.adapter)
        res, rep = gatekeeper.apply(job, res, m, staging)
        with self.store.lock():
            res = self.store.finalize(staging, res)
            self.graph.commit_result(job, res, input_nodes=_refs(job.key.inputs), supersedes=supersedes)
            if not rep.ok:
                self.graph.event("rejected", res.id, {"errors": rep.errors})
        return res

    def submit(self, name: str, params=None, inputs=None, member="base", seed=None, fidelity="screening",
               budget: Optional[Budget] = None, force: bool = False, supersedes: Optional[str] = None,
               priority="interactive") -> ResultEnvelope:
        m = A.manifest(name)
        params = dict(params or {})
        attempts: List[str] = []
        fallbacks: Dict[str, List[Dict[str, Any]]] = {k: list(v) for k, v in m.fallbacks.items()}
        res = None
        for attempt in range(MAX_ATTEMPTS):
            job = self.make_job(name, params, inputs, member, seed, fidelity, budget, priority)
            cached = None if force else self.graph.result_for_job(job.id)
            if cached and self.store.has(cached):
                res = self.store.envelope(cached)
                res.diagnostics.setdefault("cache", "hit")
                return res
            staging = self.store.staging(job.id)
            res = self._run(job, staging)
            res.diagnostics["attempts"] = attempts + [job.id]
            res = self._commit(job, res, staging, supersedes)
            attempts.append(job.id)
            if res.status == "ok":
                return res
            nxt = fallbacks.get(res.status, [])
            if not nxt:
                return res
            params = {**params, **nxt.pop(0)}
        return res

    # ------------------------------------------------------------------ queries
    def result(self, rid: str) -> ResultEnvelope:
        return self.store.envelope(rid)

    def quantity(self, ref: str):
        rid, name = ref.split(":", 1)
        return self.store.envelope(rid).q(name)
