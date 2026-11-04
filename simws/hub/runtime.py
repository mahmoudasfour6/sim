"""Parallel runtime for hub jobs (0.9c): many hub.submit calls on a Dask cluster.

    from simws.hub.runtime import Runtime
    with Runtime(n_workers=4) as rt:                       # LocalCluster, one process per worker
        results = rt.map("devsim.hbt_gummel", [{"VBE": ...}, {"VBE": ...}], fidelity="paper")
        ens = rt.members("devsim.hbt_gummel", members, params=base)
    Runtime(address="tcp://host:8786")                     # an existing scheduler (dask-scheduler / dask-worker)

Each task is one Hub.submit in a worker process, so the job semantics do not change: the same key, the same cache,
the gatekeeper, one atomic commit under the store's file lock. The driver resolves job IDs first and sends each
distinct job once (identical requests share the result), skips jobs already committed, and caps the workers so that
workers x threads-per-job never exceeds the machine's cores. Memory: workers get memory_limit = the job budget's
memory_gb, so Dask pauses or restarts a worker that runs away instead of the laptop swapping.

Without dask installed, the same API runs on concurrent.futures.ProcessPoolExecutor.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from simws.hub.schema import Budget


def _submit(root: str, name: str, kw: Dict[str, Any], mode: str = "subprocess"):
    from simws.hub.jobs import Hub
    return Hub(root, mode=mode).submit(name, **kw).model_dump_json()


class Runtime:
    def __init__(self, root=None, n_workers: Optional[int] = None, budget: Optional[Budget] = None,
                 address: Optional[str] = None, backend: str = "auto", mode: str = "subprocess"):
        """mode: how each worker's Hub runs the adapter, as Hub(mode=...): "subprocess" (default, the adapter's
        own conda env) or "inprocess" (the worker's interpreter; tests and pure-Python adapters)."""
        from simws.hub.jobs import Hub
        self.mode = mode
        self.hub = Hub(root, mode=mode)
        self.root = str(self.hub.store.root)
        self.budget = budget or Budget()
        cores = os.cpu_count() or 1
        cap = max(1, cores // max(1, self.budget.threads))
        self.n_workers = min(n_workers or cap, cap)
        self.address = address
        self.backend = backend
        self._client = self._cluster = self._pool = None

    # --------------------------------------------------------------------------------------------- lifecycle
    def __enter__(self):
        use_dask = self.backend in ("auto", "dask")
        if use_dask:
            try:
                from dask.distributed import Client, LocalCluster
            except ImportError:
                if self.backend == "dask":
                    raise
                use_dask = False
        if use_dask:
            if self.address:
                self._client = Client(self.address)
            else:
                # workers are spawned processes: the calling script needs an `if __name__ == "__main__":` guard
                mem = self.budget.memory_gb
                try:
                    import psutil
                    mem = min(mem, 0.8 * psutil.virtual_memory().total / 2 ** 30 / self.n_workers)
                except Exception:  # noqa: BLE001
                    pass
                self._cluster = LocalCluster(n_workers=self.n_workers, threads_per_worker=1, processes=True,
                                             memory_limit=f"{mem:.2f}GB", dashboard_address=None)
                self._client = Client(self._cluster)
            self.kind = "dask"
        else:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing as mp
            self._pool = ProcessPoolExecutor(self.n_workers, mp_context=mp.get_context("spawn"))
            self.kind = "processes"
        return self

    def __exit__(self, *exc):
        for x in (self._client, self._cluster):
            if x is not None:
                x.close()
        if self._pool is not None:
            self._pool.shutdown()

    # --------------------------------------------------------------------------------------------- running
    def _go(self, tasks: List[tuple]):
        """tasks: [(name, submit kwargs)] -> ResultEnvelopes in order; one execution per distinct job id."""
        from simws.hub.schema import ResultEnvelope
        ids, uniq = [], {}
        for name, kw in tasks:
            j = self.hub.make_job(name, kw.get("params"), kw.get("inputs"), kw.get("member", "base"),
                                  kw.get("seed"), kw.get("fidelity", "screening"), kw.get("budget"))
            ids.append(j.id)
            uniq.setdefault(j.id, (name, kw))
        done = {}
        pending = {}
        for jid, (name, kw) in uniq.items():
            cached = None if kw.get("force") else self.hub.graph.result_for_job(jid)
            if cached and self.hub.store.has(cached):
                r = self.hub.store.envelope(cached)
                r.diagnostics.setdefault("cache", "hit")
                done[jid] = r
            elif self.kind == "dask":
                pending[jid] = self._client.submit(_submit, self.root, name, kw, self.mode, pure=False, key=f"simws-{jid}")
            else:
                pending[jid] = self._pool.submit(_submit, self.root, name, kw, self.mode)
        for jid, fut in pending.items():
            done[jid] = ResultEnvelope.model_validate_json(fut.result())
        return [done[j] for j in ids]

    def map(self, name: str, param_list: List[Dict[str, Any]], **kw):
        """One job per params dict (same adapter); other submit() keywords apply to all."""
        kw.setdefault("budget", self.budget)
        return self._go([(name, {**kw, "params": p}) for p in param_list])

    def members(self, name: str, members: Dict[str, Dict[str, Any]], params=None, **kw):
        """An ensemble in parallel (one job per member; batch_members adapters run as one job as usual)."""
        from simws.hub import adapter as A
        if A.manifest(name).batch_members:
            return self.hub.submit_members(name, members, params=params, **kw)
        kw.setdefault("budget", self.budget)
        return self._go([(name, {**kw, "params": {**dict(params or {}), **ov}, "member": mid})
                         for mid, ov in sorted(members.items())])

    def jobs(self, tasks: List[tuple]):
        """Mixed adapters: [(name, submit kwargs)]."""
        return self._go(tasks)
