"""Local parameter sweeps and uncertainty quantification.

    from simws.sweep import Space, Store, run_design
    space = Space({"thermal_k": (1.3, 1.6), "mu_exp": (2.2, 2.8)})
    design = space.sobol(n=64)                      # Saltelli design for Sobol indices (scipy QMC)
    store = Store("results/sweeps/et.sqlite")
    run_design(stage, design, store, workers=4)     # stage(params: dict) -> dict of scalar outputs
    print(space.sobol_indices(store, "T_max_K", n=64))

Every run is keyed by a hash of its parameters, so an interrupted sweep resumes where it stopped and points are
never computed twice. Workers are separate processes (DEVSIM keeps global state, so threads cannot be used) and
each is capped to `threads_per_worker` BLAS/OpenMP threads, so workers x threads stays within the CPU budget.
Records also carry status, wall time, error text and the provenance stamp.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


def _key(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=float).encode()).hexdigest()[:20]


class Store:
    """SQLite run table: one row per parameter point."""

    def __init__(self, path):
        self.path = str(path)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with sqlite3.connect(self.path) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS runs (key TEXT PRIMARY KEY, params TEXT, outputs TEXT,
                         status TEXT, wall_s REAL, error TEXT, provenance TEXT, created REAL)""")

    def done_keys(self):
        with sqlite3.connect(self.path) as c:
            return {r[0] for r in c.execute("SELECT key FROM runs WHERE status='ok'")}

    def put(self, params, outputs, status, wall_s, error="", provenance=None):
        with sqlite3.connect(self.path) as c:
            c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?)",
                      (_key(params), json.dumps(params, default=float), json.dumps(outputs, default=float), status,
                       wall_s, error, json.dumps(provenance or {}, default=str), time.time()))

    def rows(self, status="ok") -> List[Tuple[dict, dict]]:
        with sqlite3.connect(self.path) as c:
            return [(json.loads(p), json.loads(o)) for p, o in
                    c.execute("SELECT params, outputs FROM runs WHERE status=?", (status,))]

    def lookup(self, params):
        with sqlite3.connect(self.path) as c:
            r = c.execute("SELECT outputs FROM runs WHERE key=? AND status='ok'", (_key(params),)).fetchone()
        return json.loads(r[0]) if r else None

    def failures(self):
        with sqlite3.connect(self.path) as c:
            return [(json.loads(p), e) for p, e in c.execute("SELECT params, error FROM runs WHERE status='error'")]


@dataclass
class Space:
    bounds: Dict[str, Tuple[float, float]]

    @property
    def names(self):
        return list(self.bounds)

    def _scale(self, unit):
        lo = np.array([b[0] for b in self.bounds.values()]); hi = np.array([b[1] for b in self.bounds.values()])
        return lo + unit * (hi - lo)

    def _rows(self, X):
        return [dict(zip(self.names, map(float, x))) for x in X]

    def grid(self, n_per_dim):
        axes = [np.linspace(lo, hi, n_per_dim) for lo, hi in self.bounds.values()]
        return self._rows(np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1).T)

    def lhs(self, n, seed=0):
        from scipy.stats import qmc
        return self._rows(self._scale(qmc.LatinHypercube(d=len(self.bounds), seed=seed).random(n)))

    def sobol(self, n, seed=0):
        """Saltelli design for first- and total-order Sobol indices: n*(d+2) points (n a power of 2).
        Row order per base sample j: A_j, AB_1j ... AB_dj, B_j (same layout as SALib)."""
        from scipy.stats import qmc
        d = len(self.bounds)
        base = qmc.Sobol(d=2 * d, scramble=True, seed=seed).random(n)
        A, B = base[:, :d], base[:, d:]
        rows = []
        for j in range(n):
            rows.append(A[j])
            for i in range(d):
                ab = A[j].copy(); ab[i] = B[j, i]
                rows.append(ab)
            rows.append(B[j])
        return self._rows(self._scale(np.array(rows)))

    def sobol_indices(self, store: Store, output: str, n: int, seed=0, num_resamples=200, conf_level=0.95):
        """Saltelli (2010) first-order and Jansen total-order estimators with bootstrap confidence half-widths."""
        from scipy.stats import norm
        d = len(self.bounds)
        y = []
        for p in self.sobol(n, seed=seed):
            o = store.lookup(p)
            if o is None:
                raise RuntimeError(f"missing sweep point {p}; run the design first")
            y.append(o[output])
        Y = np.array(y, float).reshape(n, d + 2)
        fA, fB, fAB = Y[:, 0], Y[:, -1], Y[:, 1:-1]

        def est(idx):
            var = np.var(np.r_[fA[idx], fB[idx]])
            s1 = np.mean(fB[idx, None] * (fAB[idx] - fA[idx, None]), axis=0) / var
            st = 0.5 * np.mean((fA[idx, None] - fAB[idx]) ** 2, axis=0) / var
            return s1, st

        s1, st = est(np.arange(n))
        rng = np.random.default_rng(seed)
        boot = [est(rng.integers(0, n, n)) for _ in range(num_resamples)]
        z = norm.ppf(0.5 + conf_level / 2)
        s1_c = z * np.std([b[0] for b in boot], axis=0, ddof=1)
        st_c = z * np.std([b[1] for b in boot], axis=0, ddof=1)
        return {name: {"S1": float(s1[i]), "S1_conf": float(s1_c[i]), "ST": float(st[i]), "ST_conf": float(st_c[i])}
                for i, name in enumerate(self.names)}


def _worker_init(threads):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(threads)


def _call(stage, params):
    t0 = time.perf_counter()
    try:
        out = stage(dict(params))
        return params, out, "ok", time.perf_counter() - t0, ""
    except Exception:
        return params, {}, "error", time.perf_counter() - t0, traceback.format_exc()[-4000:]


def run_design(stage: Callable[[dict], dict], design: List[dict], store: Store, workers: int = 1,
               threads_per_worker: int = 1, progress: bool = True):
    """Run every point of `design` not already in `store`. `stage` must be a top-level (picklable) function."""
    from simws.provenance import stamp
    prov = stamp(parameters={"design_size": len(design), "workers": workers, "threads_per_worker": threads_per_worker})
    done = store.done_keys()
    todo = [p for p in design if _key(p) not in done]
    if progress:
        print(f"sweep: {len(design)} points, {len(design) - len(todo)} cached, {len(todo)} to run on {workers} workers")
    if not todo:
        return
    if workers <= 1:
        _worker_init(threads_per_worker)
        results = (_call(stage, p) for p in todo)
        for i, r in enumerate(results, 1):
            store.put(*r, provenance=prov)
            if progress and i % max(1, len(todo) // 10) == 0:
                print(f"  {i}/{len(todo)}")
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(threads_per_worker,)) as ex:
        futs = [ex.submit(_call, stage, p) for p in todo]
        for i, f in enumerate(as_completed(futs), 1):
            store.put(*f.result(), provenance=prov)
            if progress and i % max(1, len(todo) // 10) == 0:
                print(f"  {i}/{len(todo)}")


def dakota_input(space: Space, driver: str, responses: List[str], method="sampling", samples=64, seed=1) -> str:
    """A Dakota input deck (fork interface) for the same space; `driver` is the analysis driver command.
    Dakota passes a params file and expects a results file; simws.sweep.dakota_driver handles both sides."""
    lines = ["environment", "  tabular_data", "    tabular_data_file = 'dakota_table.dat'", "", "method"]
    if method == "sampling":
        lines += [f"  sampling", f"    sample_type lhs", f"    samples = {samples}", f"    seed = {seed}"]
    else:
        lines += [f"  {method}"]
    lines += ["", "variables", f"  continuous_design = {len(space.names)}",
              "    lower_bounds " + " ".join(f"{b[0]:.10g}" for b in space.bounds.values()),
              "    upper_bounds " + " ".join(f"{b[1]:.10g}" for b in space.bounds.values()),
              "    descriptors " + " ".join(f"'{n}'" for n in space.names), "",
              "interface", "  fork", f"    analysis_drivers = '{driver}'", "    parameters_file = 'params.in'",
              "    results_file = 'results.out'", "    file_tag", "    file_save", "",
              "responses", f"  response_functions = {len(responses)}",
              "    descriptors " + " ".join(f"'{r}'" for r in responses), "  no_gradients", "  no_hessians", ""]
    return "\n".join(lines)


def dakota_driver(stage: Callable[[dict], dict], responses: List[str], params_file: str, results_file: str):
    """Analysis-driver body: read Dakota's params file (standard format), run the stage, write results."""
    params = {}
    with open(params_file) as f:
        n = int(f.readline().split()[0])
        for _ in range(n):
            val, name = f.readline().split()[:2]
            params[name] = float(val)
    out = stage(params)
    with open(results_file, "w") as f:
        for r in responses:
            f.write(f"{float(out[r]):.15e} {r}\n")
