#!/usr/bin/env python3
"""Benchmark this machine and write config/local_tuning.yaml (read by the workflow and sweeps).

Measures, on the actual hardware (Intel Core Ultra 5 125U under WSL2: mixed P/E cores, so scaling is not linear):
  1. QE pw.x Si SCF wall time for 1, 2, 4, 6, 8 MPI ranks         -> best rank count for DFT stages
  2. sweep throughput: N concurrent single-threaded DEVSIM jobs   -> best worker count for sweeps / UQ
     (N = 1, 2, 4, 6, 8, 12), with peak memory per job            -> memory cap per worker
Run:  PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/bench_local.py      (about 10-15 minutes)
"""
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simws import config  # noqa: E402

ENVS = config.path("envs_root")


def qe_scaling(ranks=(1, 2, 4, 6, 8)):
    pw, mpirun = ENVS / "materials" / "bin" / "pw.x", ENVS / "materials" / "bin" / "mpirun"
    if not mpirun.exists():
        mpirun = shutil.which("mpirun")
        ranks = ranks if mpirun else (1,)
    deck = (ROOT / "01_materials_dft_md" / "si_real_scf.pwi").read_text()
    deck = re.sub(r"pseudo_dir\s*=\s*'[^']*'", f"pseudo_dir = '{(ROOT / config.load()['data']['pseudopotentials']).resolve()}'", deck)
    out = {}
    for n in ranks:
        if n > os.cpu_count():
            continue
        with tempfile.TemporaryDirectory() as d:
            Path(d, "si.pwi").write_text(deck)
            env = dict(os.environ, OMP_NUM_THREADS="1", PATH=f"{ENVS / 'materials' / 'bin'}:{os.environ['PATH']}")
            cmd = ([str(mpirun), "--oversubscribe", "-np", str(n)] if n > 1 else []) + [str(pw), "-in", "si.pwi"]
            t0 = time.perf_counter()
            r = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True, timeout=1800)
            if r.returncode != 0 and n > 1:   # MPICH has no --oversubscribe
                cmd.remove("--oversubscribe"); t0 = time.perf_counter()
                r = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True)
            out[n] = round(time.perf_counter() - t0, 2) if r.returncode == 0 else None
            print(f"  QE ranks={n}: {out[n]} s")
    return out


def _devsim_job(_):
    from simws.coupling import electrothermal as et
    t0 = time.perf_counter()
    et.run(et.ResistorCase(biases=[1.0, 3.0]))
    return time.perf_counter() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def sweep_throughput(workers=(1, 2, 4, 6, 8, 12), jobs_per_worker=2):
    out = {}
    for w in workers:
        if w > os.cpu_count():
            continue
        os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        n = w * jobs_per_worker
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=w) as ex:
            res = list(ex.map(_devsim_job, range(n)))
        wall = time.perf_counter() - t0
        out[w] = {"jobs_per_min": round(60 * n / wall, 2), "mean_job_s": round(sum(r[0] for r in res) / n, 2),
                  "peak_rss_MB": round(max(r[1] for r in res), 1)}
        print(f"  sweep workers={w}: {out[w]}")
    return out


def main():
    mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    info = {"cpus_visible": os.cpu_count(), "memory_GB": round(mem_gb, 1)}
    print(f"machine: {info}")
    print("QE scaling:"); qe = qe_scaling()
    print("DEVSIM sweep throughput:"); sw = sweep_throughput()
    ok_qe = {k: v for k, v in qe.items() if v}
    best_qe = min(ok_qe, key=ok_qe.get) if ok_qe else 1
    best_w = max(sw, key=lambda w: sw[w]["jobs_per_min"])
    rss = max(v["peak_rss_MB"] for v in sw.values())
    mem_cap = max(1, int(0.7 * mem_gb * 1024 / rss))
    tuning = {"measured": time.strftime("%Y-%m-%d %H:%M"), "machine": info,
              "qe": {"seconds_by_ranks": qe, "best_ranks": best_qe},
              "sweep": {"by_workers": sw, "best_workers": min(best_w, mem_cap), "peak_rss_MB_per_job": rss,
                        "memory_limited_workers": mem_cap},
              "snakemake": {"cores": info["cpus_visible"], "mem_mb": int(0.8 * mem_gb * 1024)}}
    path = ROOT / "config" / "local_tuning.yaml"
    path.write_text(yaml.safe_dump(tuning, sort_keys=False))
    print(f"\nrecommendations: QE ranks = {best_qe}, sweep workers = {tuning['sweep']['best_workers']}, "
          f"snakemake --cores {info['cpus_visible']} --resources mem_mb={tuning['snakemake']['mem_mb']}\nwrote {path}")


if __name__ == "__main__":
    main()
