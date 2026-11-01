"""Shared fixtures for the sim_workspace regression suites.

Fast tier (default):  pytest tests/
Slow tier (re-runs the real solvers):  pytest -m slow tests/
Record golden values for checks that have none yet:  pytest -m slow --update-golden tests/

--update-golden only ADDS missing values. It never overwrites an existing one, so a
regression can't be silently accepted. To change a golden value on purpose, edit
tests/golden_slow.json by hand and say why in the commit message.
"""
import datetime
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ENVS = Path.home() / "miniforge3" / "envs"
GOLDEN_PATH = Path(__file__).with_name("golden_slow.json")


def pytest_addoption(parser):
    parser.addoption("--update-golden", action="store_true",
                     help="record golden values that are missing from tests/golden_slow.json")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: re-runs a real solver in its conda env (minutes)")


class Golden:
    def __init__(self, path: Path, update: bool):
        self.path, self.update, self.dirty = path, update, False
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def verify(self, measured: dict):
        """Compare {key: value} against golden entries; report every mismatch at once."""
        problems, missing = [], []
        for key, value in measured.items():
            entry = self.data.get(key)
            if entry is None:
                if self.update:
                    self.data[key] = {"value": value, "rtol": 1e-6, "atol": 0.0,
                                      "recorded": datetime.date.today().isoformat()}
                    self.dirty = True
                else:
                    missing.append(f"{key} = {value!r}")
                continue
            if value != pytest.approx(entry["value"], rel=entry.get("rtol", 1e-6), abs=entry.get("atol", 0.0)):
                problems.append(f"{key}: measured {value!r}, golden {entry['value']!r} "
                                f"(rtol={entry.get('rtol')}, atol={entry.get('atol')})")
        msg = []
        if problems:
            msg.append("REGRESSION:\n  " + "\n  ".join(problems))
        if missing:
            msg.append("No golden value yet (run with --update-golden to record):\n  " + "\n  ".join(missing))
        if msg:
            pytest.fail("\n".join(msg), pytrace=False)

    def save(self):
        if self.dirty:
            self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")


@pytest.fixture(scope="session")
def golden(request):
    g = Golden(GOLDEN_PATH, request.config.getoption("--update-golden"))
    yield g
    g.save()


def run_in_env(env: str, args, cwd, timeout=1800, threads=2, extra_env=None) -> str:
    """Run a command with a conda env's bin/ first on PATH (no `conda run` overhead)."""
    prefix = ENVS / env
    if not (prefix / "bin").is_dir():
        pytest.skip(f"conda env '{env}' not found at {prefix}")
    e = os.environ.copy()
    e["PATH"] = f"{prefix / 'bin'}:{e.get('PATH', '')}"
    e["CONDA_PREFIX"] = str(prefix)
    e["MPLBACKEND"] = "Agg"
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        e[v] = str(threads)
    if extra_env:
        e.update(extra_env)
    args = [str(a) for a in args]
    r = subprocess.run(args, cwd=str(cwd), env=e, capture_output=True, text=True, timeout=timeout)
    out = r.stdout + "\n" + r.stderr
    if r.returncode != 0:
        pytest.fail(f"[{env}] {' '.join(args)} exited with {r.returncode}\n"
                    f"--- output tail ---\n{out[-3000:]}", pytrace=False)
    return out


def env_python(env: str) -> Path:
    return ENVS / env / "bin" / "python"


def grab(pattern: str, text: str, group: int = 1) -> float:
    """Return the float captured by `pattern` (last match), or fail showing the output tail."""
    hits = re.findall(pattern, text, flags=re.MULTILINE)
    if not hits:
        pytest.fail(f"pattern not found: {pattern}\n--- output tail ---\n{text[-2500:]}", pytrace=False)
    hit = hits[-1]
    return float(hit[group - 1] if isinstance(hit, tuple) else hit)
