"""Compare two results folders of the workflow (old code against new code): every JSON, CSV and Markdown output.

    python tools/legacy_compare.py RESULTS_OLD RESULTS_NEW [--rtol 1e-9]

Numbers are compared with a relative tolerance (default 1e-9; exact for integers and strings). Provenance is
skipped: the keys "provenance", "git", "commit", "created", "timestamp", "duration_s", "host", "python", "env",
"inputs" (hashes of input files), "seconds", "wall_s", any key ending in "_utc", and file paths under the results
folders. Prints one line
per file: SAME, EXPECTED (the only differences are the model-form provenance and flag that contract amendment A2
added on purpose), DIFF (with the first differences), ONLY-OLD or ONLY-NEW. Exit code 1 when any file differs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

SKIP_KEYS = {"provenance", "git", "commit", "git_commit", "created", "timestamp", "time", "date", "duration_s",
             "host", "hostname", "python", "env", "inputs", "seconds", "wall_s", "elapsed_s", "runtime_s",
             "elapsed", "run_time_s", "cpu_s", "user", "platform", "versions", "stamp"}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def diff(a, b, rtol, path, out, roots):
    if len(out) >= 8:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in SKIP_KEYS or k.endswith("_utc"):
                continue
            if k not in a or k not in b:
                v = b.get(k) if k not in a else a.get(k)
                out.append(f"{path}/{k}: only in {'new' if k not in a else 'old'}: {str(v)[:80]}")
                continue
            diff(a[k], b[k], rtol, f"{path}/{k}", out, roots)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            extra = [str(x) for x in b if x not in a] + [str(x) for x in a if x not in b]
            out.append(f"{path}: length {len(a)} -> {len(b)}; differing items: {'; '.join(extra)[:200]}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, rtol, f"{path}[{i}]", out, roots)
    elif _num(a) and _num(b):
        if isinstance(a, float) or isinstance(b, float):
            if math.isnan(a) and math.isnan(b):
                return
            if not (a == b or abs(a - b) <= rtol * max(abs(a), abs(b))):
                out.append(f"{path}: {a!r} -> {b!r} (rel {abs(a - b) / max(abs(a), abs(b)):.2e})")
        elif a != b:
            out.append(f"{path}: {a!r} -> {b!r}")
    elif isinstance(a, str) and isinstance(b, str):
        x, y = a, b
        for r in roots:
            x, y = x.replace(r[0], "<RES>"), y.replace(r[1], "<RES>")
        if x != y:
            out.append(f"{path}: {a[:60]!r} -> {b[:60]!r}")
    elif a != b:
        out.append(f"{path}: {a!r} -> {b!r}")


def _cells(p: Path):
    rows = list(csv.reader(p.open(newline="")))
    conv = []
    for r in rows:
        cr = []
        for c in r:
            try:
                cr.append(float(c))
            except ValueError:
                cr.append(c)
        conv.append(cr)
    return conv


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--rtol", type=float, default=1e-9)
    a = ap.parse_args(argv[1:])
    old, new = Path(a.old).resolve(), Path(a.new).resolve()
    roots = [(str(old), str(new))]
    files = sorted({p.relative_to(old) for p in old.rglob("*") if p.is_file()} |
                   {p.relative_to(new) for p in new.rglob("*") if p.is_file()})
    bad = 0
    for rel in files:
        po, pn = old / rel, new / rel
        if rel.suffix not in (".json", ".csv", ".md") or ".meta" in rel.name:
            continue
        if not po.exists() or not pn.exists():
            print(f"{'ONLY-NEW' if not po.exists() else 'ONLY-OLD'} {rel}")
            bad += 1
            continue
        out: list = []
        try:
            if rel.suffix == ".json":
                diff(json.loads(po.read_text()), json.loads(pn.read_text()), a.rtol, "", out, roots)
            elif rel.suffix == ".csv":
                diff(_cells(po), _cells(pn), a.rtol, "", out, roots)
            else:
                lo = [l for l in po.read_text().splitlines() if "commit" not in l.lower() and "generated" not in l.lower()]
                ln = [l for l in pn.read_text().splitlines() if "commit" not in l.lower() and "generated" not in l.lower()]
                if lo != ln:
                    out.append(f"text differs ({len(lo)} -> {len(ln)} lines)")
        except Exception as e:  # noqa: BLE001
            out.append(f"could not compare: {e}")
        if out and all("model_form" in o for o in out):
            print(f"EXPECTED {rel}  (model-form provenance and flag added by amendment A2)")
        elif out:
            bad += 1
            print(f"DIFF {rel}")
            for o in out:
                print(f"     {o}")
        else:
            print(f"SAME {rel}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
