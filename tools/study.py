"""Experiment log: one Markdown note per study, tied to the commit and the result files it produced.

    python tools/study.py new "Kr strikes at 0.8 V supply"            # notes/2026-11-02_kr-strikes-at-0-8-v-supply.md
    python tools/study.py link notes/2026-11-02_kr-....md results/axisymmetric/summary.json
    python tools/study.py index                                        # rebuild notes/README.md

`new` writes a template (question, hypothesis, setup, results, conclusion) with today's date, the current commit
and whether the tree is dirty. `link` appends a result with its own provenance (commit that produced it, input
hashes) and a short digest, so in a year the note still says which run answered which question. Notes are plain
Markdown in git; edit them freely.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes"


def _git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


def new(title):
    NOTES.mkdir(exist_ok=True)
    today = dt.date.today().isoformat()
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
    p = NOTES / f"{today}_{slug}.md"
    if p.exists():
        raise SystemExit(f"{p} exists")
    dirty = "yes" if _git("status", "--porcelain", "--untracked-files=no") else "no"
    p.write_text(f"""# {title}

- date: {today}
- commit: {_git('rev-parse', '--short', 'HEAD')} (uncommitted changes: {dirty})
- status: open

## Question
What do we want to know, and what number would answer it?

## Hypothesis / expectation
What we expect and why (a value or a trend), so the result can surprise us.

## Setup
Device, models, parameters that differ from the defaults, commands run (snakemake target or script + args).

## Results
<!-- tools/study.py link appends result files here -->

## Conclusion
Answer to the question, with its uncertainty. What it changes. Next step.
""")
    print(p)
    index()
    return p


def link(note, result):
    note, res = Path(note), Path(result)
    digest = hashlib.sha256(res.read_bytes()).hexdigest()[:12]
    line = f"- `{res.relative_to(ROOT) if res.is_absolute() else res}` (sha256 {digest}"
    if res.suffix == ".json":
        try:
            d = json.loads(res.read_text())
            g = d.get("provenance") or {}
            line += f", produced at commit {str(g.get('git_commit', '?'))[:10]}{' (dirty)' if g.get('git_dirty') else ''}"
            r = d.get("results", {})
            keys = [k for k, v in r.items() if isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) < 30)][:6]
            if keys:
                line += "; " + ", ".join(f"{k}={r[k]:.4g}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in keys)
        except Exception:  # noqa: BLE001
            pass
    line += f"; linked {dt.date.today().isoformat()})"
    text = note.read_text()
    marker = "<!-- tools/study.py link appends result files here -->"
    text = text.replace(marker, f"{marker}\n{line}") if marker in text else text + "\n" + line + "\n"
    note.write_text(text)
    print(line)


def index():
    NOTES.mkdir(exist_ok=True)
    rows = ["# Study log", "", "| date | study | status | commit |", "|---|---|---|---|"]
    for p in sorted(NOTES.glob("20*.md"), reverse=True):
        t = p.read_text()
        title = t.splitlines()[0].lstrip("# ").strip()
        status = re.search(r"- status: (.+)", t); commit = re.search(r"- commit: (\S+)", t)
        rows.append(f"| {p.name[:10]} | [{title}]({p.name}) | {status.group(1) if status else '?'} | "
                    f"{commit.group(1) if commit else '?'} |")
    (NOTES / "README.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("new", "link", "index"):
        raise SystemExit(__doc__)
    {"new": lambda: new(" ".join(sys.argv[2:])), "link": lambda: link(sys.argv[2], sys.argv[3]),
     "index": index}[sys.argv[1]]()
