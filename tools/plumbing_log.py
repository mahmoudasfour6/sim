"""Plumbing budget (Build Plan, 0.9 onwards): time spent on hub, install and environment maintenance vs research.
Above about 30 % plumbing, no new engines are added.

    python tools/plumbing_log.py add 1.5 plumbing "conda package name fix (libprecice)" --release 0.9
    python tools/plumbing_log.py add 3 research "Schottky transport with image force inside WKB"
    python tools/plumbing_log.py report [--release 0.9]

The log is notes/plumbing_log.csv (tracked in git). Categories: plumbing (installs, environments, packaging, CI,
hub maintenance) and research (physics, numerics, validation, studies).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "notes" / "plumbing_log.csv"
FIELDS = ["date", "release", "hours", "category", "note"]


def add(hours, category, note, release):
    if category not in ("plumbing", "research"):
        raise SystemExit("category must be plumbing or research")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.is_file()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow({"date": dt.date.today().isoformat(), "release": release, "hours": hours, "category": category,
                    "note": note})


def report(release=None):
    if not LOG.is_file():
        print("no plumbing log yet"); return
    rows = [r for r in csv.DictReader(open(LOG)) if release in (None, r["release"])]
    tot = {"plumbing": 0.0, "research": 0.0}
    for r in rows:
        tot[r["category"]] += float(r["hours"])
    s = sum(tot.values())
    share = tot["plumbing"] / s if s else 0.0
    print(f"{'release ' + release if release else 'all releases'}: plumbing {tot['plumbing']:.1f} h, research "
          f"{tot['research']:.1f} h -> plumbing share {100 * share:.0f} % {'(ABOVE the 30 % limit: no new engines)' if share > 0.30 else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("hours", type=float); a.add_argument("category"); a.add_argument("note")
    a.add_argument("--release", default="0.9")
    r = sub.add_parser("report"); r.add_argument("--release", default=None)
    x = ap.parse_args()
    add(x.hours, x.category, x.note, x.release) if x.cmd == "add" else report(x.release)
