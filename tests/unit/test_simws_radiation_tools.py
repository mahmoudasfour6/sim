"""Unit tests for simws 0.4: upset-rate engine, Geant4 per-event loader and remap, release bundle, study notes."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pint")
pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[2]


def test_ser_engine_matches_closed_form():
    from simws.radiation import ser
    for lc in (0.5, 3.0, 30.0):
        assert abs(ser.verify(l_crit=lc)["rel_diff"]) < 2e-3


def test_ser_units_and_critical_let(tmp_path):
    from simws.radiation import ser
    f = tmp_path / "spec.txt"
    f.write_text("LET  Flux\n1000 1e-2\n10000 1e-4\n100000 1e-7\n")               # MeV cm2/g, per m2 s sr
    s = ser.LETSpectrum.from_file(f, let_unit="MeV cm2/g", flux_unit="1/m2/s/sr")
    assert s.let.tolist() == [1.0, 10.0, 100.0] and s.integral_flux[0] == pytest.approx(1e-6)
    assert ser.critical_let([1, 10], [10, 100], 31.6227766) == pytest.approx(math.sqrt(10), rel=1e-6)
    assert ser.critical_let([1, 10], [10, 100], 1000) is None
    w = ser.WeibullCrossSection(l0=1, w=10, s=1.5, sat_cm2=1e-8)
    assert w(0.5) == 0 and w(1e4) == pytest.approx(1e-8)


def test_geant4_event_loader_and_conservative_remap(tmp_path):
    from simws.materials import load
    from simws.radiation.geant4 import axisymmetric_footprint, load_events
    f = tmp_path / "ev.csv"
    f.write_text("# g4track per-event deposition; nz=2 nr=4 thickness_um=2 rmax_um=0.4 ion_Z=8 ion_A=16 energy_MeV=100\n"
                 "event,iz,ir,edep_MeV\n0,0,0,1.0\n0,0,1,0.5\n0,1,0,1.0\n0,1,4,0.1\n1,1,2,0.2\n")
    meta, evs = load_events(f)
    assert len(evs) == 2 and evs[0].edep_MeV.sum() == pytest.approx(2.5) and evs[0].overflow_MeV.sum() == pytest.approx(0.1)
    # tensor-product r-z grid with exact ring volumes as node volumes
    rl, zl = np.linspace(0, 1.0, 51), np.linspace(0, 2.0, 21)
    R, Z = np.meshgrid(rl, zl)
    rmid = np.concatenate([[0], 0.5 * (rl[1:] + rl[:-1]), [rl[-1]]]); zmid = np.concatenate([[0], 0.5 * (zl[1:] + zl[:-1]), [zl[-1]]])
    ring = math.pi * (rmid[1:] ** 2 - rmid[:-1] ** 2); slab = np.diff(zmid)
    vol = (slab[:, None] * ring[None, :]).ravel() * 1e-12
    dens, factor, total = axisymmetric_footprint(evs[0], load("silicon"), R.ravel(), Z.ravel(), vol)
    assert total == pytest.approx(2.6 / 3.6e-6)
    assert factor == pytest.approx(2.6 / 2.5, rel=1e-6)      # only the overflow (beyond rmax) needs rescaling
    assert (dens * vol).sum() == pytest.approx(total, rel=1e-9)


def test_release_bundle_and_study_notes(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    for t in ("release.py", "study.py"):
        (repo / "tools" / t).write_text((ROOT / "tools" / t).read_text())
    (repo / "envs").mkdir(); (repo / "envs" / "tcad.yml").write_text("name: tcad\n")
    (repo / "results" / "ser").mkdir(parents=True)
    (repo / "results" / "ser" / "summary.json").write_text(json.dumps(
        {"results": {"critical_let": 1.2}, "provenance": {"git_commit": "abc", "git_dirty": False}}))
    run = lambda *a: subprocess.run(a, cwd=repo, capture_output=True, text=True, check=True)
    run("git", "init", "-q"); run("git", "add", "-A")
    run("git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    run(sys.executable, "tools/release.py", "--title", "T", "--version", "1.0", "--author", "Doe, Jane|Uni")
    out = repo / "release" / "sim_workspace-v1.0"
    assert (repo / "release" / "sim_workspace-v1.0.tar.gz").exists()
    assert (out / "results" / "ser" / "summary.json").exists() and (out / "envs" / "tcad.yml").exists()
    assert "family-names: \"Doe\"" in (out / "CITATION.cff").read_text()
    assert subprocess.run(["sha256sum", "-c", "MANIFEST.sha256", "--quiet"], cwd=out).returncode == 0
    run(sys.executable, "tools/study.py", "new", "Test study")
    note = next((repo / "notes").glob("*_test-study.md"))
    run(sys.executable, "tools/study.py", "link", str(note), "results/ser/summary.json")
    assert "critical_let=1.2" in note.read_text() and "produced at commit abc" in note.read_text()
    assert "Test study" in (repo / "notes" / "README.md").read_text()
