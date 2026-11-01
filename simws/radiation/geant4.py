"""Build and run the g4track Geant4 application (tools/geant4_track) and load its (depth, radius) deposition.

Everything runs locally in the process_rad conda env. Geant4 dataset paths are resolved here because
`geant4-config --datasets` reports names with a "G4" prefix (e.g. G4ENSDFSTATE3.0) while conda installs the
directories without it (ENSDFSTATE3.0); outside an activated env Geant4 would abort.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simws.config import ROOT, path as cfg_path

TOOL = ROOT / "tools" / "geant4_track"


def _prefix(prefix=None) -> Path:
    return Path(prefix) if prefix else cfg_path("envs_root") / "process_rad"


def dataset_env(prefix=None) -> dict:
    pre = _prefix(prefix)
    out = subprocess.run([str(pre / "bin" / "geant4-config"), "--datasets"], capture_output=True, text=True, check=True).stdout
    env = {}
    data_root = pre / "share" / "Geant4" / "data"
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        _, var, p = parts
        cand = [Path(p), data_root / Path(p).name, data_root / Path(p).name.removeprefix("G4")]
        found = next((c for c in cand if c.is_dir()), None)
        if found is None:
            raise FileNotFoundError(f"Geant4 dataset for {var} not found (tried {', '.join(map(str, cand))})")
        env[var] = str(found)
    return env


def build(prefix=None, force=False) -> Path:
    exe = TOOL / "g4track"
    if exe.exists() and not force and exe.stat().st_mtime > (TOOL / "g4track.cc").stat().st_mtime:
        return exe
    subprocess.run(["bash", str(TOOL / "build.sh"), str(_prefix(prefix))], check=True)
    return exe


def run(out_csv, Z, A, energy_MeV, material="G4_Si", thickness_um=20.0, rmax_um=2.0, nr=40, nz=40,
        events=100, cut_um=0.1, seed=1, prefix=None) -> Path:
    exe = build(prefix)
    env = os.environ.copy()
    env.update(dataset_env(prefix))
    env["LD_LIBRARY_PATH"] = f"{_prefix(prefix) / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    args = [str(exe), "--Z", Z, "--A", A, "--energy-MeV", energy_MeV, "--material", material,
            "--thickness-um", thickness_um, "--rmax-um", rmax_um, "--nr", nr, "--nz", nz, "--events", events,
            "--cut-um", cut_um, "--seed", seed, "--out", out_csv]
    subprocess.run([str(a) for a in args], env=env, check=True, capture_output=True, text=True)
    return Path(out_csv)


@dataclass
class Deposition:
    meta: dict
    z_edges_um: np.ndarray
    r_edges_um: np.ndarray            # nr+1 edges; the overflow bin (r >= rmax) is stored separately
    edep_MeV: np.ndarray              # (nz, nr) per primary
    overflow_MeV: np.ndarray          # (nz,)  r >= rmax, per primary

    @property
    def dz_um(self):
        return np.diff(self.z_edges_um)

    def dEdz_MeV_per_um(self):
        return (self.edep_MeV.sum(axis=1) + self.overflow_MeV) / self.dz_um

    def let_profile(self, mat):
        from simws.radiation.let import let_from_dedx
        zc = 0.5 * (self.z_edges_um[1:] + self.z_edges_um[:-1])
        return zc, let_from_dedx(self.dEdz_MeV_per_um(), mat)

    def radial_fraction(self, z_lo_um=None, z_hi_um=None):
        """Cumulative fraction of deposited energy inside radius r (r = r_edges[1:]) for a depth window."""
        zc = 0.5 * (self.z_edges_um[1:] + self.z_edges_um[:-1])
        sel = np.ones_like(zc, bool)
        if z_lo_um is not None:
            sel &= zc >= z_lo_um
        if z_hi_um is not None:
            sel &= zc <= z_hi_um
        e = self.edep_MeV[sel].sum(axis=0)
        return np.cumsum(e) / (e.sum() + self.overflow_MeV[sel].sum())


def load(csv_path) -> Deposition:
    meta = {}
    with open(csv_path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            for tok in line[1:].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        meta[k] = float(v)
                    except ValueError:
                        meta[k] = v
    lines = [l for l in open(csv_path) if not l.startswith("#")]
    cols = lines[0].strip().split(",")
    d = np.loadtxt(lines[1:], delimiter=",", ndmin=2)
    c = {name: d[:, i] for i, name in enumerate(cols)}
    nz, nr = int(meta["nz"]), int(meta["nr"])
    e = c["edep_MeV"].reshape(nz, nr + 1)
    z_edges = np.append(c["z_lo_um"].reshape(nz, nr + 1)[:, 0], c["z_hi_um"].reshape(nz, nr + 1)[-1, 0])
    r_edges = np.append(c["r_lo_um"].reshape(nz, nr + 1)[0, :nr], c["r_hi_um"].reshape(nz, nr + 1)[0, nr - 1])
    return Deposition(meta, z_edges, r_edges, e[:, :nr], e[:, nr])
