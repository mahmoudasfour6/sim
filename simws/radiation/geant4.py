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
        events=100, cut_um=0.1, seed=1, prefix=None, events_out=None) -> Path:
    """Run g4track; `events_out` (path) also writes each event's own deposition (see load_events)."""
    exe = build(prefix)
    env = os.environ.copy()
    env.update(dataset_env(prefix))
    env["LD_LIBRARY_PATH"] = f"{_prefix(prefix) / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    args = [str(exe), "--Z", Z, "--A", A, "--energy-MeV", energy_MeV, "--material", material,
            "--thickness-um", thickness_um, "--rmax-um", rmax_um, "--nr", nr, "--nz", nz, "--events", events,
            "--cut-um", cut_um, "--seed", seed, "--out", out_csv]
    if events_out:
        args += ["--events-out", events_out]
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


def load_events(csv_path):
    """Per-event depositions written by g4track --events-out. Returns (meta, list of Deposition), one per event,
    on the same (z, r) bins as the averaged file; events that deposited nothing are kept (all zeros)."""
    meta = {}
    with open(csv_path) as f:
        first = f.readline()
    for tok in first[1:].split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                meta[k] = float(v)
            except ValueError:
                meta[k] = v
    d = np.loadtxt(csv_path, delimiter=",", comments="#", skiprows=2, ndmin=2)
    nz, nr = int(meta["nz"]), int(meta["nr"])
    n_ev = int(d[:, 0].max()) + 1 if len(d) else 0
    z_edges = np.linspace(0, meta["thickness_um"], nz + 1)
    r_edges = np.linspace(0, meta["rmax_um"], nr + 1)
    out = []
    for ev in range(n_ev):
        sel = d[:, 0] == ev
        h = np.zeros((nz, nr + 1))
        h[d[sel, 1].astype(int), d[sel, 2].astype(int)] = d[sel, 3]
        out.append(Deposition(dict(meta, event=ev), z_edges, r_edges, h[:, :nr], h[:, nr]))
    return meta, out


def axisymmetric_footprint(dep: "Deposition", mat, r_um, z_um, node_volume_cm3, z_offset_um=0.0):
    """Time-integrated pair density (pairs/cm^3) at the nodes (r, z) of a tensor-product r-z mesh from one event's
    (or the average) Geant4 deposition, remapped CONSERVATIVELY: each node receives the pairs of the Geant4
    (z, r) bins that overlap its control ring [r-, r+] x [z-, z+] (midpoints to the neighbouring mesh lines),
    exactly, then divided by its NodeVolume. Sampling the bin density at the node instead over-weights the steep
    near-axis bins (~25 % on a 20 nm bin grid). The remaining correction factor (returned) covers energy beyond
    rmax or outside the mesh and the triangle-vs-ring volume difference. Returns (density, factor, total pairs)."""
    E_pair_MeV = mat.get("pair_energy", "eV") * 1e-6
    r = np.asarray(r_um, float); z = np.asarray(z_um, float) - z_offset_um
    vol = np.asarray(node_volume_cm3, float)
    re, ze = dep.r_edges_um, dep.z_edges_um
    pairs_bins = dep.edep_MeV / E_pair_MeV                                   # (nz, nr) pairs per bin
    ring_bins = np.pi * (re[1:] ** 2 - re[:-1] ** 2)                         # um^2 per radial bin
    dz_bins = np.diff(ze)

    def control(lines):
        u = np.unique(np.round(lines, 9))
        mid = 0.5 * (u[1:] + u[:-1])
        return u, np.concatenate([[u[0]], mid]), np.concatenate([mid, [u[-1]]])

    ru, rlo, rhi = control(r)
    zu, zlo, zhi = control(z)
    # area fraction of each radial bin inside each node's ring, length fraction of each z bin inside each node's slab
    Wr = np.clip(np.minimum(rhi[:, None], re[None, 1:]) ** 2 - np.maximum(rlo[:, None], re[None, :-1]) ** 2, 0, None)
    Wr = np.where(np.minimum(rhi[:, None], re[None, 1:]) > np.maximum(rlo[:, None], re[None, :-1]), Wr, 0.0) * np.pi / ring_bins
    Wz = np.clip(np.minimum(zhi[:, None], ze[None, 1:]) - np.maximum(zlo[:, None], ze[None, :-1]), 0, None) / dz_bins
    P = Wz @ pairs_bins @ Wr.T                                               # (n_z_lines, n_r_lines) pairs per ring
    ir = np.searchsorted(ru, np.round(r, 9)); iz = np.searchsorted(zu, np.round(z, 9))
    node_pairs = P[iz, ir]
    dens = np.where(vol > 0, node_pairs / np.where(vol > 0, vol, 1.0), 0.0)
    total = float((dep.edep_MeV.sum() + dep.overflow_MeV.sum()) / E_pair_MeV)
    on_mesh = float((dens * vol).sum())
    factor = total / on_mesh if on_mesh > 0 else float("nan")
    return dens * factor, factor, total
