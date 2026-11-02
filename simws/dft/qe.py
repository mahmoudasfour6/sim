"""Minimal Quantum ESPRESSO (pw.x) driver: write an input, run it, read the XML result. No ASE needed.

    from simws.dft.qe import Structure, PW, run
    s = Structure.diamond("Si", a_angstrom=5.431)
    r = run(PW(s, pseudos={"Si": "Si_r.upf"}, ecutwfc=40, kgrid=(8, 8, 8)), workdir="runs/si_scf")
    r.energy_Ry, r.homo_eV, r.kpoints_tpiba, r.eigenvalues_eV

Calculations:
  scf                 kgrid (Monkhorst-Pack, shifted by default)
  nscf / bands        an explicit k list in units of 2 pi / a (cartesian), run from the scf charge density
                      (same workdir and prefix). "bands" ignores weights, "nscf" uses them.
Spin-orbit: soc=True (noncollinear, lspinorb) needs a fully-relativistic pseudopotential. A fully-relativistic
pseudopotential also works without soc (pw.x averages the j-components).
Functional: functional=None uses the pseudopotential's own; any other value is written as input_dft (consistent
only if the pseudopotential was generated with it: the result records the mismatch).
Hybrids: functional="hse" with nqx (EXX q-mesh); norm-conserving pseudopotentials strongly recommended.

pw.x is found on PATH (or SIMWS_PW). MPI: ranks > 1 runs `mpirun --oversubscribe -np N`, then plain
`mpirun -np N` (MPICH), as in the workflow's qe_si_scf rule.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HA_EV = 27.211386245988
RY_EV = HA_EV / 2
BOHR_A = 0.529177210903
HA_BOHR3_GPA = 29421.01569650548


@dataclass
class Structure:
    """Cell vectors (Angstrom, rows) and atoms (symbol, fractional coordinates)."""
    cell: np.ndarray
    symbols: List[str]
    frac: np.ndarray
    alat_angstrom: float          # the length unit "a" of 2 pi / a k-points

    @classmethod
    def diamond(cls, symbol: str, a_angstrom: float):
        a = a_angstrom
        cell = 0.5 * a * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])   # QE ibrav=2 vectors
        return cls(cell, [symbol, symbol], np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]), a)

    @classmethod
    def zincblende(cls, s1: str, s2: str, a_angstrom: float):
        st = cls.diamond(s1, a_angstrom)
        st.symbols = [s1, s2]
        return st

    def strained(self, eps: np.ndarray):
        """Apply the (symmetric) strain tensor eps to the cell; fractional coordinates kept (no relaxation)."""
        F = np.eye(3) + np.asarray(eps, float)
        return Structure(self.cell @ F.T, list(self.symbols), self.frac.copy(), self.alat_angstrom)

    @property
    def volume_A3(self):
        return float(abs(np.linalg.det(self.cell)))

    def reciprocal_tpiba(self):
        """Reciprocal vectors (rows) in units of 2 pi / alat."""
        return np.linalg.inv(self.cell).T * self.alat_angstrom


@dataclass
class PW:
    structure: Structure
    pseudos: Dict[str, str]                         # element -> UPF file name (in pseudo_dir)
    ecutwfc: float = 40.0
    ecutrho: Optional[float] = None
    calculation: str = "scf"
    kgrid: Tuple[int, int, int] = (8, 8, 8)
    kshift: Tuple[int, int, int] = (1, 1, 1)
    klist: Optional[np.ndarray] = None              # (N, 3) in 2 pi / a, or (N, 4) with weights
    nbnd: Optional[int] = None
    functional: Optional[str] = None
    soc: bool = False
    nqx: Tuple[int, int, int] = (2, 2, 2)
    tstress: bool = False
    relax_atoms: bool = False                       # calculation='relax' (fixed cell) when set with scf
    conv_thr: float = 1e-10
    prefix: str = "pw"
    pseudo_dir: Optional[str] = None
    extra_system: Dict[str, object] = field(default_factory=dict)
    masses: Dict[str, float] = field(default_factory=lambda: {"Si": 28.085, "Ge": 72.630, "C": 12.011, "O": 15.999})

    def text(self, pseudo_dir: str, outdir: str = "./out") -> str:
        s = self.structure
        species = list(dict.fromkeys(s.symbols))
        system = {"ibrav": 0, "nat": len(s.symbols), "ntyp": len(species), "ecutwfc": self.ecutwfc}
        if self.ecutrho:
            system["ecutrho"] = self.ecutrho
        if self.nbnd:
            system["nbnd"] = self.nbnd
        if self.functional:
            f = self.functional.lower()
            system["input_dft"] = f"'{f}'"
            if f in ("hse", "pbe0", "b3lyp", "gaupbe"):
                system.update({"nqx1": self.nqx[0], "nqx2": self.nqx[1], "nqx3": self.nqx[2]})
        if self.soc:
            system.update({"noncolin": ".true.", "lspinorb": ".true."})
        system.update(self.extra_system)
        alat_bohr = s.alat_angstrom / BOHR_A
        system["celldm(1)"] = f"{alat_bohr:.10f}"
        calc = "relax" if (self.relax_atoms and self.calculation == "scf") else self.calculation
        lines = ["&CONTROL", f"  calculation = '{calc}'", f"  prefix = '{self.prefix}'",
                 f"  outdir = '{outdir}'", f"  pseudo_dir = '{pseudo_dir}'", "  verbosity = 'high'",
                 "  tprnfor = .true." if self.relax_atoms else "  tprnfor = .false.",
                 f"  tstress = {'.true.' if self.tstress else '.false.'}",
                 "  forc_conv_thr = 1.0d-5", "  etot_conv_thr = 1.0d-7", "/", "&SYSTEM"]
        lines += [f"  {k} = {v}" for k, v in system.items()]
        lines += ["/", "&ELECTRONS", f"  conv_thr = {self.conv_thr:.1e}",
                  "  diago_full_acc = .true." if self.calculation in ("nscf", "bands") else "  mixing_beta = 0.7", "/"]
        if calc == "relax":
            lines += ["&IONS", "/"]
        lines.append("ATOMIC_SPECIES")
        lines += [f"  {e} {self.masses.get(e, 1.0)} {self.pseudos[e]}" for e in species]
        lines.append("CELL_PARAMETERS alat")
        lines += ["  " + " ".join(f"{x / s.alat_angstrom:.12f}" for x in row) for row in s.cell]
        lines.append("ATOMIC_POSITIONS crystal")
        lines += [f"  {sym} " + " ".join(f"{x:.12f}" for x in r) for sym, r in zip(s.symbols, s.frac)]
        if self.klist is not None:
            k = np.asarray(self.klist, float)
            if k.shape[1] == 3:
                k = np.column_stack([k, np.ones(len(k))])
            lines.append(f"K_POINTS tpiba\n  {len(k)}")
            lines += ["  " + " ".join(f"{x:.10f}" for x in row[:3]) + f" {row[3]:.10g}" for row in k]
        else:
            lines.append("K_POINTS automatic")
            lines.append("  " + " ".join(str(n) for n in (*self.kgrid, *self.kshift)))
        return "\n".join(lines) + "\n"


@dataclass
class Result:
    energy_Ry: float
    kpoints_tpiba: np.ndarray        # (nk, 3), 2 pi / a
    weights: np.ndarray
    eigenvalues_eV: np.ndarray       # (nk, nbnd)
    homo_eV: Optional[float]
    lumo_eV: Optional[float]
    n_electrons: float
    alat_angstrom: float
    functional: str
    pw_version: str
    workdir: str
    wall_s: float
    notes: List[str] = field(default_factory=list)

    def band_edges(self):
        """(VBM, CBM) in eV from the occupied/empty split (spin-degenerate bands unless SOC)."""
        nocc = int(round(self.n_electrons / (1 if self.noncolin else 2)))
        vb = self.eigenvalues_eV[:, :nocc].max()
        cb = self.eigenvalues_eV[:, nocc:].min() if self.eigenvalues_eV.shape[1] > nocc else float("nan")
        return float(vb), float(cb)

    noncolin: bool = False
    stress_GPa: Optional[np.ndarray] = None     # QE sign convention as written in the XML (see simws.dft.strain)


def _find_pw():
    exe = os.environ.get("SIMWS_PW") or shutil.which("pw.x")
    if not exe:
        raise FileNotFoundError("pw.x not found: activate the QE environment or set SIMWS_PW")
    return exe


def default_pseudo_dir():
    """config data.pseudopotentials if set and present, else the Debian/Ubuntu QE package directory."""
    try:
        from simws.config import load as cfg_load
        p = Path(cfg_load()["data"]["pseudopotentials"])
        if p.is_dir():
            return str(p.resolve())
    except Exception:  # noqa: BLE001
        pass
    for p in (Path("/usr/share/espresso/pseudo"),):
        if p.is_dir():
            return str(p)
    raise FileNotFoundError("no pseudopotential directory: set data.pseudopotentials in config/config.yaml")


def run(calc: PW, workdir, ranks: int = 1, timeout: float = 7200.0) -> Result:
    import time
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    pdir = calc.pseudo_dir or default_pseudo_dir()
    inp = wd / f"{calc.prefix}.{calc.calculation}.pwi"
    inp.write_text(calc.text(str(Path(pdir).resolve())))
    pw = [_find_pw(), "-in", inp.name]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    attempts = ([["mpirun", "--oversubscribe", "-np", str(ranks)] + pw, ["mpirun", "-np", str(ranks)] + pw]
                if ranks > 1 else [pw])
    if os.geteuid() == 0:                       # containers run as root; Open MPI refuses that by default
        env.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    t0 = time.time()
    err = ""
    for cmd in attempts:
        r = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, env=env, timeout=timeout)
        if r.returncode == 0 and "JOB DONE" in r.stdout:
            break
        err += f"{' '.join(cmd)} -> {r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}\n"
    else:
        raise RuntimeError(f"pw.x failed in {wd}:\n{err}")
    (wd / f"{calc.prefix}.{calc.calculation}.pwo").write_text(r.stdout)
    res = parse_xml(wd / "out" / f"{calc.prefix}.save" / "data-file-schema.xml")
    res.workdir, res.wall_s = str(wd), time.time() - t0
    m = re.search(r"Program PWSCF v\.(\S+)", r.stdout)
    res.pw_version = m.group(1) if m else "?"
    if calc.functional and not _pseudo_functional_matches(Path(pdir), calc):
        res.notes.append(f"input_dft={calc.functional} differs from the pseudopotential functional")
    return res


def _pseudo_functional_matches(pdir: Path, calc: PW) -> bool:
    want = calc.functional.lower()
    for f in calc.pseudos.values():
        txt = (pdir / f).read_text(errors="ignore")[:20000].lower()
        if want not in txt:
            return False
    return True


def parse_xml(path) -> Result:
    root = ET.parse(path).getroot()
    out = root.find("output")
    bs = out.find("band_structure")
    alat_bohr = float(out.find("atomic_structure").get("alat"))
    ks = bs.findall("ks_energies")
    k = np.array([[float(x) for x in e.find("k_point").text.split()] for e in ks])
    w = np.array([float(e.find("k_point").get("weight")) for e in ks])
    ev = np.array([[float(x) for x in e.find("eigenvalues").text.split()] for e in ks]) * HA_EV
    homo = bs.find("highestOccupiedLevel")
    lumo = bs.find("lowestUnoccupiedLevel")
    nel = float(bs.find("nelec").text)
    noncolin = (bs.find("noncolin") is not None and bs.find("noncolin").text.strip() == "true")
    etot = float(out.find("total_energy").find("etot").text) * 2.0
    st = out.find("stress")
    stress = (np.array([float(x) for x in st.text.split()]).reshape(3, 3) * HA_BOHR3_GPA) if st is not None else None
    dft = out.find("dft")
    func = dft.find("functional").text if dft is not None and dft.find("functional") is not None else "?"
    res = Result(etot, k, w, ev, float(homo.text) * HA_EV if homo is not None else None,
                 float(lumo.text) * HA_EV if lumo is not None else None, nel, alat_bohr * BOHR_A, func, "?", "", 0.0)
    res.noncolin = noncolin
    res.stress_GPa = stress
    return res


# ----------------------------------------------------------------------------------------------- equation of state
def birch_murnaghan(V, E):
    """Third-order Birch-Murnaghan fit E(V), done as linear least squares (E is a cubic in x = V^(-2/3)), so no
    SciPy is needed. Returns dict(V0_A3, E0_eV, B0_GPa, B0p, fit_rms_meV, sigma) with V in A^3, E in eV."""
    P = np.polynomial.polynomial.Polynomial
    V, E = np.asarray(V, float), np.asarray(E, float)
    x = V ** (-2.0 / 3.0)
    A = np.column_stack([np.ones_like(x), x, x ** 2, x ** 3])
    c, *_ = np.linalg.lstsq(A, E, rcond=None)
    p = P(c)
    roots = [r.real for r in p.deriv().roots() if abs(r.imag) < 1e-12 and r.real > 0 and p.deriv(2)(r.real) > 0]
    x0 = min(roots, key=lambda r: abs(r - x[np.argmin(E)]))
    V0 = x0 ** -1.5
    d2, d3 = p.deriv(2)(x0), p.deriv(3)(x0)
    xp = -2.0 / 3.0 * V0 ** (-5.0 / 3.0)                     # dx/dV, d2x/dV2 at V0
    xpp = 10.0 / 9.0 * V0 ** (-8.0 / 3.0)
    e2 = d2 * xp ** 2                                         # d2E/dV2 (dE/dx = 0 at x0)
    e3 = d3 * xp ** 3 + 3 * d2 * xp * xpp                     # d3E/dV3
    B0, Bp = V0 * e2, -1.0 - V0 * e3 / e2
    resid = E - A @ c
    s2 = float(resid @ resid) / max(len(V) - 4, 1)
    cov = s2 * np.linalg.pinv(A.T @ A)
    g = np.array([0.0, 1.0, 2 * x0, 3 * x0 ** 2])             # d(dE/dx at x0)/dc
    sV0 = 1.5 * V0 / x0 * math.sqrt(max(float(g @ cov @ g), 0.0)) / abs(d2)
    return {"E0_eV": float(p(x0)), "V0_A3": float(V0), "B0_GPa": float(B0 * 160.21766208), "B0p": float(Bp),
            "fit_rms_meV": float(np.sqrt(np.mean(resid ** 2)) * 1e3), "sigma": {"V0_A3": float(sV0)}}


def equation_of_state(symbol, a_guess, pseudos, functional=None, ecutwfc=40.0, kgrid=(8, 8, 8), npts=7, span=0.03,
                      workdir="eos", ranks=1, structure="diamond", second=None, pseudo_dir=None):
    """Lattice constant and bulk modulus from a Birch-Murnaghan fit over a_guess*(1 +- span)."""
    a = a_guess * np.linspace(1 - span, 1 + span, npts)
    E, V = [], []
    notes = []
    for i, ai in enumerate(a):
        st = (Structure.diamond(symbol, ai) if structure == "diamond" else Structure.zincblende(symbol, second, ai))
        r = run(PW(st, pseudos, ecutwfc=ecutwfc, kgrid=kgrid, functional=functional, prefix=f"eos{i}",
                   pseudo_dir=pseudo_dir),
                Path(workdir) / f"a{i}", ranks=ranks)
        E.append(r.energy_Ry * RY_EV); V.append(st.volume_A3)
        notes += r.notes
        shutil.rmtree(Path(workdir) / f"a{i}" / "out", ignore_errors=True)
    fit = birch_murnaghan(V, E)
    a0 = (4 * fit["V0_A3"]) ** (1 / 3)          # fcc primitive cell: V = a^3 / 4
    fit.update({"a0_A": a0, "a0_sigma_A": a0 / 3 * fit["sigma"]["V0_A3"] / fit["V0_A3"],
                "a_A": a.tolist(), "E_eV": E, "notes": sorted(set(notes))})
    return fit
