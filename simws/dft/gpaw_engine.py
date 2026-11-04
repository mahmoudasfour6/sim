"""GPAW (PAW, plane waves) beside Quantum ESPRESSO (0.9c): the same reference calculations through a second, independent
DFT code, so that the code-to-code spread enters the model-form budget.

    from simws.dft import gpaw_engine as G
    G.equation_of_state("Si", a_values, ecut_eV=500, kgrid=(8, 8, 8))     # same a grid as the QE run
    G.band_energies("Si", 5.431, klist_tpiba)                             # eigenvalues on an explicit k list

k-points are given as in simws.dft.qe: cartesian, units of 2 pi / a. GPAW runs in-process (ASE calculator); its PAW
setups come from the gpaw-data package (or GPAW_SETUP_PATH). Exchange-correlation: "PBE", "LDA".
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def _atoms(symbol, a, structure="diamond"):
    from ase.build import bulk
    return bulk(symbol, structure, a=a)


def _calc(xc, ecut_eV, kpts, txt, **kw):
    from gpaw import GPAW, PW
    return GPAW(mode=PW(ecut_eV), xc=xc, kpts=kpts, txt=txt, convergence={"energy": 1e-7, "density": 1e-7},
                occupations={"name": "fermi-dirac", "width": 0.0}, **kw)


def equation_of_state(symbol, a_values: Sequence[float], xc="PBE", ecut_eV=500.0, kgrid=(8, 8, 8),
                      workdir="eos_gpaw", structure="diamond") -> dict:
    from simws.dft.qe import birch_murnaghan
    wd = Path(workdir); wd.mkdir(parents=True, exist_ok=True)
    E, V = [], []
    for i, a in enumerate(a_values):
        at = _atoms(symbol, a, structure)
        at.calc = _calc(xc, ecut_eV, {"size": kgrid, "gamma": False}, str(wd / f"a{i}.txt"))
        E.append(float(at.get_potential_energy())); V.append(float(at.get_volume()))
    fit = birch_murnaghan(V, E)
    a0 = (4 * fit["V0_A3"]) ** (1 / 3)
    fit.update({"a0_A": a0, "a0_sigma_A": a0 / 3 * fit["sigma"]["V0_A3"] / fit["V0_A3"], "a_A": list(a_values),
                "E_eV": E})
    import gpaw
    fit["engine"] = f"gpaw {gpaw.__version__}"
    return fit


def band_energies(symbol, a, klist_tpiba, xc="PBE", ecut_eV=500.0, kgrid=(8, 8, 8), nbands=8, workdir="bands_gpaw",
                  structure="diamond") -> dict:
    """SCF on kgrid, then fixed-density eigenvalues on the k list. Returns eigenvalues (nk, nbands) in eV."""
    wd = Path(workdir); wd.mkdir(parents=True, exist_ok=True)
    at = _atoms(symbol, a, structure)
    at.calc = _calc(xc, ecut_eV, {"size": kgrid, "gamma": False}, str(wd / "scf.txt"))
    at.get_potential_energy()
    k = np.asarray(klist_tpiba, float)[:, :3]
    frac = k @ at.cell.array.T / a                        # k . a_i / a  (k in 2 pi / a)
    nscf = at.calc.fixed_density(kpts=frac, symmetry="off", nbands=nbands, txt=str(wd / "nscf.txt"),
                                 convergence={"bands": nbands})
    eig = np.array([nscf.get_eigenvalues(kpt=i) for i in range(len(frac))])
    return {"k_tpiba": k.tolist(), "eigenvalues_eV": eig, "n_electrons": float(nscf.get_number_of_electrons())}
