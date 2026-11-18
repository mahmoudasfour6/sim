"""Geant4 implant translator (architecture v1, contract amendments A13 and A14): an ImplantProblem on Geant4 11.2.2.

The program tools/g4implant/g4implant.cc (built by tools/install_g4implant.sh into the g4see env) follows each ion
with TestEm7's G4ScreenedNuclearRecoil (Mendenhall and Weller, NIM B 227 (2005) 420) for the nuclear collisions and
either Lindhard-Scharff electronic stopping (the formula RustBCA uses, written as a continuous process) or Geant4's
own ion stopping (G4IonParametrisedLossModel, "empirical"). It is different code from RustBCA, so a comparison of
the two is a verification (A13.5 R65). Importing this module does not need Geant4.

Geometry as RustBCA's (A7.4): x is depth below the front surface, the plane of incidence is x-y, the lateral
displacement is z. With models.recoils (A14), every recoil the program creates is a displaced target atom.
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from simws.engines.base import EngineNotAvailable, ImplantCapability, ImplantResult, Unsupported

FAMILY = "implant"
CODE = "geant4"
NAME = "g4implant"
GEANT4_VERSION = "11.2.2"
PHYSICS_CUTOFF_EV = 10.0          # A13.3: G4ScreenedNuclearRecoil's own default
BATCHES = 20                      # A14.3

CAPABILITY = ImplantCapability(
    engine=NAME, env="g4see", in_process=False,
    description="Geant4 11.2.2 with G4ScreenedNuclearRecoil: ion ranges and displacement counts in layered "
                "amorphous targets",
    structures=["amorphous"], potentials=["zbl", "moliere"], electronic_stopping=["lindhard_scharff", "empirical"],
    recoils=[False, True], max_layers=20, max_keV_per_amu=25.0,
    # A13.6: what passed acceptance: R65 (cross-code, Lindhard-Scharff) for boron at 50 and 100 keV only; every R64
    # case ("empirical" against the tables) failed, so nothing is tested in that mode
    tested_keV_per_amu=(4.5, 9.1), tested_target_symbols=["Si"], tested_ion_symbols=["B"],
    tested_ion_symbols_by_stopping={"empirical": ["none"]},
    notes="Collisions transferring less than 10 eV are ignored (G4ScreenedNuclearRecoil's physics cutoff). No "
          "surface or bulk binding energies. Damage (A14): one target species only, Ed of at least 10 eV, no bulk "
          "binding. Serial run: about 1 s per 2000 ions without recoils.")

_SCREENING = {"zbl": "zbl", "moliere": "mol"}
AMORPHOUS_FLAG = ("implant:amorphous_target: no channeling (a crystalline wafer shows a deeper tail) and no damage "
                  "build-up (the profile is linear in dose)")
CUTOFF_FLAG = ("implant:geant4_nuclear_cutoff: collisions transferring less than 10 eV are ignored "
               "(G4ScreenedNuclearRecoil physics cutoff, A13.3)")
EMPIRICAL_FLAG = ("implant:empirical_stopping: Geant4 G4IonParametrisedLossModel (its ion data, scaled with the "
                  "square root of the energy below the lowest table energy; A13)")
BORON_FLAG = ("implant:known_failure: boron ranges in silicon with Lindhard-Scharff stopping are 45 % too deep here "
              "(A13.7, as RustBCA's in A7.9: the stopping is too weak): trends only, not numbers")
ION_STOP_FLAG = ("implant:damage_ion_cutoff: with recoils the ion stops below Ed x m_ion / m_target instead of its "
                 "cutoff_eV (one recoil cutoff per nucleon in G4ScreenedNuclearRecoil, A14.3)")


NEAR_CUTOFF_FLAG = ("implant:damage_near_cutoff: Ed below 30 eV (three times the 10 eV physics cutoff): there the "
                    "Geant4 count barely follows Ed (A14.7: R71 failed at 15 eV); use the RustBCA count")


def _used_atoms(problem):
    used = {sym for la in problem.target.layers for sym in la.density_cm3}
    return [a for a in problem.target.atoms if a.symbol in used]


def unsupported(problem) -> List[str]:
    out = CAPABILITY.unsupported(problem)
    if problem.models.recoils:
        atoms = _used_atoms(problem)
        if len(atoms) != 1:
            out.append(f"recoils: damage in this engine needs one target species (one recoil cutoff); the target "
                       f"holds {sorted(a.symbol for a in atoms)}")
        for a in atoms:
            if a.displacement_eV is not None and a.displacement_eV < PHYSICS_CUTOFF_EV:
                out.append(f"recoils: {a.symbol} has displacement_eV {a.displacement_eV:g} below the "
                           f"{PHYSICS_CUTOFF_EV:g} eV physics cutoff (no smaller transfer is ever made)")
            if a.bulk_binding_eV > 0:
                out.append(f"recoils: {a.symbol} has bulk_binding_eV {a.bulk_binding_eV:g}; this engine has no "
                           f"bulk binding")
    return out


def prefix() -> Path:
    """The conda env the program lives in: as the G4SEE engine (SIMWS_G4SEE_PREFIX, else <envs_root>/g4see)."""
    from simws.engines import g4see_engine
    return g4see_engine.prefix()


def executable() -> Path:
    return prefix() / "bin" / "g4implant"


def available() -> bool:
    try:
        exe = executable()
        return exe.is_file() and os.access(exe, os.X_OK)
    except Exception:  # noqa: BLE001
        return False


def cutoffs(problem) -> Tuple[float, float]:
    """(recoil cutoff per nucleon in eV/u, physics cutoff in eV) for the program (A13.3, A14.3)."""
    if problem.models.recoils:
        a = _used_atoms(problem)[0]
        return float(a.displacement_eV) / float(a.mass_amu), PHYSICS_CUTOFF_EV
    ion = problem.beam.ion
    return float(ion.cutoff_eV) / float(ion.mass_amu), PHYSICS_CUTOFF_EV


def nbins(problem) -> int:
    return int(math.ceil(problem.target.thickness_um / problem.depth_bin_um - 1e-9))


def build_input(problem, dedx_keV: Tuple[float, ...] = ()) -> str:
    """The program's input file for `problem` (pure Python)."""
    ion = problem.beam.ion
    rc, pc = cutoffs(problem)
    L = [f"# simws contract A13, problem {problem.problem_id()}",
         f"seed {int(problem.statistics.seed)}", f"n {int(problem.statistics.n_ions)}",
         f"batches {BATCHES if problem.models.recoils else 1}",
         f"ion {int(ion.Z)} {int(round(ion.mass_amu))}", f"energy_keV {float(problem.beam.energy_keV)!r}",
         f"tilt_deg {float(problem.beam.tilt_deg)!r}", f"screening {_SCREENING[problem.models.potential]}",
         f"stopping {'empirical' if problem.models.electronic_stopping == 'empirical' else 'ls'}",
         f"recoil_cutoff_eV_per_u {rc!r}", f"physics_cutoff_eV {pc!r}",
         f"recoils {1 if problem.models.recoils else 0}"]
    if problem.models.recoils:
        L.append(f"bins {float(problem.depth_bin_um)!r} {nbins(problem)}")
    atoms = {a.symbol: a for a in problem.target.atoms}
    for la in problem.target.layers:
        L.append(f"layer {la.name} {float(la.thickness_um)!r} {len(la.density_cm3)}")
        for sym, n in la.density_cm3.items():
            a = atoms[sym]
            L.append(f"{sym} {int(a.Z)} {float(a.mass_amu)!r} {float(n)!r}")
    if dedx_keV:
        L.append("dedx_keV " + " ".join(repr(float(e)) for e in dedx_keV))
    return "\n".join(L) + "\n"


def parse_output(text: str) -> Dict[str, Any]:
    """The program's output: header values, the ion end points and the damage histograms."""
    out: Dict[str, Any] = {"status": [], "x": [], "z": [], "batches": [], "batch_ions": [], "outside": 0,
                           "dedx": {}, "materials": {}, "header": ""}
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith("#"):
            w = line[1:].split()
            if not w:
                continue
            if w[0] == "g4implant":
                out["header"] = " ".join(w)
            elif w[0] == "material":
                out["materials"][w[1]] = float(w[2])
            elif w[0] == "ion":
                out["ion_mass_u"] = float(w[4])
            elif w[0] == "dedx_e":
                out["dedx"][float(w[1])] = float(w[3])
            continue
        w = line.split()
        if w[0] == "D":
            out["batch_ions"].append(int(w[2]))
            out["outside"] += int(w[3])
            out["batches"].append([float(c) for c in w[4:]])
        else:
            out["status"].append(int(w[0]))
            out["x"].append(float(w[1]))
            out["z"].append(float(w[2]))
    return out


def _env() -> Dict[str, str]:
    from simws.radiation.geant4 import dataset_env
    pre = prefix()
    env = dict(os.environ)
    env.update(dataset_env(pre))
    env["LD_LIBRARY_PATH"] = f"{pre / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def run_program(text: str, folder: Path) -> str:
    """Run the program on an input text in `folder`; return its output text."""
    if not available():
        raise EngineNotAvailable(f"no g4implant program at {executable()}; build it with tools/install_g4implant.sh "
                                 f"(it needs the g4see env of tools/install_g4see.sh, or SIMWS_G4SEE_PREFIX)")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "input.txt").write_text(text)
    log = folder / "g4implant.log"
    with open(log, "w") as fh:
        rc = subprocess.run([str(executable()), "input.txt", "output.txt"], cwd=folder, env=_env(), stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-20:])
        raise RuntimeError(f"g4implant exited with code {rc}; last lines of {log}:\n{tail}")
    return (folder / "output.txt").read_text()


def dedx(problem, energies_keV) -> Dict[float, float]:
    """Electronic dE/dx (MeV cm^2/g) of the beam ion in the first layer, as the program uses it (R63)."""
    p = problem.with_changes({"statistics.n_ions": 100})
    with tempfile.TemporaryDirectory(prefix="simws_g4implant_") as d:
        txt = build_input(p, tuple(energies_keV)).replace("n 100\n", "n 1\n")
        return parse_output(run_program(txt, Path(d)))["dedx"]


def run(problem, workdir=None) -> ImplantResult:
    reasons = unsupported(problem)
    if reasons:
        raise Unsupported(NAME, reasons)
    pid = problem.problem_id()
    tmp = None
    if workdir is None:
        tmp = tempfile.TemporaryDirectory(prefix="simws_g4implant_")
        folder = Path(tmp.name)
    else:
        folder = Path(workdir) / f"g4implant_{pid[:12]}"
    try:
        o = parse_output(run_program(build_input(problem), folder))
    finally:
        if tmp is not None:
            tmp.cleanup()
    n = int(problem.statistics.n_ions)
    if len(o["status"]) != n:
        raise RuntimeError(f"g4implant wrote {len(o['status'])} ion lines for {n} ions")
    depth = [x for s, x in zip(o["status"], o["x"]) if s == 0]
    lateral = [z for s, z in zip(o["status"], o["z"]) if s == 0]
    flags = [AMORPHOUS_FLAG, CUTOFF_FLAG]
    if problem.models.electronic_stopping == "empirical":
        flags.append(EMPIRICAL_FLAG)
    elif problem.beam.ion.Z == 5 and any("Si" in la.density_cm3 for la in problem.target.layers):
        flags.append(BORON_FLAG)
    if problem.models.recoils:
        flags.append(ION_STOP_FLAG)
        if any((a.displacement_eV or 0) < 3 * PHYSICS_CUTOFF_EV for a in _used_atoms(problem)):
            flags.append(NEAR_CUTOFF_FLAG)
    rc, pc = cutoffs(problem)
    diag = {"geant4": o["header"], "ion_mass_u": o.get("ion_mass_u"), "physics_cutoff_eV": pc,
            "recoil_cutoff_eV_per_u": rc, "materials_g_cm3": o["materials"]}
    extra: Dict[str, Any] = {}
    if problem.models.recoils:
        if len(o["batches"]) != BATCHES or sum(o["batch_ions"]) != n:
            raise RuntimeError(f"g4implant damage output: {len(o['batches'])} batches for {sum(o['batch_ions'])} "
                               f"ions, expected {BATCHES} for {n}")
        extra = {"vacancy_bin_um": float(problem.depth_bin_um), "vacancy_batches": o["batches"],
                 "vacancy_batch_ions": o["batch_ions"]}
        diag["vacancies_outside"] = int(o["outside"])
        diag["damage_batches"] = BATCHES
    return ImplantResult(engine=NAME, problem_id=pid, n_ions=n, n_reflected=o["status"].count(1),
                         n_transmitted=o["status"].count(2), depth_um=depth, lateral_um=lateral, flags=flags,
                         diagnostics=diag, **extra)
