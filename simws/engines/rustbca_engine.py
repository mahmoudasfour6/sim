"""RustBCA translator (architecture v1, contract amendment A7.4): an ImplantProblem on RustBCA 3.0.0.

RustBCA (Drobny and Curreli, JOSS, doi 10.21105/joss.03298; GPL-3) is a binary-collision Monte Carlo code. It is
built from source as the Python module `libRustBCA` (tools/install_rustbca.sh) and called in process through
`rustbca_local_py(input, "1D")`, which returns where every ion ended. Importing this module does not import the
engine.

Geometry: x is depth below the front surface (x = 0), the plane of incidence is x-y, so the axis perpendicular to
it is z. The target is a stack of laterally infinite amorphous layers.

What it cannot do, and so refuses or flags (contract A7.6): crystalline targets (no channeling), beams above 25 keV
per nucleon (Lindhard-Scharff stopping only), dose effects. Damage (contract A14): with models.recoils the ions run
through RustBCA's own command-line program in 20 batches, and every recoil that started with more than its Ed
counts as a vacancy.
Known failure (contract A7.9): boron ranges in silicon are about 35 % too deep with Lindhard-Scharff stopping;
such results carry a flag. Contract A12: electronic_stopping "empirical" multiplies Lindhard-Scharff by a factor
taken from measured stopping powers (simws.radiation.empirical_stopping); it exists for boron in silicon only.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from simws.engines.base import EngineNotAvailable, ImplantCapability, ImplantResult, Unsupported

FAMILY = "implant"
NAME = "rustbca"

CAPABILITY = ImplantCapability(
    engine=NAME, env="tcad", in_process=True,
    description="RustBCA 3.0.0: binary-collision Monte Carlo, ion ranges in layered amorphous targets",
    structures=["amorphous"], potentials=["zbl", "kr_c", "moliere"], electronic_stopping=["lindhard_scharff", "empirical"],
    recoils=[False, True], max_layers=20,
    max_keV_per_amu=25.0,                 # RustBCA README: its low-energy stopping models hold below 25 keV/nucleon
    tested_keV_per_amu=(1.6, 3.3),        # what passed contract A7.7 R16: P at 50 and 100 keV (A7.9)
    tested_target_symbols=["Si"], tested_ion_symbols=["P"],
    tested_ion_symbols_by_stopping={"empirical": ["B"]},   # A12.6: all four R16e checks passed (A12.8)
    notes="Boron ranges in silicon FAILED acceptance R16: about 35 % too deep, because Lindhard-Scharff electronic "
          "stopping is too weak for boron in silicon (contract A7.9). electronic_stopping 'empirical' (A12) scales it "
          "by a factor from measured stopping powers, boron in silicon only. Damage (A14): recoils followed in 20 "
          "batches through RustBCA's command-line program (Lindhard-Scharff stopping only), memory bounded by a "
          "chunk of 2 x threads ions. Cross-code check against Geant4 (A13.7, R65): boron agrees within 6 to 10 %, "
          "phosphorus does not (13 % in Rp, 16 % in dRp).")

_POTENTIAL = {"zbl": "ZBL", "kr_c": "KR_C", "moliere": "MOLIERE"}
_STOPPING = {"lindhard_scharff": "LOW_ENERGY_NONLOCAL", "empirical": "LOW_ENERGY_NONLOCAL"}
AMORPHOUS_FLAG = ("implant:amorphous_target: no channeling (a crystalline wafer shows a deeper tail) and no damage "
                  "build-up (the profile is linear in dose)")
BORON_FLAG = ("implant:known_failure: boron ranges in silicon failed acceptance R16, about 35 % too deep "
              "(electronic stopping too weak; contract A7.9): trends only, not numbers")


def _empirical_reasons(problem) -> List[str]:
    from simws.radiation import empirical_stopping as es
    if problem.models.electronic_stopping != "empirical":
        return []
    out = []
    ion = problem.beam.ion.symbol
    if es.entry(ion) is None:
        out.append(f"electronic_stopping: no measured-data factor for {ion} in silicon (contract A12.2: needs at "
                   f"least 3 measured rows from 2 publications at or below 10 keV/u)")
    other = sorted({s for la in problem.target.layers for s in la.density_cm3} - {"Si"})
    if other:
        out.append(f"electronic_stopping: 'empirical' needs layers of silicon only (the factor is for silicon); "
                   f"the target also holds {other}")
    return out


BATCHES = 20
SEED_LIMIT = 2_000_000


def _damage_reasons(problem) -> List[str]:
    if not problem.models.recoils:
        return []
    out = []
    if problem.models.electronic_stopping == "empirical":
        out.append("electronic_stopping: 'empirical' with recoils is refused (RustBCA would scale the recoils' "
                   "stopping by the ion's factor too; contract A14.3)")
    used = {sym for la in problem.target.layers for sym in la.density_cm3}
    for a in problem.target.atoms:
        if a.symbol in used and a.displacement_eV is not None and a.displacement_eV < a.cutoff_eV:
            out.append(f"recoils: {a.symbol} has displacement_eV {a.displacement_eV:g} below its cutoff_eV "
                       f"{a.cutoff_eV:g} (RustBCA makes no recoil below the cutoff, so displacements would be lost)")
    if problem.statistics.seed >= SEED_LIMIT:
        out.append(f"recoils: the seed must be below {SEED_LIMIT} (batch b runs with seed * 1000 + b)")
    return out


def unsupported(problem) -> List[str]:
    return CAPABILITY.unsupported(problem) + _empirical_reasons(problem) + _damage_reasons(problem)


def _boron_in_si(problem) -> bool:
    return problem.beam.ion.Z == 5 and any("Si" in la.density_cm3 for la in problem.target.layers)


def cli() -> Optional[str]:
    """RustBCA's command-line program: SIMWS_RUSTBCA_BIN, else <this Python's env>/bin/RustBCA, else PATH."""
    v = os.environ.get("SIMWS_RUSTBCA_BIN")
    if v:
        return v if Path(v).is_file() else None
    p = Path(sys.prefix) / "bin" / "RustBCA"
    if p.is_file():
        return str(p)
    return shutil.which("RustBCA")


def stopping_factor(problem, ck=None):
    """(ck, entry) for the problem: 1.0 and None in Lindhard-Scharff mode (A12.5)."""
    if problem.models.electronic_stopping != "empirical":
        return 1.0, None
    from simws.radiation import empirical_stopping as es
    e = es.entry(problem.beam.ion.symbol)
    return (float(ck) if ck is not None else float(e["ck"])), e


def available() -> bool:
    try:
        import libRustBCA  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def damage_available() -> bool:
    return cli() is not None


def _threads() -> int:
    v = os.environ.get("SIMWS_THREADS")
    if v:
        return max(1, int(v))
    return max(1, min(4, os.cpu_count() or 1))


def build_input(problem, ck=None) -> Dict[str, Any]:
    """The RustBCA input dictionary for `problem` (pure Python; lengths in micrometres, energies in eV, masses in
    amu, densities in atoms per cubic micrometre)."""
    ion, tgt = problem.beam.ion, problem.target
    tilt = math.radians(problem.beam.tilt_deg)
    atoms = tgt.atoms
    factor = stopping_factor(problem, ck)[0]
    return {
        "options": {
            "name": f"simws_{problem.problem_id()[:12]}",
            "track_trajectories": False, "track_recoils": bool(problem.models.recoils),
            "track_recoil_trajectories": False, "weak_collision_order": 0,
            "electronic_stopping_mode": _STOPPING[problem.models.electronic_stopping],
            "mean_free_path_model": "LIQUID",
            "interaction_potential": [[_POTENTIAL[problem.models.potential]]],
            "scattering_integral": [["MENDENHALL_WELLER"]],
            "num_threads": _threads(), "num_chunks": 1, "seed": int(problem.statistics.seed),
        },
        "particle_parameters": {
            "length_unit": "MICRON", "energy_unit": "EV", "mass_unit": "AMU",
            "N": [int(problem.statistics.n_ions)], "m": [float(ion.mass_amu)], "Z": [int(ion.Z)],
            "E": [float(problem.beam.energy_keV) * 1000.0],
            "Ec": [float(ion.cutoff_eV)], "Es": [float(ion.surface_binding_eV)],
            "interaction_index": [0],
            "pos": [[0.0, 0.0, 0.0]], "dir": [[math.cos(tilt), math.sin(tilt), 0.0]],
        },
        "geometry_input": {
            "length_unit": "MICRON",
            "layer_thicknesses": [float(la.thickness_um) for la in tgt.layers],
            "densities": [[float(la.density_cm3.get(a.symbol, 0.0)) * 1e-12 for a in atoms] for la in tgt.layers],
            "electronic_stopping_correction_factors": [factor] * len(tgt.layers),
        },
        "material_parameters": {
            "energy_unit": "EV", "mass_unit": "AMU",
            "Z": [int(a.Z) for a in atoms], "m": [float(a.mass_amu) for a in atoms],
            "Ec": [float(a.cutoff_eV) for a in atoms], "Es": [float(a.surface_binding_eV) for a in atoms],
            "Eb": [float(a.bulk_binding_eV) for a in atoms],
            "interaction_index": [0] * len(atoms),
        },
    }


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("RustBCA")
    except Exception:  # noqa: BLE001
        return "unknown"


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, float):
        return repr(v)
    return str(int(v))


def to_toml(inp: Dict[str, Any]) -> str:
    """The input dictionary as the TOML file RustBCA's command-line program reads (tables of scalars and lists)."""
    lines = []
    for sec, kv in inp.items():
        lines.append(f"[{sec}]")
        lines += [f"{k} = {_toml_value(v)}" for k, v in kv.items()]
        lines.append("")
    return "\n".join(lines)


def batch_sizes(n: int, batches: int = BATCHES) -> List[int]:
    """n ions in `batches` batches whose sizes differ by at most one."""
    return [n // batches + (1 if b < n % batches else 0) for b in range(batches)]


def _read(path: Path, cols) -> "Any":
    import numpy as np
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, len(cols)))
    return np.atleast_2d(np.loadtxt(path, delimiter=",", usecols=cols, ndmin=2))


def _run_damage(problem, workdir, ck=None) -> ImplantResult:
    import numpy as np
    exe = cli()
    if exe is None:
        raise EngineNotAvailable("RustBCA's command-line program (needed for damage, contract A14.3) is not "
                                 "installed; run tools/install_rustbca.sh again, or set SIMWS_RUSTBCA_BIN")
    n = int(problem.statistics.n_ions)
    half = 0.5 * problem.target.thickness_um
    T = problem.target.thickness_um
    bin_um = float(problem.depth_bin_um)
    nb = int(math.ceil(T / bin_um - 1e-9))
    threads = _threads()
    base = build_input(problem, ck)
    base["material_parameters"]["Ed"] = [float(a.displacement_eV or 0.0) for a in problem.target.atoms]
    depth: List[float] = []
    lateral: List[float] = []
    n_refl = n_trans = outside = 0
    hists: List[List[float]] = []
    sizes = batch_sizes(n)
    t0 = time.time()
    tmp = None
    if workdir is None:
        tmp = tempfile.TemporaryDirectory(prefix="simws_rustbca_")
        folder = Path(tmp.name)
    else:
        folder = Path(workdir)
        folder.mkdir(parents=True, exist_ok=True)
    try:
        for b, m in enumerate(sizes):
            inp = {k: dict(v) for k, v in base.items()}
            name = f"b{b:02d}_"
            inp["options"].update(name=name, track_recoils=True, track_displacements=True, num_threads=threads,
                                  num_chunks=max(1, math.ceil(m / (2 * threads))),
                                  seed=int(problem.statistics.seed) * 1000 + b)
            inp["particle_parameters"]["N"] = [int(m)]
            (folder / f"{name}input.toml").write_text(to_toml(inp))
            with open(folder / f"{name}log.txt", "w") as log:
                rc = subprocess.run([exe, "1D", f"{name}input.toml"], cwd=folder, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
            if rc != 0:
                tail = (folder / f"{name}log.txt").read_text(errors="replace").strip().splitlines()[-5:]
                raise RuntimeError(f"RustBCA batch {b} exited with {rc}: " + " | ".join(tail))
            dep = _read(folder / f"{name}deposited.output", (2, 4))          # x, z of the stopped ions
            ref = _read(folder / f"{name}reflected.output", (3,))           # x of the ions that left
            disp = _read(folder / f"{name}displacements.output", (3,))      # x where each vacancy was made
            depth += dep[:, 0].tolist()
            lateral += dep[:, 1].tolist()
            if dep.shape[0] + ref.shape[0] != m:
                raise RuntimeError(f"RustBCA batch {b}: {dep.shape[0]} stopped + {ref.shape[0]} left is not {m}")
            n_trans += int(np.sum(ref[:, 0] > half))
            n_refl += int(np.sum(ref[:, 0] <= half))
            x = disp[:, 0]
            inside = (x >= 0) & (x <= T)
            outside += int(np.sum(~inside))
            idx = np.minimum(np.floor(x[inside] / bin_um).astype(int), nb - 1)
            hists.append(np.bincount(idx, minlength=nb).astype(float).tolist())
            for f in folder.glob(f"{name}*.output"):
                f.unlink()
    finally:
        if tmp is not None:
            tmp.cleanup()
    return ImplantResult(engine=NAME, problem_id=problem.problem_id(), n_ions=n, n_reflected=n_refl,
                         n_transmitted=n_trans, depth_um=depth, lateral_um=lateral, vacancy_bin_um=bin_um,
                         vacancy_batches=hists, vacancy_batch_ions=sizes,
                         flags=[AMORPHOUS_FLAG] + ([BORON_FLAG] if _boron_in_si(problem) else []),
                         diagnostics={"rustbca_version": _version(), "threads": threads, "rustbca_cli": exe,
                                      "damage_batches": len(sizes), "vacancies_outside": outside,
                                      "seconds": round(time.time() - t0, 3)})


def run(problem, workdir=None, ck=None) -> ImplantResult:
    reasons = unsupported(problem)
    if reasons:
        raise Unsupported(NAME, reasons)
    if problem.models.recoils:
        return _run_damage(problem, workdir, ck)
    try:
        from libRustBCA import rustbca_local_py
    except Exception as e:  # noqa: BLE001
        raise EngineNotAvailable(f"the RustBCA Python module libRustBCA cannot be imported ({e}); "
                                 f"build it with tools/install_rustbca.sh") from e
    inp = build_input(problem, ck)
    out = rustbca_local_py(inp, "1D")
    n = int(problem.statistics.n_ions)
    half = 0.5 * problem.target.thickness_um
    depth: List[float] = []
    lateral: List[float] = []
    n_refl = n_trans = 0
    for implanted, x, z in zip(out["implanted"], out["x"], out["z"]):
        if implanted:
            depth.append(float(x))
            lateral.append(float(z))
        elif x > half:
            n_trans += 1
        else:
            n_refl += 1
    flags = [AMORPHOUS_FLAG]
    factor, entry = stopping_factor(problem, ck)
    extra: Dict[str, Any] = {}
    if entry is not None:
        u = float(entry["u"])
        flags.append(f"implant:empirical_stopping: Lindhard-Scharff x {factor:.3f} +- {u:.3f} from {entry['n']} "
                     f"measured points (IAEA database, A12); the data sets disagree (Birge ratio "
                     f"{entry['birge_ratio']:.1f})")
        extra = {"stopping_ck": factor, "stopping_ck_u": u}
    elif _boron_in_si(problem):
        flags.append(BORON_FLAG)
    return ImplantResult(engine=NAME, problem_id=problem.problem_id(), n_ions=n, n_reflected=n_refl,
                         n_transmitted=n_trans, depth_um=depth, lateral_um=lateral, flags=flags,
                         diagnostics={"rustbca_version": _version(), "threads": inp["options"]["num_threads"],
                                      "rows_returned": len(out["x"]), **extra})
