"""RustBCA translator (architecture v1, contract amendment A7.4): an ImplantProblem on RustBCA 3.0.0.

RustBCA (Drobny and Curreli, JOSS, doi 10.21105/joss.03298; GPL-3) is a binary-collision Monte Carlo code. It is
built from source as the Python module `libRustBCA` (tools/install_rustbca.sh) and called in process through
`rustbca_local_py(input, "1D")`, which returns where every ion ended. Importing this module does not import the
engine.

Geometry: x is depth below the front surface (x = 0), the plane of incidence is x-y, so the axis perpendicular to
it is z. The target is a stack of laterally infinite amorphous layers.

What it cannot do, and so refuses or flags (contract A7.6): crystalline targets (no channeling), recoil tracking
(damage profiles), beams above 25 keV per nucleon (Lindhard-Scharff stopping only), dose effects.
Known failure (contract A7.9): boron ranges in silicon are about 35 % too deep; such results carry a flag.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List

from simws.engines.base import EngineNotAvailable, ImplantCapability, ImplantResult, Unsupported

FAMILY = "implant"
NAME = "rustbca"

CAPABILITY = ImplantCapability(
    engine=NAME, env="tcad", in_process=True,
    description="RustBCA 3.0.0: binary-collision Monte Carlo, ion ranges in layered amorphous targets",
    structures=["amorphous"], potentials=["zbl", "kr_c", "moliere"], electronic_stopping=["lindhard_scharff"],
    recoils=[False], max_layers=20,
    max_keV_per_amu=25.0,                 # RustBCA README: its low-energy stopping models hold below 25 keV/nucleon
    tested_keV_per_amu=(1.6, 3.3),        # what passed contract A7.7 R16: P at 50 and 100 keV (A7.9)
    tested_target_symbols=["Si"], tested_ion_symbols=["P"],
    notes="Boron ranges in silicon FAILED acceptance R16: about 35 % too deep, because Lindhard-Scharff electronic "
          "stopping is too weak for boron in silicon (contract A7.9). Range profiles only. Following every recoil in memory exhausted an 8 GB machine in a 300-ion trial, so "
          "recoil tracking (damage) is refused until a streaming path exists.")

_POTENTIAL = {"zbl": "ZBL", "kr_c": "KR_C", "moliere": "MOLIERE"}
_STOPPING = {"lindhard_scharff": "LOW_ENERGY_NONLOCAL"}
AMORPHOUS_FLAG = ("implant:amorphous_target: no channeling (a crystalline wafer shows a deeper tail) and no damage "
                  "build-up (the profile is linear in dose)")
BORON_FLAG = ("implant:known_failure: boron ranges in silicon failed acceptance R16, about 35 % too deep "
              "(electronic stopping too weak; contract A7.9): trends only, not numbers")


def unsupported(problem) -> List[str]:
    return CAPABILITY.unsupported(problem)


def available() -> bool:
    try:
        import libRustBCA  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _threads() -> int:
    v = os.environ.get("SIMWS_THREADS")
    if v:
        return max(1, int(v))
    return max(1, min(4, os.cpu_count() or 1))


def build_input(problem) -> Dict[str, Any]:
    """The RustBCA input dictionary for `problem` (pure Python; lengths in micrometres, energies in eV, masses in
    amu, densities in atoms per cubic micrometre)."""
    ion, tgt = problem.beam.ion, problem.target
    tilt = math.radians(problem.beam.tilt_deg)
    atoms = tgt.atoms
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
            "electronic_stopping_correction_factors": [1.0] * len(tgt.layers),
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


def run(problem, workdir=None) -> ImplantResult:
    reasons = unsupported(problem)
    if reasons:
        raise Unsupported(NAME, reasons)
    try:
        from libRustBCA import rustbca_local_py
    except Exception as e:  # noqa: BLE001
        raise EngineNotAvailable(f"the RustBCA Python module libRustBCA cannot be imported ({e}); "
                                 f"build it with tools/install_rustbca.sh") from e
    inp = build_input(problem)
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
    if problem.beam.ion.Z == 5 and any("Si" in la.density_cm3 for la in problem.target.layers):
        flags.append(BORON_FLAG)
    return ImplantResult(engine=NAME, problem_id=problem.problem_id(), n_ions=n, n_reflected=n_refl,
                         n_transmitted=n_trans, depth_um=depth, lateral_um=lateral, flags=flags,
                         diagnostics={"rustbca_version": _version(), "threads": inp["options"]["num_threads"],
                                      "rows_returned": len(out["x"])})
