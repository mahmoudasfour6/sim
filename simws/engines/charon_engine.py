"""Charon translator (architecture v1, section 4.5). Charon runs out of process (laptop only); this module turns a
Problem into the decks of simws.validation.charon_xcheck and reads Charon's outputs with that module's reader.

Units: the Problem's (um, cm^-3, V, K, cm^2/V/s, s, eV); results in A/cm^2 positive into the device at a contact.
Nothing here imports Charon or exodus at import time (the reader imports exodus when collect() runs).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from simws.engines.base import Capability, EngineNotAvailable, EngineResult, Unsupported
from simws.spec.schema import DCSweep, Problem

NAME = "charon"

CAPABILITY = Capability(
    engine=NAME, env="tcad",
    description="Sandia Charon (CVFEM-SG drift-diffusion, Fermi-Dirac, constant-energy incomplete ionization), "
                "out of process; only the pn-diode configuration of the 0.9a cross-check is exercised",
    in_process=False, dimensions=[1], max_regions=1, materials=["silicon"], grading_elements=[],
    contact_kinds=["ohmic"], statistics=["fermi_dirac"], mobility=["constant"], velocity_saturation=[False],
    srh=["off", "constant"], auger=[False], band_gap_narrowing=["off"], incomplete_ionization=["constant_energy"],
    impact_ionization=["off"], analyses=["dc_sweep"], tested_temperature_K=(77.0, 300.0),
    notes="Abrupt pn step on the 1.0 um / 0.5 um meshes of charon_xcheck.MESHES (the 1.25 nm 'fine' mesh is used); "
          "sweep on the x_min contact, no tracking or fixed contacts. Temperatures below 250 K are reached by "
          "0 V temperature continuation from 300 K through the temperatures of charon_xcheck.CHAIN "
          "(200, 150, 110, 95, 85, 77 K, those above the target); targets below 77 K or above 300 K have no "
          "continuation chain beyond that and are not exercised. One Charon deck per sweep bias (plus a 0 V start "
          "deck when the sweep does not begin at 0): the biases are not subdivided, so keep the spacing at or below "
          "0.1 V as in charon_xcheck.biases (larger jumps are flagged 'charon_bias_step>0.1V'). Sweep values must "
          "be distinct multiples of 1 mV (deck names). Degeneracy factors 2 (donor) and 4 (acceptor) as in "
          "charon_xcheck.case. Charon's reader returns |J|; collect() restores the sign from the bias.")

_CONT_T = (200.0, 150.0, 110.0, 95.0, 85.0, 77.0)       # temperatures of charon_xcheck.CHAIN, in order
_MAX_STEP_V = 0.1                                       # largest bias step used by charon_xcheck.biases


# ------------------------------------------------------------------------------------------------ description checks
def _pn_step(problem: Problem) -> Optional[Tuple[float, float, float]]:
    """(NA, ND, x_junction_um) when the doping is exactly one acceptor term on x < xj and one donor term on
    x >= xj (p side at the region's x_min), else None. Never raises."""
    dev = problem.device
    try:
        if len(dev.doping) != 2 or dev.dimension != 1 or dev.profiles:      # A16.5: no tabulated profiles
            return None
        acc = [t for t in dev.doping if t.species == "acceptor"]
        don = [t for t in dev.doping if t.species == "donor"]
        if len(acc) != 1 or len(don) != 1:
            return None
        a, d = acc[0], don[0]
        if a.include_upper or d.include_upper or a.box.y is not None or d.box.y is not None:
            return None
        xs = [b for r in dev.regions for b in r.box.x]
        lo, hi = min(xs), max(xs)
        xj = a.box.x[1]
        if xj is None or d.box.x[0] is None or not math.isclose(xj, d.box.x[0], rel_tol=0, abs_tol=1e-12):
            return None
        if a.box.x[0] is not None and a.box.x[0] > lo:
            return None
        if d.box.x[1] is not None and d.box.x[1] < hi:
            return None
        return a.concentration_cm3, d.concentration_cm3, float(xj)
    except Exception:  # noqa: BLE001  an unusual description is simply "not a pn step"
        return None


def _sweep_values(problem: Problem) -> List[float]:
    a = problem.analysis
    return [float(v) for v in a.values_V] if isinstance(a, DCSweep) else []


def _start_values(problem: Problem) -> List[float]:
    """The biases of the decks, in order: the sweep values, preceded by a 0 V start deck unless the sweep begins at 0."""
    v = _sweep_values(problem)
    return v if (v and v[0] == 0.0) else [0.0] + v


def _mV(v: float) -> int:
    return int(round(v * 1000))


def unsupported(problem: Problem) -> List[str]:
    """CAPABILITY.unsupported plus the translator's own limits, checked from the description alone."""
    from simws.validation import charon_xcheck as X
    out = CAPABILITY.unsupported(problem)
    dev, m, a = problem.device, problem.models, problem.analysis
    step = _pn_step(problem)
    if step is None:
        out.append("doping: Charon translator supports an abrupt two-term pn step only"
                   + (" (no tabulated profiles, A16.5)" if dev.profiles else ""))
    if m.tau_n_s != m.tau_p_s:
        out.append("srh: Charon deck takes one lifetime for both carriers")
    if not isinstance(a, DCSweep):
        out.append("analysis: Charon translator runs a DC sweep on the x_min contact only")
    else:
        xmin = [c.name for c in dev.contacts if c.side == "x_min"]
        if a.contact not in xmin:
            out.append(f"analysis: the swept contact '{a.contact}' must be the x_min contact")
        if a.tracking or a.fixed_V:
            out.append("analysis: tracking and fixed contacts are not supported")
        vals = _start_values(problem)
        if any(abs(v * 1000 - round(v * 1000)) > 1e-6 for v in vals) or len({_mV(v) for v in vals}) != len(vals):
            out.append("analysis: sweep values must be distinct multiples of 1 mV (a 0 V start deck is implied "
                       "unless the sweep begins at 0)")
    xs = [b for r in dev.regions for b in r.box.x]
    length = max(xs) - min(xs) if xs else float("nan")
    ok_mesh = bool(X.MESHES) and math.isclose(length, X.L_UM, abs_tol=1e-9) and \
        (step is None or math.isclose(step[2] - min(xs), X.XJ_UM, abs_tol=1e-9))
    if not ok_mesh:
        out.append("mesh: no Charon mesh for this geometry")
    return out


def _config_exe() -> Optional[Path]:
    try:
        from simws import config
        p = config.path("toolchain.charon_exe")
        return p if p.exists() else None
    except Exception:  # noqa: BLE001  key missing, config unreadable, ...
        return None


def available() -> bool:
    """True when toolchain.charon_exe exists. Never raises."""
    try:
        return _config_exe() is not None
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------------------------ translation
def _label(problem: Problem) -> str:
    return "p" + problem.problem_id()[:8]


def _case(problem: Problem, T: float) -> dict:
    """The dict charon_xcheck's writers expect, with band parameters from charon_xcheck.case(T) and every value the
    Problem carries taking precedence over the constants hard-coded there."""
    from simws.validation import charon_xcheck as X
    na, nd, _ = _pn_step(problem)
    m = problem.models
    c = X.case(T)
    c.update(NA=na, ND=nd, mu_n=m.mu_n_cm2_Vs, mu_p=m.mu_p_cm2_Vs, E_ion=m.ii_energy_eV,
             N_crit=m.ii_critical_doping_cm3)
    if m.srh == "constant":
        c["tau"] = m.tau_n_s
    return c


def _plan_decks(problem: Problem) -> Tuple[List[dict], List[float]]:
    """Decks in run order: dicts {name, imp, out, V, T, ic ('mesh'|'file'), srh, kind}, and the target temperature
    chain. Mirrors charon_xcheck.write_decks for one temperature."""
    from simws.validation import charon_xcheck as X
    T = problem.models.temperature_K
    L = _label(problem)
    mesh = X.MESHES["fine"][0]
    srh = problem.models.srh != "off"
    decks: List[dict] = []
    prev, src_is_mesh = mesh, True
    if T < 250.0:
        cont = [t for t in _CONT_T if t > T]
        steps = [("base300", 300.0)] + [(f"cont_T{t:g}", t) for t in cont]
        for name, t in steps:
            full = f"xc_{L}_{name}"
            decks.append(dict(name=full, imp=prev, out=f"{full}.exo", V=0.0, T=t, ic="mesh" if src_is_mesh else "file",
                              srh=srh, kind="chain"))
            prev, src_is_mesh = f"{full}.exo", False
    for V in _start_values(problem):
        name = f"xc_{L}_dd_{_mV(V):04d}mV"
        decks.append(dict(name=name, imp=prev, out=f"{name}.exo", V=V, T=T, ic="mesh" if src_is_mesh else "file",
                          srh=srh, kind="bias"))
        prev, src_is_mesh = f"{name}.exo", False
    return decks, [d["T"] for d in decks if d["kind"] == "chain"]


def _place_mesh(out: Path) -> None:
    """Put the 'fine' mesh next to the decks: copy it from simws/validation/data, else write it with charon_mesh
    (needs netCDF4); if neither works the decks are still written and the mesh must be supplied before running."""
    from simws.validation import charon_xcheck as X
    f, dx = X.MESHES["fine"]
    if (X.DATA / f).exists():
        shutil.copy(X.DATA / f, out / f)
        return
    try:
        from simws.validation.charon_mesh import write_diode_mesh
        write_diode_mesh(out / f, L_um=X.L_UM, W_um=X.W_UM, dx_um=dx)
    except Exception:  # noqa: BLE001  e.g. netCDF4 missing on this machine
        pass


def prepare(problem: Problem, workdir) -> List[Path]:
    """Write the meshes, one deck per bias (preceded by the continuation chain below 250 K), runlist_all.txt (run
    order), runlist_<label>.txt (the bias decks) and case_<label>.json into `workdir`. Returns the decks written."""
    from simws.validation import charon_xcheck as X
    reasons = unsupported(problem)
    if reasons:
        raise Unsupported(NAME, reasons)
    out = Path(workdir)
    out.mkdir(parents=True, exist_ok=True)
    _place_mesh(out)
    L = _label(problem)
    decks, _ = _plan_decks(problem)
    paths, cache = [], {}
    for d in decks:
        if d["T"] not in cache:
            cache[d["T"]] = _case(problem, d["T"])
        c = cache[d["T"]]
        ic = X.IC_MESH if d["ic"] == "mesh" else X.IC_FILE
        p = out / f"{d['name']}.inp"
        p.write_text(X._deck(c, d["imp"], d["out"], d["V"], ic, d["srh"]))
        paths.append(p)
    order = [d["name"] for d in decks]
    (out / "runlist_all.txt").write_text("\n".join(order) + "\n")
    (out / f"runlist_{L}.txt").write_text("\n".join(d["name"] for d in decks if d["kind"] == "bias") + "\n")
    (out / f"case_{L}.json").write_text(json.dumps(
        {**_case(problem, problem.models.temperature_K), "label": L, "problem_id": problem.problem_id(),
         "biases": _start_values(problem), "decks": order}, indent=1))
    return paths


# ------------------------------------------------------------------------------------------------ execution
_INTERP_CACHE: Dict[str, Path] = {}


def _charon_root() -> Path:
    """Where Charon lives: CHARON_ROOT, else the top folder of the configured executable (~/tcad/build/src/
    charon_mp.exe -> ~/tcad), else ~/tcad. The same default as tools/run_charon_xcheck.sh."""
    if os.environ.get("CHARON_ROOT"):
        return Path(os.environ["CHARON_ROOT"]).expanduser()
    exe = _config_exe()
    home = Path.home()
    if exe is not None:
        try:
            return home / exe.resolve().relative_to(home).parts[0]
        except (ValueError, IndexError):
            pass
    return home / "tcad"


def _interpreter() -> Path:
    """charonInterpreter.py, found as tools/run_charon_xcheck.sh finds it: CHARON_INTERP, then PATH, then a search
    under the Charon root (following links, skipping the parseGenerator copies)."""
    for cand in (os.environ.get("CHARON_INTERP"), shutil.which("charonInterpreter.py")):
        if cand and Path(cand).expanduser().is_file():
            return Path(cand).expanduser()
    root = _charon_root()
    key = str(root)
    if key not in _INTERP_CACHE:
        for dirpath, _dirs, files in os.walk(root, followlinks=True):
            if "charonInterpreter.py" in files and "parseGenerator" not in dirpath:
                _INTERP_CACHE[key] = Path(dirpath) / "charonInterpreter.py"
                break
    if key in _INTERP_CACHE:
        return _INTERP_CACHE[key]
    raise EngineNotAvailable(f"charonInterpreter.py not found on PATH or under {root} (set CHARON_INTERP)")


def _tail(text: str, n: int = 12) -> str:
    return "\n".join((text or "").strip().splitlines()[-n:])


def _execute(workdir: Path) -> None:
    """Run every deck in runlist_all.txt the way tools/run_charon_xcheck.sh does: the interpreter with this
    Python (the tcad env), then charon_mp.exe under mpirun, with the env's bin folder first on PATH (what
    `conda activate tcad` gives the shell script). Logs: <deck>.interp.log and <deck>.log in the work folder."""
    import sys
    from simws.hub.adapter import NotConverged
    exe, interp = _config_exe(), _interpreter()
    envbin = str(Path(sys.executable).parent)
    env = dict(os.environ, PATH=envbin + os.pathsep + os.environ.get("PATH", ""))
    mpi = shutil.which("mpirun", path=env["PATH"])
    for name in (workdir / "runlist_all.txt").read_text().split():
        inp = workdir / f"{name}.inp"
        src = inp.read_text().splitlines()[0].split()[3]
        if not (workdir / src).exists():
            raise NotConverged(f"Charon deck {name} skipped: start file {src} is missing in {workdir} "
                               f"(the mesh was not written, or an earlier deck failed)")
        r = subprocess.run([sys.executable, str(interp), "-i", f"{name}.inp"], cwd=workdir, capture_output=True,
                           text=True, env=env)
        (workdir / f"{name}.interp.log").write_text((r.stdout or "") + (r.stderr or ""))
        if r.returncode:
            raise NotConverged(f"Charon interpreter failed on {name} (log: {workdir / (name + '.interp.log')}):\n"
                               + _tail((r.stdout or "") + (r.stderr or "")))
        cmd = ([mpi, "-np", "1"] if mpi else []) + [str(exe), f"--i={name}.inp.xml", "--current"]
        r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, env=env)
        (workdir / f"{name}.log").write_text((r.stdout or "") + (r.stderr or ""))
        if r.returncode:
            (workdir / f"{name}.exo").unlink(missing_ok=True)       # an aborted run leaves a partial file
            raise NotConverged(f"Charon failed on {name} (exit {r.returncode}; log: {workdir / (name + '.log')}):\n"
                               + _tail((r.stdout or "") + (r.stderr or "")))


def collect(problem: Problem, workdir) -> EngineResult:
    """Read the bias decks' outputs with charon_xcheck.read_charon. UNTESTED here (needs Charon outputs). The
    reader returns |J|, so the sign is taken from the bias (a diode's current has the sign of its voltage); the
    cathode current is minus the anode current (two terminals, steady state)."""
    from simws.hub.adapter import NotConverged
    from simws.validation import charon_xcheck as X
    ch = X.read_charon(workdir, _label(problem))
    want = _sweep_values(problem)
    J: List[float] = []
    for v in want:
        hits = [j for vv, j in zip(ch["V"], ch["J_A_cm2"]) if abs(vv - v) < 5e-4]
        if not hits:
            reached = [vv for vv in ch["V"]]
            raise NotConverged(f"no Charon output at V={v:g} V (outputs exist for {reached})")
        J.append(math.copysign(hits[0], v) if v != 0.0 else hits[0])
    a = problem.analysis
    cathode = [c.name for c in problem.device.contacts if c.name != a.contact]
    currents: Dict[str, List[float]] = {a.contact: J}
    flags = ["current_sign_from_bias"]
    if len(cathode) == 1:
        currents[cathode[0]] = [-j for j in J]
        flags.append(f"{cathode[0]}_current_from_kirchhoff")
    if max([abs(b - a_) for a_, b in zip([0.0] + want[:-1], want)] or [0.0]) > _MAX_STEP_V + 1e-9:
        flags.append("charon_bias_step>0.1V")
    return EngineResult(engine=NAME, problem_id=problem.problem_id(), analysis="dc_sweep", sweep_contact=a.contact,
                        sweep_V=want, currents=currents, current_unit="A/cm**2", flags=flags,
                        diagnostics={"label": _label(problem), "current_variable": ch.get("current_variable"),
                                     "unreadable": ch.get("unreadable", []), "mesh": "fine"})


def run(problem: Problem, workdir: Optional[Path] = None) -> EngineResult:
    """prepare + execute + collect. Raises Unsupported, EngineNotAvailable (after prepare, so the decks can be
    inspected), NotConverged."""
    reasons = unsupported(problem)
    if reasons:
        raise Unsupported(NAME, reasons)
    wd = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="charon_"))
    prepare(problem, wd)
    if not available():
        raise EngineNotAvailable(f"Charon executable not found (toolchain.charon_exe); decks written to {wd}")
    _execute(wd)
    return collect(problem, wd)
