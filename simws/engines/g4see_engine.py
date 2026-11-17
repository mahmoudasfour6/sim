"""G4SEE translator (architecture v1, contract amendment A8.4): a SeeProblem on G4SEE v1.0 / Geant4 11.2.2.

G4SEE (D. Lucsanyi et al., IEEE Trans. Nucl. Sci. 69(3) 273-281, 2022, doi 10.1109/TNS.2022.3149989; GPL-3) is a
Geant4 application for single-event effects: a bulk block with one sensitive volume (SV) and back-end-of-line
(BEOL) layers on top, a General Particle Source, and histograms of what each event leaves in the SV. It runs as an
executable from its own conda env (tools/install_g4see.sh); this module writes the macro, runs it and reads the
energy-deposition histogram back. Importing this module does not need Geant4.

Geometry as G4SEE defines it: the centre of the top bulk surface is the origin, z points up (out of the target),
the SV position is the centre of its top surface, BEOL layers stack upwards in the order they are added.
"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import List, Tuple

from simws.engines.base import EngineNotAvailable, SeeCapability, SeeResult, Unsupported

FAMILY = "see"
NAME = "g4see"
VERSION = "v1.0"                 # git tag; commit 8c6fc45
GEANT4_VERSION = "11.2.2"
HIST_FILE = "Edep_0_hist_t0.out"
# Digits after the point in G4SEE's histogram file (A8.9): its default (3) loses counts above 9999; 10 makes a
# number 16 characters wide, which runs into the next 15-character column. 8 keeps 9 significant digits and a space.
PRECISION = 8

CAPABILITY = SeeCapability(
    engine=NAME, env="g4see", in_process=False,
    description="G4SEE v1.0 on Geant4 11.2.2: energy deposited per event in one sensitive volume",
    particles=["proton", "alpha", "neutron", "e-", "gamma", "ion"], em=["option4", "option3", "livermore"],
    hadronic=["none", "ftfp_bert_hp"], max_beol_layers=20,
    tested_particles=["alpha", "proton"], tested_energy_MeV=(5.0, 10.0), tested_materials=["G4_Si"],
    tested_em=["option4"], tested_hadronic=["none"],
    notes="Single-threaded build. Box geometry, one SV, point source of one energy and direction.")

_EM = {"option3": "G4EmStandardPhysics_option3", "livermore": "G4EmLivermorePhysics"}
_HADRONIC = {"ftfp_bert_hp": ["G4HadronElasticPhysicsHP", "G4HadronPhysicsFTFP_BERT_HP", "G4IonElasticPhysics",
                              "G4IonPhysics"]}
EM_ONLY_FLAG = ("see:em_only: hadronic physics off, no nuclear reactions (neutrons deposit nothing; protons and ions "
                "lose their nuclear-reaction events)")


def unsupported(problem) -> List[str]:
    return CAPABILITY.unsupported(problem)


def prefix() -> Path:
    """The conda env that holds g4see: SIMWS_G4SEE_PREFIX, else <envs_root>/g4see from the simws configuration."""
    if os.environ.get("SIMWS_G4SEE_PREFIX"):
        return Path(os.environ["SIMWS_G4SEE_PREFIX"]).expanduser()
    from simws import config
    return config.path("envs_root") / "g4see"


def executable() -> Path:
    return prefix() / "bin" / "g4see"


def available() -> bool:
    try:
        exe = executable()
        return exe.is_file() and os.access(exe, os.X_OK)
    except Exception:  # noqa: BLE001
        return False


def _g(x: float) -> str:
    return repr(float(x))


def build_macro(problem) -> str:
    """The G4SEE macro for `problem` (pure Python)."""
    s, g, ph, h = problem.source, problem.geometry, problem.physics, problem.histogram
    seed = int(problem.statistics.seed)
    L: List[str] = [f"# simws contract A8.4, problem {problem.problem_id()}",
                    "/control/verbose 0", "/run/verbose 0", "/tracking/verbose 0",
                    f"/random/setSeeds {2 * seed + 1} {2 * seed + 2}",
                    f"/SEE/geometry/Bulk {g.bulk_material} {_g(g.bulk_width_um)} um {_g(g.bulk_thickness_um)} um",
                    f"/SEE/geometry/SV 0 0 {_g(-g.sv_depth_um)} um {_g(g.sv_width_um[0])} {_g(g.sv_width_um[1])} um "
                    f"{_g(g.sv_thickness_um)} um"]
    for b in reversed(g.beol):            # G4SEE puts each new layer on top; the description lists the beam side first
        L.append(f"/SEE/geometry/BEOL/addLayer {b.material} {_g(g.bulk_width_um)} um {_g(b.thickness_um)} um {b.name}")
    if ph.em in _EM:
        L.append(f"/SEE/physics/addPhysics {_EM[ph.em]}")
    for mod in _HADRONIC.get(ph.hadronic, []):
        L.append(f"/SEE/physics/addPhysics {mod}")
    L.append("/run/initialize")
    t = math.radians(s.tilt_deg)
    z0 = g.beol_thickness_um + 1.0
    x0 = -math.tan(t)                     # 1 um above the stack: enters the top at x = 0
    L += [f"/gps/particle {s.particle}"]
    if s.particle == "ion":
        L.append(f"/gps/ion {s.ion_Z} {s.ion_A}")
    L += ["/gps/ene/type Mono", f"/gps/ene/mono {_g(s.energy_MeV)} MeV", "/gps/pos/type Point",
          f"/gps/direction {_g(math.sin(t))} 0 {_g(-math.cos(t))}",
          f"/gps/pos/centre {_g(x0)} 0 {_g(z0)} um",
          "/SEE/scoring/standard/Edep 0",
          f"/SEE/scoring/setHistogram {h.scale} {h.bins} {_g(h.low_MeV)} {_g(h.high_MeV)} MeV",
          f"/SEE/scoring/setPrecision {PRECISION}",
          f"/run/beamOn {int(problem.statistics.n_events)}"]
    return "\n".join(L) + "\n"


def edges(problem) -> List[float]:
    """Bin edges in MeV with G4SEE's own formulas (bins + 1 values)."""
    h = problem.histogram
    if h.scale == "lin":
        return [h.low_MeV + i * ((h.high_MeV - h.low_MeV) / h.bins) for i in range(h.bins + 1)]
    lo, hi = math.log10(h.low_MeV), math.log10(h.high_MeV)
    return [10 ** (i * ((hi - lo) / h.bins) + lo) for i in range(h.bins + 1)]


def parse_histogram(path, problem) -> Tuple[List[float], float, float]:
    """(counts, underflow, overflow) from a G4SEE Edep histogram file; checks the bins against the description."""
    lines = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if len(lines) < 4 or lines[1][0] != "underflow" or lines[2][0] != "overflow":
        raise ValueError(f"{path}: not a G4SEE histogram file")
    under, over = float(lines[1][1]), float(lines[2][1])
    rows = lines[3:]
    want = edges(problem)
    if len(rows[-1]) != 1 or any(len(r) != 2 for r in rows[:-1]):
        raise ValueError(f"{path}: every bin line must hold an edge and a count, and the last line the upper edge only")
    if len(rows) != len(want):
        raise ValueError(f"{path}: {len(rows) - 1} bins, the description has {len(want) - 1}")
    for r, e in zip(rows, want):
        got = float(r[0])
        if abs(got - e) > (1e-6 * abs(e) if e != 0 else 1e-12):
            raise ValueError(f"{path}: bin edge {got!r} MeV, expected {e!r} MeV")
    return [float(r[1]) for r in rows[:-1]], under, over


def _info(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip()
            if k in ("Geant4 release", "G4SEE version", "Total CPU time (s)"):
                out[k] = v.strip()
    return out


def run(problem, workdir) -> SeeResult:
    reasons = unsupported(problem)
    if reasons:
        raise Unsupported(NAME, reasons)
    if not available():
        raise EngineNotAvailable(f"no G4SEE executable at {executable()}; install it with tools/install_g4see.sh "
                                 f"(or point SIMWS_G4SEE_PREFIX at the conda env that has it)")
    from simws.radiation.geant4 import dataset_env
    pid = problem.problem_id()
    folder = Path(workdir) / f"g4see_{pid[:12]}"
    (folder / "out").mkdir(parents=True, exist_ok=True)
    (folder / "run.mac").write_text(build_macro(problem))
    pre = prefix()
    env = dict(os.environ)
    env.update(dataset_env(pre))
    env["LD_LIBRARY_PATH"] = f"{pre / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    log = folder / "g4see.log"
    with open(log, "w") as fh:
        rc = subprocess.run([str(executable()), "run.mac", "-o", "out"], cwd=folder, env=env, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    hist = folder / "out" / HIST_FILE
    if rc != 0 or not hist.exists():
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-20:])
        raise RuntimeError(f"g4see exited with code {rc}" + ("" if hist.exists() else ", no histogram written")
                           + f"; last lines of {log}:\n{tail}")
    counts, under, over = parse_histogram(hist, problem)
    info = _info((folder / "out" / "g4see.out").read_text(errors="replace")) \
        if (folder / "out" / "g4see.out").exists() else {}
    flags = [EM_ONLY_FLAG] if problem.physics.hadronic == "none" else []
    return SeeResult(engine=NAME, problem_id=pid, n_events=int(problem.statistics.n_events), edges_MeV=edges(problem),
                     counts=counts, underflow=under, overflow=over, flags=flags,
                     diagnostics={"g4see_version": info.get("G4SEE version", VERSION),
                                  "geant4_release": info.get("Geant4 release", ""),
                                  "cpu_time_s": info.get("Total CPU time (s)", "")})
