"""Slow regression tier: actually re-runs each verified simulation in its own conda env
and compares the numbers it produces against tests/golden_slow.json (or against the
committed output files, for benchmarks whose outputs live in git).

Run:     pytest -m slow tests/                 (whole tier, ~10-20 min on the 125U)
         pytest -m slow tests/ -k devsim       (one check)
Record:  pytest -m slow --update-golden tests/ (adds golden values that are missing)

Every check runs in a scratch copy (tmp_path) unless noted, so nothing in the repo changes.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import ENVS, REPO, env_python, grab, run_in_env

pytestmark = pytest.mark.slow
FLOAT = r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"


def _require_env(env):
    if not (ENVS / env / "bin").is_dir():
        pytest.skip(f"conda env '{env}' not found")


def _copy(tmp_path, *rel_paths):
    for rel in rel_paths:
        src = REPO / rel
        if not src.exists():
            pytest.skip(f"missing input: {rel}")
        shutil.copy2(src, tmp_path / src.name)


# --------------------------------------------------------------------------- materials
def test_qe_si_bulk_scf(tmp_path, golden):
    """Quantum ESPRESSO Si SCF (2 atoms, 24 Ry, 4x4x4 k). Pseudopotential comes from sim_data."""
    _copy(tmp_path, "01_materials_dft_md/si_real_scf.pwi")
    pw = ENVS / "materials" / "bin" / "pw.x"
    if not pw.exists():
        pytest.skip("pw.x not found in materials env")
    out = run_in_env("materials", [pw, "-in", "si_real_scf.pwi"], tmp_path, timeout=900, threads=1)
    e_tot = grab(r"^!\s+total energy\s+=\s+" + FLOAT + r"\s+Ry", out)
    n_iter = grab(r"convergence has been achieved in\s+(\d+)\s+iterations", out)
    golden.verify({"qe.si_scf.total_energy_Ry": e_tot, "qe.si_scf.scf_iterations": n_iter})


# --------------------------------------------------------------------------- TCAD
def test_devsim_diode_1d(tmp_path, golden):
    """DEVSIM 1D Si pn diode, forward sweep 0 -> 0.5 V (tau = 10 ns)."""
    _copy(tmp_path, "03_tcad_devices/decks/diode_1d.py", "03_tcad_devices/decks/diode_common.py")
    out = run_in_env("tcad", [env_python("tcad"), "diode_1d.py"], tmp_path, timeout=600)
    rows = [l.split("\t") for l in out.splitlines() if l.startswith("top\t")]
    if not rows:
        pytest.fail(f"no 'top' current lines in DEVSIM output\n{out[-2000:]}", pytrace=False)
    v, i_e, i_h, i_tot = (float(x) for x in rows[-1][1:5])
    assert abs(v - 0.5) < 1e-6, f"last bias point was {v} V, expected 0.5 V"
    # Physics invariant independent of the golden file: current continuity between contacts.
    bot = [l.split("\t") for l in out.splitlines() if l.startswith("bot\t")][-1]
    assert abs(i_tot + float(bot[4])) / abs(i_tot) < 1e-3, "top/bot contact currents do not balance"
    golden.verify({"devsim.diode_1d.I_top_0p5V": i_tot, "devsim.diode_1d.Ie_top_0p5V": i_e,
                   "devsim.diode_1d.Ih_top_0p5V": i_h})


def test_charon_sic_diode_iv(tmp_path, golden):
    """Sandia Charon 4H-SiC pn diode: equilibrium (nlp) solve, then forward I-V sweep to 3.4 V.
    Runs on a scratch copy of the decks folder (including the untracked .exo mesh)."""
    exe = Path.home() / "tcad" / "build" / "src" / "charon_mp.exe"
    decks = REPO / "03_tcad_devices" / "decks"
    if not exe.exists():
        pytest.skip(f"{exe} not built")
    if not (decks / "diode2d.exo").exists():
        pytest.skip("decks/diode2d.exo mesh not present (it is gitignored; regenerate it first)")
    work = tmp_path / "decks"
    shutil.copytree(decks, work, ignore=shutil.ignore_patterns("__pycache__", "*.log", "sic_diode.*.exo"))
    lib = f"{ENVS / 'charon' / 'lib'}:{Path.home() / 'tcad' / 'install' / 'lib'}"
    for deck in ("sic_diode.nlp.inp.xml", "sic_diode.iv.inp.xml"):
        run_in_env("charon", [exe, f"--i={deck}"], work, timeout=1800, threads=1,
                   extra_env={"LD_LIBRARY_PATH": lib})
    dat = work / "currents-loca.dat"
    if not dat.exists():
        pytest.fail(f"Charon did not write currents-loca.dat; files: {sorted(p.name for p in work.iterdir())}",
                    pytrace=False)
    rows = [l.split() for l in dat.read_text().splitlines() if l.strip() and not l.lstrip().startswith(("#", "%"))]
    numeric = []
    for r in rows:
        try:
            numeric.append([float(x) for x in r])
        except ValueError:
            continue  # header line
    assert numeric, f"no numeric rows in {dat}"
    last = numeric[-1]
    golden.verify({f"charon.sic_diode_iv.final_row.col{i}": x for i, x in enumerate(last)})


def test_ihp_npn13g2_baseline_matches_committed():
    """IHP SG13G2 npn13G2 ngspice baseline. Re-runs in place, then compares against the
    version committed in git (git diff will also show any change to the PNG/JSON)."""
    _require_env("eda")
    bench = REPO / "benchmarks" / "ihp_npn13g2_baseline"
    rel = "benchmarks/ihp_npn13g2_baseline/npn13g2_baseline_300K.json"
    committed = json.loads(subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO,
                                          capture_output=True, text=True, check=True).stdout)
    run_in_env("eda", [env_python("eda"), "run_npn13g2_baseline.py"], bench, timeout=1800, threads=4)
    fresh = json.loads((REPO / rel).read_text())
    bad = []
    for corner, vals in committed["corners"].items():
        for k, v in vals.items():
            if isinstance(v, (int, float)) and fresh["corners"][corner][k] != pytest.approx(v, rel=1e-4):
                bad.append(f"{corner}.{k}: committed {v}, now {fresh['corners'][corner][k]}")
    assert not bad, "IHP baseline changed:\n  " + "\n  ".join(bad)


# --------------------------------------------------------------------------- FEM
def test_fem_electrothermal_joule(tmp_path, golden):
    _copy(tmp_path, "04_fem_multiphysics/test_electrothermal_fem.py")
    out = run_in_env("fem", [env_python("fem"), "test_electrothermal_fem.py"], tmp_path, timeout=600)
    golden.verify({
        "fem.electrothermal.V_max_V": grab(r"V_max = " + FLOAT + " V", out),
        "fem.electrothermal.T_max_K": grab(r"T_max = " + FLOAT + " K", out),
    })


# --------------------------------------------------------------------------- quantum / III-V
def test_quantum_iii_v_suite(tmp_path, golden):
    _copy(tmp_path, "05_quantum_iii_v/test_quantum_iii_v.py")
    out = run_in_env("quantum_iii_v", [env_python("quantum_iii_v"), "test_quantum_iii_v.py"], tmp_path, timeout=900)
    if "[III-V / Solcore] Note:" in out:
        pytest.fail("Solcore quantum-well solve failed:\n" + out[-1500:], pytrace=False)
    golden.verify({
        "quantum.kwant.T_max": grab(r"T\(E\) max = " + FLOAT, out),
        "quantum.sp.E1_meV": grab(r"E1 = " + FLOAT + " meV", out),
        "quantum.sp.E2_meV": grab(r"E2 = " + FLOAT + " meV", out),
        "quantum.solcore.E_e1_eV": grab(r"E_e1 = " + FLOAT + " eV", out),
        "quantum.solcore.E_hh1_eV": grab(r"E_hh1 = " + FLOAT + " eV", out),
    })


# --------------------------------------------------------------------------- photonics
def test_photonics_suite(tmp_path, golden):
    _copy(tmp_path, "06_photonics_spl/test_photonics.py")
    out = run_in_env("spl", [env_python("spl"), "test_photonics.py"], tmp_path, timeout=900)
    R = grab(r"R = " + FLOAT + ",", out)
    T = grab(r"T = " + FLOAT + ",", out)
    assert abs(R + T - 1.0) < 1e-6, "lossless metasurface must conserve energy (R+T=1)"
    golden.verify({
        "photonics.ceviche.output_intensity": grab(r"Output intensity = " + FLOAT, out),
        "photonics.ceviche.max_grad": grab(r"max \|dJ/deps\| = " + FLOAT, out),
        "photonics.grcwa.R": R,
        "photonics.grcwa.T": T,
    })


# --------------------------------------------------------------------------- characterization
def test_charz_metrology_matches_committed():
    """abTEM / py4DSTEM / xrayutilities suite. Re-runs in place; compares to the committed JSON."""
    _require_env("charz")
    rel = "07_characterization_charz/charz_metrology_summary.json"
    committed = json.loads(subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO,
                                          capture_output=True, text=True, check=True).stdout)
    run_in_env("charz", [env_python("charz"), "test_charz_stem_hrxrd.py"], REPO / "07_characterization_charz",
               timeout=3600, threads=6)
    fresh = json.loads((REPO / rel).read_text())
    bad = [f"{k}: committed {v}, now {fresh.get(k)}" for k, v in committed.items()
           if isinstance(v, (int, float)) and not isinstance(v, bool)
           and fresh.get(k) != pytest.approx(v, rel=1e-3)]
    assert not bad, "charz metrology changed:\n  " + "\n  ".join(bad)


# --------------------------------------------------------------------------- ML potentials
CHGNET_SI = '''
import torch, mace, nequip
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model import CHGNet
s = AseAtomsAdaptor.get_structure(bulk("Si", "diamond", a=5.431))
p = CHGNet.load(use_device="cpu", verbose=False).predict_structure(s)
print(f"CHGNet Si E = {float(p['e']):.6f} eV/atom")
'''


def test_chgnet_si_energy(tmp_path, golden):
    """CHGNet on bulk Si (CPU); also imports MACE and NequIP. Guards the torch/MLIP stack."""
    script = tmp_path / "chgnet_si.py"
    script.write_text(CHGNET_SI)
    out = run_in_env("materials", [env_python("materials"), script], tmp_path, timeout=600)
    golden.verify({"mlip.chgnet.si_energy_eV_per_atom": grab(r"CHGNet Si E = " + FLOAT, out)})
