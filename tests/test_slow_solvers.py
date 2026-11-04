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
        "quantum.sp.E1p_D4_meV": grab(r"E1' = " + FLOAT + " meV", out),
        "quantum.sp.EF_meV": grab(r"E_F = " + FLOAT + " meV", out),
        "quantum.solcore.e1_hh1_transition_eV": grab(r"transition = " + FLOAT + " eV", out),
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


# --------------------------------------------------------------------------- simws package (infrastructure)
def _simws_env():
    return {"PYTHONPATH": str(REPO)}


def test_simws_unit_suite_in_tcad_env():
    """Package unit tests in the tcad env, where every dependency exists: nothing may be skipped there."""
    out = run_in_env("tcad", [env_python("tcad"), "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", "tests/unit"],
                     REPO, timeout=900, extra_env=_simws_env())
    assert re.search(r"\d+ passed", out) and "skipped" not in out and "failed" not in out, out[-1500:]


def test_simws_particle_strike_matches_charon(tmp_path, golden):
    gold = REPO / "03_tcad_devices" / "particle_strike_ref" / "gold_currents.csv"
    if not gold.exists():
        pytest.skip("Charon gold currents not present")
    run_in_env("tcad", [env_python("tcad"), REPO / "workflow" / "scripts" / "particle_strike.py",
                        "--gold", gold, "--outdir", tmp_path], REPO, timeout=900, threads=1, extra_env=_simws_env())
    cmp = json.loads((tmp_path / "comparison.json").read_text())["results"]
    assert cmp["max_abs_dev_in_pulse_frac_of_peak"] < 0.015, cmp
    golden.verify({"simws.particle_strike.devsim_peak_A_per_cm": cmp["devsim"]["peak"],
                   "simws.particle_strike.devsim_dc_A_per_cm": cmp["devsim"]["dc"]})


def test_simws_electrothermal_resistor(tmp_path, golden):
    run_in_env("tcad", [env_python("tcad"), REPO / "workflow" / "scripts" / "electrothermal.py",
                        "--outdir", tmp_path, "--biases", "1,5"], REPO, timeout=900, threads=1, extra_env=_simws_env())
    pts = json.loads((tmp_path / "summary.json").read_text())["results"]["points"]
    for p in pts:
        assert p["converged"] and p["joule_vs_IV_rel_error"] < 1e-10 and p["energy_balance_rel_error"] < 1e-8
    golden.verify({"simws.electrothermal.I_5V_A_per_cm": pts[-1]["current_A_per_cm"],
                   "simws.electrothermal.Tmax_5V_K": pts[-1]["T_max_K"]})


# --------------------------------------------------------------------------- simws items 6-9 (physics, UQ, verification)
def test_simws_device_model_validation(tmp_path, golden):
    """Velocity saturation vs Caughey-Thomas closed form; avalanche breakdown vs ionization integral."""
    run_in_env("tcad", [env_python("tcad"), REPO / "workflow" / "scripts" / "validate_models.py", "--outdir", tmp_path],
               REPO, timeout=1500, threads=1, extra_env=_simws_env())
    res = json.loads((tmp_path / "validation.json").read_text())["results"]
    assert res["pass"], res
    golden.verify({"simws.models.breakdown_V_devsim": res["breakdown_V_devsim"]})


def test_simws_mixed_mode_rc_and_seu(tmp_path, golden):
    run_in_env("tcad", [env_python("tcad"), REPO / "workflow" / "scripts" / "seu.py", "--outdir", tmp_path,
                        "--lets", "0.1"], REPO, timeout=2400, threads=1, extra_env=_simws_env())
    res = json.loads((tmp_path / "summary.json").read_text())["results"]
    assert res["rc_check"]["worst_abs_dev_frac_of_vinf"] < 2e-3, res["rc_check"]
    s = res["seu"][0]
    assert abs(s["track_mesh_correction"] - 1) < 0.05
    assert 0 < s["collected_charge_substrate_fC"] < 1.2 * s["deposited_charge_fC"]
    golden.verify({"simws.seu.let0p1.v_node_min_V": s["v_node_min_V"],
                   "simws.seu.let0p1.collected_fC": s["collected_charge_substrate_fC"]})


def test_simws_geant4_track(tmp_path, golden):
    if not (ENVS / "process_rad" / "bin" / "geant4-config").exists():
        pytest.skip("Geant4 not installed in process_rad")
    run_in_env("process_rad", [env_python("process_rad"), REPO / "workflow" / "scripts" / "geant4_track.py",
                               "--outdir", tmp_path, "--events", "100"], REPO, timeout=1500, extra_env=_simws_env())
    res = json.loads((tmp_path / "summary.json").read_text())["results"]
    # energy deposited near the entrance agrees with the G4EmCalculator stopping power (delta electrons escape
    # the first bins, cut 1 um, so a few % either way is expected)
    assert abs(res["LET_ratio_entrance_deposit_to_G4EmCalculator"] - 1) < 0.08, res
    assert res["radius_90pct_energy_um"] < 0.5
    golden.verify({"simws.geant4.O16_100MeV.LET_G4EmCalculator": res["meta"]["entrance_LET_MeV_cm2_per_mg"]})


# --------------------------------------------------------------------------- simws 0.3 (roadmap items 5-8)
def test_simws_physics_checks(tmp_path, golden):
    """BGN-consistent equilibrium, Masetti mobility, SRH+Auger lifetimes, n_i(T) vs measured, r-z volumes."""
    run_in_env("tcad", [env_python("tcad"), REPO / "workflow" / "scripts" / "physics_checks.py", "--outdir", tmp_path],
               REPO, timeout=900, threads=1, extra_env=_simws_env())
    res = json.loads((tmp_path / "physics_checks.json").read_text())["results"]
    assert res["pass"], res
    golden.verify({"simws.physics.DeltaEg_max_meV": res["equilibrium_bgn"]["DeltaEg_max_meV"]})


def test_simws_axisymmetric_seu(tmp_path, golden):
    run_in_env("tcad", [env_python("tcad"), REPO / "workflow" / "scripts" / "axisymmetric.py", "--outdir", tmp_path,
                        "--lets", "10"], REPO, timeout=2400, threads=1, extra_env=_simws_env())
    r = json.loads((tmp_path / "summary.json").read_text())["results"]
    assert r["track_mesh_correction"] > 0.9
    assert r["collected_charge_drain_fC"] < r["deposited_charge_fC"]
    assert abs(r["collected_charge_drain_fC"] / r["collected_charge_sub_fC"] - 1) < 0.01   # charge conservation
    golden.verify({"simws.axi.let10.collected_drain_fC": r["collected_charge_drain_fC"],
                   "simws.axi.let10.peak_drain_A": r["peak_drain_current_A"]})


def test_simws_calibration_selftest(tmp_path):
    run_in_env("tcad", [env_python("tcad"), REPO / "workflow" / "scripts" / "calibration_selftest.py", "--outdir",
                        tmp_path], REPO, timeout=1800, threads=1, extra_env=_simws_env())
    res = json.loads((tmp_path / "selftest.json").read_text())["results"]
    assert res["pass"], res


# --------------------------------------------------------------------------- simws 0.4 (roadmap items 9, 10, 12, 13, 20)
def test_simws_geant4_event_seu(tmp_path, golden):
    """Two Geant4 Cl-35 events -> one event-driven axisymmetric strike + Gaussian reference."""
    if not (ENVS / "process_rad" / "bin" / "geant4-config").exists():
        pytest.skip("Geant4 not installed in process_rad")
    run_in_env("process_rad", [env_python("process_rad"), REPO / "workflow" / "scripts" / "geant4_events.py", "--outdir",
                               tmp_path / "g4", "--ions", "17:35:300", "--events", "2"], REPO, timeout=1200, extra_env=_simws_env())
    run_in_env("tcad", [env_python("tcad"), REPO / "workflow" / "scripts" / "event_seu.py", "--index", tmp_path / "g4" / "index.json",
                        "--outdir", tmp_path / "seu", "--n-events", "1"], REPO, timeout=2400, threads=1, extra_env=_simws_env())
    r = next(iter(json.loads((tmp_path / "seu" / "summary.json").read_text())["results"].values()))
    assert 0.05 < r["collection_efficiency_mean"] < 0.6
    assert abs(r["gaussian_bias_in_efficiency"]) < 0.3
    golden.verify({"simws.event.cl35.collected_fC": r["collected_fC_mean"]})


def test_simws_ser_engine_and_critical_let(tmp_path):
    axi = REPO / "results" / "axisymmetric" / "summary.json"
    if not axi.exists():
        pytest.skip("run the workflow first (results/axisymmetric)")
    run_in_env("tcad", [env_python("tcad"), REPO / "workflow" / "scripts" / "ser.py", "--axi", axi, "--outdir", tmp_path,
                        "--qcrit-fC", "20"], REPO, timeout=300, extra_env=_simws_env())
    r = json.loads((tmp_path / "summary.json").read_text())["results"]
    assert abs(r["engine_verification"]["rel_diff"]) < 2e-3
    assert r["critical_let"] and 0.1 < r["critical_let"] < 5


def test_simws_figures_build(tmp_path):
    if not (REPO / "results" / "axisymmetric" / "summary.json").exists():
        pytest.skip("run the workflow first")
    run_in_env("tcad", [env_python("tcad"), REPO / "tools" / "make_figures.py", "--out", tmp_path], REPO, timeout=600,
               extra_env=_simws_env())
    idx = json.loads((tmp_path / "INDEX.json").read_text())
    assert len(idx["made"]) >= 6, idx
    assert (tmp_path / "methods_parameters.tex").exists()


# --------------------------------------------------------------------------- simws 0.5 (roadmap items 11, 16, 17)
def test_simws_mos_and_tid(tmp_path, golden):
    run_in_env("tcad", [env_python("tcad"), REPO / "workflow" / "scripts" / "mos_tid.py", "--outdir", tmp_path],
               REPO, timeout=1800, threads=1, extra_env=_simws_env())
    r = json.loads((tmp_path / "summary.json").read_text())["results"]
    assert r["checks"]["pass"], r["checks"]
    golden.verify({"simws.tid.tox20.dVmg_1Mrad_V": r["tid"]["tox_20nm"]["dV_mg"][-1]})


def test_simws_mos_variability_surrogate(tmp_path):
    run_in_env("tcad", [env_python("tcad"), REPO / "workflow" / "scripts" / "mos_variability.py", "--outdir", tmp_path,
                        "--n-train", "40", "--n-test", "8"], REPO, timeout=2400, threads=1, extra_env=_simws_env())
    r = json.loads((tmp_path / "summary.json").read_text())["results"]
    for o in ("V_fb", "V_inv"):
        assert r[o]["loo_q2"] > 0.99 and r[o]["holdout_rmse"] < 0.01 * r[o]["std"] + 1e-4, r[o]


def test_simws_sic_checks(tmp_path, golden):
    run_in_env("tcad", [env_python("tcad"), REPO / "workflow" / "scripts" / "sic_checks.py", "--outdir", tmp_path],
               REPO, timeout=3000, threads=1, extra_env=_simws_env())
    r = json.loads((tmp_path / "sic_checks.json").read_text())["results"]
    assert r["pass"], {k: v.get("pass") for k, v in r.items() if isinstance(v, dict)}
    golden.verify({"simws.sic.BV_2e17_V": r["breakdown"]["BV_devsim"]})


# --------------------------------------------------------------------------- simws 0.6 (SiGe)
def test_simws_sige_checks(tmp_path, golden):
    run_in_env("tcad", [env_python("tcad"), REPO / "workflow" / "scripts" / "sige.py", "--what", "checks",
                        "--outdir", tmp_path], REPO, timeout=3600, threads=1, extra_env=_simws_env())
    r = json.loads((tmp_path / "sige_checks.json").read_text())["results"]
    assert r["pass"], {k: v.get("pass") for k, v in r.items() if isinstance(v, dict)}
    golden.verify({"simws.sige.IC_ratio_0.6V": r["collector_enhancement"]["cases"]["VBE_0.6"]["Jn_ratio_devsim"],
                   "simws.sige.buried_shift_V": r["buried_channel"]["shift_V"],
                   "simws.sige.strike_QC_let0.1_fC": r["strike_conservation"]["Q_C_fC"][0]})


# --------------------------------------------------------------------------- simws 0.7 (first-principles parameters)
def test_simws_dft_params_quick(tmp_path, golden):
    """PBE-only pass of the DFT parameter study (no convergence variants, no CHGNet): runs Quantum ESPRESSO."""
    import yaml as _y
    env = _y.safe_load((REPO / "config" / "config.yaml").read_text()).get("dft", {}).get("env", "materials")
    run_in_env(env, [env_python(env), REPO / "workflow" / "scripts" / "dft_params.py", "--outdir", tmp_path, "--quick",
                     "--ranks", "4"], REPO, timeout=3600, threads=1, extra_env={**_simws_env(), "SIMWS_DFT_PSEUDOS": "bundled"})
    p = json.loads((tmp_path / "params.json").read_text())["results"]["parameters"]
    c = json.loads((tmp_path / "checks.json").read_text())["results"]
    # PBE alone (quick mode) is known to be soft: the C_ij tolerances apply to the ensemble mean (full rule)
    soft = {"C11", "C12", "C44"}
    assert all(v["pass"] for k, v in c["parameters"].items() if k not in soft), c["parameters"]
    assert c["intrinsic"]["pass"] and c["gap"]["pass"] and c["elastic_consistency"]["pass"], c
    golden.verify({"simws.dft.PBE.m_l": p["m_l"]["value"], "simws.dft.PBE.m_t": p["m_t"]["value"],
                   "simws.dft.PBE.Nc_300": p["Nc_300"]["value"], "simws.dft.PBE.Nv_300": p["Nv_300"]["value"]})


# --------------------------------------------------------------------------- simws 0.8 (hub core contract)
def test_simws_hub_connections_in_tcad_env():
    """Connection tests (producer -> consumer hand-overs) in the tcad env, where every engine exists."""
    out = run_in_env("tcad", [env_python("tcad"), "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                              "tests/connections"], REPO, timeout=1800, extra_env=_simws_env())
    assert re.search(r"\d+ passed", out) and "skipped" not in out and "failed" not in out, out[-1500:]


def test_simws_hub_gate_devices(tmp_path, golden):
    """Upgrade gate: hub results equal the direct 0.7 calls (TID, SiGe enhancement, HBT strike) to round-off."""
    run_in_env("tcad", [env_python("tcad"), REPO / "workflow" / "scripts" / "hub_gate.py", "--part", "devices",
                        "--outdir", tmp_path], REPO, timeout=3600, threads=1, extra_env=_simws_env())
    r = json.loads((tmp_path / "hub_gate_devices.json").read_text())
    assert r["pass"], r
    golden.verify({"simws.hub.tid.tox20.dVmg_1Mrad_V": r["tid.tox20.dVmg_1Mrad_V"]["hub"],
                   "simws.hub.sige.IC_ratio_0.6V": r["sige.IC_ratio_0.6V"]["hub"],
                   "simws.hub.sige.strike_QC_let0.1_fC": r["sige.strike_QC_let0.1_fC"]["hub"],
                   "simws.hub.mms.poisson_order": r["mms.poisson_order"]["hub"]})


def test_simws_hub_gate_dft(tmp_path, golden):
    """Upgrade gate, first-principles part: dft.si_bandedge (PBE member) equals a direct band-edge run."""
    import yaml as _y
    env = _y.safe_load((REPO / "config" / "config.yaml").read_text()).get("dft", {}).get("env", "materials")
    run_in_env(env, [env_python(env), REPO / "workflow" / "scripts" / "hub_gate.py", "--part", "dft",
                     "--outdir", tmp_path, "--ranks", "4"], REPO, timeout=7200, threads=1, extra_env=_simws_env())
    r = json.loads((tmp_path / "hub_gate_dft.json").read_text())
    assert r["pass"], r
    golden.verify({"simws.hub.dft.PBE.m_l": r["dft.PBE.m_l"]["hub"], "simws.hub.dft.PBE.m_t": r["dft.PBE.m_t"]["hub"]})


def test_simws_pseudodojo_vs_bundled(tmp_path, golden):
    """The 0.7 Si re-run with the PseudoDojo set: same PBE band edges as the bundled set within physics tolerances
    (masses 3 %, conduction-minimum position 1 %, gap 50 meV)."""
    import yaml as _y
    if not (REPO / "sim_data" / "pseudopotentials" / "pseudodojo_v0.4").is_dir():
        import pytest as _pt
        _pt.skip("PseudoDojo set not installed (tools/fetch_pseudodojo.sh)")
    env = _y.safe_load((REPO / "config" / "config.yaml").read_text()).get("dft", {}).get("env", "materials")
    run_in_env(env, [env_python(env), REPO / "workflow" / "scripts" / "hub_gate.py", "--part", "pseudos",
                     "--outdir", tmp_path, "--ranks", "4"], REPO, timeout=4 * 3600, threads=1, extra_env=_simws_env())
    r = json.loads((tmp_path / "hub_gate_pseudos.json").read_text())
    assert r["pass"], r
    golden.verify({"simws.hub.dft.pseudodojo.PBE.m_l": r["pseudos.dft.m_l"]["pseudodojo"],
                   "simws.hub.dft.pseudodojo.PBE.m_t": r["pseudos.dft.m_t"]["pseudodojo"]})


# --------------------------------------------------------------------------- simws 0.9a (cryogenic core)
def test_simws_cryo_gate(tmp_path, golden):
    """0.9a gate: closed-form cryogenic checks, HBT/BJT convergence 77-400 K (flagship A v2), paper-fidelity
    mesh error bars."""
    flag = REPO / "results" / "flagship_a" / "flagship_a.json"
    run_in_env("tcad", [env_python("tcad"), REPO / "workflow" / "scripts" / "cryo_gate.py", "--outdir", tmp_path,
                        "--flagship", flag], REPO, timeout=4 * 3600, threads=1, extra_env=_simws_env())
    r = json.loads((tmp_path / "cryo_gate.json").read_text())
    assert r["pass"], r
    ch = r["checks"]
    golden.verify({"simws.cryo.fd_statistics_error": ch["statistics_error"]["max_density_error"]})


def test_simws_charon_xcheck():
    """0.9a gate: DEVSIM vs Charon on a cold pn diode within 2 % (needs Charon on this laptop:
    tools/run_charon_xcheck.sh writes the comparison files)."""
    import pathlib as _pl
    import pytest as _pt
    d = _pl.Path.home() / "sim_workspace" / "runs" / "charon_xcheck"
    files = sorted(d.glob("comparison_T*.json"))
    if not files:
        _pt.skip("no Charon comparison yet: run bash tools/run_charon_xcheck.sh")
    for f in files:
        r = json.loads(f.read_text())
        assert r["pass"], (f.name, r["worst_rel_diff"], r.get("hint"))


# --------------------------------------------------------------------------- simws 0.9b (contacts, AC/noise, geometry, coupling)
def test_simws_gate_0p9b(tmp_path, golden):
    """0.9b gate: thermionic emission vs closed form to 1e-4, f_T within 5 % of 1/(2 pi tau), field transfers
    conserving integrals to 1e-8, preCICE electro-thermal coupling (tolerances in workflow/scripts/gate_0p9b.py)."""
    run_in_env("tcad", [env_python("tcad"), REPO / "workflow" / "scripts" / "gate_0p9b.py", "--outdir", tmp_path],
               REPO, timeout=2 * 3600, threads=1, extra_env=_simws_env())
    r = json.loads((tmp_path / "gate_0p9b.json").read_text())
    assert r["pass"], {k: v.get("pass") for k, v in r.items() if isinstance(v, dict)}
    ft = {f"{row['ge_peak']}_{row['VBE']}": row["fT_ac_GHz"] for row in r["small_signal"]["hbt_ft"]["rows"]}
    golden.verify({"simws.0p9b.fT_SiGe_0p85_GHz": ft["0.15_0.85"], "simws.0p9b.fT_Si_0p85_GHz": ft["0.0_0.85"],
                   "simws.0p9b.precice_Tmax_5V_K": r["geometry"]["precice"]["same_mesh"][-1]["T_max_K"]})


# --------------------------------------------------------------------------- simws 0.9c (engine pairs, held-out gate)
def test_simws_engine_pairs():
    """0.9c gate: each engine pair within the tolerances fixed on 2026-11-04 (results/engine_pairs, written by the
    workflow rules engine_pairs_dft and engine_pairs)."""
    import pytest as _pt
    f = REPO / "results" / "engine_pairs" / "engine_pairs.json"
    if not f.is_file():
        _pt.skip("run: snakemake studies (rules engine_pairs_dft, engine_pairs)")
    r = json.loads(f.read_text())
    assert r["pass"], {k: r[k].get("pass") for k in ("dft", "fem", "dd") if isinstance(r.get(k), dict)}


def test_simws_sic_schottky_heldout(tmp_path):
    """0.9c gate: SiC Schottky I-V(T) against the approved held-out dataset (skipped until one is approved)."""
    import subprocess as _sp
    import pytest as _pt
    p = _sp.run([str(env_python("tcad")), str(REPO / "workflow" / "scripts" / "sic_heldout.py"), "--outdir",
                 str(tmp_path)], cwd=REPO, env={**__import__("os").environ, **_simws_env()}, capture_output=True,
                text=True, timeout=3600)
    if p.returncode == 3:
        _pt.skip("no approved SiC Schottky dataset yet (simws/validation/data/sic_schottky/<name>/meta.json)")
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
