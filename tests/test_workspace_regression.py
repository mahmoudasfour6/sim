#!/usr/bin/env python3
"""
Automated Physical Invariant & Toolchain Regression Suite (`pytest`)
====================================================================
Guards against silent physical regressions across all 10 Conda environments:
  1. 3C-SiC crystal density must match experimental ~3.214 g/cm^3 (guards against unit-cell bugs).
  2. SRAM 6T mixed-mode node voltage under heavy-ion strike must remain clamped by the
     drain-substrate PN junction (V_min >= -0.95 V, never unphysically <-1.5 V).
  3. Multiscale pipeline must report unambiguous SRH recombination vs total current ratios.
  4. IHP SG13G2 `npn13G2` foundry VBIC 300 K baseline must satisfy foundry corner ordering
     (hbt_wcs < hbt_typ < hbt_bcs for both peak beta and peak f_T).
  5. Characterization suite (`charz`: abTEM, py4DSTEM, HyperSpy, xrayutilities) outputs
     must match strained Si0.78Ge0.22 / Si(001) elasticity & Takagi-Taupin diffraction bounds.
  6. SEACAS `exodus` Python binding and OpenVAF-compiled `.osdi` models must be present and loadable.

Run with:
    /home/mahmoudasfour/miniforge3/envs/workflow/bin/pytest -v /home/mahmoudasfour/sim_workspace/tests/test_workspace_regression.py
"""
import json
import os

WORKSPACE = "/home/mahmoudasfour/sim_workspace"
PIPE = f"{WORKSPACE}/pipeline"


def test_materials_sic_density_and_qe_energy():
    with open(f"{PIPE}/w1_materials_out.json") as f:
        w1 = json.load(f)
    # 3C-SiC theoretical density is 3.214 g/cm^3
    assert 3.15 <= w1["sic_density_g_cm3"] <= 3.26, f"Unphysical SiC density: {w1['sic_density_g_cm3']}"
    assert abs(w1["si_volume_A3"] - 5.431**3) < 0.05
    assert -22.90 <= w1["qe_si_scf_energy_Ry"] <= -22.80


def test_sram_mixedmode_substrate_diode_clamping():
    with open(f"{PIPE}/w5_tcad_out.json") as f:
        w5 = json.load(f)
    # Struck drain node is clamped by forward-biased drain-substrate PN diode around -0.76 V
    assert -0.95 <= w5["sram_v_min_V"] <= 0.1, f"Unphysical SRAM node undershoot: {w5['sram_v_min_V']} V"
    assert w5["seu_bit_flipped"] is True


def test_multiscale_recombination_vs_total_current_consistency():
    with open(f"{PIPE}/multiscale_results.json") as f:
        ms = json.load(f)
    s3 = ms["stage3_tcad"]
    assert abs(s3["recomb_leakage_increase_factor"] - 26.0) < 0.1
    assert 1.05 <= s3["total_current_increase_ratio"] <= 1.15


def test_ihp_npn13g2_foundry_baseline():
    path = f"{WORKSPACE}/benchmarks/ihp_npn13g2_baseline/npn13g2_baseline_300K.json"
    assert os.path.exists(path)
    with open(path) as f:
        d = json.load(f)
    c = d["corners"]
    # Verify monotonic corner ordering: wcs < typ < bcs
    assert c["hbt_wcs"]["beta_peak"] < c["hbt_typ"]["beta_peak"] < c["hbt_bcs"]["beta_peak"]
    assert c["hbt_wcs"]["fT_peak_GHz"] < c["hbt_typ"]["fT_peak_GHz"] < c["hbt_bcs"]["fT_peak_GHz"]
    assert 700.0 <= c["hbt_typ"]["beta_peak"] <= 900.0
    assert 0.64 <= c["hbt_typ"]["VBE_at_IC_1uA_V"] <= 0.70


def test_charz_stem_nbed_eels_hrxrd_metrology():
    path = f"{WORKSPACE}/07_characterization_charz/charz_metrology_summary.json"
    assert os.path.exists(path)
    with open(path) as f:
        c = json.load(f)
    assert -0.90 <= c["sige_eps_xx_pct"] <= -0.75
    assert 0.55 <= c["sige_eps_zz_pct"] <= 0.70
    assert -2200.0 <= c["hrxrd_sige_004_peak_arcsec"] <= -1850.0
    assert -220.0 <= c["hrxrd_algaas_004_peak_arcsec"] <= -100.0


def test_openvaf_osdi_and_seacas_exodus_installed():
    osdi_dir = f"{WORKSPACE}/ihp_sg13g2_pdk/ihp-sg13g2/libs.tech/ngspice/osdi"
    for model in ("psp103.osdi", "psp103_nqs.osdi", "r3_cmc.osdi", "mosvar.osdi", "cap_cmomi.osdi", "cap_cmomf.osdi"):
        p = os.path.join(osdi_dir, model)
        assert os.path.exists(p) and os.path.getsize(p) > 10000, f"Missing OSDI binary: {p}"
    assert os.path.exists("/home/mahmoudasfour/tcad/install/lib/libexodus.so")
