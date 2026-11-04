"""0.9c engine pairs that run in the tcad env (DEVSIM vs Solcore; the scikit-fem side of the FEM pair) and the SiC
held-out harness on synthetic data. The DFT pair (QE vs GPAW) and DOLFINx run in the 0.9c workflow rules."""
import numpy as np
import pytest

from simws.validation import engine_pairs as E


def test_tolerances_fixed_in_advance():
    assert E.FIXED_ON == "2026-11-04"
    assert E.TOLERANCES["dd.dark_J"] == 0.02 and E.TOLERANCES["fem.poisson_nodal"] == 1e-9


def test_devsim_vs_solcore():
    pytest.importorskip("solcore")
    r = E.dd_pair()
    assert r["pass"], r
    assert r["model_form_rel_dark"] < E.TOLERANCES["dd.dark_J"]


def test_skfem_side_of_fem_pair(tmp_path):
    g = E._fem_geometry()
    a = E.skfem_solve(g.mesh(tmp_path / "f.msh", size=0.05))
    u, T = np.array(a["u"]), np.array(a["T"])
    assert abs(u.min()) < 1e-12 or u.min() >= 0
    assert T.max() > E.FEM_PROBLEM["heat"]["T0"]


def test_sic_heldout_harness_round_trip():
    from simws.validation import sic_schottky_heldout as H
    r = H.self_test()
    assert r["pass"], r


def test_sic_heldout_refuses_unapproved(tmp_path, monkeypatch):
    from simws.validation import sic_schottky_heldout as H
    d = tmp_path / "x"; d.mkdir()
    (d / "meta.json").write_text('{"approved": false, "fit_T_K": [300], "heldout_T_K": [350], "N_D_cm3": 1e16}')
    (d / "iv.csv").write_text("T_K,V,I_A\n300,0.5,1e-6\n")
    monkeypatch.setattr(H, "DATA", tmp_path)
    with pytest.raises(PermissionError):
        H.run("x")


def test_sic_protocol2_split_rule():
    from simws.validation import sic_schottky_heldout as H
    assert H.split_rule_2([298.15, 323.15, 348.15, 373.15, 398.15]) == ([298.15, 348.15], [323.15, 373.15, 398.15])
    with pytest.raises(ValueError):
        H.split_rule_2([300.0, 400.0])
    lo, hi = H.window_2(300.0, {"R_s_ohm": 2.0})
    assert lo == 1e-7 and abs(hi - 0.025852 / 2.0) < 1e-4
