"""Unit tests for items 6-9: radiation (LET, SRIM, track), sweeps/UQ store and solution verification.
Fast (seconds); DEVSIM/Geant4 runs are in the slow tier."""
import math

import numpy as np
import pytest

pytest.importorskip("pint")
pytest.importorskip("yaml")

SRIM_SAMPLE = """ ==================================================================
              Calculation using SRIM-2006
 Ion = Oxygen [8] , Mass = 15.995 amu
 Target Density =  2.3212E+00 g/cm3 = 4.9766E+22 atoms/cm3
 Stopping Units =  MeV / (mg/cm2)
   Ion        dE/dx      dE/dx     Projected  Longitudinal   Lateral
  Energy      Elec.      Nuclear     Range     Straggling   Straggling
-----------  ---------- ---------- ----------  ----------  ----------
  1.00 MeV   5.000E+00  1.000E-02     1.00 um       100 A       100 A
 10.00 MeV   7.000E+00  2.000E-03     8.00 um      2000 A      1000 A
100.00 MeV   3.000E+00  3.000E-04    80.00 um      1.00 um     0.50 um
-----------------------------------------------------------
"""


def test_let_charge_conversion_si():
    from simws.materials import load
    from simws.radiation.let import charge_fC_per_um, let_from_charge
    si = load("silicon")
    assert charge_fC_per_um(1.0, si) == pytest.approx(10.36, rel=0.01)     # textbook 10.3-10.4 fC/um per MeV cm2/mg
    assert let_from_charge(charge_fC_per_um(37.0, si), si) == pytest.approx(37.0)


def test_srim_parse_units_and_csda():
    from simws.radiation import srim
    t = srim.parse(SRIM_SAMPLE)
    assert t.ion.startswith("Oxygen")
    assert t.energy_MeV.tolist() == [1.0, 10.0, 100.0]
    assert t.proj_range_um.tolist() == [1.0, 8.0, 80.0]
    assert t.let(10.0) == pytest.approx(7.0)
    prof = t.let_vs_depth(100.0, np.array([0.0, 1.0]))
    assert prof[0] == pytest.approx(3.0, rel=1e-6)
    assert 3.0 < prof[1] < 3.1                       # LET rises as the ion slows (below the Bragg peak energy)
    kev_um = SRIM_SAMPLE.replace("MeV / (mg/cm2)", "keV / micron")
    assert srim.parse(kev_um).let(10.0) == pytest.approx(7.0 * 1e-3 * 1e4 / 2321.2)


def test_track_conserves_charge_on_mesh():
    from simws.materials import load
    from simws.radiation.track import IonTrack
    si = load("silicon")
    tr = IonTrack(let=10.0, x0_um=1.0, length_um=2.0, radius_um=0.05, depth_um=1.0)
    assert tr.total_charge_fC(si) == pytest.approx(2.0 * 10.36 * 10.0, rel=0.01)
    xs, ys = np.linspace(0, 2, 201), np.linspace(0, 3, 61)
    X, Y = np.meshgrid(xs, ys)
    vol = np.full(X.size, (xs[1] - xs[0]) * (ys[1] - ys[0]) * 1e-8)
    foot, factor = tr.footprint(si, X.ravel(), Y.ravel(), vol)
    assert abs(factor - 1) < 0.05                    # mesh resolves the 50 nm track to a few %
    assert (foot * vol).sum() * 1e-4 == pytest.approx(tr.total_pairs(si), rel=1e-9)
    t = np.linspace(0, 2e-11, 4001)
    assert np.trapezoid([tr.temporal(v) for v in t], t) == pytest.approx(1.0, rel=1e-4)


def _quadratic(p):
    return {"y": p["a"] ** 2 + 0.1 * p["b"]}


def test_sweep_store_resume_and_sobol(tmp_path):
    from simws.sweep import Space, Store, run_design
    space = Space({"a": (-1.0, 1.0), "b": (-1.0, 1.0)})
    store = Store(tmp_path / "s.sqlite")
    design = space.sobol(64)
    run_design(_quadratic, design, store, workers=2, progress=False)
    assert len(store.rows()) == len(design)
    run_design(_quadratic, design, store, workers=2, progress=False)          # resume: nothing recomputed
    assert len(store.rows()) == len(design)
    s = space.sobol_indices(store, "y", 64)
    assert s["a"]["ST"] > 0.8 and s["b"]["ST"] < 0.2
    deck = __import__("simws.sweep", fromlist=["dakota_input"]).dakota_input(space, "drv.sh", ["y"])
    assert "continuous_design = 2" in deck and "'a' 'b'" in deck


def test_sweep_records_failures(tmp_path):
    from simws.sweep import Store, run_design
    store = Store(tmp_path / "f.sqlite")
    run_design(_fails, [{"a": 1.0}], store, progress=False)
    (p, err), = store.failures()
    assert p == {"a": 1.0} and "ZeroDivisionError" in err


def _fails(p):
    return {"y": 1 / 0}


def test_verify_observed_order_and_gci():
    from simws.verify import analyse, refinement_study
    exact = 1.0
    res = refinement_study(lambda lv: exact + 0.3 * (0.5 ** lv) ** 2, levels=4)
    assert res.order == pytest.approx(2.0, abs=1e-9)
    assert res.richardson == pytest.approx(exact, abs=1e-12)
    assert res.asymptotic_ratio == pytest.approx(1.0, abs=0.05)
    assert analyse([0, 1, 2], [1.0, 0.9, 0.95], [0, 0, 0]).oscillatory


def _ishigami(p):
    a, b, c = p["x1"], p["x2"], p["x3"]
    return {"y": math.sin(a) + 7 * math.sin(b) ** 2 + 0.1 * c ** 4 * math.sin(a)}


def test_sobol_indices_ishigami(tmp_path):
    """Analytic Ishigami indices: S1 = 0.314, 0.442, 0; ST = 0.558, 0.442, 0.244."""
    from simws.sweep import Space, Store, run_design
    space = Space({k: (-math.pi, math.pi) for k in ("x1", "x2", "x3")})
    store = Store(tmp_path / "i.sqlite")
    run_design(_ishigami, space.sobol(1024), store, progress=False)
    s = space.sobol_indices(store, "y", 1024)
    for k, s1, st in (("x1", 0.314, 0.558), ("x2", 0.442, 0.442), ("x3", 0.0, 0.244)):
        assert s[k]["S1"] == pytest.approx(s1, abs=0.04) and s[k]["ST"] == pytest.approx(st, abs=0.04)


def test_verify_noise_level_convergence():
    from simws.verify import analyse
    r = analyse([0, 1, 2, 3], [0.0165212653, 0.01652145767, 0.01651995831, 0.01652052186], [0] * 4)
    assert r.oscillatory and r.order is None and r.gci_fine < 2e-4
