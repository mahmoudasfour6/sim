"""Unit tests for upgrades 5-8: full drift-diffusion physics, axisymmetric geometry, calibration.
Fast checks only (seconds); the full validation and the strike/calibration runs are in the slow tier."""
import math

import numpy as np
import pytest

pytest.importorskip("pint")
pytest.importorskip("yaml")


def test_masetti_limits_and_values():
    from simws.materials import load
    from simws.tcad.physics import masetti_numpy
    si = load("silicon")
    assert masetti_numpy(si, 1e12, carrier="n") == pytest.approx(si.get("mu_n", "cm**2/V/s"), rel=2e-3)
    assert masetti_numpy(si, 1e12, carrier="p") == pytest.approx(si.get("mu_p", "cm**2/V/s"), rel=2e-3)
    mu = masetti_numpy(si, np.array([1e15, 1e17, 1e19, 1e21]), carrier="n")
    assert np.all(np.diff(mu) < 0)                         # falls with doping
    assert 650 < mu[1] < 800 and 80 < mu[2] < 130          # textbook range (Masetti Fig. 3)


def test_temperature_helpers():
    from simws.materials import load
    from simws.tcad.physics import impact_gamma, intrinsic_density, saturation_velocity
    si = load("silicon")
    assert intrinsic_density(si, 300.0) == pytest.approx(si.get("n_i", "cm**-3"))
    ref400 = 5.29e19 * (400 / 300) ** 2.54 * math.exp(-6726 / 400)          # Misiakos & Tsamakis (1993)
    assert intrinsic_density(si, 400.0) == pytest.approx(ref400, rel=0.08)
    assert impact_gamma(si, 300.0) == pytest.approx(1.0)
    assert impact_gamma(si, 400.0) > 1.0                   # ionization falls with temperature
    vn300, _ = saturation_velocity(si, 300.0)
    vn400, _ = saturation_velocity(si, 400.0)
    assert vn400 < vn300


def test_physics_resistor_matches_masetti():
    pytest.importorskip("devsim")
    from simws.validation.physics_checks import mobility_resistors
    r = mobility_resistors()
    assert r["pass"], r


def test_equilibrium_with_bandgap_narrowing_carries_no_current():
    pytest.importorskip("devsim")
    from simws.validation.physics_checks import equilibrium_bgn
    r = equilibrium_bgn()
    assert r["pass"] and r["DeltaEg_max_meV"] > 50, r


def test_axisymmetric_volume_convention():
    pytest.importorskip("devsim")
    from simws.tcad.axisym import resistor_check
    assert abs(resistor_check(radius_um=3.0, length_um=6.0)["rel_diff"]) < 1e-6


def test_axisymmetric_track_conserves_charge():
    from simws.materials import load
    from simws.radiation.track import IonTrack
    si = load("silicon")
    tr = IonTrack(let=10.0, x0_um=0.0, length_um=2.0, radius_um=0.05)
    r = np.linspace(0, 1.0, 401); z = np.linspace(0, 3.0, 61)
    R, Z = np.meshgrid(r, z)
    dr, dz = (r[1] - r[0]) * 1e-4, (z[1] - z[0]) * 1e-4
    vol = 2 * math.pi * np.maximum(R.ravel() * 1e-4, dr / 4) * dr * dz          # ring volumes
    foot, factor = tr.footprint_axisymmetric(si, R.ravel(), Z.ravel(), vol)
    assert abs(factor - 1) < 0.05
    assert (foot * vol).sum() == pytest.approx(tr.total_pairs(si), rel=1e-12)


def _line(p):
    x = np.linspace(0, 1, 12)
    return {"iv": p["a"] * np.exp(p["b"] * x)}


def test_calibration_recovers_parameters_with_uncertainty():
    from simws.calibrate import Calibration, Parameter
    rng = np.random.default_rng(3)
    truth = {"a": 2e-9, "b": 3.0}
    y = _line(truth)["iv"] * np.exp(0.05 * rng.standard_normal(12))
    cal = Calibration([Parameter("a", 1e-12, 1e-6, log=True), Parameter("b", 0.5, 10.0)], _line,
                      {"iv": {"V": np.linspace(0, 1, 12), "y": y}}, {"iv": 0.05})
    fit = cal.fit()
    for k, v in truth.items():
        lo, hi = fit.interval_95[k]
        assert lo <= v <= hi, (k, fit.table())
    assert 0.2 < fit.chi2_per_dof < 3
    assert abs(fit.correlation[0, 1]) > 0.5                 # a and b trade off along the curve
    post = cal.sample(n_steps=3000, seed=1)
    assert 0.1 < post["acceptance"] < 0.7
    s = post["samples"]["b"]
    assert abs(np.median(s) - fit.best["b"]) < 2 * np.std(s)


def test_measurement_loader(tmp_path):
    from simws.calibrate import load_measurement
    f = tmp_path / "m.csv"
    f.write_text("# diode D3, 300 K\nV,I,C\n-1,,2.5e-12\n0,,3.1e-12\n0.5,1e-6,\n0.6,2e-5,\n")
    d = load_measurement(f)
    assert d["iv"]["V"].tolist() == [0.5, 0.6] and d["cv"]["y"].tolist() == [2.5e-12, 3.1e-12]
