"""Unit tests for simws 0.5 (Batch C): MOS capacitor + total dose, polynomial-chaos surrogates, 4H-SiC."""
import dataclasses
import math

import numpy as np
import pytest

pytest.importorskip("pint")
pytest.importorskip("yaml")


def test_pce_ishigami_sobol_and_gaussian_moments():
    from simws.surrogate import PCE, Variable
    V = [Variable(n, "uniform", -math.pi, math.pi) for n in ("x1", "x2", "x3")]
    d = PCE.design(V, 400)
    f = lambda r: math.sin(r["x1"]) + 7 * math.sin(r["x2"]) ** 2 + 0.1 * r["x3"] ** 4 * math.sin(r["x1"])
    p = PCE.fit(V, d, [f(r) for r in d], max_degree=12)
    s = p.sobol()
    assert p.loo_q2 > 0.999
    for k, s1, st in (("x1", 0.314, 0.558), ("x2", 0.442, 0.442), ("x3", 0.0, 0.244)):
        assert s[k]["S1"] == pytest.approx(s1, abs=0.01) and s[k]["ST"] == pytest.approx(st, abs=0.01)
    V2 = [Variable("a", "normal", 1, 0.5), Variable("b", "normal", 0, 1)]
    d2 = PCE.design(V2, 40)
    p2 = PCE.fit(V2, d2, [2 * r["a"] + r["b"] ** 2 for r in d2], max_degree=3)
    assert p2.mean == pytest.approx(3.0, abs=1e-9) and p2.variance == pytest.approx(3.0, abs=1e-9)


def test_tid_buildup_limits():
    from simws.radiation.tid import buildup, charge_yield
    assert charge_yield(1e3) == pytest.approx(1.0, abs=1e-3) and charge_yield(0.5) == pytest.approx(2 ** -0.7)
    N_ot, N_it = buildup([0, 1e3, 1e9], 20.0, 1.0)
    assert N_ot[0] == 0 and N_ot[2] == pytest.approx(2e12, rel=1e-6)          # saturates at N_sat
    lin = 0.05 * 8.1e12 * charge_yield(1.0) * 20e-7 * 1e3
    assert N_ot[1] == pytest.approx(lin, rel=1e-3)                            # linear at low dose
    assert buildup([1e5], 20.0, -1.0)[1][0] == 0                              # no interface traps at negative bias


def test_mos_flatband_and_fixed_charge_exact():
    pytest.importorskip("devsim")
    from simws.tcad.mos import MOSCase, flatband
    from simws.units import EPS0_F_PER_CM, Q_E
    v0 = flatband(MOSCase())
    v1 = flatband(MOSCase(Q_f=5e11))
    cox = 3.9 * EPS0_F_PER_CM / 10e-7
    assert (v1 - v0) == pytest.approx(-Q_E * 5e11 / cox, rel=1e-6)
    assert v0 == pytest.approx(-0.4159, abs=2e-3)                             # midgap gate on p-Si 1e17


def test_sic_ionization_and_mobility():
    pytest.importorskip("devsim")
    from simws.validation import sic_checks
    assert sic_checks.single_level_closed_form()["pass"]
    assert sic_checks.devsim_ionization()["pass"]
    r = sic_checks.anisotropic_mobility()
    assert r["pass"], r


def test_sic_band_values():
    from simws.materials import load
    from simws.tcad.sic import band, equilibrium
    Eg, Nc, Nv, ni = band(load("sic_4h"), 300.0)
    assert 3.2 < Eg < 3.3 and 1e-9 < ni < 1e-7
    e = equilibrium(load("sic_4h"), 1e18, 0.0)
    assert 0.4 < e["ND+"] / 1e18 < 0.7                                         # nitrogen half ionized at 1e18
