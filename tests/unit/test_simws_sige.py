"""Unit tests for simws 0.6: SiGe material laws, heterojunction equilibrium, HBT surface and track bookkeeping."""
import math

import numpy as np
import pytest

pytest.importorskip("pint")
pytest.importorskip("yaml")


def test_sige_parameters_and_offsets():
    from simws.materials import load
    m = load("sige")
    assert m.get("strained_dEg_per_x", "eV") == pytest.approx(0.75)
    x = 0.25
    gam = math.exp(-math.sqrt(m.get("dos_gamma_k", "dimensionless") * x))
    dEv = 0.75 * x + 0.025852 * math.log(gam / m.get("dos_Nc_ratio", "dimensionless"))
    assert 0.15 < dEv < 0.19                       # band offset minus the DOS reduction
    Eg = m.get("relaxed_Eg_a", "eV") + m.get("relaxed_Eg_b", "eV") * 0.5 + m.get("relaxed_Eg_c", "eV") * 0.25
    assert Eg == pytest.approx(0.917, abs=1e-3)    # relaxed Si0.5Ge0.5 (Delta minimum)


def test_graded_sige_equilibrium_is_current_free():
    pytest.importorskip("devsim")
    from simws.validation.physics_checks import equilibrium_sige
    r = equilibrium_sige()
    assert r["pass"], r


def test_hbt_surface_factor_integrates_to_spacer_length():
    pytest.importorskip("devsim")
    from devsim import get_node_model_values
    from simws.radiation.hbt_rad import _surface_factor
    from simws.tcad.hbt import HBTCase, UM, build
    c = HBTCase()
    build(c)
    f = _surface_factor(c)
    vol = np.array(get_node_model_values(device="d", region="si", name="NodeVolume"))
    assert float((f * vol).sum()) == pytest.approx((c.x_bc - c.w_e) * UM, rel=1e-9)


def test_hbt_track_footprint_conserves_charge():
    pytest.importorskip("devsim")
    from devsim import get_node_model_values
    from simws.materials import load
    from simws.radiation.track import IonTrack
    from simws.tcad.hbt import HBTCase, UM, build
    build(HBTCase(), x_lines=[(0.05, 0.02), (0.25, 0.02), (0.45, 0.02)])
    g = lambda k: np.array(get_node_model_values(device="d", region="si", name=k))
    t = IonTrack(let=1.0, x0_um=0.25, length_um=1.5, radius_um=0.05)
    foot, factor = t.footprint(load("silicon"), g("x") / UM, g("y") / UM, g("NodeVolume"))
    assert abs(factor - 1) < 0.03
    assert float((foot * g("NodeVolume")).sum()) * 1e-4 == pytest.approx(t.total_pairs(load("silicon")), rel=1e-9)
