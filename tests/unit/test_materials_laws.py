"""Unit tests for simws 0.8: material laws with validity boxes, oxide sets, interface distributions."""
import numpy as np
import pytest

pytest.importorskip("pint")
pytest.importorskip("pydantic")

from simws.materials import available, laws, load  # noqa: E402


def test_new_material_sets_load_and_are_flagged():
    assert {"al2o3", "hfo2"} <= set(available()) and "interfaces" not in available()
    for m, eps in (("al2o3", 9.0), ("hfo2", 25.0)):
        mat = load(m)
        assert mat.get("eps_r", "dimensionless") == eps
        assert "eps_r" in mat.provisional()                 # secondary-source check only: stays provisional


def test_varshni_law_matches_material_and_flags():
    v = laws.evaluate("silicon", "Eg", T=300.0)
    assert v.value == pytest.approx(1.1245, abs=1e-3) and v.flags == []
    v4 = laws.evaluate("silicon", "Eg", T=4.0)
    assert v4.value == pytest.approx(1.170, abs=1e-4)


def test_extrapolation_flag_and_unknown_validity():
    v = laws.evaluate("silicon", "mu_n", T=[77.0, 300.0])
    assert v.value[1] == pytest.approx(1414.0)
    assert any(f.startswith("extrapolated:T") for f in v.flags)
    assert laws.evaluate("silicon", "mu_n", T=300.0).flags == []
    assert "validity_unknown" in laws.evaluate("sic_4h", "Eg", T=573.0).flags
    assert laws.evaluate("silicon", "n_i", T=300.0).value == pytest.approx(9.65e9, rel=1e-6)


def test_law_spec_for_graph():
    s = laws.spec("silicon", "Eg")
    assert s.expression == "varshni" and s.status == "verified" and s.validity["T"] == (0.0, 1000.0)
    assert s.coefficients["beta"].unit == "K"
    assert laws.spec("hfo2", "Eg").status == "provisional"
    with pytest.raises(KeyError):
        laws.get("silicon", "nonsense")


def test_interface_distributions():
    d = laws.sample_interface("Si/SiO2", "Dit_midgap", 2000, seed=0)
    assert d.min() >= 1e9 and d.max() <= 1e11
    assert np.median(np.log10(d)) == pytest.approx(10.0, abs=0.1)     # log-uniform centre
    assert all(e["status"] == "provisional" for i in laws.interfaces().values() for e in i.values())
