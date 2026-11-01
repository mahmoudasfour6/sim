"""Unit tests for the simws package. Run in the tcad env, which has every dependency:
    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python -m pytest -q tests/unit
In other envs, tests whose dependencies are missing are skipped (the slow tier runs them in tcad)."""
import json

import numpy as np
import pytest

pytest.importorskip("pint")
pytest.importorskip("yaml")


def test_material_units_and_guards():
    from simws.materials import load, available
    assert {"silicon", "sic_4h", "gaas", "sio2"} <= set(available())
    si = load("silicon")
    assert si.get("mu_n", "cm**2/V/s") == pytest.approx(1414)
    assert si.get("mu_n", "m**2/V/s") == pytest.approx(0.1414)
    assert si.get("thermal_k", "W/m/K") == pytest.approx(148)
    with pytest.raises(Exception):
        si.get("mu_n", "eV")                         # wrong dimensions must fail loudly
    with pytest.raises(KeyError):
        si.get("not_a_parameter", "eV")
    assert load("silicon", profile="charon").get("n_i", "cm**-3") == pytest.approx(1.44712e10)
    assert load("silicon").get("n_i", "cm**-3") == pytest.approx(9.65e9)


def test_every_parameter_has_unit_reference_and_status():
    from simws.materials import available, load
    for name in available():
        for key, p in load(name).parameters.items():
            assert p.ref, f"{name}.{key} has no reference"
            assert p.status in ("verified", "provisional"), f"{name}.{key} status {p.status!r}"


def test_overrides_are_recorded():
    from simws.materials import load
    m = load("silicon").with_overrides(n_i=(1e10, "cm**-3"))
    assert m.get("n_i", "cm**-3") == 1e10 and m.overrides == {"n_i": 1e10}


def test_interpolation_exact_for_linear_fields():
    pytest.importorskip("scipy"); pytest.importorskip("matplotlib")
    from matplotlib.tri import Triangulation
    from simws.io.interp import Transfer
    xs, ys = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 0.5, 5))
    src = np.column_stack([xs.ravel(), ys.ravel()])
    tri = Triangulation(src[:, 0], src[:, 1]).triangles
    dst = np.random.default_rng(0).random((400, 2)) * [1, 0.5]
    f = lambda p: 3 * p[:, 0] - 2 * p[:, 1] + 0.5
    assert np.max(np.abs(Transfer(src, tri, dst)(f(src)) - f(dst))) < 1e-12


def test_field_file_roundtrip_with_units(tmp_path):
    pytest.importorskip("meshio"); pytest.importorskip("matplotlib")
    from matplotlib.tri import Triangulation
    from simws.io import fields
    xs, ys = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 3))
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    tri = Triangulation(pts[:, 0], pts[:, 1]).triangles
    fs = fields.FieldSet(pts * 1e-4, tri, "cm", {"T": 300 + pts[:, 0]}, {"T": "K"}, {"case": "unit-test"})
    back = fields.read(fields.write(tmp_path / "f", fs))
    assert np.allclose(back.get("T"), 300 + pts[:, 0]) and back.meta == {"case": "unit-test"}
    assert np.allclose(back.points_in("um"), pts)
    with pytest.raises(ValueError):
        fields.write(tmp_path / "bad", fields.FieldSet(pts, tri, "cm", {"x": pts[:, 0]}, {}))


def test_provenance_record(tmp_path):
    from simws.provenance import stamp, write_json
    f = tmp_path / "in.txt"; f.write_text("x")
    rec = stamp(inputs=[f], parameters={"a": 1})
    assert {"git_commit", "conda_env", "inputs", "parameters", "created_utc"} <= set(rec)
    assert len(rec["inputs"][str(f)]) == 64
    write_json(tmp_path / "o.json", {"v": 1.0}, rec)
    assert json.loads((tmp_path / "o.json").read_text())["results"]["v"] == 1.0


def test_si_film_reproduces_validated_subbands():
    pytest.importorskip("scipy")
    from simws.quantum.si_film import solve
    r = solve()
    assert r["converged"]
    assert r["E_D2_meV"][0] == pytest.approx(114.6, abs=0.05)
    assert r["E_D2_meV"][1] == pytest.approx(145.6, abs=0.05)
    assert r["E_D4_meV"][0] == pytest.approx(180.1, abs=0.05)
    assert r["EF_meV"] == pytest.approx(135.4, abs=0.05)
    assert r["n_s_cm2"] == pytest.approx(1e13, rel=1e-6)


def test_electrothermal_verification_against_analytic():
    """No feedback (constant mu, n_i, k): uniform heating with a sink at x = L has T_max = T0 + qL^2/2k.
    Also checks the discrete energy identities: Joule heat = I*V and heat into the sink = Joule heat."""
    pytest.importorskip("devsim"); pytest.importorskip("skfem")
    from simws.coupling import electrothermal as et
    from simws.materials import load
    si = load("silicon")
    case = et.ResistorCase(biases=[2.0], couple_mobility=False, couple_ni=False, couple_k=False)
    p = et.run(case, si)["points"][0]
    Ta = et.analytic_max_temperature(p["joule_W_per_cm"], case, si.get("thermal_k", "W/cm/K"))
    assert (p["T_max_K"] - Ta) / (Ta - case.T_sink) == pytest.approx(0, abs=1e-8)
    assert p["joule_vs_IV_rel_error"] < 1e-10
    assert p["energy_balance_rel_error"] < 1e-8


def test_electrothermal_coupled_converges_and_conserves_energy():
    pytest.importorskip("devsim"); pytest.importorskip("skfem")
    from simws.coupling import electrothermal as et
    pts = et.run(et.ResistorCase(biases=[1.0, 3.0]))["points"]
    for p in pts:
        assert p["converged"] and p["iterations"] < 30
        assert p["joule_vs_IV_rel_error"] < 1e-10 and p["energy_balance_rel_error"] < 1e-8
        assert p["current_A_per_cm"] < p["current_isothermal_A_per_cm"]     # self-heating lowers mobility
    assert pts[1]["T_max_K"] > pts[0]["T_max_K"] > 300
