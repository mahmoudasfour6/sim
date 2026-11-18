"""Contract amendment A9.5 - A9.8: the hub adapter circuit.ngspice (manifest, quantities, axes, flags, self-checks,
trust), the device-to-circuit link (simws.chain.diode: fit_shockley, circuit_from_fit), the study layer on a circuit
problem, and the acceptance checks R30 to R37. Written from docs/ARCHITECTURE_V1.md (A9, with the shared conventions
of A7 / A8) only.

Adapter arithmetic is tested with a FAKE engine run (hand-made CircuitResult objects monkeypatched over the ngspice
engine module's run(), available() and openvaf_available()). Acceptance runs need ngspice (and OpenVAF for R34, R37,
DEVSIM for R35, the IHP PDK for R36) and are skipped without them. Closed forms use CODATA 2018 k and q."""
import math
import os

import numpy as np
import pytest

ADAPTER = "circuit.ngspice"
K_B = 1.380649e-23                       # J/K, CODATA 2018 (exact)
Q_E = 1.602176634e-19                    # C, CODATA 2018 (exact)


def _lib():
    from simws.spec import library
    return library


def _ng():
    from simws import engines
    return engines.module("ngspice")


def _params(p):
    return {"problem": p.model_dump(mode="json")}


def _cp(d):
    from simws.spec import CircuitProblem
    return CircuitProblem.model_validate(d)


def _as_problem(problem):
    from simws.spec import CircuitProblem
    return problem if isinstance(problem, CircuitProblem) else CircuitProblem.model_validate(problem)


def _need_engine():
    try:
        ok = _ng().available()
    except Exception as e:                                      # noqa: BLE001
        pytest.skip(f"ngspice engine module not importable: {e!r}")
    if not ok:
        pytest.skip("ngspice not available (SIMWS_SPICE_PREFIX / <envs_root>/spice)")


def _need_openvaf():
    _need_engine()
    if not _ng().openvaf_available():
        pytest.skip("OpenVAF not available")


def _result(problem, analysis, signals, units, axis=(), axis_name="", axis_unit="", imag=None, flags=()):
    from simws.engines.base import CircuitResult
    return CircuitResult(engine="ngspice", problem_id=problem.problem_id(), analysis=analysis, axis_name=axis_name,
                         axis_unit=axis_unit, axis=list(map(float, axis)),
                         signals={k: list(map(float, v)) for k, v in signals.items()},
                         signals_imag={k: list(map(float, v)) for k, v in (imag or {}).items()}, units=units,
                         flags=list(flags))


@pytest.fixture
def fake_engine(monkeypatch):
    """install(make): the ngspice module's run() becomes make(problem) -> CircuitResult; available() and
    openvaf_available() return True. The fake refuses what the real capability refuses (A9.4: run raises
    Unsupported). Returns the list of problems it was called with."""
    from simws.engines.base import Unsupported
    mod = _ng()
    real_unsupported = mod.unsupported
    calls = []

    def install(make):
        def run(problem, workdir=None, *a, **k):
            p = _as_problem(problem)
            reasons = real_unsupported(p)
            if reasons:
                raise Unsupported("ngspice", reasons)
            calls.append(p)
            return make(p)
        monkeypatch.setattr(mod, "run", run)
        monkeypatch.setattr(mod, "available", lambda: True)
        monkeypatch.setattr(mod, "openvaf_available", lambda: True)
        return calls
    return install


def _names(r):
    return sorted(q.name for q in r.quantities)


def _no_uncertainty(r):
    for q in r.quantities:
        assert q.uncertainty.kind == "none", (q.name, q.uncertainty)
        assert q.uncertainty.sigma is None and not q.uncertainty.components, q.name


def _axis(q):
    assert len(q.axes) == 1, q.name
    a = q.axes[0]
    return a.name, a.unit, np.asarray(a.values, float)


# ------------------------------------------------------------------------------------------------ manifest
def test_manifest():
    from simws.hub import adapter as A
    from simws.hub import citations
    from simws.units import Q_
    m = A.manifest(ADAPTER)
    assert ADAPTER in A.names()
    assert m.name == ADAPTER and m.version == "1.0" and m.env == "tcad" and m.engine == "ngspice"
    assert m.inputs == ["problem"]
    assert "ngspice47" in m.citations and "openvaf" in m.citations
    assert citations.unknown(["ngspice47", "openvaf"]) == []
    assert [o.name for o in m.outputs] == ["circuit.n_points"]
    assert Q_(1.0, m.outputs[0].unit).dimensionality == Q_(1.0, "dimensionless").dimensionality


def test_adapter_classes():
    from simws.adapters import circuit as C
    from simws.hub import adapter as A
    assert issubclass(C.CircuitNgspice, C.CircuitAdapter)
    assert isinstance(A.get(ADAPTER), C.CircuitNgspice)


def test_invalid_description_raised_from_submit(hub):
    bad = _lib().divider().model_dump(mode="json")
    bad["elements"][1]["value"] = -5.0
    with pytest.raises(Exception):
        hub.submit(ADAPTER, params={"problem": bad})
    bad = _lib().divider().model_dump(mode="json")
    bad["probes"] = ["v(nowhere)"]
    with pytest.raises(Exception):
        hub.submit(ADAPTER, params={"problem": bad})
    assert hub.graph.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


# ------------------------------------------------------------------------------------------------ fake-engine arithmetic
def test_op_scalars(hub, fake_engine):
    p = _lib().divider()
    calls = fake_engine(lambda q: _result(q, "op", {"v(mid)": [0.75], "i(vin)": [-2.5e-4]},
                                          {"v(mid)": "V", "i(vin)": "A"}))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    assert len(calls) == 1 and calls[0].problem_id() == p.problem_id()
    assert _names(r) == ["circuit.i.vin", "circuit.n_points", "circuit.v.mid"]
    v, i, n = r.q("circuit.v.mid"), r.q("circuit.i.vin"), r.q("circuit.n_points")
    assert v.unit == "V" and i.unit == "A"
    for q in (v, i, n):
        assert q.axes == [] and isinstance(q.value, (int, float)), (q.name, q.value)
    assert v.value == 0.75 and i.value == -2.5e-4                           # SPICE sign kept
    assert n.value == 1
    from simws.units import Q_
    assert Q_(1.0, n.unit).dimensionality == Q_(1.0, "dimensionless").dimensionality
    _no_uncertainty(r)
    assert "budget:none" in r.flags
    assert not any(f.startswith("extrapolated:") for f in r.flags), r.flags
    job = hub.store.job(r.id)
    assert job.key.adapter == ADAPTER and job.key.engine == "ngspice"


def test_dc_axis_and_values(hub, fake_engine):
    p = _lib().diode_iv()
    V = np.linspace(0.30, 0.65, 8)
    I = -1e-14 * np.expm1(V / 0.0258)
    fake_engine(lambda q: _result(q, "dc", {"i(vin)": I}, {"i(vin)": "A"}, axis=V, axis_name="vin", axis_unit="V"))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    assert _names(r) == ["circuit.i.vin", "circuit.n_points"]
    q = r.q("circuit.i.vin")
    name, unit, vals = _axis(q)
    assert (name, unit) == ("vin", "V") and np.allclose(vals, V, rtol=1e-14)
    assert q.unit == "A" and np.allclose(q.array(), I, rtol=1e-14, atol=0)
    assert r.q("circuit.n_points").value == 8
    _no_uncertainty(r)
    assert "budget:none" in r.flags


def test_dc_source_name_lower_case_and_isource_axis(hub, fake_engine):
    d = {"elements": [{"name": "Ibias", "kind": "isource", "nodes": ["0", "A"], "value": 0.0},
                      {"name": "R1", "kind": "resistor", "nodes": ["A", "0"], "value": 1e3},
                      {"name": "Vsense", "kind": "vsource", "nodes": ["A", "B"], "value": 0.0},
                      {"name": "R2", "kind": "resistor", "nodes": ["B", "0"], "value": 1e3}],
         "analysis": {"kind": "dc", "source": "Ibias", "start": 0.0, "stop": 1e-3, "step": 2.5e-4},
         "probes": ["v(A)", "v(A,B)", "i(Vsense)"]}
    p = _cp(d)
    x = np.linspace(0.0, 1e-3, 5)
    fake_engine(lambda q: _result(q, "dc", {"v(A)": 500 * x, "v(A,B)": 0 * x, "i(Vsense)": x / 2},
                                  {"v(A)": "V", "v(A,B)": "V", "i(Vsense)": "A"}, axis=x, axis_name="Ibias",
                                  axis_unit="A"))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    assert _names(r) == ["circuit.i.vsense", "circuit.n_points", "circuit.v.a", "circuit.v.a-b"]
    for name, unit in (("circuit.v.a", "V"), ("circuit.v.a-b", "V"), ("circuit.i.vsense", "A")):
        q = r.q(name)
        assert q.unit == unit
        an, au, av = _axis(q)
        assert (an, au) == ("ibias", "A") and np.allclose(av, x)
    assert np.allclose(r.q("circuit.v.a").array(), 500 * x)
    assert np.allclose(r.q("circuit.i.vsense").array(), x / 2)
    assert r.q("circuit.n_points").value == 5


def test_ac_four_quantities_per_probe(hub, fake_engine):
    p = _lib().rc_lowpass().with_changes({"probes": ["v(out)", "v(in,out)", "i(vin)"]})
    f = 10.0 ** (1 + np.arange(71) / 10)
    h = 1 / (1 + 2j * np.pi * f * 1e-6)
    hd = 1 - h
    cur = -(1 - h) / 1e3
    sig = {"v(out)": h, "v(in,out)": hd, "i(vin)": cur}
    fake_engine(lambda q: _result(q, "ac", {k: v.real for k, v in sig.items()},
                                  {"v(out)": "V", "v(in,out)": "V", "i(vin)": "A"}, axis=f, axis_name="f",
                                  axis_unit="Hz", imag={k: v.imag for k, v in sig.items()}))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    exp = ["circuit.n_points"]
    for base in ("circuit.v.out", "circuit.v.in-out", "circuit.i.vin"):
        exp += [base + s for s in (".re", ".im", ".mag", ".phase")]
    assert _names(r) == sorted(exp)
    for probe, base, unit in (("v(out)", "circuit.v.out", "V"), ("v(in,out)", "circuit.v.in-out", "V"),
                              ("i(vin)", "circuit.i.vin", "A")):
        z = sig[probe]
        for suf, val, u in ((".re", z.real, unit), (".im", z.imag, unit), (".mag", np.abs(z), unit),
                            (".phase", np.angle(z), "rad")):
            q = r.q(base + suf)
            assert q.unit == u, (q.name, q.unit)
            an, au, av = _axis(q)
            assert (an, au) == ("f", "Hz") and np.allclose(av, f, rtol=1e-14)
            assert np.allclose(q.array(), val, rtol=1e-12, atol=1e-300), q.name
    assert r.q("circuit.n_points").value == 71
    # the phase is numpy.angle: in (-pi, pi], negative for the low-pass output, near pi for the current
    assert np.all(r.q("circuit.v.out.phase").array() < 0)
    assert np.all(np.abs(r.q("circuit.i.vin.phase").array()) > math.pi / 2)
    _no_uncertainty(r)


def test_tran_axis(hub, fake_engine):
    p = _lib().rc_lowpass(analysis="tran")
    t = np.linspace(0.0, 1e-5, 1001)
    y = 1 - np.exp(-t / 1e-6)
    fake_engine(lambda q: _result(q, "tran", {"v(out)": y}, {"v(out)": "V"}, axis=t, axis_name="t", axis_unit="s"))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    assert _names(r) == ["circuit.n_points", "circuit.v.out"]
    an, au, av = _axis(r.q("circuit.v.out"))
    assert (an, au) == ("t", "s") and np.allclose(av, t)
    assert np.allclose(r.q("circuit.v.out").array(), y)
    assert r.q("circuit.n_points").value == 1001


def test_probe_names_lower_case(hub, fake_engine):
    p = _lib().divider().with_changes({"probes": ["v(MID)", "i(VIN)", "v(In,Mid)"]})
    fake_engine(lambda q: _result(q, "op", {"v(MID)": [0.75], "i(VIN)": [-2.5e-4], "v(In,Mid)": [0.25]},
                                  {"v(MID)": "V", "i(VIN)": "A", "v(In,Mid)": "V"}))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    assert _names(r) == ["circuit.i.vin", "circuit.n_points", "circuit.v.in-mid", "circuit.v.mid"]
    assert r.q("circuit.v.in-mid").value == 0.25


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_self_check_non_finite(hub, fake_engine, bad):
    p = _lib().diode_iv(IS_A=1e-14 * (2 if bad != bad else 3 if bad > 0 else 4))
    V = np.linspace(0.30, 0.65, 8)
    I = -1e-14 * np.expm1(V / 0.0258)
    I[3] = bad
    fake_engine(lambda q: _result(q, "dc", {"i(vin)": I}, {"i(vin)": "A"}, axis=V, axis_name="vin", axis_unit="V"))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "check_failed", (r.status, r.message)


def test_self_check_non_finite_imag(hub, fake_engine):
    p = _lib().rc_lowpass(points_per_decade=1, f_start_Hz=10.0, f_stop_Hz=1e3)
    f = np.array([10.0, 100.0, 1000.0])
    fake_engine(lambda q: _result(q, "ac", {"v(out)": [1.0, 0.9, 0.5]}, {"v(out)": "V"}, axis=f, axis_name="f",
                                  axis_unit="Hz", imag={"v(out)": [0.0, float("nan"), -0.5]}))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "check_failed", (r.status, r.message)


def test_self_check_non_finite_op(hub, fake_engine):
    p = _lib().divider(V=2.0)
    fake_engine(lambda q: _result(q, "op", {"v(mid)": [float("nan")], "i(vin)": [-5e-4]},
                                  {"v(mid)": "V", "i(vin)": "A"}))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "check_failed", (r.status, r.message)


@pytest.mark.parametrize("n", [7, 9])
def test_self_check_dc_point_count(hub, fake_engine, n):
    p = _lib().diode_iv(N=1.0 + n / 100)                               # 0.30 to 0.65 V in 0.05 V steps: 8 points
    V = np.linspace(0.30, 0.30 + 0.05 * (n - 1), n)
    I = -1e-14 * np.expm1(V / 0.0258)
    fake_engine(lambda q: _result(q, "dc", {"i(vin)": I}, {"i(vin)": "A"}, axis=V, axis_name="vin", axis_unit="V"))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "check_failed", (r.status, r.message)


def test_self_check_dc_point_count_exact(hub, fake_engine):
    """0 to 1 V in 0.25 V steps is 5 points: 5 passes."""
    d = {"elements": [{"name": "vin", "kind": "vsource", "nodes": ["a", "0"]},
                      {"name": "r1", "kind": "resistor", "nodes": ["a", "0"], "value": 1e3}],
         "analysis": {"kind": "dc", "source": "vin", "start": 0.0, "stop": 1.0, "step": 0.25}, "probes": ["i(vin)"]}
    p = _cp(d)
    x = np.linspace(0.0, 1.0, 5)
    fake_engine(lambda q: _result(q, "dc", {"i(vin)": -x / 1e3}, {"i(vin)": "A"}, axis=x, axis_name="vin",
                                  axis_unit="V"))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", (r.status, r.message)


def _lib_problem(tmp_path):
    return _cp({"elements": [{"name": "vin", "kind": "vsource", "nodes": ["in", "0"], "value": 1.0},
                             {"name": "x1", "kind": "subckt", "nodes": ["in", "0"], "model": "myres"}],
                "libraries": [{"path": str(tmp_path / "res.lib"), "section": "typ"}],
                "analysis": {"kind": "op"}, "probes": ["i(vin)"]})


def test_external_library_flag_passes_through(hub, fake_engine, tmp_path):
    from simws import engines
    p = _lib_problem(tmp_path)
    flag = f"circuit:external_library: {tmp_path / 'res.lib'} sha256 0123456789ab"
    fake_engine(lambda q: _result(q, "op", {"i(vin)": [-1e-3]}, {"i(vin)": "A"}, flags=[flag]))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    assert flag in r.flags
    s = engines.capability("ngspice").outside_tested_range(p)
    assert [x.split(":", 1)[0] for x in s] == ["libraries"], s
    for x in s:
        assert "extrapolated:" + x in r.flags, (x, r.flags)


def test_extrapolated_flags(hub, fake_engine):
    from simws import engines
    d = {"elements": [{"name": "vin", "kind": "vsource", "nodes": ["in", "0"], "value": 1.0},
                      {"name": "l1", "kind": "inductor", "nodes": ["in", "out"], "value": 1e-6},
                      {"name": "r1", "kind": "resistor", "nodes": ["out", "0"], "value": 50.0}],
         "analysis": {"kind": "op"}, "probes": ["v(out)"], "temperature_C": 85.0}
    p = _cp(d)
    s = engines.capability("ngspice").outside_tested_range(p)
    assert sorted(x.split(":", 1)[0] for x in s) == ["element_kinds", "temperature_C"], s
    fake_engine(lambda q: _result(q, "op", {"v(out)": [1.0]}, {"v(out)": "V"}))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    for x in s:
        assert "extrapolated:" + x in r.flags, (x, r.flags)


def test_engine_flags_pass_through(hub, fake_engine):
    p = _lib().divider(V=3.0)
    fake_engine(lambda q: _result(q, "op", {"v(mid)": [2.25], "i(vin)": [-7.5e-4]}, {"v(mid)": "V", "i(vin)": "A"},
                                  flags=["circuit:test flag"]))
    r = hub.submit(ADAPTER, params=_params(p))
    assert "circuit:test flag" in r.flags


def _big_problem(n):
    els = [{"name": "vin", "kind": "vsource", "nodes": ["n0", "0"], "value": 1.0}]
    els += [{"name": f"r{i}", "kind": "resistor", "nodes": [f"n{i - 1}", f"n{i}"], "value": 1.0}
            for i in range(1, n - 1)]
    els.append({"name": "rend", "kind": "resistor", "nodes": [f"n{n - 2}", "0"], "value": 1.0})
    return _cp({"elements": els, "analysis": {"kind": "op"}, "probes": ["v(n1)"]})


def test_unsupported_is_invalid_input(hub, fake_engine):
    calls = fake_engine(lambda q: pytest.fail("the engine must not run an unsupported problem"))
    r = hub.submit(ADAPTER, params=_params(_big_problem(10001)))
    assert r.status == "invalid_input", (r.status, r.message)
    assert "unsupported by ngspice: " in r.message and "max_elements:" in r.message
    assert r.quantities == [] and calls == []


def test_engine_missing_is_engine_error(hub, monkeypatch, tmp_path):
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv("SIMWS_SPICE_PREFIX", str(tmp_path / "empty"))
    r = hub.submit(ADAPTER, params=_params(_lib().divider()))
    assert r.status == "engine_error", (r.status, r.message)
    assert "not installed" in r.message.lower(), r.message


def test_openvaf_missing_is_engine_error(hub, monkeypatch, tmp_path):
    b = tmp_path / "pre" / "bin"
    b.mkdir(parents=True)
    (b / "ngspice").write_text("#!/bin/sh\nexit 0\n")
    (b / "ngspice").chmod(0o755)
    monkeypatch.setenv("SIMWS_SPICE_PREFIX", str(tmp_path / "pre"))
    monkeypatch.setenv("SIMWS_CACHE", str(tmp_path / "cache"))
    r = hub.submit(ADAPTER, params=_params(_lib().diode_iv(model="va")))
    assert r.status == "engine_error", (r.status, r.message)
    assert "not installed" in r.message.lower(), r.message


def test_trust_unverified(hub, fake_engine):
    from simws.hub.trust import assess_stored
    p = _lib().divider(R2_ohm=1e3)
    fake_engine(lambda q: _result(q, "op", {"v(mid)": [0.5], "i(vin)": [-5e-4]}, {"v(mid)": "V", "i(vin)": "A"}))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok"
    assert assess_stored(hub, r.id).level == "unverified"


# ------------------------------------------------------------------------------------------------ study (A7.3 rule)
def test_study_submits_to_circuit_adapter(hub, fake_engine):
    from simws.study import Study

    def make(q):
        r1, r2 = q.elements[1].value, q.elements[2].value
        return _result(q, "op", {"v(mid)": [r2 / (r1 + r2)], "i(vin)": [-1.0 / (r1 + r2)]},
                       {"v(mid)": "V", "i(vin)": "A"})
    fake_engine(make)
    res = Study(hub, _lib().divider()).sweep({"elements.1.value": [1e3, 3e3]})
    assert [pt["changes"] for pt in res.points] == [{"elements.1.value": 1e3}, {"elements.1.value": 3e3}]
    for pt in res.points:
        assert pt["engine"] == "ngspice" and pt["status"] == "ok", pt
        assert hub.store.job(pt["result_id"]).key.adapter == ADAPTER
        assert pt["trust"] == "unverified"


def test_study_refuses_device_engine_on_circuit_problem(hub):
    from simws.study import Study
    res = Study(hub, _lib().divider(), engines=["devsim"]).sweep({})
    (pt,) = res.points
    assert pt["status"] == "refused" and pt["result_id"] is None
    assert len(pt["reasons"]) == 1 and pt["reasons"][0].startswith("problem kind:")


# ------------------------------------------------------------------------------------------------ A9.6 chain
def _chain():
    from simws.chain import diode
    return diode


def _synthetic(Js, N, T, V, noise=None):
    J = Js * np.exp(Q_E * V / (N * K_B * T))
    if noise is not None:
        J = J * (1 + noise)
    return J


@pytest.mark.parametrize("Js,N,T", [(2.5e-7, 1.0, 300.0), (1e-4, 1.37, 350.0), (3.3e-12, 1.02, 250.0),
                                    (1e-9, 2.0, 300.15)])
def test_fit_shockley_recovers_parameters(Js, N, T):
    V = np.array([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    J = _synthetic(Js, N, T, V)
    fit = _chain().fit_shockley(V, J, T)
    for k in ("Js_A_m2", "N", "T_K", "max_rel_residual"):
        assert k in fit, fit
    assert fit["Js_A_m2"] == pytest.approx(Js, rel=1e-8)
    assert fit["N"] == pytest.approx(N, rel=1e-10)
    assert fit["T_K"] == T
    assert 0.0 <= fit["max_rel_residual"] <= 1e-8
    fl = _chain().fit_shockley(list(V), list(J), T)                      # lists work too
    assert fl["N"] == pytest.approx(N, rel=1e-10)


def test_fit_shockley_least_squares_and_residual():
    """With perturbed data: the least-squares line in (V, ln J) over all points (computed here with numpy), and the
    residual max |J_fit / J - 1|."""
    rng = np.random.default_rng(5)
    V = np.linspace(0.2, 0.7, 11)
    T = 300.0
    J = _synthetic(4e-8, 1.15, T, V, noise=rng.normal(0.0, 0.03, V.size))
    slope, icpt = np.polyfit(V, np.log(J), 1)
    N = Q_E / (K_B * T * slope)
    Js = math.exp(icpt)
    Jfit = Js * np.exp(Q_E * V / (N * K_B * T))
    res = float(np.max(np.abs(Jfit / J - 1)))
    fit = _chain().fit_shockley(V, J, T)
    assert fit["N"] == pytest.approx(N, rel=1e-9)
    assert fit["Js_A_m2"] == pytest.approx(Js, rel=1e-8)
    assert fit["max_rel_residual"] == pytest.approx(res, rel=1e-6)
    assert res > 0.01                                                    # the data are not ideal


def test_fit_shockley_uses_codata_2018():
    """N is q / (k T slope) with CODATA 2018 constants: an older k (ngspice's 1.38064852e-23) would move N by 3.5e-7
    relative, so N must match the CODATA value to 1e-10."""
    V = np.array([0.3, 0.4, 0.5])
    T = 300.0
    J = 1e-6 * np.exp(V / 0.025)                                          # slope 40 per volt
    fit = _chain().fit_shockley(V, J, T)
    assert fit["N"] == pytest.approx(Q_E / (K_B * T * 40.0), rel=1e-10)


def test_circuit_from_fit():
    lib = _lib()
    fit = {"Js_A_m2": 3.7e-7, "N": 1.08, "T_K": 300.0, "max_rel_residual": 0.01}
    p = _chain().circuit_from_fit(fit, area_m2=2e-12, start=0.30, stop=0.60, step=0.05)
    ref = lib.diode_iv(IS_A=3.7e-7 * 2e-12, N=1.08, start=0.30, stop=0.60, step=0.05, model="builtin",
                       temperature_C=300.0 - 273.15, tnom_C=300.0 - 273.15)
    _same_diode_problem(p, ref)
    d = _chain().circuit_from_fit(fit, start=0.30, stop=0.60, step=0.05)   # default area 1e-12 m^2
    _same_diode_problem(d, lib.diode_iv(IS_A=3.7e-7 * 1e-12, N=1.08, start=0.30, stop=0.60, step=0.05,
                                        model="builtin", temperature_C=26.85, tnom_C=26.85))
    q = _chain().circuit_from_fit(dict(fit, T_K=77.0), start=0.5, stop=1.0, step=0.1)
    assert q.temperature_C == pytest.approx(77.0 - 273.15, rel=1e-12)
    assert q.tnom_C == pytest.approx(77.0 - 273.15, rel=1e-12)
    # added by the lead (A9.11): the link sets gmin_S = 0 unless told otherwise
    assert d.numerics.gmin_S == 0.0
    assert _chain().circuit_from_fit(fit, gmin_S=1e-15).numerics.gmin_S == 1e-15


def _same_diode_problem(p, ref):
    from simws.spec import CircuitProblem
    assert isinstance(p, CircuitProblem)
    assert [(e.name, e.kind, e.nodes, e.model) for e in p.elements] == \
        [(e.name, e.kind, e.nodes, e.model) for e in ref.elements]
    assert [(m.name, m.kind) for m in p.models] == [(m.name, m.kind) for m in ref.models]
    assert sorted(p.models[0].params) == sorted(ref.models[0].params)
    for k, v in ref.models[0].params.items():
        assert p.models[0].params[k] == pytest.approx(v, rel=1e-12), k
    a, b = p.analysis, ref.analysis
    assert a.kind == "dc" and a.source == b.source
    assert (a.start, a.stop, a.step) == pytest.approx((b.start, b.stop, b.step), rel=1e-12)
    assert p.probes == ref.probes
    assert p.temperature_C == pytest.approx(ref.temperature_C, rel=1e-12)
    assert p.tnom_C == pytest.approx(ref.tnom_C, rel=1e-12)
    # amended by the lead (A9.11): circuit_from_fit sets gmin_S itself (default 0); the other numerics are the library's
    assert p.numerics.model_dump(exclude={"gmin_S"}) == ref.numerics.model_dump(exclude={"gmin_S"})
    assert p.libraries == ref.libraries


# ------------------------------------------------------------------------------------------------ acceptance R30-R37
@pytest.fixture(scope="module")
def rhub(tmp_path_factory):
    from simws.hub.jobs import Hub
    return Hub(tmp_path_factory.mktemp("circuit_acceptance") / "store", mode="inprocess")


def _ok(hub, name, p):
    r = hub.submit(name, params=_params(p))
    assert r.status == "ok", (r.status, r.message)
    return r


def _report(label, worst, gate, unit=""):
    ok = worst <= gate
    print(f"\n[{label}] worst difference {worst:.3e}{unit} against gate {gate:.0e}{unit}: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_R30_divider(rhub):
    _need_engine()
    r = _ok(rhub, ADAPTER, _lib().divider())
    dv = abs(r.q("circuit.v.mid").value - 0.75)
    di = abs(r.q("circuit.i.vin").value - (-2.5e-4))
    a = _report("R30 v(mid)", dv, 1e-9, " V")
    b = _report("R30 i(vin)", di, 1e-12, " A")
    assert a and b, (dv, di)


def test_R31_rc_ac(rhub):
    _need_engine()
    r = _ok(rhub, ADAPTER, _lib().rc_lowpass())
    R, C = 1e3, 1e-9
    mag, ph = r.q("circuit.v.out.mag"), r.q("circuit.v.out.phase")
    an, au, f = _axis(mag)
    assert (an, au) == ("f", "Hz")
    k = np.arange(f.size)
    nominal = 10.0 ** (1 + k / 10)
    print(f"\n[R31] {f.size} points from {f[0]:.6g} to {f[-1]:.6g} Hz (10^(1 + k/10), k = 0 .. 70: 71 points)")
    df = float(np.max(np.abs(f / nominal - 1)))
    x = 2 * np.pi * f * R * C
    dm = float(np.max(np.abs(mag.array() / (1 / np.sqrt(1 + x ** 2)) - 1)))
    dp = float(np.max(np.abs(ph.array() - (-np.arctan(x)))))
    a = _report("R31 frequency (relative)", df, 1e-9)
    b = _report("R31 |v(out)| (relative)", dm, 1e-6)
    c = _report("R31 phase", dp, 1e-6, " rad")
    assert f.size == 71 and nominal[-1] == pytest.approx(1e8)
    assert a and b and c


def test_R32_rc_tran(rhub):
    _need_engine()
    r = _ok(rhub, ADAPTER, _lib().rc_lowpass(analysis="tran"))
    q = r.q("circuit.v.out")
    an, au, t = _axis(q)
    assert (an, au) == ("t", "s")
    d = float(np.max(np.abs(q.array() - (1 - np.exp(-t / (1e3 * 1e-9))))))
    print(f"\n[R32] {t.size} output times from {t[0]:.3g} to {t[-1]:.3g} s")
    assert _report("R32 |v(out) - (1 - exp(-t/RC))|", d, 1e-3, " V"), d


def _diode_gate(r, IS, N, T, label, area=None):
    q = r.q("circuit.i.vin")
    an, au, V = _axis(q)
    assert (an, au) == ("vin", "V")
    I = -q.array()
    ref = IS * (np.exp(Q_E * V / (N * K_B * T)) - 1)
    rel = np.abs(I / ref - 1)
    i = int(np.argmax(rel))
    print(f"\n[{label}] {V.size} points {V[0]:.3f} .. {V[-1]:.3f} V; worst at {V[i]:.3f} V: -i(vin) = {I[i]:.6e} A, "
          f"closed form {ref[i]:.6e} A")
    return _report(label + " (relative)", float(rel.max()), 1e-4), V, I


def test_R33_builtin_diode(rhub):
    _need_engine()
    r = _ok(rhub, ADAPTER, _lib().diode_iv())
    ok, V, _ = _diode_gate(r, 1e-14, 1.0, 300.15, "R33 built-in diode")
    assert V.size == 8
    assert ok


def test_R34_va_diode(rhub):
    _need_openvaf()
    r = _ok(rhub, ADAPTER, _lib().diode_iv(model="va"))
    ok, V, _ = _diode_gate(r, 1e-14, 1.0, 300.15, "R34 Verilog-A diode (OpenVAF)")
    assert V.size == 8
    assert ok


# Marked by the lead after the checker's report (contract A9.11): R35 as written FAILED (1.84e-3 against 1e-4, at
# 0.30 V) because ngspice puts gmin (1e-15 S, the description's default) across the junction, and with IS = 2.6e-18 A
# the gmin current is part of the low-bias current. The gate is unchanged and the case is run as it was specified
# (gmin_S = 1e-15); strict, so a pass is noticed. The post hoc R35b below runs the link with gmin_S = 0.
@pytest.mark.xfail(strict=True, reason="recorded FAIL of R35 (contract A9.11): gmin current across the junction")
def test_R35_device_to_circuit(rhub):
    _need_engine()
    pytest.importorskip("devsim")
    from simws.spec.library import pn_diode_1d
    Vd = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    dev = pn_diode_1d(V=Vd)
    rd = _ok(rhub, "device.devsim", dev)
    qa = rd.q("dc.I.anode")
    assert qa.unit == "A/m**2"
    Vdev = np.asarray(qa.axes[0].values, float)
    J = qa.array()
    T = float(dev.models.temperature_K)
    fit = _chain().fit_shockley(Vdev, J, T)
    area = 1e-12
    # as specified in A9.8 (amended by the lead, A9.11: circuit_from_fit's default is now gmin_S = 0, so the
    # specified configuration is given explicitly)
    circ = _chain().circuit_from_fit(fit, start=0.30, stop=0.60, step=0.05, gmin_S=1e-15)
    rc = _ok(rhub, ADAPTER, circ)
    ok, V, I = _diode_gate(rc, fit["Js_A_m2"] * area, fit["N"], fit["T_K"], "R35 circuit vs fitted closed form")
    # reported, not gated: how well an ideal Shockley diode describes the DEVSIM device
    j_dev = np.array([J[int(np.argmin(np.abs(Vdev - v)))] for v in V])
    mf = float(np.max(np.abs((I / area) / j_dev - 1)))
    print(f"[R35] reported: N = {fit['N']:.6f}, Js = {fit['Js_A_m2']:.6e} A/m^2, T = {fit['T_K']} K, fit residual "
          f"max|J_fit/J - 1| = {fit['max_rel_residual']:.4e}; model-form difference max|(I_circuit/area)/J_DEVSIM - 1| "
          f"= {mf:.4e} (over {V.size} points)")
    print(f"[R35] DEVSIM J = {np.array2string(J, precision=6)} A/m^2 at V = {Vdev.tolist()}")
    assert V.size == 7
    assert ok


IHP_PDK = "~/sim_workspace/ihp_sg13g2_pdk"
IHP_LIB = IHP_PDK + "/ihp-sg13g2/libs.tech/ngspice/models/cornerHBT.lib"


def r36_problem():
    """The legacy Gummel set-up of benchmarks/ihp_npn13g2_baseline (A9.8 R36), with the legacy numerics."""
    return _cp({"title": "IHP SG13G2 npn13G2 Gummel (hbt_typ, 27 C)",
                "libraries": [{"path": IHP_LIB, "section": "hbt_typ"}],
                "elements": [{"name": "vc", "kind": "vsource", "nodes": ["c", "0"], "value": 0.0},
                             {"name": "vb", "kind": "vsource", "nodes": ["b", "0"], "value": 0.0},
                             {"name": "ve", "kind": "vsource", "nodes": ["e", "0"], "value": -0.6},
                             {"name": "x1", "kind": "subckt", "nodes": ["c", "b", "e", "0"], "model": "npn13G2",
                              "params": {"Nx": 1}}],
                "analysis": {"kind": "dc", "source": "ve", "start": -0.35, "stop": -1.02, "step": -0.005},
                "probes": ["i(vc)", "i(vb)"], "temperature_C": 27.0, "tnom_C": 27.0,
                "numerics": {"reltol": 1e-5, "abstol_A": 1e-16, "gmin_S": 1e-15}})


def test_R36_foundry_model(rhub):
    if not os.path.isdir(os.path.expanduser(IHP_PDK)):
        pytest.skip(f"IHP SG13G2 PDK not at {IHP_PDK}")
    _need_engine()
    r = _ok(rhub, ADAPTER, r36_problem())
    assert any(f.startswith("circuit:external_library: ") for f in r.flags), r.flags
    assert any(f.startswith("extrapolated:libraries:") for f in r.flags), r.flags
    qc, qb = r.q("circuit.i.vc"), r.q("circuit.i.vb")
    an, au, ve = _axis(qc)
    assert (an, au) == ("ve", "V")
    vbe = -ve
    ic, ib = -qc.array(), -qb.array()
    beta = ic / np.maximum(ib, 1e-30)
    mask = (ic > 1e-6) & (ic < 5e-3)
    peak = float(np.max(beta[mask]))
    i1 = int(np.argmin(np.abs(ic - 1e-6)))
    rb = abs(peak / 800.17 - 1)
    dv = abs(vbe[i1] - 0.665)
    print(f"\n[R36] beta peak {peak:.3f} (legacy 800.17), V_BE at I_C nearest 1 uA {vbe[i1]:.4f} V (legacy 0.665)")
    a = _report("R36 beta peak (relative)", rb, 1e-2)
    b = _report("R36 V_BE at 1 uA", dv, 0.0051, " V")
    assert a and b


def test_R37_cache_and_repeatability(tmp_path, monkeypatch):
    _need_openvaf()
    monkeypatch.setenv("SIMWS_CACHE", str(tmp_path / "cache"))
    p = _lib().diode_iv(model="va")
    r1 = _ng().run(p, tmp_path / "run1")
    r2 = _ng().run(p, tmp_path / "run2")
    diff = max(abs(a - b) for k in r1.signals for a, b in zip(r1.signals[k], r2.signals[k]))
    print(f"\n[R37] first run osdi_cached = {r1.diagnostics.get('osdi_cached')}, second run osdi_cached = "
          f"{r2.diagnostics.get('osdi_cached')}; largest difference between the two runs' signals {diff:.3e} A "
          f"(gate: identical)")
    assert r1.signals == r2.signals
    assert r2.diagnostics.get("osdi_cached") is True


# ------------------------------------------------------------------------------------------------ contract A9.11
def test_R35b_post_hoc_link_without_gmin(rhub):
    """POST HOC (lead, A9.11, after R35 failed): the same link with circuit_from_fit's new default gmin_S = 0.
    Same gate as R35 (1e-4 relative against the fitted Shockley diode). The checker's diagnostic run gave 7.5e-6."""
    _need_engine()
    pytest.importorskip("devsim")
    from simws.spec.library import pn_diode_1d
    dev = pn_diode_1d(V=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    rd = _ok(rhub, "device.devsim", dev)
    qa = rd.q("dc.I.anode")
    fit = _chain().fit_shockley(np.asarray(qa.axes[0].values, float), qa.array(), float(dev.models.temperature_K))
    circ = _chain().circuit_from_fit(fit, start=0.30, stop=0.60, step=0.05)
    assert circ.numerics.gmin_S == 0.0
    rc = _ok(rhub, ADAPTER, circ)
    ok, V, I = _diode_gate(rc, fit["Js_A_m2"] * 1e-12, fit["N"], fit["T_K"], "R35b (post hoc, gmin 0)")
    assert ok
