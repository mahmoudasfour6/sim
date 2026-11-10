"""simws.adapters.device (contract section 5): device.devsim and device.charon through the hub."""
import numpy as np
import pytest

from simws.spec.library import npn_hbt_2d, pn_diode_1d

FLOOR_1D = 1e-4          # A/m^2 (contract section 5)
FLOOR_2D = 1e-8          # A/m


def _params(p):
    return {"problem": p.model_dump(mode="json")}


# ------------------------------------------------------------------------------------------------ manifests / resolve
@pytest.mark.parametrize("name,engine", [("device.devsim", "devsim"), ("device.charon", "charon")])
def test_manifest(name, engine):
    from simws.hub import adapter as A
    m = A.manifest(name)
    assert m.name == name and m.env == "tcad" and m.engine == engine
    assert m.version == "1.0" and m.engine_version == "env-locked"
    # amended by the lead (contract A1.8): one declared output, the Kirchhoff residual
    assert m.inputs == ["problem"] and [o.name for o in m.outputs] == ["dc.kirchhoff_residual"]
    assert m.native_conventions == {"terminal_current": "into_device+"}
    assert name in A.names()


def test_resolve_fills_defaults_and_accepts_problem_objects():
    from simws.hub import adapter as A
    from simws.spec.schema import Problem
    ad = A.get("device.devsim")
    p = pn_diode_1d(V=[0.3])
    full = p.model_dump(mode="json")
    assert ad.resolve({"problem": p})["problem"] == full
    assert ad.resolve({"problem": full})["problem"] == full
    sparse = {k: v for k, v in full.items() if k not in ("numerics", "spec_version")}
    out = ad.resolve({"problem": sparse})["problem"]
    assert out == Problem.model_validate(sparse).model_dump(mode="json")
    assert "numerics" in out and "spec_version" in out
    with pytest.raises(Exception):          # the exception type is not fixed by the contract
        ad.resolve({"problem": {**full, "colour": "red"}})


def test_same_problem_spelled_differently_is_one_job(hub):
    p = pn_diode_1d(V=[0.3])
    import copy
    full = p.model_dump(mode="json")
    sparse = copy.deepcopy(full)                       # omit fields that are at their defaults
    del sparse["spec_version"]
    del sparse["models"]["velocity_saturation"]
    del sparse["models"]["extended_precision"]
    a = hub.make_job("device.devsim", params={"problem": full})
    b = hub.make_job("device.devsim", params={"problem": sparse})
    c = hub.make_job("device.devsim", params={"problem": p})
    assert a.id == b.id == c.id
    d = hub.make_job("device.devsim", params=_params(pn_diode_1d(V=[0.4])))
    assert d.id != a.id


# ------------------------------------------------------------------------------------------------ device.devsim
def test_devsim_acceptance(hub):
    """Contract section 5 acceptance: ok, dc.I.anode / dc.I.cathode in A/m^2 summing to ~0, second submit cached.
    Values: the legacy diode at the same biases x 1e4 (A/cm^2 -> A/m^2), within R1's 1e-6 relative."""
    pytest.importorskip("devsim")
    from simws.validation.engine_pairs import _devsim_diode
    V = [0.3, 0.5]
    p = pn_diode_1d(V=V)
    r = hub.submit("device.devsim", params=_params(p))
    assert r.status == "ok", r.message
    names = sorted(q.name for q in r.quantities)
    assert names == ["dc.I.anode", "dc.I.cathode", "dc.kirchhoff_residual"]      # A1.8
    assert 0 <= r.q("dc.kirchhoff_residual").value <= 1e-4
    a, c = r.q("dc.I.anode"), r.q("dc.I.cathode")
    for q in (a, c):
        assert q.unit == "A/m**2"
        assert q.conventions == {"terminal_current": "into_device+"}
        assert len(q.axes) == 1
        ax = q.axes[0]
        assert ax.name == "V_anode" and ax.unit == "V" and ax.values == V
        assert len(q.value) == len(V)
    ia, ic = a.array(), c.array()
    for k in range(len(V)):
        assert abs(ia[k] + ic[k]) <= 1e-4 * max(abs(ia[k]), abs(ic[k])) + FLOOR_1D
    J_legacy, _ = _devsim_diode(V)
    assert np.abs(ia / (np.asarray(J_legacy) * 1e4) - 1).max() <= 1e-6
    # diagnostics
    d = r.diagnostics
    assert d["problem_id"] == p.problem_id()
    assert d["engine"] == "devsim" and d["device"] == "pn_diode_1d" and d["temperature_K"] == 300.0
    # cache
    r2 = hub.submit("device.devsim", params=_params(p))
    assert r2.id == r.id and r2.diagnostics.get("cache") == "hit"
    job = hub.store.job(r.id)
    assert job.key.adapter == "device.devsim" and job.key.engine == "devsim" and "problem" in job.key.params


def test_devsim_equilibrium_scalars(hub):
    pytest.importorskip("devsim")
    r = hub.submit("device.devsim", params=_params(pn_diode_1d(nodes=201)))
    assert r.status == "ok", r.message
    for name in ("dc.I.anode", "dc.I.cathode"):
        q = r.q(name)
        assert q.axes == [] and q.unit == "A/m**2"
        assert isinstance(q.value, (int, float)), q.value
        assert abs(q.value) < FLOOR_1D


def test_devsim_unsupported_is_invalid_input(hub):
    p = npn_hbt_2d(VBE=[0.6]).with_changes({"device.contacts.2.side": "x_max"})
    r = hub.submit("device.devsim", params=_params(p))
    assert r.status == "invalid_input"
    assert "unsupported by devsim" in r.message
    assert "contacts: 2D DEVSIM contacts must be on y_min or y_max ('collector' is on x_max)" in r.message
    assert r.quantities == []


def test_devsim_outside_tested_range_flagged(hub):
    """240 K is outside DEVSIM's tested [250, 400] K: the run goes ahead and is flagged, not refused."""
    pytest.importorskip("devsim")
    from simws import engines
    p = pn_diode_1d(nodes=201, T=240.0, V=[0.3, 0.4])
    expected = ["extrapolated:" + s for s in engines.capability("devsim").outside_tested_range(p)]
    assert expected
    r = hub.submit("device.devsim", params=_params(p))
    assert r.status == "ok", r.message
    for f in expected:
        assert f in r.flags, (f, r.flags)
    assert r.diagnostics["temperature_K"] == 240.0


def test_devsim_inside_range_not_flagged(hub):
    pytest.importorskip("devsim")
    r = hub.submit("device.devsim", params=_params(pn_diode_1d(nodes=201, V=[0.3])))
    assert r.status == "ok"
    assert not any(f.startswith("extrapolated:") for f in r.flags), r.flags


@pytest.mark.slow
def test_devsim_2d_units(hub):
    pytest.importorskip("devsim")
    r = hub.submit("device.devsim", params=_params(npn_hbt_2d()))
    assert r.status == "ok", r.message
    cur = [q for q in r.quantities if q.name.startswith("dc.I.")]
    assert sorted(q.name for q in cur) == ["dc.I.base", "dc.I.collector", "dc.I.emitter"]
    assert sorted(q.name for q in r.quantities if q not in cur) == ["dc.kirchhoff_residual"]      # contract A1.8
    for q in cur:
        assert q.unit == "A/m" and q.axes == [] and abs(q.value) < FLOOR_2D


# ------------------------------------------------------------------------------------------------ device.charon
def test_charon_not_installed(hub, charon_problem):
    from simws.config import path
    if path("toolchain.charon_exe").exists():
        pytest.skip("Charon is installed on this machine")
    r = hub.submit("device.charon", params=_params(charon_problem()))
    assert r.status != "ok"
    assert r.status == "engine_error"
    assert "not installed" in r.message.lower(), r.message
    assert hub.graph.result_for_job(r.job_id) is None            # failures are never served from cache


def test_charon_unsupported_is_invalid_input(hub):
    r = hub.submit("device.charon", params=_params(pn_diode_1d(V=[0.1])))
    assert r.status == "invalid_input"
    assert "unsupported by charon" in r.message
    assert "mesh: no Charon mesh for this geometry" in r.message


def test_invalid_problem_refused_at_submit(hub):
    bad = pn_diode_1d().model_dump(mode="json")
    bad["models"]["temperature_K"] = -5.0
    # resolve() runs in make_job, outside the adapter's failure mapping, so the error propagates from submit; the
    # contract does not fix its type
    with pytest.raises(Exception):
        hub.submit("device.devsim", params={"problem": bad})
    assert hub.graph.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


# ------------------------------------------------------------------------------------------------ A2.4
def _fake_run(engine, flags):
    """An engine run() that returns a fixed EngineResult carrying `flags`: the cathode current is minus the anode
    current (true by construction, as Charon's collect does, A1.6)."""
    def run(problem, workdir=None):
        from simws.engines.base import EngineResult
        from simws.spec.schema import Problem
        p = problem if isinstance(problem, Problem) else Problem.model_validate(problem)
        V = list(p.analysis.values_V)
        ia = [1.0 + k for k in range(len(V))]                   # A/cm^2, test values only
        return EngineResult(engine=engine, problem_id=p.problem_id(), analysis="dc_sweep", sweep_contact="anode",
                            sweep_V=V, currents={"anode": ia, "cathode": [-x for x in ia]},
                            current_unit="A/cm**2", flags=list(flags))
    return run


@pytest.mark.parametrize("engine", ["devsim", "charon"])
@pytest.mark.parametrize("flags,expect_flag", [
    (["cathode_current_from_kirchhoff", "current_sign_from_bias"], True),
    (["current_sign_from_bias"], False),
])
def test_a2_4_kirchhoff_residual_not_independent(hub, monkeypatch, charon_problem, engine, flags, expect_flag):
    """A2.4: an engine flag ending in _current_from_kirchhoff marks dc.kirchhoff_residual "not_independent"; its
    value stays finite (0 for these currents). The engine module's run() is replaced, so no engine runs."""
    import math
    from simws import engines
    monkeypatch.setattr(engines.module(engine), "run", _fake_run(engine, flags))
    p = pn_diode_1d(V=[0.3, 0.5], nodes=201) if engine == "devsim" else charon_problem(V=(0.1, 0.2))
    r = hub.submit(f"device.{engine}", params=_params(p))
    assert r.status == "ok", r.message
    for f in flags:
        assert f in r.flags, (f, r.flags)
    q = r.q("dc.kirchhoff_residual")
    assert isinstance(q.value, (int, float)) and math.isfinite(q.value) and q.value == pytest.approx(0.0, abs=1e-12)
    assert ("not_independent" in q.flags) is expect_flag, q.flags


def test_a2_4_devsim_residual_has_no_flag(hub):
    pytest.importorskip("devsim")
    r = hub.submit("device.devsim", params=_params(pn_diode_1d(V=[0.3, 0.5], nodes=201)))
    assert r.status == "ok", r.message
    assert not any(f.endswith("_current_from_kirchhoff") for f in r.flags), r.flags
    assert "not_independent" not in r.q("dc.kirchhoff_residual").flags
