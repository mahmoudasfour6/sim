"""Contract amendment A18 (docs/ARCHITECTURE_V1.md): mixed mode, the strike with the circuit node solved together
with the device.

Written by the checker from the contract text (A18 items 0 to 7 and the lead's clarification 7a; A17 for the strike
path and its reference device A17.6; A10 for the bisection and its fields) and the public interfaces only:
StrikeTransient / StrikeLoad / Problem, chain.see (reference_strike_diode, strike_problem, with_load,
critical_charge_mixed, critical_charge_tcad, critical_charge, Q_E), the engine registry, the Hub and
trust.assess_stored.

Acceptance items R110 to R116 of A18.5 use the A18.5 tolerances unchanged. R115 is reported (printed, no gate).
Every DEVSIM run goes through one module-scoped hub and a cache, so a run is made once per module.

Readings of the contract used here (none changes a tolerance):
- R110: "I the reported tr.I.top" in A/m^2, times the area A (1 um^2 = 1e-12 m^2) gives amperes; v_k is tr.V.top at
  the time point t_k of the same axis; the bound is 1e-3 x max over all k of |I_k A|.
- R111: "the peak current" is the largest |tr.I.top|; the A17 run is reference_strike_diode(pairs_cm2=1e9), top at
  +1 V fixed.
- R113: the dip is supply_V - tr.V_min.top; each of the two variants within 1 % of the default's dip.
- R114: "no deposit to speak of (pairs 1e6)" is checked with the loads of R110 (10 kOhm) and R112 (100 kOhm), both
  with 1 fF; I_0 is the first value of tr.I.top (t = 0).
- R116: "the load contact also in bias" and "a load on an unknown contact" are invalid descriptions (pydantic
  ValidationError); the unknown contact's message starts "analysis:".
- A18.4: "the A10 fields" are qcrit_C, q_lo_C, q_hi_C, runs, result_ids, node, threshold_V, polarity; "v_min at
  q_crit" is looked for as any returned field whose name contains "v_min".
- The bisection rule is checked with a small fake hub (A10.3: "fake hub where useful") that answers tr.V_min from
  the trial's pairs_cm2; the device physics is checked with real DEVSIM runs.
"""
import math
import types

import numpy as np
import pydantic
import pytest

Q_E = 1.602176634e-19
FLAG_FIXED_PREFIX = "strike:fixed_bias"
FLAG_MIXED = "strike:mixed_mode: the device and its load (supply, R, C) are solved together in one Newton system"
FLAG_1D = "strike:1d_column: the pairs are spread over the device area (no track radius, no 2D funnelling)"
DEVSIM = "device.devsim"
AREA = 1e-12                  # m^2, 1 um^2 (A18.5)
VS = 1.0                      # V, supply (A18.5)
REF_ID_A17 = "ff0714fb4325f7f403ef4969c97f233b"
A10_FIELDS = ("qcrit_C", "q_lo_C", "q_hi_C", "runs", "result_ids", "node", "threshold_V", "polarity")


def _S():
    from simws.chain import see
    return see


def _ref(**kw):
    return _S().reference_strike_diode(**kw)


def _mixed(R, C=1e-15, **kw):
    """The reference device of A17.6 with "top" on a load (supply 1 V, area 1 um^2), fixed biases empty (A18.5)."""
    return _S().with_load(_ref(**kw), "top", VS, R, C, AREA)


def _load(**kw):
    from simws.spec.schema import StrikeLoad
    base = dict(contact="top", supply_V=1.0, R_ohm=1e4, C_F=1e-15, area_um2=1.0)
    base.update(kw)
    return StrikeLoad(**base)


def _strike(**kw):
    from simws.spec.schema import StrikeTransient
    base = dict(pairs_cm2=1e11, depth_um=(0.25, 0.6), t_peak_s=20e-12, sigma_s=5e-12, t_stop_s=1e-6)
    base.update(kw)
    return StrikeTransient(**base)


# ------------------------------------------------------------------------------------------------ shared runs
@pytest.fixture(scope="module")
def shub(tmp_path_factory):
    from simws.hub.jobs import Hub
    return Hub(tmp_path_factory.mktemp("mixed_store"), mode="inprocess")


class _Runs:
    """name -> (record, envelope), one DEVSIM run each, made on first use."""
    SPECS = {
        "r110": lambda: _mixed(1e4, pairs_cm2=1e11),
        "r111_mixed": lambda: _mixed(1e4, pairs_cm2=1e9),
        "r111_fixed": lambda: _ref(pairs_cm2=1e9),
        "r112": lambda: _mixed(1e5, pairs_cm2=1e11),
        "r113": lambda: _mixed(1e5, pairs_cm2=5e11),
        "r113_time": lambda: _mixed(1e5, pairs_cm2=5e11, steps_per_sigma=20, points_per_decade=40),
        "r113_mesh": lambda: _mixed(1e5, pairs_cm2=5e11, mesh_scale=0.5),
        "r114_10k": lambda: _mixed(1e4, pairs_cm2=1e6),
        "r114_100k": lambda: _mixed(1e5, pairs_cm2=1e6),
        "fixed_1e11": lambda: _ref(),
    }

    def __init__(self, hub):
        self.hub, self.done = hub, {}

    def problem(self, name):
        return self.SPECS[name]()

    def rec(self, name):
        self.get(name, require_ok=False)
        return self.done[name][0]

    def get(self, name, require_ok=True):
        if name not in self.done:
            rec = self.hub.submit(DEVSIM, params={"problem": self.problem(name)})
            env = self.hub.result(rec.id) if rec.status == "ok" else None
            self.done[name] = (rec, env)
        rec, env = self.done[name]
        if require_ok:
            assert rec.status == "ok", (name, rec.status, rec.message)
        return env


@pytest.fixture(scope="module")
def runs(shub):
    return _Runs(shub)


def _arr(env, name, unit=None):
    q = env.q(name)
    return np.asarray(q.to(unit) if unit else q.array(), float)


def _time(env, name="tr.I.top"):
    return np.asarray(env.q(name).axes[0].values, float)


def _I(env, contact="top"):
    return _arr(env, f"tr.I.{contact}", "A/m**2")


def _V(env, contact="top"):
    return _arr(env, f"tr.V.{contact}", "V")


def _Vmin(env, contact="top"):
    return float(np.asarray(env.q(f"tr.V_min.{contact}").to("V")))


def _Q(env, contact="top"):
    return float(np.asarray(env.q(f"tr.Q.{contact}").to("C/m**2")))


def _Qgen(env):
    return float(np.asarray(env.q("tr.Q_generated").to("C/m**2")))


def _peak(env, contact="top"):
    return float(np.max(np.abs(_I(env, contact))))


def _rel(a, b):
    return abs(a / b - 1.0)


# ================================================================================================ A18.1 description
def test_a18_1_load_fields_and_default_step():
    ld = _load()
    assert ld.contact == "top" and ld.supply_V == 1.0 and ld.R_ohm == 1e4 and ld.C_F == 1e-15
    assert ld.area_um2 == 1.0
    assert ld.supply_step_V == 0.1


def test_a18_1_C_zero_allowed():
    assert _load(C_F=0.0).C_F == 0.0


@pytest.mark.parametrize("kw", [{"R_ohm": 0.0}, {"R_ohm": -1.0}, {"C_F": -1e-15}, {"area_um2": 0.0},
                                {"area_um2": -1.0}])
def test_a18_1_load_limits(kw):
    with pytest.raises(pydantic.ValidationError):
        _load(**kw)


@pytest.mark.parametrize("missing", ["contact", "supply_V", "R_ohm", "area_um2"])
def test_a18_1_load_required_fields(missing):
    from simws.spec.schema import StrikeLoad
    d = dict(contact="top", supply_V=1.0, R_ohm=1e4, C_F=1e-15, area_um2=1.0)
    d.pop(missing)
    with pytest.raises(pydantic.ValidationError):
        StrikeLoad(**d)


def test_a18_1_load_unknown_field_forbidden():
    from simws.spec.schema import StrikeLoad
    with pytest.raises(pydantic.ValidationError):
        StrikeLoad(contact="top", supply_V=1.0, R_ohm=1e4, C_F=0.0, area_um2=1.0, L_H=1e-9)


def test_a18_1_load_contact_not_in_bias():
    from simws.spec.schema import Bias
    with pytest.raises(pydantic.ValidationError):
        _strike(bias=[Bias(contact="top", value_V=1.0)], load=_load(contact="top"))
    # another contact in bias keeps its fixed bias
    an = _strike(bias=[Bias(contact="back", value_V=0.0)], load=_load(contact="top"))
    assert an.load.contact == "top" and [b.contact for b in an.bias] == ["back"]


def test_a18_1_load_left_out_when_absent():
    an = _strike()
    assert an.load is None
    for mode in ("python", "json"):
        assert "load" not in an.model_dump(mode=mode)
    p = _ref()
    assert "load" not in p.model_dump(mode="json")["analysis"]
    assert p.problem_id() == REF_ID_A17


def test_a18_1_load_serialised_when_present():
    p = _mixed(1e4)
    d = p.model_dump(mode="json")["analysis"]
    assert d["load"]["contact"] == "top" and d["load"]["R_ohm"] == 1e4
    assert p.problem_id() != REF_ID_A17
    # round trip keeps the id
    from simws.spec.schema import Problem
    assert Problem.model_validate(p.model_dump(mode="json")).problem_id() == p.problem_id()


def test_r116_reference_strike_diode_keeps_a17_id():
    assert _ref().problem_id() == REF_ID_A17


# ================================================================================================ A18.4 with_load
def test_a18_4_with_load_moves_contact_out_of_bias_and_area():
    from simws.spec.schema import Bias
    S = _S()
    base = _ref()
    d = base.model_dump(mode="json")
    d["analysis"]["bias"] = [{"contact": "top", "value_V": 1.0}, {"contact": "back", "value_V": 0.0}]
    two = type(base).model_validate(d)
    p = S.with_load(two, "top", 1.2, 2.5e4, 3e-15, 3.5e-12)
    an = p.analysis
    assert [b.contact for b in an.bias] == ["back"] and an.bias[0].value_V == 0.0
    ld = an.load
    assert ld.contact == "top" and ld.supply_V == 1.2 and ld.R_ohm == 2.5e4 and ld.C_F == 3e-15
    assert ld.area_um2 == pytest.approx(3.5, rel=1e-12)                  # m^2 -> um^2
    # everything else unchanged
    a0, a1 = two.model_dump(mode="json"), p.model_dump(mode="json")
    for k in ("device", "models", "numerics"):
        assert a0[k] == a1[k]
    for k, v in a0["analysis"].items():
        if k not in ("bias", "load"):
            assert a1["analysis"][k] == v, k
    assert isinstance(an.bias[0], Bias)


def test_a18_5_reference_mixed_fixed_biases_empty():
    p = _mixed(1e4)
    assert p.analysis.bias == []
    assert p.analysis.load.contact == "top" and p.analysis.load.supply_V == VS
    assert p.analysis.load.area_um2 == pytest.approx(1.0, rel=1e-12)


def test_a18_4_with_load_needs_strike():
    from simws.spec.library import pn_diode_1d
    with pytest.raises(ValueError):
        _S().with_load(pn_diode_1d(), "anode", 1.0, 1e4, 1e-15, 1e-12)


# ================================================================================================ A18.3 result
def test_a18_3_node_voltage_output(runs):
    env = runs.get("r110")
    q = env.q("tr.V.top")
    assert q.axes[0].name == "time" and q.axes[0].unit == "s"
    assert str(q.unit) == "V"
    t = _time(env, "tr.V.top")
    assert t[0] == 0.0
    assert np.array_equal(t, _time(env, "tr.I.top"))                   # V read after every step
    ref_t = np.concatenate(([0.0], runs.problem("r110").analysis.time_points_s()))
    assert t.size == ref_t.size and np.allclose(t, ref_t, rtol=1e-12, atol=0)
    v = _V(env)
    assert v.size == t.size
    assert str(env.q("tr.V_min.top").unit) == "V"
    assert _Vmin(env) == pytest.approx(float(v.min()), rel=0, abs=1e-15)
    assert _Vmin(env) < VS                                             # the strike pulls the node down
    print(f"\nA18.3: {t.size} points, v(0) {v[0]:.9f} V, V_min {_Vmin(env):.6f} V")


def test_a18_3_flags(runs):
    mixed = runs.get("r110")
    fixed = runs.get("fixed_1e11")
    assert FLAG_MIXED in mixed.flags, mixed.flags
    assert not any(f.startswith(FLAG_FIXED_PREFIX) for f in mixed.flags), mixed.flags
    assert FLAG_1D in mixed.flags and FLAG_1D in fixed.flags
    assert any(f.startswith(FLAG_FIXED_PREFIX) for f in fixed.flags), fixed.flags
    assert not any(f.startswith("strike:mixed_mode") for f in fixed.flags), fixed.flags


def test_a18_3_no_node_voltage_without_load(runs):
    env = runs.get("fixed_1e11")
    for name in ("tr.V.top", "tr.V_min.top", "tr.V.back", "tr.V_min.back"):
        with pytest.raises(KeyError):
            env.q(name)


def test_a18_3_strike_outputs_kept(runs):
    env = runs.get("r110")
    for name in ("tr.I.top", "tr.I.back", "tr.Q.top", "tr.Q.back", "tr.Q_generated"):
        env.q(name)
    assert "generated_pairs_quadrature_cm2" in env.diagnostics


# ================================================================================================ A18.5 acceptance
def test_r110_kirchhoff_at_node(runs):
    env = runs.get("r110")
    R, C = 1e4, 1e-15
    t, v, i = _time(env), _V(env), _I(env) * AREA
    res = (VS - v[1:]) / R - C * (v[1:] - v[:-1]) / (t[1:] - t[:-1]) - i[1:]
    lim = 1e-3 * float(np.max(np.abs(i)))
    worst = float(np.max(np.abs(res)))
    k = int(np.argmax(np.abs(res))) + 1
    print(f"\nR110: worst |residual| {worst:.4e} A at k={k} (t={t[k]:.4e} s) = {worst / (lim / 1e-3):.3e} of peak "
          f"|I A| {lim / 1e-3:.4e} A (limit 1e-3)")
    assert worst <= lim


def test_r111_fixed_bias_limit(runs):
    m, f = runs.get("r111_mixed"), runs.get("r111_fixed")
    dq = _rel(_Q(m), _Q(f))
    dp = _rel(_peak(m), _peak(f))
    move = (VS - _Vmin(m)) / VS
    print(f"\nR111: node moves by {move:.4%} of the supply; Q_top mixed {_Q(m):.6e} fixed {_Q(f):.6e} C/m^2 "
          f"(diff {dq:.3e}, limit 1e-2); peak mixed {_peak(m):.6e} fixed {_peak(f):.6e} A/m^2 "
          f"(diff {dp:.3e}, limit 2e-2)")
    assert dq <= 0.01


# amended by the lead (A18.8): the peak gate of R111 is a recorded FAIL. With the node free, part of the fast
# current is supplied by the device's own junction capacitance and never passes the external circuit; a stiff node
# (10 Ohm, or 100 fF) brings the mixed-mode peak to the fixed-bias one (7.6e-5, 1.0e-3). Gate kept, strict xfail.
@pytest.mark.xfail(strict=True, reason="A18.8: R111 peak 5.5 % below fixed bias (limit 2 %): the junction "
                                       "capacitance supplies part of the fast current when the node is free")
def test_r111_fixed_bias_limit_peak(runs):
    m, f = runs.get("r111_mixed"), runs.get("r111_fixed")
    assert _rel(_peak(m), _peak(f)) <= 0.02


def test_r112_collection(runs):
    env = runs.get("r112")
    frac = _Q(env) / _Qgen(env)
    print(f"\nR112: Q_top / Q_generated {frac:.6f} (limit 1 +- 0.01); V_min {_Vmin(env):.6f} V")
    assert abs(frac - 1.0) <= 0.01


def test_r113_convergence(runs):
    d0 = VS - _Vmin(runs.get("r113"))
    dt = VS - _Vmin(runs.get("r113_time"))
    dm = VS - _Vmin(runs.get("r113_mesh"))
    print(f"\nR113: dip default {d0:.6f} V, time {dt:.6f} V (diff {_rel(dt, d0):.3e}), mesh {dm:.6f} V "
          f"(diff {_rel(dm, d0):.3e}); limit 1e-2")
    assert _rel(dm, d0) <= 0.01


# amended by the lead (A18.8): the time-step half of R113 is a recorded FAIL: the default grid (10 steps per sigma,
# 20 points per decade) leaves the dip 1.3 % below the refined grid's (about 2 % below 40 / 80); the tail's
# points per decade drive it. Gate kept, strict xfail.
@pytest.mark.xfail(strict=True, reason="A18.8: R113 dip with the refined time grid differs by 1.27 % (limit 1 %)")
def test_r113_convergence_time(runs):
    d0 = VS - _Vmin(runs.get("r113"))
    dt = VS - _Vmin(runs.get("r113_time"))
    assert _rel(dt, d0) <= 0.01


@pytest.mark.parametrize("name,R", [("r114_10k", 1e4), ("r114_100k", 1e5)])
def test_r114_dc_point(runs, name, R):
    env = runs.get(name)
    t, v, i = _time(env), _V(env), _I(env)
    assert t[0] == 0.0
    expect = VS - i[0] * AREA * R
    err = abs(v[0] - expect)
    print(f"\nR114 ({R:g} Ohm): v(0) {v[0]:.12f} V, V_s - I_0 A R {expect:.12f} V, |diff| {err:.3e} V (limit 1e-6); "
          f"I_0 A {i[0] * AREA:.4e} A")
    assert err <= 1e-6


# ================================================================================================ R116 refusals
def test_r116_load_contact_in_bias_invalid():
    from simws.spec.schema import Problem
    d = _ref().model_dump(mode="json")              # top at +1 V in bias
    d["analysis"]["load"] = {"contact": "top", "supply_V": 1.0, "R_ohm": 1e4, "C_F": 1e-15, "area_um2": 1.0}
    with pytest.raises(pydantic.ValidationError):
        Problem.model_validate(d)


def test_r116_load_unknown_contact_invalid():
    from simws.spec.schema import Problem
    d = _mixed(1e4).model_dump(mode="json")
    d["analysis"]["load"]["contact"] = "gate"
    with pytest.raises(pydantic.ValidationError) as ei:
        Problem.model_validate(d)
    assert "analysis:" in str(ei.value), str(ei.value)


def _mixed_2d():
    from simws.spec.library import npn_hbt_2d
    from simws.spec.schema import Problem
    d = npn_hbt_2d().model_dump(mode="json")
    a = _strike(depth_um=(0.1, 0.2)).model_dump(mode="json")
    a["bias"] = []
    a["load"] = {"contact": "collector", "supply_V": 1.0, "R_ohm": 1e4, "C_F": 1e-15, "area_um2": 1.0}
    d["analysis"] = a
    return Problem.model_validate(d)


def test_r116_2d_device_refused(shub):
    from simws import engines
    p = _mixed_2d()
    reasons = engines.unsupported("devsim", p)
    assert any(r.startswith("strike:") for r in reasons), reasons
    r = shub.submit(DEVSIM, params={"problem": p})
    assert r.status == "invalid_input" and "strike:" in r.message, (r.status, r.message)


@pytest.mark.parametrize("adapter,engine", [("device.charon", "charon"), ("device.devsim_eb", "devsim_eb")])
def test_r116_other_engines_refuse(shub, adapter, engine):
    from simws import engines
    p = _mixed(1e4)
    assert engines.unsupported("devsim", p) == []
    reasons = engines.unsupported(engine, p)
    assert any(r.startswith("analyses:") for r in reasons), (engine, reasons)
    r = shub.submit(adapter, params={"problem": p})
    assert r.status == "invalid_input" and "analyses:" in r.message, (r.status, r.message)


# ================================================================================================ A18.6 trust
def test_a18_6_trust_unverified(shub, runs):
    from simws.hub.trust import assess_stored
    rec = runs.rec("r110")
    assert rec.status == "ok"
    t = assess_stored(shub, rec.id)
    assert t.level == "unverified", t
    print(f"\nA18.6 trust: {t.level}, reasons {t.reasons}")


# ================================================================================================ A18.4 critical_charge_mixed
class _FakeHub:
    """Answers device.devsim jobs with tr.V_min.<load contact> = vmin(q), q the trial's generated charge (C)."""

    def __init__(self, vmin, area_m2):
        self.vmin, self.area, self.jobs, self.out = vmin, area_m2, [], {}

    def submit(self, name, params=None, **kw):
        p = params["problem"]
        rid = f"r{len(self.jobs)}"
        q = p.analysis.pairs_cm2 * Q_E * self.area * 1e4
        self.jobs.append((name, p, q))
        self.out[rid] = (p.analysis.load.contact, self.vmin(q))
        return types.SimpleNamespace(status="ok", id=rid, message="")

    def result(self, rid):
        contact, v = self.out[rid]

        def q(name, member=None):
            if name != f"tr.V_min.{contact}":
                raise KeyError(name)
            return types.SimpleNamespace(name=name, value=v, unit="V", array=lambda: np.asarray(v))
        return types.SimpleNamespace(q=q, flags=[FLAG_MIXED, FLAG_1D], diagnostics={})


def _vkeys(res):
    return [k for k in res if "v_min" in k.lower()]


def test_a18_4_critical_charge_mixed_bisection_fake():
    S = _S()
    qc_true = 1.0e-15
    hub = _FakeHub(lambda q: 1.0 - 0.5 * q / qc_true, AREA)            # v_min = 0.5 at q = 1 fC
    p = _mixed(1e5)
    res = S.critical_charge_mixed(hub, p, 0.5, AREA, 0.1e-15, 10e-15)
    for k in A10_FIELDS:
        assert k in res, (k, sorted(res))
    assert "tau_rise_s" not in res and "tau_fall_s" not in res
    assert res["node"] == "top" and res["threshold_V"] == 0.5 and res["polarity"] == "discharge"
    # first q_low, then q_high, then bisection midpoints; every trial a device.devsim run with the load
    qs = [q for _, _, q in hub.jobs]
    assert all(n == DEVSIM for n, _, _ in hub.jobs)
    assert all(pp.analysis.load is not None and pp.analysis.load.contact == "top" for _, pp, _ in hub.jobs)
    assert qs[0] == pytest.approx(0.1e-15, rel=1e-9) and qs[1] == pytest.approx(10e-15, rel=1e-9)
    lo, hi = 0.1e-15, 10e-15
    for q in qs[2:]:
        assert q == pytest.approx(0.5 * (lo + hi), rel=1e-9)
        if 1.0 - 0.5 * q / qc_true <= 0.5:
            hi = q
        else:
            lo = q
    assert res["q_lo_C"] == pytest.approx(lo, rel=1e-9) and res["q_hi_C"] == pytest.approx(hi, rel=1e-9)
    assert (res["q_hi_C"] - res["q_lo_C"]) / res["q_hi_C"] <= 1e-3
    # stops as soon as the rule is met (the previous bracket was wider than rel_tol)
    assert res["qcrit_C"] == pytest.approx(0.5 * (res["q_lo_C"] + res["q_hi_C"]), rel=1e-12)
    assert res["q_lo_C"] < qc_true <= res["q_hi_C"]
    assert abs(res["qcrit_C"] / qc_true - 1) <= 1e-3
    assert res["runs"] == len(qs) == len(res["result_ids"])
    dev_keys = [k for k in res if "device" in k and "id" in k]
    assert dev_keys, f"no device result ids in {sorted(res)}"
    assert list(res[dev_keys[0]]) == [f"r{i}" for i in range(len(qs))]
    assert _vkeys(res), f"no v_min field in {sorted(res)}"
    print(f"\ncritical_charge_mixed (fake hub): qcrit {res['qcrit_C']:.6e} C in {res['runs']} runs, fields "
          f"{sorted(res)}")


def test_a18_4_critical_charge_mixed_at_threshold_is_upset():
    """'upset when tr.V_min of the load contact is at or below the threshold': a v_min exactly at the threshold."""
    S = _S()
    qstep = 2.0e-15
    hub = _FakeHub(lambda q: 0.5 if q >= qstep * (1 - 1e-12) else 0.9, AREA)
    res = S.critical_charge_mixed(hub, _mixed(1e5), 0.5, AREA, 0.1e-15, 10e-15, rel_tol=1e-2)
    assert res["q_lo_C"] < qstep <= res["q_hi_C"] * (1 + 1e-9)
    assert (res["q_hi_C"] - res["q_lo_C"]) / res["q_hi_C"] <= 1e-2


def test_a18_4_critical_charge_mixed_bracket_errors():
    S = _S()
    with pytest.raises(ValueError):            # q_low already upsets
        S.critical_charge_mixed(_FakeHub(lambda q: 0.0, AREA), _mixed(1e5), 0.5, AREA, 0.1e-15, 10e-15)
    with pytest.raises(ValueError):            # q_high does not upset
        S.critical_charge_mixed(_FakeHub(lambda q: 0.9, AREA), _mixed(1e5), 0.5, AREA, 0.1e-15, 10e-15)
    with pytest.raises(ValueError):            # no load
        S.critical_charge_mixed(_FakeHub(lambda q: 0.9, AREA), _ref(), 0.5, AREA, 0.1e-15, 10e-15)


def test_a18_4_critical_charge_mixed_failed_job_raises():
    S = _S()

    class Bad(_FakeHub):
        def submit(self, name, params=None, **kw):
            return types.SimpleNamespace(status="failed", id="x", message="boom")
    with pytest.raises(RuntimeError):
        S.critical_charge_mixed(Bad(lambda q: 0.9, AREA), _mixed(1e5), 0.5, AREA, 0.1e-15, 10e-15)


def test_a18_4_critical_charge_mixed_real(shub):
    """A real bisection (coarse rel_tol to keep it short): the stored trials are mixed-mode device results whose
    generated charge is the trial charge, and q_lo / q_hi bracket the threshold."""
    S = _S()
    p = _mixed(1e5)
    res = S.critical_charge_mixed(shub, p, 0.5, AREA, 0.05e-15, 20e-15, rel_tol=0.05)
    for k in A10_FIELDS:
        assert k in res, (k, sorted(res))
    assert (res["q_hi_C"] - res["q_lo_C"]) / res["q_hi_C"] <= 0.05
    envs = {rid: shub.result(rid) for rid in res["result_ids"]}
    gen = {rid: _Qgen(e) * AREA for rid, e in envs.items()}
    for e in envs.values():
        assert FLAG_MIXED in e.flags
    lo_env = [e for rid, e in envs.items() if abs(gen[rid] / res["q_lo_C"] - 1) < 1e-6]
    hi_env = [e for rid, e in envs.items() if abs(gen[rid] / res["q_hi_C"] - 1) < 1e-6]
    assert lo_env and hi_env
    assert _Vmin(lo_env[0]) > 0.5 >= _Vmin(hi_env[0])
    vk = _vkeys(res)
    assert vk, f"no v_min field in {sorted(res)}"
    print(f"\ncritical_charge_mixed real (rel_tol 0.05): qcrit {res['qcrit_C']:.4e} C, runs {res['runs']}, "
          f"V_min at q_lo {_Vmin(lo_env[0]):.4f} V, at q_hi {_Vmin(hi_env[0]):.4f} V; "
          + ", ".join(f"{k} {res[k]}" for k in vk))


# ================================================================================================ R115 (reported, slow)
def _rc_node(C, R):
    """A10.5's circuit: vsource "vdd" 1.0 V on node "vdd", resistor "r1" (vdd-n) R, capacitor "c1" (n-0) C,
    transient step 0.1 ps, stop 300 ps, max step 0.1 ps; probe v(n)."""
    from simws.spec.circuit import CircuitProblem, Element, Transient
    return CircuitProblem(
        elements=[Element(name="vdd", kind="vsource", nodes=["vdd", "0"], value=1.0),
                  Element(name="r1", kind="resistor", nodes=["vdd", "n"], value=float(R)),
                  Element(name="c1", kind="capacitor", nodes=["n", "0"], value=float(C))],
        analysis=Transient(step_s=0.1e-12, stop_s=300e-12, max_step_s=0.1e-12),
        probes=["v(n)"])


@pytest.mark.slow
def test_r115_reported(shub):
    from simws import engines
    S = _S()
    lo, hi = 0.05e-15, 100e-15
    out = []

    def attempt(label, fn):
        try:
            return fn()
        except ValueError as ex:
            out.append(f"    {label}: no upset up to {hi:.3e} C ({ex})")
            return None

    for R in (1e4, 1e5):
        out.append(f"  node {R / 1e3:g} kOhm / 1 fF, threshold 0.5 V, 1 um^2 on 'top':")
        res = attempt("mixed", lambda: S.critical_charge_mixed(shub, _mixed(R), 0.5, AREA, lo, hi))
        if res:
            r = shub.submit(DEVSIM, params={"problem": S.strike_problem(_mixed(R), res["qcrit_C"], AREA)})
            vq = _Vmin(shub.result(r.id)) if r.status == "ok" else float("nan")
            out.append(f"    mixed: qcrit {res['qcrit_C']:.5e} C ({res['runs']} runs); v_min at q_crit {vq:.5f} V; "
                       + ", ".join(f"{k} {res[k]:.5f}" for k in _vkeys(res)))
        if engines.module("ngspice").available():
            circ = _rc_node(1e-15, R)
            rt = attempt("A17 fixed-bias TCAD (rerun)",
                         lambda: S.critical_charge_tcad(shub, circ, "n", 0.5, _ref(), "top", AREA, lo, hi,
                                                        rerun=True))
            if rt:
                out.append(f"    A17 fixed-bias TCAD (rerun): qcrit {rt['qcrit_C']:.5e} C ({rt['runs']} runs)")
            rm = attempt("Messenger (1 ps, 10 ps)",
                         lambda: S.critical_charge(shub, circ, "n", 0.5, 1e-12, 10e-12, q_low_C=lo, q_high_C=hi))
            if rm:
                out.append(f"    Messenger (1 ps, 10 ps): qcrit {rm['qcrit_C']:.5e} C ({rm['runs']} runs)")
        else:
            out.append("    ngspice not available: A17 TCAD and Messenger not run")
    print("\nR115 (reported, no gate):\n" + "\n".join(out))
