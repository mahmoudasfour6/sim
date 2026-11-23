"""Contract amendment A17 (docs/ARCHITECTURE_V1.md): a TCAD strike current as the circuit pulse.

Written by the checker from the contract text (A17 items 0 to 9, with A6.6, A10 and section 6.1 for conventions) and
the public interfaces only: StrikeTransient (time_points_s, g, window), Problem, chain.see (reference_strike_diode,
strike_problem, tcad_pulse, critical_charge_tcad, critical_charge, inject, Q_E, LINEAR_FLAG), the engine registry and
planner, the Hub and trust.assess_stored. Item 9's corrections are tested as written: tr.Q.<contact> is the backward
sum of (I_k - I_0)(t_k - t_{k-1}) and tcad_pulse is the staircase I_k held over (t_{k-1}, t_k].

The acceptance items R100 to R107 of A17.6 use the A17.6 tolerances unchanged. R103's high-injection numbers and
R106 are reported (printed), not gated. Every DEVSIM strike run is shared through one module-scoped hub and cache.

Readings of the contract used here (none changes a tolerance):
- "peak |I_top|" is the largest |tr.I.top| over the time points; "collected fraction" is tr.Q.top / tr.Q_generated.
- Item 1's tail "geometric at points_per_decade to t_stop_s" cannot land on t_stop exactly unless the number of
  decades times points_per_decade is an integer; the test requires a geometric tail that ends at t_stop with a
  number of steps within 1 of decades x points_per_decade, and exactly points_per_decade per decade when that
  product is an integer.
- Item 1's "10 points geometric from 1e-3 (t_peak - 4 sigma)" is empty when t_peak = 4 sigma (the reference
  device): the first point is then t = 0 (item 4), added by the engine; time_points_s() starts after it.
- "The A10 fields" of critical_charge_tcad are those of A10.1 that apply to any pulse (qcrit_C, q_lo_C, q_hi_C,
  runs, result_ids, node, threshold_V, polarity); tau_rise_s / tau_fall_s describe Messenger's pulse only.
"""
import math

import numpy as np
import pydantic
import pytest

Q_E = 1.602176634e-19
FLAG_FIXED = ("strike:fixed_bias: the contacts are held at their bias; a circuit node's voltage change does not act "
              "back on the collection (no mixed mode)")
FLAG_1D = "strike:1d_column: the pairs are spread over the device area (no track radius, no 2D funnelling)"
DEVSIM = "device.devsim"
AREA_1UM2 = 1e-12            # m^2 (R106)

# R107: problem ids of every zero-argument builder of simws/spec/library.py, computed by the checker at commit
# 748f5fc (a git worktree of that commit, PYTHONPATH pointing at it, f().problem_id()).
IDS_748F5FC = {
    "diode_iv": "c70674e64e582db8d71e6abb0ea81470",
    "divider": "650cd885614fdaa14c23b0b8e1129eab",
    "implant_si": "90285c5cb1b9a21088805bf9e60ba682",
    "n_bar_1d": "3595ce69f90e9cfaaaf89d025b745563",
    "npn_hbt_2d": "55cf78ed7e298d344d60b782acad29b8",
    "pn_diode_1d": "52b23f57732ec9aa197c56d61f3fb049",
    "rc_lowpass": "2e04b4f07e1b802a0786b19a1a763498",
    "see_si_slab": "a4e0942f20122f9bba3ca49b45871df9",
}


def _S():
    from simws.chain import see
    return see


def _ref(**kw):
    return _S().reference_strike_diode(**kw)


def _strike(**kw):
    from simws.spec.schema import StrikeTransient
    base = dict(pairs_cm2=1e11, depth_um=(0.25, 0.6), t_peak_s=20e-12, sigma_s=5e-12, t_stop_s=1e-6)
    base.update(kw)
    return StrikeTransient(**base)


# ------------------------------------------------------------------------------------------------ shared runs
@pytest.fixture(scope="module")
def shub(tmp_path_factory):
    from simws.hub.jobs import Hub
    return Hub(tmp_path_factory.mktemp("strike_store"), mode="inprocess")


class _Runs:
    """name -> (record, envelope) of one DEVSIM strike run each, computed on first use."""
    SPECS = {
        "base": {},
        "fine_time": {"steps_per_sigma": 20, "points_per_decade": 40},
        "fine_mesh": {"mesh_scale": 0.5},
        "p1e10": {"pairs_cm2": 1e10},
        "p1e12": {"pairs_cm2": 1e12},
        "p1e13": {"pairs_cm2": 1e13},
    }

    def __init__(self, hub):
        self.hub, self.done = hub, {}

    def problem(self, name):
        return _ref(**self.SPECS[name])

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


def _time(env, contact="top"):
    return np.asarray(env.q(f"tr.I.{contact}").axes[0].values, float)


def _peak(env, contact="top"):
    return float(np.max(np.abs(_arr(env, f"tr.I.{contact}", "A/m**2"))))


def _Q(env, contact="top"):
    return float(np.asarray(env.q(f"tr.Q.{contact}").to("C/m**2")))


def _Qgen(env):
    return float(np.asarray(env.q("tr.Q_generated").to("C/m**2")))


def _rel(a, b):
    return abs(a / b - 1.0)


# ================================================================================================ A17.6 reference device
def test_a17_6_reference_device_matches_contract():
    from simws.spec.schema import ModelSpec, StrikeTransient
    p = _ref()
    d = p.device
    assert d.dimension == 1
    assert len(d.regions) == 1 and d.regions[0].material == "silicon"
    assert tuple(d.regions[0].box.x) == (0.0, 5.0)                                  # 5 um
    assert d.extent().x[0] == 0.0 and d.extent().x[1] == 5.0
    donors = [t for t in d.doping if t.species == "donor"]
    acceptors = [t for t in d.doping if t.species == "acceptor"]
    assert len(donors) == 1 and len(acceptors) == 1 and not d.grading
    n, a = donors[0], acceptors[0]
    assert n.concentration_cm3 == 1e19 and n.box.x[0] in (None, 0.0) and n.box.x[1] == 0.2 and not n.include_upper
    assert a.concentration_cm3 == 1e16
    assert (a.box.x[0] is None or a.box.x[0] <= 0.0) and (a.box.x[1] is None or a.box.x[1] >= 5.0)  # p everywhere
    c = {k.name: k for k in d.contacts}
    assert set(c) == {"top", "back"}
    assert c["top"].side == "x_min" and c["back"].side == "x_max"
    assert c["top"].kind == "ohmic" and c["back"].kind == "ohmic"
    assert p.models == ModelSpec()
    m = p.models
    assert (m.mobility, m.srh, m.auger, m.band_gap_narrowing) == ("masetti", "scharfetter", True, "slotboom")
    an = p.analysis
    assert isinstance(an, StrikeTransient) and an.kind == "strike"
    assert [(b.contact, b.value_V) for b in an.bias] == [("top", 1.0)]               # top +1 V, back at 0 V
    assert tuple(an.depth_um) == (0.25, 0.6)
    assert (an.pairs_cm2, an.t_peak_s, an.sigma_s, an.t_stop_s) == (1e11, 20e-12, 5e-12, 1e-6)
    assert (an.steps_per_sigma, an.points_per_decade) == (10, 20)                    # the defaults
    lines = [tuple(x) for x in d.mesh.x_lines]
    assert d.mesh.scale == 1.0
    assert lines[0] == (0.0, 0.001) and (1.0, 0.001) in lines                        # 1 nm to 1 um
    assert all(h == 0.001 for x, h in lines if x <= 1.0)
    assert lines[-1] == (5.0, 0.02)                                                  # 20 nm after
    print(f"\nreference mesh x_lines: {lines}")


def test_a17_6_reference_variants_change_only_their_field():
    base = _ref().model_dump(mode="json")
    v = _ref(mesh_scale=0.5).model_dump(mode="json")
    assert v["device"]["mesh"]["scale"] == 0.5
    v["device"]["mesh"]["scale"] = 1.0
    assert v == base
    v = _ref(pairs_cm2=1e10).model_dump(mode="json")
    assert v["analysis"]["pairs_cm2"] == 1e10
    v["analysis"]["pairs_cm2"] = 1e11
    assert v == base
    v = _ref(steps_per_sigma=20, points_per_decade=40).model_dump(mode="json")
    assert (v["analysis"]["steps_per_sigma"], v["analysis"]["points_per_decade"]) == (20, 40)


# ================================================================================================ A17.1 description
def _expected_points(t_peak, sigma, t_stop, sps, ppd):
    a, b = t_peak - 4 * sigma, t_peak + 4 * sigma
    geo = [1e-3 * a * (1e3) ** (j / 9) for j in range(10)] if a > 0 else []
    n = int(round(8 * sps))                              # (b - a) / (sigma / sps)
    uni = [a + k * (sigma / sps) for k in range(1, n + 1)]
    return np.array(geo), np.array(uni), b


@pytest.mark.parametrize("t_peak,sigma,sps", [(20e-12, 5e-12, 10), (50e-12, 5e-12, 10), (20e-12, 5e-12, 20),
                                               (1e-9, 0.1e-9, 7)])
def test_a17_1_time_points_head_and_uniform(t_peak, sigma, sps):
    st = _strike(t_peak_s=t_peak, sigma_s=sigma, t_stop_s=1e-6, steps_per_sigma=sps)
    t = np.asarray(st.time_points_s(), float)
    geo, uni, b = _expected_points(t_peak, sigma, 1e-6, sps, 20)
    head = np.concatenate([geo, uni])
    assert np.all(np.diff(t) > 0)
    assert t[0] > 0
    np.testing.assert_allclose(t[:head.size], head, rtol=1e-12, atol=0)
    assert t[head.size - 1] == pytest.approx(b, rel=1e-14)
    # uniform part: steps of sigma / steps_per_sigma exactly
    a0 = t_peak - 4 * sigma
    uniform = t[geo.size:head.size]
    assert uniform.size == 8 * sps
    np.testing.assert_allclose(np.diff(np.concatenate([[a0], uniform])), sigma / sps, rtol=1e-9)
    if geo.size:
        assert t[geo.size - 1] == pytest.approx(a0, rel=1e-14) and t[0] == pytest.approx(1e-3 * a0, rel=1e-14)
        np.testing.assert_allclose(t[1:geo.size] / t[:geo.size - 1], 1e3 ** (1 / 9), rtol=1e-12)
    a, bb = st.window()
    assert (a, bb) == pytest.approx((t_peak - 4 * sigma, t_peak + 4 * sigma), rel=1e-15)


@pytest.mark.parametrize("t_peak,sigma,t_stop,ppd", [(20e-12, 5e-12, 1e-6, 20), (20e-12, 5e-12, 1e-6, 40),
                                                      (20e-12, 5e-12, 40e-12 * 1e4, 20),
                                                      (20e-12, 5e-12, 40e-12 * 1e3, 7),
                                                      (50e-12, 5e-12, 3e-7, 13)])
def test_a17_1_time_points_tail(t_peak, sigma, t_stop, ppd):
    st = _strike(t_peak_s=t_peak, sigma_s=sigma, t_stop_s=t_stop, points_per_decade=ppd)
    t = np.asarray(st.time_points_s(), float)
    geo, uni, b = _expected_points(t_peak, sigma, t_stop, 10, ppd)
    tail = t[geo.size + uni.size:]
    assert tail[-1] == pytest.approx(t_stop, rel=1e-12)
    r = np.diff(np.log10(np.concatenate([[b], tail])))
    np.testing.assert_allclose(r, r[0], rtol=1e-9)                    # geometric from b
    decades = math.log10(t_stop / b)
    m = tail.size
    assert abs(m - decades * ppd) < 1.0, (m, decades * ppd)
    if abs(decades * ppd - round(decades * ppd)) < 1e-9:              # integer: exactly ppd per decade
        assert m == round(decades * ppd)
        assert r[0] == pytest.approx(1.0 / ppd, rel=1e-9)
    print(f"\ntail: {m} steps for {decades:.4f} decades x {ppd} = {decades * ppd:.3f}")


def test_a17_1_time_points_reference_count():
    st = _ref().analysis
    t = st.time_points_s()
    # reference: t_peak = 4 sigma -> no geometric head; 80 uniform steps of 0.5 ps; tail to 1 us
    assert t[0] == pytest.approx(0.5e-12, rel=1e-12)
    assert t[79] == pytest.approx(40e-12, rel=1e-12)
    assert t[-1] == pytest.approx(1e-6, rel=1e-12)
    print(f"\nreference time points: {len(t)} (+ t = 0)")


def test_a17_1_g_truncated_unit_area():
    from math import erf, sqrt
    for t_peak, sigma in [(20e-12, 5e-12), (50e-12, 5e-12), (1e-9, 0.1e-9)]:
        st = _strike(t_peak_s=t_peak, sigma_s=sigma, t_stop_s=1e-6)
        a, b = st.window()
        x = np.linspace(a, b, 400001)
        g = np.asarray(st.g(x), float)
        area = float(np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(x)))
        assert area == pytest.approx(1.0, rel=1e-8)
        peak = 1.0 / (sigma * sqrt(2 * math.pi) * erf(4 / sqrt(2)))
        assert float(st.g(t_peak)) == pytest.approx(peak, rel=1e-12)
        mid = float(st.g(t_peak + 2 * sigma))
        assert mid == pytest.approx(peak * math.exp(-2.0), rel=1e-12)
        assert float(st.g(t_peak - 1.3 * sigma)) == pytest.approx(float(st.g(t_peak + 1.3 * sigma)), rel=1e-12)
        outside = np.asarray(st.g([a - 1e-3 * sigma, b + 1e-3 * sigma, b + sigma, 10 * b, 0.0 if a > 0 else -1.0]))
        assert np.all(outside == 0.0)


@pytest.mark.parametrize("kw", [
    {"t_peak_s": 19.9e-12},                                       # t_peak < 4 sigma
    {"t_peak_s": 10e-12},
    {"t_stop_s": 40e-12},                                         # t_stop == t_peak + 4 sigma
    {"t_stop_s": 30e-12},                                         # t_stop < t_peak + 4 sigma
    {"pairs_cm2": 0.0}, {"pairs_cm2": -1e10},
    {"sigma_s": 0.0}, {"t_peak_s": 0.0},
    {"depth_um": (0.6, 0.25)}, {"depth_um": (0.3, 0.3)},          # start < stop
])
def test_a17_1_invalid_descriptions(kw):
    with pytest.raises((pydantic.ValidationError, ValueError)):
        _strike(**kw)
    args = {k: v for k, v in kw.items()}
    with pytest.raises((pydantic.ValidationError, ValueError)):
        _ref(**args)


def test_a17_1_boundaries_valid():
    _strike(t_peak_s=20e-12, sigma_s=5e-12)                       # t_peak == 4 sigma is allowed
    _strike(t_stop_s=40.0001e-12)
    _ref(depth_um=(0.0, 5.0))                                     # the whole device is inside it


@pytest.mark.parametrize("seg", [(0.25, 5.5), (-0.1, 0.5), (5.0, 6.0), (6.0, 7.0)])
def test_r107_segment_outside_device_invalid(seg):
    with pytest.raises((pydantic.ValidationError, ValueError)) as ei:
        _ref(depth_um=seg)
    print(f"\n{seg}: {str(ei.value).splitlines()[-1][:160]}")


def test_r107_invalid_times_messages():
    with pytest.raises((pydantic.ValidationError, ValueError)):
        _ref(t_peak_s=19e-12)
    with pytest.raises((pydantic.ValidationError, ValueError)):
        _ref(t_stop_s=40e-12)


def test_a17_1_new_kind_in_union():
    from simws.spec.schema import Problem
    p = _ref()
    p2 = Problem.model_validate(p.model_dump(mode="json"))
    assert p2.analysis.kind == "strike" and p2.problem_id() == p.problem_id()


# ================================================================================================ A17.4 the result
def test_a17_4_result_quantities_and_flags(runs):
    env = runs.get("base")
    p = runs.problem("base")
    for c in ("top", "back"):
        q = env.q(f"tr.I.{c}")
        assert q.unit == "A/m**2"
        assert len(q.axes) == 1 and q.axes[0].name == "time" and q.axes[0].unit == "s"
        assert env.q(f"tr.Q.{c}").unit == "C/m**2"
    assert env.q("tr.Q_generated").unit == "C/m**2"
    t = _time(env)
    assert t[0] == 0.0                                                     # first point t = 0
    np.testing.assert_allclose(t[1:], p.analysis.time_points_s(), rtol=1e-12, atol=0)
    np.testing.assert_array_equal(_time(env, "back"), t)
    assert FLAG_FIXED in env.flags and FLAG_1D in env.flags
    i0 = abs(_arr(env, "tr.I.top", "A/m**2")[0])
    assert i0 <= 1e-6 * _peak(env), "I(0) is the leakage at the bias, before any generation"
    print(f"\nI_top(0) = {i0:.3e} A/m^2, peak {_peak(env):.4e} A/m^2, {t.size} time points")


def test_a17_4_Q_is_backward_sum_and_Q_generated_exact(runs):
    env = runs.get("base")
    t = _time(env)
    for c in ("top", "back"):
        i = _arr(env, f"tr.I.{c}", "A/m**2")
        back = float(np.sum((i[1:] - i[0]) * np.diff(t)))
        trap = float(np.sum(0.5 * ((i[1:] - i[0]) + (i[:-1] - i[0])) * np.diff(t)))
        assert _Q(env, c) == pytest.approx(back, rel=1e-9, abs=1e-30)
        print(f"\nQ.{c}: backward sum {back:.6e} C/m^2 = reported {_Q(env, c):.6e}; trapezoid would be {trap:.6e} "
              f"({trap / back - 1:+.3%})")
    assert _Qgen(env) == pytest.approx(Q_E * 1e11 * 1e4, rel=1e-12)


def test_a17_4_flags_on_every_strike_result(runs):
    runs.get("base")
    checked = 0
    for name, (rec, env) in runs.done.items():
        if env is None:
            continue
        assert FLAG_FIXED in env.flags and FLAG_1D in env.flags, name
        assert FLAG_FIXED in rec.flags and FLAG_1D in rec.flags, name
        checked += 1
    print(f"\nflags checked on {checked} stored strike results: {sorted(runs.done)}")


def test_a17_9b_terminal_current_includes_displacement(runs):
    env = runs.get("base")
    pc = env.diagnostics.get("particle_currents")
    assert isinstance(pc, dict) and {"top", "back"} <= set(pc), "item 9(b): particle currents in the diagnostics"
    t = _time(env)
    pt, pb = np.asarray(pc["top"], float), np.asarray(pc["back"], float)
    assert pt.size == t.size and pb.size == t.size
    it, ib = _arr(env, "tr.I.top", "A/m**2"), _arr(env, "tr.I.back", "A/m**2")
    pk = np.max(np.abs(it))
    pkp = max(np.max(np.abs(pt)), np.max(np.abs(pb)))
    print(f"\nterminal: max|I_top + I_back| / peak = {np.max(np.abs(it + ib)) / pk:.3e}; particle currents (engine "
          f"units): max|top + back| / peak = {np.max(np.abs(pt + pb)) / pkp:.3e}")


# ================================================================================================ A17.6 acceptance
def test_r100_collection(runs):
    env = runs.get("base")
    ft, fb = _Q(env, "top") / _Qgen(env), _Q(env, "back") / _Qgen(env)
    print(f"\nR100: Q_top/Q_gen = {ft:.6f} (|dev| {abs(ft - 1):.2e} <= 1e-2), Q_back/Q_gen = {fb:.6f} "
          f"(|dev| {abs(fb + 1):.2e} <= 1e-2)")
    assert abs(ft - 1.0) <= 0.01
    assert abs(fb + 1.0) <= 0.01


def test_r101_time_steps(runs):
    a, b = runs.get("base"), runs.get("fine_time")
    dq, dp = _rel(_Q(b), _Q(a)), _rel(_peak(b), _peak(a))
    print(f"\nR101: Q_top {_Q(a):.6e} vs {_Q(b):.6e} C/m^2: {dq:.2e} <= 5e-3; peak {_peak(a):.5e} vs {_peak(b):.5e} "
          f"A/m^2: {dp:.2e} <= 2e-2")
    assert dq <= 0.005
    assert dp <= 0.02


def test_r102_mesh(runs):
    a, b = runs.get("base"), runs.get("fine_mesh")
    dq, dp = _rel(_Q(b), _Q(a)), _rel(_peak(b), _peak(a))
    print(f"\nR102: Q_top {_Q(a):.6e} vs {_Q(b):.6e} C/m^2: {dq:.2e} <= 5e-3; peak {_peak(a):.5e} vs {_peak(b):.5e} "
          f"A/m^2: {dp:.2e} <= 2e-2")
    assert dq <= 0.005
    assert dp <= 0.02


def test_r103_low_injection_linearity(runs):
    a, b = runs.get("p1e10"), runs.get("base")
    fa, fb = _Q(a) / _Qgen(a), _Q(b) / _Qgen(b)
    df = _rel(fa, fb)
    ratio = _peak(b) / _peak(a)
    dr = abs(ratio / 10.0 - 1.0)
    print(f"\nR103: collected fraction 1e10 {fa:.6f}, 1e11 {fb:.6f}: {df:.2e} <= 1e-2; peak ratio {ratio:.5f}: "
          f"{dr:.2e} <= 2e-2")
    assert df <= 0.01


# amended by the lead (A17.10): the peak-ratio gate of R103 is a recorded FAIL. 1e11 pairs over 0.35 um is about
# 0.3 of the p doping, already moderate injection for the peak (ratio 9.615 against 10, limit 2 %); the gate stays
# as fixed and the result is kept as a strict xfail.
@pytest.mark.xfail(strict=True, reason="A17.10: R103 peak ratio 1e11/1e10 is 9.615 (limit 10 within 2 %): "
                                       "1e11 cm^-2 is not low injection for the peak current")
def test_r103_low_injection_peak_ratio(runs):
    a, b = runs.get("p1e10"), runs.get("base")
    ratio = _peak(b) / _peak(a)
    assert abs(ratio / 10.0 - 1.0) <= 0.02


def test_r103_reported_high_injection(runs):
    lines = []
    for name, pairs in (("p1e10", 1e10), ("base", 1e11), ("p1e12", 1e12), ("p1e13", 1e13)):
        env = runs.get(name, require_ok=False)
        rec = runs.done[name][0]
        if env is None:
            lines.append(f"  {pairs:.0e}: run {rec.status}: {rec.message}")
            continue
        lines.append(f"  {pairs:.0e}: collected fraction {_Q(env) / _Qgen(env):.6f}, peak |I_top| {_peak(env):.5e} "
                     f"A/m^2, peak per pair {_peak(env) / pairs:.5e} A/m^2 per cm^-2, "
                     f"Q_back/Q_gen {_Q(env, 'back') / _Qgen(env):.6f}")
    print("\nR103 reported (no gate):\n" + "\n".join(lines))


def test_r104_kirchhoff(runs):
    worst = {}
    for name in ("base", "fine_time", "fine_mesh", "p1e10"):
        env = runs.get(name)
        it, ib = _arr(env, "tr.I.top", "A/m**2"), _arr(env, "tr.I.back", "A/m**2")
        worst[name] = float(np.max(np.abs(it + ib)) / np.max(np.abs(it)))
    print(f"\nR104: max_t |I_top + I_back| / max|I_top|: {worst} (<= 2e-2)")
    assert worst["base"] <= 0.02
    assert all(w <= 0.02 for w in worst.values())


def test_r105_quadrature(runs):
    out = {}
    for name in ("base", "fine_time", "p1e10"):
        env = runs.get(name)
        pairs = runs.problem(name).analysis.pairs_cm2
        s = float(env.diagnostics["generated_pairs_quadrature_cm2"])
        out[name] = (s, abs(s / pairs - 1.0))
    print(f"\nR105: quadrature sums (value, rel. dev.): {out} (<= 1e-3)")
    assert all(d <= 1e-3 for _, d in out.values())


# ================================================================================================ A17.5 chain functions
@pytest.mark.parametrize("charge,area", [(1e-15, 1e-12), (2.5e-14, 4e-12), (1e-13, 1e-10)])
def test_a17_5_strike_problem_pairs(charge, area):
    S = _S()
    p = _ref()
    p2 = S.strike_problem(p, charge, area)
    expect = charge / (S.Q_E * area * 1e4)
    assert S.Q_E == Q_E
    assert p2.analysis.pairs_cm2 == pytest.approx(expect, rel=1e-11)
    d1, d2 = p.model_dump(mode="json"), p2.model_dump(mode="json")
    d2["analysis"]["pairs_cm2"] = d1["analysis"]["pairs_cm2"]
    assert d1 == d2                                                    # nothing else changes
    assert p.analysis.pairs_cm2 == 1e11                                # the original is untouched


def test_a17_5_tcad_pulse_staircase(runs):
    env = runs.get("base")
    S = _S()
    t = _time(env)
    for c in ("top", "back"):
        i = _arr(env, f"tr.I.{c}", "A/m**2")
        pulse = S.tcad_pulse(env, c, AREA_1UM2)
        pts = np.asarray(pulse.points, float)
        exp = [(t[0], 0.0)]
        for k in range(1, t.size):
            dt = t[k] - t[k - 1]
            v = (i[k] - i[0]) * AREA_1UM2
            exp += [(t[k - 1] + 1e-4 * dt, v), (t[k], v)]
        exp = np.asarray(exp)
        assert pts.shape == exp.shape, (pts.shape, exp.shape)
        np.testing.assert_allclose(pts[:, 0], exp[:, 0], rtol=1e-9, atol=0)
        scale = np.max(np.abs(exp[:, 1]))
        np.testing.assert_allclose(pts[:, 1], exp[:, 1], rtol=1e-9, atol=1e-12 * scale)
        integ = float(np.sum(0.5 * (pts[1:, 1] + pts[:-1, 1]) * np.diff(pts[:, 0])))
        ref = _Q(env, c) * AREA_1UM2
        print(f"\ntcad_pulse {c}: integral {integ:.6e} C vs tr.Q x area {ref:.6e} C: {_rel(integ, ref):.2e} (~1e-3)")
        assert _rel(integ, ref) <= 1e-3
    # sign as collected: positive at top (drawn from the node), integral = collected charge
    assert float(np.sum(np.asarray(S.tcad_pulse(env, "top", AREA_1UM2).points)[:, 1])) > 0


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


def _need_ngspice():
    from simws import engines
    if not engines.module("ngspice").available():
        pytest.skip("ngspice not available (SIMWS_SPICE_PREFIX)")


A10_FIELDS = ("qcrit_C", "q_lo_C", "q_hi_C", "runs", "result_ids", "node", "threshold_V", "polarity")


def _collected_key(res):
    keys = [k for k in res if "collected" in k and "charge" in k or k.startswith("collected_at")]
    assert keys, f"no collected-charge field in {sorted(res)}"
    return keys[0]


def _check_tcad_result(hub, res, rel_tol, rerun):
    S = _S()
    for k in A10_FIELDS + ("device_result_ids",):
        assert k in res, (k, sorted(res))
    assert res["q_lo_C"] < res["qcrit_C"] < res["q_hi_C"]
    assert res["qcrit_C"] == pytest.approx(0.5 * (res["q_lo_C"] + res["q_hi_C"]), rel=1e-12)
    assert (res["q_hi_C"] - res["q_lo_C"]) / res["q_hi_C"] <= rel_tol
    assert res["runs"] == len(res["result_ids"]) and res["node"] == "n" and res["threshold_V"] == 0.5
    assert res["polarity"] == "discharge"
    for rid in res["result_ids"]:
        assert hub.result(rid).q("circuit.v.n") is not None
    dev = [hub.result(rid) for rid in res["device_result_ids"]]
    assert dev and all(FLAG_FIXED in e.flags and FLAG_1D in e.flags for e in dev)
    flags = res.get("flags", [])
    if rerun:
        assert S.LINEAR_FLAG not in flags
    else:
        assert S.LINEAR_FLAG in flags
        assert len(dev) == 1
    ck = _collected_key(res)
    return ck, dev


# The field checks of critical_charge_tcad use a 100 um^2 device (low injection at these charges, so the node is
# upset well below q_high); R106's 1 um^2 case is reported separately.
AREA_FIELDS = 1e-10          # m^2


def test_a17_5_critical_charge_tcad_scaled_fields(shub):
    """rerun=False: one device run, the pulse scaled linearly, flagged LINEAR_FLAG; the collected charge at q_crit
    is q_crit times the device run's collected fraction (the pulse integral matches tr.Q to ~1e-3)."""
    _need_ngspice()
    S = _S()
    res = S.critical_charge_tcad(shub, _rc_node(1e-15, 1e4), "n", 0.5, _ref(), "top", AREA_FIELDS,
                                 q_low_C=0.05e-15, q_high_C=10e-15, rel_tol=1e-2, rerun=False)
    ck, dev = _check_tcad_result(shub, res, 1e-2, rerun=False)
    frac = _Q(dev[0]) / _Qgen(dev[0])
    assert _Qgen(dev[0]) * AREA_FIELDS == pytest.approx(10e-15, rel=1e-9)      # run once at q_high
    assert res[ck] == pytest.approx(frac * res["qcrit_C"], rel=1e-3)
    print(f"\ncritical_charge_tcad rerun=False (rel_tol 1e-2): qcrit {res['qcrit_C']:.4e} C, {ck} {res[ck]:.4e} C, "
          f"runs {res['runs']}, keys {sorted(res)}")


@pytest.mark.slow
def test_a17_5_critical_charge_tcad_rerun_fields(shub):
    """rerun=True: a device run per trial charge, no LINEAR_FLAG; the collected charge at q_crit comes from the
    device run nearest q_crit."""
    _need_ngspice()
    S = _S()
    res = S.critical_charge_tcad(shub, _rc_node(1e-15, 1e4), "n", 0.5, _ref(), "top", AREA_FIELDS,
                                 q_low_C=0.05e-15, q_high_C=10e-15, rel_tol=1e-2, rerun=True)
    ck, dev = _check_tcad_result(shub, res, 1e-2, rerun=True)
    gen = sorted(_Qgen(e) * AREA_FIELDS for e in dev)
    assert len(dev) == res["runs"]                                   # one device run per circuit trial
    near = min(dev, key=lambda e: abs(_Qgen(e) * AREA_FIELDS - res["qcrit_C"]))
    frac = _Q(near) / _Qgen(near)
    assert res[ck] == pytest.approx(frac * res["qcrit_C"], rel=2e-3)
    print(f"\ncritical_charge_tcad rerun=True (rel_tol 1e-2): qcrit {res['qcrit_C']:.4e} C, {ck} {res[ck]:.4e} C, "
          f"{res['runs']} circuit / {len(dev)} device runs, generated charges {['%.3e' % g for g in gen]}")


# ================================================================================================ A17.3 / A17.7 engines, trust
def test_a17_3_devsim_supports_others_refuse():
    from simws import engines
    p = _ref()
    assert engines.unsupported("devsim", p) == []
    assert "strike" in engines.capability("devsim").analyses
    for e in ("charon", "devsim_eb"):
        reasons = engines.unsupported(e, p)
        assert any(r.startswith("analyses: 'strike' not in") for r in reasons), (e, reasons)


@pytest.mark.parametrize("adapter,engine", [("device.charon", "charon"), ("device.devsim_eb", "devsim_eb")])
def test_a17_3_hub_refusal(shub, adapter, engine):
    r = shub.submit(adapter, params={"problem": _ref()})
    assert r.status == "invalid_input", (r.status, r.message)
    assert "analyses:" in r.message and "strike" in r.message, r.message


def test_a17_3_planner_one_engine():
    from simws.engines.planner import plan
    pl = plan(_ref())
    assert pl.primary == "devsim" and pl.checkers == []
    assert any(r.startswith("analyses:") for r in pl.refused["charon"])
    assert any(r.startswith("analyses:") for r in pl.refused["devsim_eb"])


def test_a17_7_trust_unverified(shub, runs):
    from simws.hub.trust import assess_stored
    runs.get("base")
    rec = runs.done["base"][0]
    t = assess_stored(shub, rec.id)
    assert t.level == "unverified", t
    print(f"\ntrust: {t.level}, reasons {t.reasons}")


# ================================================================================================ R107 refusals and ids
def _strike_2d():
    from simws.spec.library import npn_hbt_2d
    from simws.spec.schema import Problem
    d = npn_hbt_2d().model_dump(mode="json")
    a = _strike(depth_um=(0.1, 0.2)).model_dump(mode="json")
    a["bias"] = [{"contact": "collector", "value_V": 1.0}]
    d["analysis"] = a
    return Problem.model_validate(d)


def test_r107_2d_device_refused(shub):
    from simws import engines
    p = _strike_2d()
    reasons = engines.unsupported("devsim", p)
    assert any(r.startswith("strike:") for r in reasons), reasons
    r = shub.submit(DEVSIM, params={"problem": p})
    assert r.status == "invalid_input" and "strike:" in r.message, (r.status, r.message)


@pytest.mark.parametrize("how", ["param", "paper"])
def test_r107_mesh_study_refused(shub, how):
    p = _ref(pairs_cm2=2e11)                     # a problem not run elsewhere here
    if how == "param":
        r = shub.submit(DEVSIM, params={"problem": p, "mesh_study": True})
    else:
        r = shub.submit(DEVSIM, params={"problem": p}, fidelity="paper")
    assert r.status == "invalid_input", (r.status, r.message)
    assert "mesh_study:" in r.message, r.message
    print(f"\nR107 mesh study ({how}): {r.message}")


def test_r107_library_ids_unchanged():
    import inspect

    import simws.spec.library as L
    now = {}
    for name, f in sorted(vars(L).items()):
        if inspect.isfunction(f) and f.__module__ == L.__name__ and not name.startswith("_"):
            ps = inspect.signature(f).parameters.values()
            if all(q.default is not inspect.Parameter.empty or q.kind in (q.VAR_POSITIONAL, q.VAR_KEYWORD)
                   for q in ps):
                now[name] = f().problem_id()
    assert set(IDS_748F5FC) <= set(now), sorted(set(IDS_748F5FC) - set(now))
    changed = {k: (IDS_748F5FC[k], now[k]) for k in IDS_748F5FC if now[k] != IDS_748F5FC[k]}
    assert not changed, changed
    print(f"\nR107: {len(IDS_748F5FC)} library ids unchanged since 748f5fc; builders now: {sorted(now)}")


# ================================================================================================ R106 (reported, slow)
@pytest.mark.slow
def test_r106_chain_reported(shub):
    _need_ngspice()
    S = _S()
    circ = _rc_node(1e-15, 1e4)
    lo, hi = 0.05e-15, 100e-15
    out = []

    def attempt(label, fn):
        try:
            return fn()
        except ValueError as ex:
            if "does not upset" in str(ex):
                out.append(f"  {label}: does not upset the node up to {hi:.3e} C ({ex})")
                return None
            raise

    res_m = attempt("Messenger (1 ps, 10 ps)",
                    lambda: S.critical_charge(shub, circ, "n", 0.5, 1e-12, 10e-12, q_low_C=lo, q_high_C=hi))
    if res_m:
        out.append(f"  Messenger (1 ps, 10 ps): qcrit {res_m['qcrit_C']:.5e} C ({res_m['runs']} runs)")
    res_s = attempt("TCAD scaled", lambda: S.critical_charge_tcad(shub, circ, "n", 0.5, _ref(), "top", AREA_1UM2,
                                                                  q_low_C=lo, q_high_C=hi, rerun=False))
    if res_s:
        ck, _ = _check_tcad_result(shub, res_s, 1e-3, rerun=False)
        out.append(f"  TCAD scaled: qcrit {res_s['qcrit_C']:.5e} C, collected at qcrit {res_s[ck]:.5e} C "
                   f"({res_s['runs']} circuit runs, {len(res_s['device_result_ids'])} device run)")
    res_r = attempt("TCAD rerun", lambda: S.critical_charge_tcad(shub, circ, "n", 0.5, _ref(), "top", AREA_1UM2,
                                                                 q_low_C=lo, q_high_C=hi, rerun=True))
    if res_r:
        ck, dev = _check_tcad_result(shub, res_r, 1e-3, rerun=True)
        out.append(f"  TCAD rerun: qcrit {res_r['qcrit_C']:.5e} C, collected at qcrit {res_r[ck]:.5e} C "
                   f"({res_r['runs']} circuit runs, {len(dev)} device runs)")
        near = min(dev, key=lambda e: abs(_Qgen(e) * AREA_1UM2 - res_r["qcrit_C"]))
        out.append(f"    device run nearest qcrit: generated {_Qgen(near) * AREA_1UM2:.5e} C, "
                   f"collected fraction {_Q(near) / _Qgen(near):.6f}")
    print("\nR106 (reported, no gate): rc node 1 fF / 10 kOhm, threshold 0.5 V, 1 um^2 on 'top'\n" + "\n".join(out))
