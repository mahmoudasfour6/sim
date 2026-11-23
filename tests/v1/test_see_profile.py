"""Contract amendment A19 (docs/ARCHITECTURE_V1.md): the G4SEE deposit's depth profile as the TCAD generation, and
the upset probability.

Written by the checker from the contract text (A19 items 1 to 6; A17 / A18 for the strike and mixed-mode paths;
A8 for G4SEE and its NIST reference method, A8.7; A10 for upset_probability) and the public interfaces only:
StrikeTransient / GenerationProfile / Problem, chain.see (reference_strike_diode, with_load, strike_problem,
critical_charge_mixed, critical_charge, upset_probability, deposit_profile, profile_from_deposit, strike_from_see,
MEAN_PROFILE_FLAG), simws.radiation.nist_star.deposit, simws.spec.library.see_si_slab, the engine registry, the Hub
and trust.assess_stored.

Acceptance items R120 to R124 of A19.4 use the A19.4 tolerances unchanged. R124 is reported (printed, no gate).

Readings and choices (none changes a tolerance):
- The G4SEE geometry of R120 / R121 / R124: see_si_slab with width_um = 20 (bulk 20 um wide), bulk_thickness_um =
  50, then geometry.sv_width_um set to (1, 1) um (R120, R124) or (10, 10) um (R121) with with_changes; tilt 0
  (normal incidence).
- Histogram (checker's choice, A19 fixes none): linear, 20 000 bins, from 0 to 1.0 MeV (50 eV bins) for the proton
  runs (R120, R124) and 0 to 5.25 MeV (1.05 x the alpha energy, 262.5 eV bins) for the alpha slices of R121 (with 0
  to 1 MeV an alpha slice had events above 1 MeV and no mean). Every run is checked to have
  see.fraction_overflow = 0 (and underflow 0, low edge 0 MeV).
- R121 (ten alpha runs of about 30 s each) is marked slow.
- R121 states no event count; 20 000 events per slice as in R120 (seed 1 everywhere).
- R120: "combined standard errors" = sqrt(sum of the slices' sigma^2 + the 0 to 5 um run's sigma^2), each sigma the
  reported see.edep_mean uncertainty (statistical and binning, A8.5).
- R121: the reference for slice [a, b] is nist_star.deposit("alpha", 5.0, a, b) (A8.7: RK4 CSDA, step 0.01 um).
- R122 (a): "every tr.I.top value within 1e-9 relative" is |I_profile(t_k) - I_uniform(t_k)| <= 1e-9 |I_uniform(t_k)|
  at every time point, taken literally. (b) "within 1e-3 of pairs_cm2" is relative (as A17 R105).
- A19.3's flag strike:mean_profile "on the chain's result": looked for on the result of critical_charge_mixed run on
  a device whose strike follows a profile, or on upset_probability's result with that q_crit.
"""
import types

import numpy as np
import pydantic
import pytest

Q_E = 1.602176634e-19
DEVSIM = "device.devsim"
G4SEE = "see.g4see"
AREA = 1e-12
HIST_HIGH_MEV = 1.0
HIST_BINS = 20000
N_EVENTS = 20000
ALPHA_HIGH_MEV = 5.25        # R121: 1.05 x 5 MeV (see_si_slab's default); 0 to 1 MeV left overflow in an alpha slice
FLAG_1D = "strike:1d_column: the pairs are spread over the device area (no track radius, no 2D funnelling)"


def _S():
    from simws.chain import see
    return see


def _ref(**kw):
    return _S().reference_strike_diode(**kw)


def _profile(depth, weight):
    from simws.spec.schema import GenerationProfile
    return GenerationProfile(depth_um=list(depth), weight=list(weight))


def _strike(**kw):
    from simws.spec.schema import StrikeTransient
    base = dict(pairs_cm2=1e11, depth_um=(0.25, 0.6), t_peak_s=20e-12, sigma_s=5e-12, t_stop_s=1e-6)
    base.update(kw)
    return StrikeTransient(**base)


def _see(particle, energy_MeV, sv_depth_um, sv_thickness_um, sv_width_um, n_events=N_EVENTS, high_MeV=HIST_HIGH_MEV):
    """A19.4's G4SEE geometry: bulk 20 um wide and 50 um thick, SV sv_width x sv_width laterally, normal incidence."""
    from simws.spec.library import see_si_slab
    p = see_si_slab(particle=particle, energy_MeV=energy_MeV, sv_thickness_um=sv_thickness_um,
                    sv_depth_um=sv_depth_um, bulk_thickness_um=50.0, width_um=20.0, tilt_deg=0.0,
                    n_events=n_events, seed=1, bins=HIST_BINS, high_MeV=high_MeV)
    return p.with_changes({"geometry.sv_width_um": [float(sv_width_um), float(sv_width_um)]})


def _need_g4see():
    from simws import engines
    if not engines.module("g4see").available():
        pytest.skip("G4SEE not available (SIMWS_G4SEE_PREFIX)")


def _need_ngspice():
    from simws import engines
    if not engines.module("ngspice").available():
        pytest.skip("ngspice not available (SIMWS_SPICE_PREFIX)")


def _arr(env, name, unit=None):
    q = env.q(name)
    return np.asarray(q.to(unit) if unit else q.array(), float)


def _Q(env, contact="top"):
    return float(np.asarray(env.q(f"tr.Q.{contact}").to("C/m**2")))


def _Qgen(env):
    return float(np.asarray(env.q("tr.Q_generated").to("C/m**2")))


def _mean_MeV(env):
    q = env.q("see.edep_mean")
    return float(q.value) / 1e6, float(q.uncertainty.sigma) / 1e6


def _no_overflow(env, label):
    over = float(env.q("see.fraction_overflow").value)
    under = float(env.q("see.fraction_underflow").value)
    assert over == 0.0 and under == 0.0, (label, over, under)


# ------------------------------------------------------------------------------------------------ shared runs
@pytest.fixture(scope="module")
def shub(tmp_path_factory):
    from simws.hub.jobs import Hub
    return Hub(tmp_path_factory.mktemp("see_profile_store"), mode="inprocess")


class _Dev:
    """name -> envelope, one DEVSIM run each, made on first use."""
    SPECS = {
        "uniform": lambda: _ref(),
        "flat": lambda: _S().strike_from_see(_ref(), _profile([0.25, 0.6], [1.0, 1.0])),
        "linear": lambda: _S().strike_from_see(_ref(), _profile([0.25, 0.6], [1.0, 3.0])),
    }

    def __init__(self, hub):
        self.hub, self.done = hub, {}

    def get(self, name):
        if name not in self.done:
            rec = self.hub.submit(DEVSIM, params={"problem": self.SPECS[name]()})
            assert rec.status == "ok", (name, rec.status, rec.message)
            self.done[name] = (rec, self.hub.result(rec.id))
        return self.done[name][1]

    def rec(self, name):
        self.get(name)
        return self.done[name][0]


@pytest.fixture(scope="module")
def dev(shub):
    return _Dev(shub)


@pytest.fixture(scope="module")
def r120(shub):
    _need_g4see()
    S = _S()
    full = _see("proton", 5.0, 0.0, 5.0, 1.0)
    dp = S.deposit_profile(shub, full, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    rec = shub.submit(G4SEE, params={"problem": full})
    assert rec.status == "ok", (rec.status, rec.message)
    return dp, shub.result(rec.id), rec


@pytest.fixture(scope="module")
def r121(shub):
    _need_g4see()
    S = _S()
    base = _see("alpha", 5.0, 0.0, 10.0, 10.0, high_MeV=ALPHA_HIGH_MEV)
    return S.deposit_profile(shub, base, [float(k) for k in range(11)])


# ================================================================================================ A19.1 description
def test_a19_1_profile_valid_and_linear_between_points():
    p = _profile([0.0, 1.0, 3.0], [1.0, 3.0, 0.0])
    x = np.array([-0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 3.5])
    np.testing.assert_allclose(p.at(x), [0.0, 1.0, 2.0, 3.0, 1.5, 0.0, 0.0], rtol=0, atol=1e-15)


@pytest.mark.parametrize("depth,weight", [
    ([0.0, 1.0, 1.0], [1.0, 1.0, 1.0]),        # not strictly increasing
    ([0.0, 2.0, 1.0], [1.0, 1.0, 1.0]),        # decreasing
    ([1.0, 0.0], [1.0, 1.0]),                  # decreasing
    ([0.0, 1.0], [0.0, 0.0]),                  # all zero
    ([0.0, 1.0], [1.0, -0.1]),                 # negative weight
    ([0.0, 1.0], [1.0, 1.0, 1.0]),             # lengths differ
])
def test_a19_1_profile_rules(depth, weight):
    with pytest.raises(pydantic.ValidationError):
        _profile(depth, weight)


def test_a19_1_profile_zero_weight_allowed_when_not_all_zero():
    p = _profile([0.0, 1.0], [0.0, 1.0])
    assert p.weight == [0.0, 1.0]


def test_a19_1_profile_ends_equal_depth_um():
    an = _strike(depth_um=(0.25, 0.6), profile=_profile([0.25, 0.4, 0.6], [1.0, 2.0, 1.0]))
    assert tuple(an.depth_um) == (0.25, 0.6)
    for seg in [(0.2, 0.6), (0.25, 0.7), (0.3, 0.5)]:
        with pytest.raises(pydantic.ValidationError):
            _strike(depth_um=seg, profile=_profile([0.25, 0.4, 0.6], [1.0, 2.0, 1.0]))


def test_a19_1_profile_left_out_when_absent():
    an = _strike()
    assert an.profile is None
    for mode in ("python", "json"):
        assert "profile" not in an.model_dump(mode=mode)
    assert "profile" not in _ref().model_dump(mode="json")["analysis"]
    assert _ref().problem_id() == "ff0714fb4325f7f403ef4969c97f233b"
    withp = _S().strike_from_see(_ref(), _profile([0.25, 0.6], [1.0, 1.0]))
    assert "profile" in withp.model_dump(mode="json")["analysis"]
    assert withp.problem_id() != _ref().problem_id()


# ================================================================================================ R123 refusals
def test_r123_profile_ends_differ_from_depth_refused():
    from simws.spec.schema import Problem
    d = _ref().model_dump(mode="json")
    d["analysis"]["profile"] = {"depth_um": [0.2, 0.6], "weight": [1.0, 1.0]}   # depth_um stays (0.25, 0.6)
    with pytest.raises(pydantic.ValidationError):
        Problem.model_validate(d)


def test_r123_all_zero_weight_refused():
    from simws.spec.schema import Problem
    d = _ref().model_dump(mode="json")
    d["analysis"]["profile"] = {"depth_um": [0.25, 0.6], "weight": [0.0, 0.0]}
    with pytest.raises(pydantic.ValidationError):
        Problem.model_validate(d)


def test_r123_decreasing_depths_refused():
    from simws.spec.schema import Problem
    d = _ref().model_dump(mode="json")
    d["analysis"]["depth_um"] = [0.25, 0.6]
    d["analysis"]["profile"] = {"depth_um": [0.6, 0.4, 0.25], "weight": [1.0, 1.0, 1.0]}
    with pytest.raises(pydantic.ValidationError):
        Problem.model_validate(d)
    d["analysis"]["profile"] = {"depth_um": [0.25, 0.5, 0.4, 0.6], "weight": [1.0, 1.0, 1.0, 1.0]}
    with pytest.raises(pydantic.ValidationError):
        Problem.model_validate(d)


# ================================================================================================ A19.3 strike_from_see
def test_a19_3_strike_from_see():
    S = _S()
    base = S.with_load(_ref(), "top", 1.0, 1e5, 1e-15, AREA)
    prof = _profile([0.0, 2.5, 5.0], [3.0, 2.0, 1.0])
    p = S.strike_from_see(base, prof)
    an = p.analysis
    assert tuple(an.depth_um) == (0.0, 5.0)
    assert an.profile.depth_um == [0.0, 2.5, 5.0] and an.profile.weight == [3.0, 2.0, 1.0]
    assert an.load == base.analysis.load                      # the load is kept
    a0, a1 = base.model_dump(mode="json"), p.model_dump(mode="json")
    for k in ("device", "models", "numerics"):
        assert a0[k] == a1[k]
    for k, v in a0["analysis"].items():
        if k not in ("depth_um", "profile"):
            assert a1["analysis"][k] == v, k
    with pytest.raises(ValueError):                           # outside the 5 um device
        S.strike_from_see(base, _profile([0.0, 6.0], [1.0, 1.0]))


def test_a19_3_strike_from_see_needs_strike():
    from simws.spec.library import pn_diode_1d
    with pytest.raises(ValueError):
        _S().strike_from_see(pn_diode_1d(), _profile([0.0, 0.5], [1.0, 1.0]))


def test_a19_3_mean_profile_flag_constant():
    assert _S().MEAN_PROFILE_FLAG.startswith("strike:mean_profile")


# ================================================================================================ A19.2 deposit_profile (fake)
class _FakeSeeHub:
    """Answers see.g4see jobs with see.edep_mean = 1e5 eV x thickness x (1 + depth) and sigma 10 eV x thickness."""

    def __init__(self):
        self.jobs, self.out = [], {}

    def submit(self, name, params=None, **kw):
        p = params["problem"]
        rid = f"s{len(self.jobs)}"
        self.jobs.append((name, p))
        g = p.geometry
        self.out[rid] = (1e5 * g.sv_thickness_um * (1 + g.sv_depth_um), 10.0 * g.sv_thickness_um)
        return types.SimpleNamespace(status="ok", id=rid, message="")

    def result(self, rid):
        v, s = self.out[rid]

        def q(name, member=None):
            if name != "see.edep_mean":
                raise KeyError(name)
            return types.SimpleNamespace(name=name, value=v, unit="eV",
                                         uncertainty=types.SimpleNamespace(sigma=s))
        return types.SimpleNamespace(q=q, flags=[], diagnostics={"n_events": 100})


def test_a19_2_deposit_profile_slices_fake():
    S = _S()
    base = _see("proton", 5.0, 0.0, 5.0, 1.0)
    edges = [0.0, 0.5, 1.5, 3.0]
    hub = _FakeSeeHub()
    dp = S.deposit_profile(hub, base, edges)
    assert len(hub.jobs) == 3 and all(n == G4SEE for n, _ in hub.jobs)
    b0 = base.model_dump(mode="json")
    for (a, b), (_, p) in zip(zip(edges[:-1], edges[1:]), hub.jobs):
        assert p.geometry.sv_depth_um == pytest.approx(a, abs=1e-12)
        assert p.geometry.sv_thickness_um == pytest.approx(b - a, abs=1e-12)
        d = p.model_dump(mode="json")
        d["geometry"]["sv_depth_um"] = b0["geometry"]["sv_depth_um"]
        d["geometry"]["sv_thickness_um"] = b0["geometry"]["sv_thickness_um"]
        assert d == b0                                         # everything else unchanged
    np.testing.assert_allclose(dp["centres_um"], [0.25, 1.0, 2.25], rtol=0, atol=1e-12)
    means = [1e5 * (b - a) * (1 + a) / 1e6 for a, b in zip(edges[:-1], edges[1:])]
    np.testing.assert_allclose(dp["edep_mean_MeV"], means, rtol=1e-12)
    np.testing.assert_allclose(dp["edep_sigma_MeV"], [10.0 * (b - a) / 1e6 for a, b in zip(edges[:-1], edges[1:])],
                               rtol=1e-12)
    np.testing.assert_allclose(dp["MeV_per_um"], [m / (b - a) for m, (a, b) in
                                                  zip(means, zip(edges[:-1], edges[1:]))], rtol=1e-12)


@pytest.mark.parametrize("edges", [[0.0], [1.0, 0.5], [0.0, 1.0, 1.0], [-1.0, 0.0]])
def test_a19_2_deposit_profile_bad_edges(edges):
    with pytest.raises(ValueError):
        _S().deposit_profile(_FakeSeeHub(), _see("proton", 5.0, 0.0, 5.0, 1.0), edges)


def test_a19_2_profile_from_deposit_rule():
    """weights = mean MeV per um at the slice centres, extended flat to the slice edges at both ends."""
    S = _S()
    hub = _FakeSeeHub()
    edges = [0.0, 0.5, 1.5, 3.0]
    dp = S.deposit_profile(hub, _see("proton", 5.0, 0.0, 5.0, 1.0), edges)
    prof = S.profile_from_deposit(dp)
    w = dp["MeV_per_um"]
    np.testing.assert_allclose(prof.depth_um, [0.0, 0.25, 1.0, 2.25, 3.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(prof.weight, [w[0]] + list(w) + [w[-1]], rtol=1e-12)
    np.testing.assert_allclose(prof.at(dp["centres_um"]), w, rtol=1e-12)
    assert prof.at(0.1) == pytest.approx(w[0], rel=1e-12)          # flat to the first edge
    assert prof.at(2.9) == pytest.approx(w[-1], rel=1e-12)         # flat to the last edge
    # usable as the strike's generation over the profile's own range
    p = S.strike_from_see(_ref(), prof)
    assert tuple(p.analysis.depth_um) == (0.0, 3.0)


# ================================================================================================ R120, R121 (G4SEE)
def test_r120_slices_add_up(r120):
    dp, full, _ = r120
    m_full, s_full = _mean_MeV(full)
    _no_overflow(full, "0-5 um")
    total = float(np.sum(dp["edep_mean_MeV"]))
    comb = float(np.sqrt(np.sum(np.square(dp["edep_sigma_MeV"])) + s_full ** 2))
    dev = abs(total - m_full) / comb
    print(f"\nR120: slices {['%.6f' % m for m in dp['edep_mean_MeV']]} MeV (sigma "
          f"{['%.6f' % s for s in dp['edep_sigma_MeV']]}); sum {total:.6f} MeV; 0-5 um run {m_full:.6f} +- "
          f"{s_full:.6f} MeV; |diff| {abs(total - m_full):.6f} MeV = {dev:.3f} combined sigma (limit 4)")
    assert dev <= 4.0


def test_r120_no_overflow_in_slices(shub, r120):
    dp, _, _ = r120
    assert len(dp["result_ids"]) == 5
    for k, rid in enumerate(dp["result_ids"]):
        env = shub.result(rid)
        _no_overflow(env, f"slice {k}")
        assert int(env.diagnostics["n_events"]) == N_EVENTS


def test_r120_deposit_profile_fields(r120):
    dp, _, _ = r120
    np.testing.assert_allclose(dp["centres_um"], [0.5, 1.5, 2.5, 3.5, 4.5], rtol=0, atol=1e-12)
    np.testing.assert_allclose(dp["MeV_per_um"], dp["edep_mean_MeV"], rtol=1e-12)     # 1 um slices
    assert all(s > 0 for s in dp["edep_sigma_MeV"])


@pytest.mark.slow
def test_r121_against_nist(shub, r121):
    from simws.radiation import nist_star
    dp = r121
    lines, worst = [], 0.0
    for k, (m, s) in enumerate(zip(dp["edep_mean_MeV"], dp["edep_sigma_MeV"])):
        ref = nist_star.deposit("alpha", 5.0, float(k), float(k + 1))
        d = m / ref - 1.0
        worst = max(worst, abs(d))
        lines.append(f"  [{k},{k + 1}] um: G4SEE {m:.5f} +- {s:.5f} MeV, NIST CSDA {ref:.5f} MeV, {d:+.3%}")
    for k, rid in enumerate(dp["result_ids"]):
        _no_overflow(shub.result(rid), f"alpha slice {k}")
    print("\nR121 (limit 3 %):\n" + "\n".join(lines) + f"\n  worst {worst:.3%}")
    assert worst <= 0.03


def test_a19_5_see_trust_unverified(shub, r120):
    from simws.hub.trust import assess_stored
    _, _, rec = r120
    t = assess_stored(shub, rec.id)
    assert t.level == "unverified", t


# ================================================================================================ R122 (a profile in TCAD)
def test_r122a_flat_profile_reproduces_uniform(dev):
    u, f = dev.get("uniform"), dev.get("flat")
    qu, qf = _Q(u), _Q(f)
    iu, iff = _arr(u, "tr.I.top", "A/m**2"), _arr(f, "tr.I.top", "A/m**2")
    assert iu.shape == iff.shape
    dq = abs(qf / qu - 1)
    rel = np.abs(iff - iu) / np.abs(iu)
    worst = float(np.max(rel))
    print(f"\nR122a: Q_top uniform {qu:.10e}, flat profile {qf:.10e} (rel {dq:.3e}, limit 1e-9); worst tr.I.top rel "
          f"{worst:.3e} at k={int(np.argmax(rel))} (limit 1e-9)")
    assert dq <= 1e-9
    assert worst <= 1e-9


def test_r122b_linear_profile(dev):
    env = dev.get("linear")
    pairs = 1e11
    quad = float(env.diagnostics["generated_pairs_quadrature_cm2"])
    frac = _Q(env) / _Qgen(env)
    print(f"\nR122b: quadrature {quad:.8e} against {pairs:.1e} (rel {abs(quad / pairs - 1):.3e}, limit 1e-3); "
          f"Q_top / Q_generated {frac:.6f} (limit 1 +- 0.01)")
    assert abs(quad / pairs - 1) <= 1e-3
    assert abs(frac - 1.0) <= 0.01
    assert _Qgen(env) == pytest.approx(Q_E * pairs * 1e4, rel=1e-12)


def test_a19_1_profile_changes_the_current(dev):
    """The generation follows the profile: a linear profile gives a different current from the flat one."""
    f, lin = dev.get("flat"), dev.get("linear")
    a, b = _arr(f, "tr.I.top", "A/m**2"), _arr(lin, "tr.I.top", "A/m**2")
    assert float(np.max(np.abs(a - b))) > 1e-6 * float(np.max(np.abs(a)))


def test_a19_5_profile_device_trust_unverified(shub, dev):
    from simws.hub.trust import assess_stored
    rec = dev.rec("linear")
    t = assess_stored(shub, rec.id)
    assert t.level == "unverified", t
    assert FLAG_1D in dev.get("linear").flags


# ================================================================================================ A19.3 mean_profile flag
class _FakeDevHub:
    """device.devsim jobs answered with tr.V_min.top = 1 - 0.5 q / 1 fC."""

    def __init__(self):
        self.n = 0
        self.v = {}

    def submit(self, name, params=None, **kw):
        p = params["problem"]
        q = p.analysis.pairs_cm2 * Q_E * AREA * 1e4
        rid = f"d{self.n}"
        self.n += 1
        self.v[rid] = 1.0 - 0.5 * q / 1e-15
        return types.SimpleNamespace(status="ok", id=rid, message="")

    def result(self, rid):
        v = self.v[rid]
        return types.SimpleNamespace(q=lambda name, member=None: types.SimpleNamespace(value=v), flags=[],
                                     diagnostics={})


def test_a19_3_mean_profile_flag_on_chain_result(shub, r120):
    """A19.3: q_crit from critical_charge_mixed on the profiled device, then upset_probability; the flag
    strike:mean_profile is on the chain's result."""
    S = _S()
    dp, full, rec = r120
    prof = S.profile_from_deposit(dp)
    device = S.strike_from_see(S.with_load(_ref(), "top", 1.0, 1e5, 1e-15, AREA), prof)
    qc = S.critical_charge_mixed(_FakeDevHub(), device, 0.5, AREA, 0.1e-15, 10e-15, rel_tol=1e-2)
    up = S.upset_probability(shub.result(rec.id), qc)
    flags = list(qc.get("flags", [])) + list(up.get("flags", []))
    print(f"\nA19.3 flags: critical_charge_mixed {qc.get('flags')}, upset_probability {up.get('flags')}")
    assert any(f.startswith("strike:mean_profile") for f in flags), flags


# ================================================================================================ R124 (reported, slow)
def _rc_node(C, R):
    from simws.spec.circuit import CircuitProblem, Element, Transient
    return CircuitProblem(
        elements=[Element(name="vdd", kind="vsource", nodes=["vdd", "0"], value=1.0),
                  Element(name="r1", kind="resistor", nodes=["vdd", "n"], value=float(R)),
                  Element(name="c1", kind="capacitor", nodes=["n", "0"], value=float(C))],
        analysis=Transient(step_s=0.1e-12, stop_s=300e-12, max_step_s=0.1e-12),
        probes=["v(n)"])


@pytest.mark.slow
def test_r124_chain_reported(shub):
    _need_g4see()
    S = _S()
    from simws import engines
    lo, hi = 0.05e-15, 100e-15
    edges = [0.5 * k for k in range(11)]
    out = []
    circ = _rc_node(1e-15, 1e5)
    qm = None
    if engines.module("ngspice").available():
        qm = S.critical_charge(shub, circ, "n", 0.5, 1e-12, 10e-12, q_low_C=lo, q_high_C=hi)
        out.append(f"  Messenger (1 ps, 10 ps) q_crit on 100 kOhm / 1 fF: {qm['qcrit_C']:.5e} C ({qm['runs']} runs)")
    else:
        out.append("  ngspice not available: Messenger q_crit not run")
    for E in (5.0, 2.0):
        full = _see("proton", E, 0.0, 5.0, 1.0)
        dp = S.deposit_profile(shub, full, edges)
        for rid in dp["result_ids"]:
            _no_overflow(shub.result(rid), f"{E} MeV slice")
        rec = shub.submit(G4SEE, params={"problem": full})
        assert rec.status == "ok", rec.message
        env = shub.result(rec.id)
        _no_overflow(env, f"{E} MeV 0-5 um")
        m, s = _mean_MeV(env)
        prof = S.profile_from_deposit(dp)
        out.append(f"  proton {E:g} MeV: slices (0.5 um) MeV/um {['%.5f' % w for w in dp['MeV_per_um']]}; "
                   f"0-5 um mean {m:.5f} +- {s:.5f} MeV, slice sum {dp['total_MeV']:.5f} MeV")
        device = S.strike_from_see(S.with_load(_ref(), "top", 1.0, 1e5, 1e-15, AREA), prof)
        try:
            qc = S.critical_charge_mixed(shub, device, 0.5, AREA, lo, hi)
        except ValueError as ex:
            out.append(f"    q_crit mixed: no bracket ({ex})")
            continue
        up = S.upset_probability(env, qc)
        vk = {k: qc[k] for k in qc if "v_min" in k}
        out.append(f"    q_crit mixed with the profile: {qc['qcrit_C']:.5e} C ({qc['runs']} runs), {vk}; "
                   f"P {up['P']:.5f} (P_low {up['P_low']:.5f}, P_high {up['P_high']:.5f}, sigma {up['sigma']:.2e}), "
                   f"E_crit {up['E_crit_eV'] / 1e3:.3f} keV, flags {qc.get('flags')} / {up['flags']}")
        if qm:
            um = S.upset_probability(env, qm)
            out.append(f"    with Messenger q_crit: P {um['P']:.5f} (P_low {um['P_low']:.5f}, P_high {um['P_high']:.5f}), "
                       f"E_crit {um['E_crit_eV'] / 1e3:.3f} keV, flags {um['flags']}")
    print("\nR124 (reported, no gate; 20 000 events, histogram 0-1 MeV x 20 000 bins):\n" + "\n".join(out))
