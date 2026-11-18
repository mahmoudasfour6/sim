"""Contract amendment A10: single event to circuit (simws.chain.see): PAIR_ENERGY_SI_EV, charge_C / edep_eV,
double_exponential, inject, critical_charge, upset_probability, and the acceptance checks R40 to R44. Written from
docs/ARCHITECTURE_V1.md (A10, with A8 and A9 as far as needed) only, by the checker; simws/chain/see.py was not read.

critical_charge is tested with (a) a FAKE hub object whose submit() returns hand-made ResultEnvelopes whose node
minimum (or maximum) is a chosen function of the charge injected, (b) the real Hub with a fake ngspice run, and
(c) the real ngspice engine (R43 and one charge-polarity check). upset_probability is tested on stored see.g4see
results made by the real Hub from hand-made histograms (fake G4SEE run), and on a real one in R44.

Closed-form references of A10.4 are recomputed here numerically (dense sampling plus a bounded maximiser) and
compared with the contract's numbers; the gates use the contract's numbers with the contract's tolerances."""
import inspect
import math

import numpy as np
import pytest

Q_E = 1.602176634e-19                     # C (A10.1)
CIRCUIT = "circuit.ngspice"
SEE = "see.g4see"

TAU_R, TAU_F = 1e-12, 10e-12              # A10.5 pulse
# Contract numbers (A10.5), used as the gates
REF_R40_C = 4.37753e-14
REF_R41_PEAK_A, REF_R41_PEAK_T = 1.548527e-4, 2.558428e-12
REF_R42_VMIN, REF_R42_DROOP = 0.8452308, 0.1547692
REF_R43_QCRIT = 6.461234e-15
REF_R44_QCRIT = {(60e-15, 1e5): 3.032237e-14, (200e-15, 1e5): 1.003812e-13}


def _S():
    from simws.chain import see
    return see


def _trapz(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _pts(pwl):
    a = np.asarray(pwl.points, float)
    return a[:, 0], a[:, 1]


def _messenger(t, Q, tr, tf, d=0.0):
    s = np.asarray(t, float) - d
    return np.where(s >= 0, Q / (tf - tr) * (np.exp(-s / tf) - np.exp(-s / tr)), 0.0)


# ------------------------------------------------------------------------------------------------ the A10.5 circuit
def rc_node(C, R):
    """A10.5: vsource "vdd" 1.0 V on node "vdd", resistor "r1" (vdd-n) R, capacitor "c1" (n-0) C, transient step
    0.1 ps, stop 300 ps, max step 0.1 ps, default numerics. A10.5 names no probe; a CircuitProblem needs one
    (A9.1), so the probe is "v(n)" (the node of the acceptance checks)."""
    from simws.spec.circuit import CircuitProblem, Element, Transient
    return CircuitProblem(
        elements=[Element(name="vdd", kind="vsource", nodes=["vdd", "0"], value=1.0),
                  Element(name="r1", kind="resistor", nodes=["vdd", "n"], value=float(R)),
                  Element(name="c1", kind="capacitor", nodes=["n", "0"], value=float(C))],
        analysis=Transient(step_s=0.1e-12, stop_s=300e-12, max_step_s=0.1e-12),
        probes=["v(n)"])


def rc_low(C, R, probes=("v(n)",)):
    """Checker's own variant for the "charge" polarity: node n held at 0 V through R."""
    from simws.spec.circuit import CircuitProblem, Element, Transient
    return CircuitProblem(
        elements=[Element(name="vss", kind="vsource", nodes=["vss", "0"], value=0.0),
                  Element(name="r1", kind="resistor", nodes=["vss", "n"], value=float(R)),
                  Element(name="c1", kind="capacitor", nodes=["n", "0"], value=float(C))],
        analysis=Transient(step_s=0.1e-12, stop_s=300e-12, max_step_s=0.1e-12),
        probes=list(probes))


# ------------------------------------------------------------------------------------------------ closed form (A10.4)
def droop(t, Q, C, R, tr=TAU_R, tf=TAU_F):
    tau = R * C
    t = np.asarray(t, float)

    def h(a):
        return a * tau * (np.exp(-t / tau) - np.exp(-t / a)) / (tau - a)
    return Q / (C * (tf - tr)) * (h(tf) - h(tr))


def droop_max(Q, C, R, t_stop=300e-12):
    from scipy.optimize import minimize_scalar
    t = np.linspace(0.0, t_stop, 300001)
    u = droop(t, Q, C, R)
    i = int(np.argmax(u))
    lo, hi = t[max(i - 1, 0)], t[min(i + 1, t.size - 1)]
    r = minimize_scalar(lambda x: -float(droop(x, Q, C, R)), bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-22})
    return max(-float(r.fun), float(u[i])), float(r.x)


def qcrit_closed(C, R, threshold=0.5, vdd=1.0):
    um, _ = droop_max(1.0, C, R)                 # linear in Q
    return (vdd - threshold) / um


def test_reference_values_recomputed():
    """A10.4 closed forms recomputed; reported when they differ from the contract's numbers by > 1e-5 relative."""
    rows = []
    um, tm = droop_max(2e-15, 10e-15, 1e4)
    rows.append(("R42 v_min", 1.0 - um, REF_R42_VMIN))
    rows.append(("R42 droop", um, REF_R42_DROOP))
    rows.append(("R43 qcrit", qcrit_closed(10e-15, 1e4), REF_R43_QCRIT))
    for (C, R), ref in REF_R44_QCRIT.items():
        rows.append((f"R44 qcrit rc_node({C:.0e}, {R:.0e})", qcrit_closed(C, R), ref))
    ts = math.log(TAU_F / TAU_R) * TAU_R * TAU_F / (TAU_F - TAU_R)
    rows.append(("R41 peak time", ts, REF_R41_PEAK_T))
    rows.append(("R41 peak value", float(_messenger(ts, 2e-15, TAU_R, TAU_F)), REF_R41_PEAK_A))
    rows.append(("R40 charge_C(1e6)", 1e6 / 3.66 * Q_E, REF_R40_C))
    # "about 1.61 MeV, 70.6 fC": 70.6 fC is the A8.7 R22 reference 1.6137 MeV (1.61 MeV exactly gives 70.48 fC)
    rows.append(("R44 70.6 fC for 1.6137 MeV", 1.6137e6 / 3.66 * Q_E, 70.6e-15))
    bad = []
    for label, mine, ref in rows:
        rel = mine / ref - 1
        tol = 1e-3 if label.startswith("R44 70.6") else 1e-5      # "about 1.61 MeV, 70.6 fC": 3 digits given
        print(f"\n[reference {label}] recomputed {mine:.9g}, contract {ref:.9g}, relative {rel:+.2e}")
        if abs(rel) > tol:
            bad.append((label, mine, ref, rel))
    assert not bad, bad


# ------------------------------------------------------------------------------------------------ A10.1 constants, charge
def test_pair_energy_constant():
    assert _S().PAIR_ENERGY_SI_EV == 3.66


def test_charge_C_scalar_and_default():
    S = _S()
    assert S.charge_C(1e6) == pytest.approx(1e6 / 3.66 * Q_E, rel=1e-12, abs=0)
    assert S.charge_C(3.66) == pytest.approx(Q_E, rel=1e-12, abs=0)
    sig = inspect.signature(S.charge_C).parameters
    assert sig["pair_energy_eV"].default == S.PAIR_ENERGY_SI_EV
    sig = inspect.signature(S.edep_eV).parameters
    assert sig["pair_energy_eV"].default == S.PAIR_ENERGY_SI_EV


def test_charge_C_custom_pair_energy():
    S = _S()
    assert S.charge_C(1e6, pair_energy_eV=3.6) == pytest.approx(1e6 / 3.6 * Q_E, rel=1e-12, abs=0)
    assert S.charge_C(1e6, 3.6) == pytest.approx(1e6 / 3.6 * Q_E, rel=1e-12, abs=0)
    assert S.edep_eV(1e-15, pair_energy_eV=4.0) == pytest.approx(1e-15 / Q_E * 4.0, rel=1e-12, abs=0)
    assert S.edep_eV(S.charge_C(2.5e5, pair_energy_eV=4.2), pair_energy_eV=4.2) == pytest.approx(2.5e5, rel=1e-12)


def test_edep_eV_scalar():
    S = _S()
    assert S.edep_eV(1e-15) == pytest.approx(1e-15 / Q_E * 3.66, rel=1e-12, abs=0)


def test_charge_arrays_and_round_trip():
    S = _S()
    E = np.array([1.0, 1e3, 1.61e6, 5e6, 1e9])
    q = np.asarray(S.charge_C(E), float)
    assert q.shape == E.shape
    assert np.allclose(q, E / 3.66 * Q_E, rtol=1e-12, atol=0)
    back = np.asarray(S.edep_eV(q), float)
    assert back.shape == E.shape and np.allclose(back, E, rtol=1e-12, atol=0)
    Q = np.array([[1e-15, 2e-15], [3e-14, 4e-13]])
    e = np.asarray(S.edep_eV(Q), float)
    assert e.shape == Q.shape and np.allclose(e, Q / Q_E * 3.66, rtol=1e-12, atol=0)
    assert np.allclose(np.asarray(S.charge_C(e), float), Q, rtol=1e-12, atol=0)


# ------------------------------------------------------------------------------------------------ double_exponential
def test_double_exponential_type_and_default_n_points():
    from simws.spec.circuit import Pwl
    S = _S()
    p = S.double_exponential(2e-15, TAU_R, TAU_F)
    assert isinstance(p, Pwl)
    assert len(p.points) == 2001                                          # A10.6
    sig = inspect.signature(S.double_exponential).parameters
    assert sig["n_points"].default == 2001
    assert sig["delay_s"].default == 0.0


@pytest.mark.parametrize("Q,tr,tf,d,n", [(2e-15, 1e-12, 10e-12, 0.0, 401), (2e-15, 1e-12, 10e-12, 0.0, 2001),
                                         (7.3e-14, 3e-12, 50e-12, 25e-12, 50), (1e-13, 2e-11, 2.1e-11, 1e-9, 5000)])
def test_double_exponential_grid_values_integral(Q, tr, tf, d, n):
    S = _S()
    p = S.double_exponential(Q, tr, tf, delay_s=d, n_points=n)
    t, v = _pts(p)
    assert t.size == n
    j = np.arange(n)
    t_ref = d + 20.0 * tf * (j / (n - 1)) ** 2
    assert np.allclose(t, t_ref, rtol=1e-12, atol=1e-30)
    assert t[0] == pytest.approx(d, rel=1e-12, abs=1e-30)
    assert v[-1] == 0.0                                                   # last value set to 0
    assert _trapz(t, v) == pytest.approx(Q, rel=1e-12, abs=0)             # integral rule
    # values proportional (one factor) to the Messenger formula, apart from the last
    s_ref = 20.0 * tf * (j / (n - 1)) ** 2                               # t - d without cancellation
    f = Q / (tf - tr) * (np.exp(-s_ref / tf) - np.exp(-s_ref / tr))
    inner = slice(1, n - 1)                                              # f(t_0 = d) = 0
    ratio = v[inner] / f[inner]
    # rtol 1e-7: with a delay the implementation may evaluate the formula at t_j - d, which loses digits where
    # t_j - d << d (float cancellation, not a contract matter)
    assert np.allclose(ratio, np.median(ratio), rtol=1e-7, atol=0), (ratio.min(), ratio.max())
    assert abs(v[0]) <= 1e-12 * np.max(v)


@pytest.mark.parametrize("kw", [dict(charge_C=0.0), dict(charge_C=-1e-15), dict(tau_rise_s=0.0),
                                dict(tau_rise_s=-1e-12), dict(tau_rise_s=10e-12), dict(tau_rise_s=11e-12),
                                dict(delay_s=-1e-15), dict(n_points=49), dict(n_points=5001), dict(n_points=0)])
def test_double_exponential_value_errors(kw):
    args = dict(charge_C=2e-15, tau_rise_s=1e-12, tau_fall_s=10e-12, delay_s=0.0, n_points=401)
    args.update(kw)
    with pytest.raises(ValueError):
        _S().double_exponential(args["charge_C"], args["tau_rise_s"], args["tau_fall_s"], delay_s=args["delay_s"],
                                n_points=args["n_points"])


def test_double_exponential_tau_fall_nonpositive():
    with pytest.raises(ValueError):
        _S().double_exponential(2e-15, 1e-12, 0.0)
    with pytest.raises(ValueError):
        _S().double_exponential(2e-15, -2e-12, -1e-12)


def test_double_exponential_bounds_accepted():
    S = _S()
    assert len(S.double_exponential(2e-15, 1e-12, 10e-12, n_points=50).points) == 50
    assert len(S.double_exponential(2e-15, 1e-12, 10e-12, n_points=5000).points) == 5000
    assert len(S.double_exponential(2e-15, 1e-12, 10e-12, delay_s=0.0, n_points=401).points) == 401


# ------------------------------------------------------------------------------------------------ inject
def _pulse():
    return _S().double_exponential(2e-15, TAU_R, TAU_F, n_points=401)


def _by_name(p):
    return {e.name.lower(): e for e in p.elements}


def test_inject_discharge_default():
    from simws.spec.circuit import CircuitProblem
    base = rc_node(10e-15, 1e4)
    before = base.model_dump(mode="json")
    pulse = _pulse()
    out = _S().inject(base, "n", pulse)
    assert isinstance(out, CircuitProblem)
    assert out is not base
    assert base.model_dump(mode="json") == before                         # a copy: the input is not changed
    els = _by_name(out)
    assert len(out.elements) == len(base.elements) + 1
    for e in base.elements:                                               # originals kept unchanged
        assert els[e.name.lower()].model_dump(mode="json") == e.model_dump(mode="json")
    s = els["set"]
    assert s.name == "set" and s.kind == "isource"
    assert [x.lower() for x in s.nodes] == ["n", "0"]                     # from the node to ground
    assert (s.value or 0.0) == 0.0                                        # DC 0
    assert s.pulse is None and s.ac is None
    assert s.pwl is not None and np.array_equal(np.asarray(s.pwl.points), np.asarray(pulse.points))
    assert out.probes == base.probes                                      # v(n) already there
    for f in ("analysis", "models", "libraries", "numerics", "temperature_C", "tnom_C", "title"):
        assert getattr(out, f) == getattr(base, f), f


def test_inject_charge_polarity_and_name():
    base = rc_node(10e-15, 1e4)
    out = _S().inject(base, "n", _pulse(), polarity="charge", name="strike")
    els = _by_name(out)
    assert "set" not in els
    s = els["strike"]
    assert s.kind == "isource" and [x.lower() for x in s.nodes] == ["0", "n"]
    assert (s.value or 0.0) == 0.0


def test_inject_defaults():
    sig = inspect.signature(_S().inject).parameters
    assert sig["polarity"].default == "discharge" and sig["name"].default == "set"


def test_inject_adds_probe_when_missing():
    base = rc_node(10e-15, 1e4).with_changes({"probes": ["v(vdd)", "i(vdd)"]})
    out = _S().inject(base, "n", _pulse())
    assert len(out.probes) == 3
    assert out.probes[:2] == ["v(vdd)", "i(vdd)"]
    assert "v(n)" in out.probes
    assert base.probes == ["v(vdd)", "i(vdd)"]
    # probe not duplicated when present
    base2 = rc_node(10e-15, 1e4).with_changes({"probes": ["v(vdd)", "v(n)"]})
    assert _S().inject(base2, "n", _pulse()).probes == ["v(vdd)", "v(n)"]


def test_inject_other_node():
    base = rc_node(10e-15, 1e4)
    out = _S().inject(base, "vdd", _pulse())
    assert [x.lower() for x in _by_name(out)["set"].nodes] == ["vdd", "0"]
    assert "v(vdd)" in out.probes and "v(n)" in out.probes


def test_inject_not_transient():
    from simws.spec import library
    with pytest.raises(ValueError):
        _S().inject(library.divider(), "mid", _pulse())
    with pytest.raises(ValueError):
        _S().inject(library.rc_lowpass(analysis="ac"), "out", _pulse())


def test_inject_unknown_node():
    with pytest.raises(ValueError):
        _S().inject(rc_node(10e-15, 1e4), "x", _pulse())


def test_inject_name_taken():
    with pytest.raises(ValueError):
        _S().inject(rc_node(10e-15, 1e4), "n", _pulse(), name="r1")
    with pytest.raises(ValueError):
        _S().inject(rc_node(10e-15, 1e4), "n", _pulse(), name="C1")      # names are case-insensitive (A9.1)
    once = _S().inject(rc_node(10e-15, 1e4), "n", _pulse())
    with pytest.raises(ValueError):
        _S().inject(once, "n", _pulse())                                  # "set" already there


def test_inject_node_case_insensitive():
    """A9.1: nodes are case-insensitive, so "N" is in the circuit."""
    out = _S().inject(rc_node(10e-15, 1e4), "N", _pulse())
    assert [x.lower() for x in _by_name(out)["set"].nodes] == ["n", "0"]


# ------------------------------------------------------------------------------------------------ critical_charge, fake hub
def _problem_of(params):
    from simws.spec.circuit import CircuitProblem
    p = params["problem"]
    return CircuitProblem.model_validate(p.model_dump(mode="json") if hasattr(p, "model_dump") else p)


def _injected_charge(p, node="n", name="set"):
    """Charge the source `name` puts INTO `node` (negative for discharge), from its PWL."""
    e = _by_name(p)[name]
    t, v = _pts(e.pwl)
    Q = _trapz(t, v)
    nodes = [x.lower() for x in e.nodes]
    if nodes == [node, "0"]:
        return -Q
    if nodes == ["0", node]:
        return Q
    raise AssertionError(f"injected source on nodes {e.nodes}")


class FakeHub:
    """Stand-in for the hub: submit() returns a ResultEnvelope whose v(n) waveform has its extreme at
    extreme(Q_removed) (discharge) or extreme(Q_added) (charge), chosen by the test."""

    def __init__(self, extreme, node="n", fail_at=None, message="solver blew up"):
        self.extreme, self.node, self.fail_at, self.message = extreme, node, fail_at, message
        self.calls = []          # (adapter name, problem, charge into the node)
        self.ids = []

    def submit(self, name, params=None, *a, **k):
        from simws.hub.schema import Axis, Quantity, ResultEnvelope
        p = _problem_of(params)
        qin = _injected_charge(p, self.node)
        self.calls.append((name, p, qin))
        n = len(self.calls)
        rid = f"fake-result-{n:03d}"
        self.ids.append(rid)
        if self.fail_at is not None and n == self.fail_at:
            return ResultEnvelope(id=rid, job_id=f"fake-job-{n:03d}", status="not_converged", message=self.message)
        ext = float(self.extreme(qin))
        base = 1.0 if qin <= 0 else 0.0
        t = [0.0, 1e-11, 2e-11, 5e-11, 3e-10]
        v = [base, 0.5 * (base + ext), ext, 0.5 * (base + ext), base]
        qs = [Quantity(name=f"circuit.v.{self.node}", value=v, unit="V", axes=[Axis(name="t", unit="s", values=t)]),
              Quantity(name="circuit.n_points", value=float(len(t)), unit="dimensionless")]
        return ResultEnvelope(id=rid, job_id=f"fake-job-{n:03d}", status="ok", quantities=qs, flags=["budget:none"])


def _bisect_reference(upset, q_low, q_high, rel_tol):
    """Arithmetic bisection as A10.1 states it; returns (q_lo, q_hi, runs, trial charges)."""
    trials = [q_low, q_high]
    lo, hi = q_low, q_high
    while (hi - lo) / hi > rel_tol:
        q = 0.5 * (lo + hi)
        trials.append(q)
        if upset(q):
            hi = q
        else:
            lo = q
    return lo, hi, len(trials), trials


QC_FAKE = 6.789123e-15                     # upset at and above this charge (fake model)


def _linear_vmin(Q_removed):               # v_min = 1 - Q / (2 QC): v_min <= 0.5 exactly when Q >= QC
    return 1.0 - Q_removed / (2.0 * QC_FAKE)


def test_critical_charge_fake_hub_discharge():
    S = _S()
    hub = FakeHub(lambda qin: _linear_vmin(-qin))
    base = rc_node(10e-15, 1e4)
    out = S.critical_charge(hub, base, "n", 0.5, TAU_R, TAU_F, 1e-15, 20e-15)
    lo, hi, runs, trials = _bisect_reference(lambda q: q >= QC_FAKE, 1e-15, 20e-15, 1e-3)
    print(f"\n[critical_charge fake hub] qcrit {out['qcrit_C']:.6e} (model {QC_FAKE:.6e}), runs {out['runs']} "
          f"(reference bisection {runs})")
    for key in ("qcrit_C", "q_lo_C", "q_hi_C", "runs", "result_ids", "node", "threshold_V", "polarity",
                "tau_rise_s", "tau_fall_s"):
        assert key in out, key
    assert out["q_lo_C"] < QC_FAKE <= out["q_hi_C"]
    assert (out["q_hi_C"] - out["q_lo_C"]) / out["q_hi_C"] <= 1e-3
    assert out["qcrit_C"] == pytest.approx(0.5 * (out["q_lo_C"] + out["q_hi_C"]), rel=1e-15)
    assert out["runs"] == len(hub.calls)
    assert sorted(out["result_ids"]) == sorted(hub.ids) and len(out["result_ids"]) == out["runs"]
    assert out["node"] == "n" and out["threshold_V"] == 0.5 and out["polarity"] == "discharge"
    assert out["tau_rise_s"] == TAU_R and out["tau_fall_s"] == TAU_F
    assert all(c[0] == CIRCUIT for c in hub.calls)
    # first q_low, then q_high
    q_seen = [-c[2] for c in hub.calls]
    assert q_seen[0] == pytest.approx(1e-15, rel=1e-12) and q_seen[1] == pytest.approx(20e-15, rel=1e-12)


def test_critical_charge_bisection_sequence():
    """Arithmetic bisection (the midpoint of the bracket) and the stopping rule (q_hi - q_lo) / q_hi <= rel_tol:
    same trial charges, same number of runs, same final bracket as the reference bisection."""
    S = _S()
    for rel_tol in (1e-3, 1e-2, 3e-5):
        hub = FakeHub(lambda qin: _linear_vmin(-qin))
        out = S.critical_charge(hub, rc_node(10e-15, 1e4), "n", 0.5, TAU_R, TAU_F, 1e-15, 20e-15, rel_tol=rel_tol)
        lo, hi, runs, trials = _bisect_reference(lambda q: q >= QC_FAKE, 1e-15, 20e-15, rel_tol)
        assert out["runs"] == runs, (rel_tol, out["runs"], runs)
        assert out["q_lo_C"] == pytest.approx(lo, rel=1e-12) and out["q_hi_C"] == pytest.approx(hi, rel=1e-12)
        assert np.allclose([-c[2] for c in hub.calls], trials, rtol=1e-12, atol=0)
        assert (out["q_hi_C"] - out["q_lo_C"]) / out["q_hi_C"] <= rel_tol


def test_critical_charge_trials_are_injected_double_exponentials():
    S = _S()
    hub = FakeHub(lambda qin: _linear_vmin(-qin))
    base = rc_node(10e-15, 1e4)
    d = 3e-12
    S.critical_charge(hub, base, "n", 0.5, 2e-12, 15e-12, 1e-15, 20e-15, delay_s=d, rel_tol=1e-2)
    for name, p, qin in hub.calls:
        exp = S.inject(base, "n", S.double_exponential(-qin, 2e-12, 15e-12, delay_s=d))
        got_e, exp_e = _by_name(p), _by_name(exp)
        assert sorted(got_e) == sorted(exp_e)
        assert np.allclose(np.asarray(got_e["set"].pwl.points), np.asarray(exp_e["set"].pwl.points),
                           rtol=1e-12, atol=0)
        assert [x.lower() for x in got_e["set"].nodes] == ["n", "0"]
        assert p.analysis == base.analysis and "v(n)" in p.probes


def test_critical_charge_threshold_is_inclusive_discharge():
    """discharge: an upset is min v(node) <= threshold, so a minimum exactly at the threshold upsets."""
    S = _S()
    hub = FakeHub(lambda qin: 0.5 if -qin >= 14.9e-15 else 0.9)
    out = S.critical_charge(hub, rc_node(10e-15, 1e4), "n", 0.5, TAU_R, TAU_F, 1e-15, 15e-15)
    assert out["q_lo_C"] < 14.9e-15 <= out["q_hi_C"] <= 15e-15 * (1 + 1e-12)
    hub = FakeHub(lambda qin: 0.5)                                         # q_low reaches the threshold
    with pytest.raises(ValueError):
        S.critical_charge(hub, rc_node(10e-15, 1e4), "n", 0.5, TAU_R, TAU_F, 1e-15, 15e-15)


def test_critical_charge_charge_polarity_fake():
    S = _S()
    qc = 4.4321e-15
    hub = FakeHub(lambda qin: qin / (2.0 * qc))                          # max v = Q / (2 qc): >= 0.5 when Q >= qc
    out = S.critical_charge(hub, rc_low(10e-15, 1e4), "n", 0.5, TAU_R, TAU_F, 1e-15, 20e-15, polarity="charge")
    assert out["polarity"] == "charge"
    assert all(c[2] > 0 for c in hub.calls)                               # current from "0" into the node
    assert out["q_lo_C"] < qc <= out["q_hi_C"]
    assert (out["q_hi_C"] - out["q_lo_C"]) / out["q_hi_C"] <= 1e-3
    # threshold inclusive for charge: max v >= threshold
    hub = FakeHub(lambda qin: 0.5)
    with pytest.raises(ValueError):
        S.critical_charge(hub, rc_low(10e-15, 1e4), "n", 0.5, TAU_R, TAU_F, 1e-15, 20e-15, polarity="charge")


def test_critical_charge_bracket_errors():
    S = _S()
    hub = FakeHub(lambda qin: _linear_vmin(-qin))
    with pytest.raises(ValueError):                                       # q_low upsets
        S.critical_charge(hub, rc_node(10e-15, 1e4), "n", 0.5, TAU_R, TAU_F, 7e-15, 20e-15)
    hub = FakeHub(lambda qin: _linear_vmin(-qin))
    with pytest.raises(ValueError):                                       # q_high does not upset
        S.critical_charge(hub, rc_node(10e-15, 1e4), "n", 0.5, TAU_R, TAU_F, 1e-15, 6e-15)
    hub = FakeHub(lambda qin: 0.0)                                        # everything upsets
    with pytest.raises(ValueError):
        S.critical_charge(hub, rc_node(10e-15, 1e4), "n", 0.5, TAU_R, TAU_F, 1e-15, 6e-15)
    hub = FakeHub(lambda qin: 1.0)                                        # nothing upsets
    with pytest.raises(ValueError):
        S.critical_charge(hub, rc_node(10e-15, 1e4), "n", 0.5, TAU_R, TAU_F, 1e-15, 6e-15)


@pytest.mark.parametrize("fail_at", [1, 2, 5])
def test_critical_charge_failed_job(fail_at):
    S = _S()
    hub = FakeHub(lambda qin: _linear_vmin(-qin), fail_at=fail_at, message="solver blew up at step 17")
    with pytest.raises(RuntimeError) as ei:
        S.critical_charge(hub, rc_node(10e-15, 1e4), "n", 0.5, TAU_R, TAU_F, 1e-15, 20e-15)
    assert "solver blew up at step 17" in str(ei.value)
    assert len(hub.calls) == fail_at


def test_critical_charge_defaults():
    sig = inspect.signature(_S().critical_charge).parameters
    assert sig["delay_s"].default == 0.0 and sig["polarity"].default == "discharge"
    assert sig["rel_tol"].default == 1e-3 and sig["max_runs"].default == 60


def test_critical_charge_max_runs_not_exceeded():
    """max_runs caps the hub jobs (what happens at the cap is not stated; only the cap is tested)."""
    S = _S()
    hub = FakeHub(lambda qin: _linear_vmin(-qin))
    try:
        S.critical_charge(hub, rc_node(10e-15, 1e4), "n", 0.5, TAU_R, TAU_F, 1e-15, 20e-15, max_runs=5)
    except Exception:                                                     # noqa: BLE001
        pass
    assert len(hub.calls) <= 5, len(hub.calls)


# ------------------------------------------------------------------------------------------------ critical_charge, real hub + fake ngspice
@pytest.fixture
def fake_ngspice(monkeypatch):
    """The ngspice module's run() computes v(n) from the A10.4 closed form for the injected charge."""
    from simws import engines
    from simws.engines.base import CircuitResult
    mod = engines.module("ngspice")
    calls = []

    def install(C, R, fail=None):
        def run(problem, workdir=None, *a, **k):
            from simws.spec.circuit import CircuitProblem
            p = problem if isinstance(problem, CircuitProblem) else CircuitProblem.model_validate(problem)
            calls.append(p)
            if fail is not None and len(calls) == fail:
                raise RuntimeError("fake ngspice: timestep too small")
            Q = -_injected_charge(p)
            t = np.linspace(0.0, 300e-12, 3001)
            v = 1.0 - droop(t, Q, C, R)
            sig = {pr: (v.tolist() if pr.lower() == "v(n)" else [1.0] * t.size) for pr in p.probes}
            return CircuitResult(engine="ngspice", problem_id=p.problem_id(), analysis="tran", axis_name="t",
                                 axis_unit="s", axis=t.tolist(), signals=sig, units={k: "V" for k in sig})
        monkeypatch.setattr(mod, "run", run)
        monkeypatch.setattr(mod, "available", lambda: True)
        monkeypatch.setattr(mod, "openvaf_available", lambda: True)
        return calls
    return install


def test_critical_charge_real_hub_fake_engine(hub, fake_ngspice):
    S = _S()
    calls = fake_ngspice(10e-15, 1e4)
    out = S.critical_charge(hub, rc_node(10e-15, 1e4), "n", 0.5, TAU_R, TAU_F, 1e-15, 20e-15)
    ref = qcrit_closed(10e-15, 1e4)
    print(f"\n[critical_charge real hub, fake engine] qcrit {out['qcrit_C']:.6e} closed form {ref:.6e}, "
          f"runs {out['runs']}")
    assert out["runs"] == len(calls) == len(out["result_ids"])
    for rid in out["result_ids"]:
        assert hub.store.has(rid)
        assert hub.store.job(rid).key.adapter == CIRCUIT
    assert out["q_lo_C"] <= ref * (1 + 1e-9) and out["q_hi_C"] >= ref * (1 - 1e-9)
    assert abs(out["qcrit_C"] / ref - 1) <= 1e-3


def test_critical_charge_real_hub_failed_job(hub, fake_ngspice):
    S = _S()
    fake_ngspice(10e-15, 1e4, fail=3)
    with pytest.raises(RuntimeError) as ei:
        S.critical_charge(hub, rc_node(10e-15, 1e4), "n", 0.5, TAU_R, TAU_F, 1e-15, 20e-15)
    assert "timestep too small" in str(ei.value)


# ------------------------------------------------------------------------------------------------ upset_probability
@pytest.fixture
def see_store(hub, monkeypatch):
    """make(counts, under, over, low_MeV, high_MeV, N) -> a stored see.g4see ResultEnvelope of that histogram."""
    from simws import engines
    from simws.engines.base import SeeResult
    from simws.spec.library import see_si_slab
    mod = engines.module("g4see")
    state = {}

    def run(problem, workdir=None, *a, **k):
        from simws.spec import SeeProblem
        p = problem if isinstance(problem, SeeProblem) else SeeProblem.model_validate(problem)
        h = p.histogram
        edges = (h.low_MeV + np.arange(h.bins + 1) * (h.high_MeV - h.low_MeV) / h.bins).tolist()
        return SeeResult(engine="g4see", problem_id=p.problem_id(), n_events=p.statistics.n_events, edges_MeV=edges,
                         counts=list(map(float, state["counts"])), underflow=float(state["under"]),
                         overflow=float(state["over"]), flags=["see:em_only (fake engine for tests)"])
    monkeypatch.setattr(mod, "run", run)
    monkeypatch.setattr(mod, "available", lambda: True)

    def make(counts, under=0, over=0, low_MeV=0.0, high_MeV=1.0, seed=1):
        counts = list(counts)
        N = int(sum(counts) + under + over)
        state.update(counts=counts, under=under, over=over)
        p = see_si_slab("alpha", 5.0, n_events=N, bins=len(counts), high_MeV=high_MeV, seed=seed)
        p = p.with_changes({"histogram.low_MeV": low_MeV})
        r = hub.submit(SEE, params={"problem": p.model_dump(mode="json")})
        assert r.status == "ok", r.message
        return r, p
    return make


def _qdict(E_crit, E_lo=None, E_hi=None, pair=3.66):
    """qcrit dict as critical_charge returns it, with q from energies (eV). E_lo <-> q_lo, E_hi <-> q_hi."""
    q = E_crit / pair * Q_E
    q_lo = (E_lo if E_lo is not None else 0.999 * E_crit) / pair * Q_E
    q_hi = (E_hi if E_hi is not None else 1.001 * E_crit) / pair * Q_E
    return {"qcrit_C": q, "q_lo_C": q_lo, "q_hi_C": q_hi, "runs": 12, "result_ids": [], "node": "n",
            "threshold_V": 0.5, "polarity": "discharge", "tau_rise_s": TAU_R, "tau_fall_s": TAU_F}


def _p_interp(r, E):
    q = r.q("see.p_above")
    return float(np.interp(E, np.asarray(q.axes[0].values, float), q.array()))


COUNTS10 = [50, 120, 80, 200, 150, 100, 90, 60, 70, 40]                  # 960 + 40 overflow = 1000 events


def test_upset_probability_interpolation(see_store):
    S = _S()
    r, _ = see_store(COUNTS10, under=0, over=40, low_MeV=0.0, high_MeV=1.0)
    N = 1000
    edges = np.linspace(0.0, 1e6, 11)
    k = np.cumsum(np.asarray(COUNTS10, float)[::-1])[::-1] + 40
    for E, E_lo, E_hi in [(0.37e6, 0.30e6, 0.45e6), (0.1e6, 0.05e6, 0.15e6), (0.555e6, 0.555e6, 0.5551e6),
                          (0.05e6, 0.01e6, 0.86e6), (0.8999e6, 0.5e6, 0.8999e6)]:
        out = S.upset_probability(r, _qdict(E, E_lo, E_hi))
        # the exact value for events spread evenly in each bin
        i = int(np.searchsorted(edges, E, side="right") - 1)
        P_exact = (k[i] - COUNTS10[i] * (E - edges[i]) / (edges[i + 1] - edges[i])) / N
        P = _p_interp(r, E)
        assert P == pytest.approx(P_exact, rel=1e-9, abs=1e-12)
        for key in ("P", "P_low", "P_high", "sigma", "E_crit_eV", "n_events", "flags"):
            assert key in out, key
        assert out["E_crit_eV"] == pytest.approx(E, rel=1e-12)
        assert out["n_events"] == N
        assert out["P"] == pytest.approx(P, rel=1e-9, abs=1e-12), (E, out["P"], P)
        assert out["P_low"] == pytest.approx(_p_interp(r, E_hi), rel=1e-9, abs=1e-12)    # at edep_eV(q_hi)
        assert out["P_high"] == pytest.approx(_p_interp(r, E_lo), rel=1e-9, abs=1e-12)   # at edep_eV(q_lo)
        assert out["P_low"] <= out["P"] <= out["P_high"]
        p = (out["P"] * N + 1) / (N + 2)
        assert out["sigma"] == pytest.approx(math.sqrt(p * (1 - p) / N), rel=1e-9)
        assert not any(f.startswith("upset:outside_histogram") for f in out["flags"]), out["flags"]


def test_upset_probability_at_edges(see_store):
    S = _S()
    r, _ = see_store(COUNTS10, under=0, over=40, low_MeV=0.0, high_MeV=1.0)
    k = np.cumsum(np.asarray(COUNTS10, float)[::-1])[::-1] + 40
    for i in (1, 4, 9):
        E = i * 1e5
        out = S.upset_probability(r, _qdict(E))
        assert out["P"] == pytest.approx(k[i] / 1000, rel=1e-9), (i, out["P"])


def test_upset_probability_last_bin(see_store):
    """Between the last p_above edge and high_MeV, "the exact value when the events are spread evenly inside a
    bin" ends at the overflow fraction at high_MeV (inferred from A10.1; p_above's axis stops at edges[bins-1])."""
    S = _S()
    r, _ = see_store(COUNTS10, under=0, over=40, low_MeV=0.0, high_MeV=1.0)
    E = 0.95e6
    exact = (40 + 40 * 0.5) / 1000
    out = S.upset_probability(r, _qdict(E, 0.92e6, 0.97e6))
    print(f"\n[upset_probability last bin] P {out['P']:.6f} (evenly spread: {exact:.6f}), flags {out['flags']}")
    assert out["P"] == pytest.approx(exact, rel=1e-9)
    assert out["P_high"] == pytest.approx((40 + 40 * 0.8) / 1000, rel=1e-9)
    assert out["P_low"] == pytest.approx((40 + 40 * 0.3) / 1000, rel=1e-9)
    assert not any(f.startswith("upset:outside_histogram") for f in out["flags"])


def test_upset_probability_outside_below(see_store):
    S = _S()
    r, _ = see_store(COUNTS10, under=25, over=40, low_MeV=0.5, high_MeV=1.5)
    N = 1025
    out = S.upset_probability(r, _qdict(0.2e6, 0.1e6, 0.3e6))
    assert out["P"] == 1.0
    assert any(f.startswith("upset:outside_histogram") for f in out["flags"]), out["flags"]
    assert out["P_low"] == 1.0 and out["P_high"] == 1.0
    assert out["n_events"] == N
    p = (1.0 * N + 1) / (N + 2)
    assert out["sigma"] == pytest.approx(math.sqrt(p * (1 - p) / N), rel=1e-9)
    # E_crit inside, E(q_lo) below the range: P_high follows the same rule (1 below the range)
    out = S.upset_probability(r, _qdict(0.6e6, 0.3e6, 0.7e6))
    assert out["P_high"] == 1.0
    assert out["P"] == pytest.approx(_p_interp(r, 0.6e6), rel=1e-9)


@pytest.mark.parametrize("E_lo,E_hi", [(0.3e6, 0.7e6), (0.55e6, 1.7e6), (0.3e6, 1.7e6)])
def test_upset_probability_flag_only_from_E_crit(see_store, E_lo, E_hi):
    """A10.1: the flag is set "when E_crit is outside the histogram's range"; here E_crit = 0.6 MeV is inside
    [0.5, 1.5] MeV and only edep_eV(q_lo) and/or edep_eV(q_hi) is outside."""
    S = _S()
    r, _ = see_store(COUNTS10, under=25, over=40, low_MeV=0.5, high_MeV=1.5)
    out = S.upset_probability(r, _qdict(0.6e6, E_lo, E_hi))
    print(f"\n[upset flag, E_crit inside, E_lo {E_lo:.3g} E_hi {E_hi:.3g}] flags {out['flags']}")
    assert not any(f.startswith("upset:outside_histogram") for f in out["flags"]), out["flags"]


def test_upset_probability_outside_above(see_store):
    S = _S()
    r, _ = see_store(COUNTS10, under=25, over=40, low_MeV=0.5, high_MeV=1.5)
    N = 1025
    out = S.upset_probability(r, _qdict(2.0e6, 1.8e6, 2.2e6))
    assert out["P"] == pytest.approx(40 / N, rel=1e-9)
    assert out["P_low"] == pytest.approx(40 / N, rel=1e-9) and out["P_high"] == pytest.approx(40 / N, rel=1e-9)
    assert any(f.startswith("upset:outside_histogram") for f in out["flags"]), out["flags"]
    p = (40.0 + 1) / (N + 2)
    assert out["sigma"] == pytest.approx(math.sqrt(p * (1 - p) / N), rel=1e-9)


def test_upset_probability_at_low_edge_is_inside(see_store):
    S = _S()
    r, _ = see_store(COUNTS10, under=25, over=40, low_MeV=0.5, high_MeV=1.5)
    E = 0.5e6 + 1.0                       # 1 eV inside: E = 0.5 MeV itself round-trips to 0.49999999999999994e6
    out = S.upset_probability(r, _qdict(E, E, 0.51e6))
    assert out["P"] == pytest.approx((1000 - 50 * 1.0 / 1e5) / 1025, rel=1e-9)     # just past the first edge
    assert not any(f.startswith("upset:outside_histogram") for f in out["flags"])


def test_upset_probability_custom_pair_energy(see_store):
    S = _S()
    r, _ = see_store(COUNTS10, under=0, over=40, low_MeV=0.0, high_MeV=1.0)
    E = 0.43e6
    qd = _qdict(E, pair=3.66)                                             # charges for 3.66 eV per pair
    out = S.upset_probability(r, qd, pair_energy_eV=3.9)
    E39 = qd["qcrit_C"] / Q_E * 3.9
    assert out["E_crit_eV"] == pytest.approx(E39, rel=1e-12)
    assert out["P"] == pytest.approx(_p_interp(r, E39), rel=1e-9)
    assert out["P_low"] == pytest.approx(_p_interp(r, qd["q_hi_C"] / Q_E * 3.9), rel=1e-9)
    assert out["P_high"] == pytest.approx(_p_interp(r, qd["q_lo_C"] / Q_E * 3.9), rel=1e-9)
    sig = inspect.signature(S.upset_probability).parameters
    assert sig["pair_energy_eV"].default == S.PAIR_ENERGY_SI_EV


# ------------------------------------------------------------------------------------------------ acceptance R40 - R44
def _need_ngspice():
    try:
        from simws.engines import ngspice_engine
        ok = ngspice_engine.available()
    except Exception as e:                                                # noqa: BLE001
        pytest.skip(f"ngspice engine module not importable: {e!r}")
    if not ok:
        pytest.skip("ngspice not available (SIMWS_SPICE_PREFIX / <envs_root>/spice)")


def _need_g4see():
    try:
        from simws.engines import g4see_engine
        ok = g4see_engine.available()
    except Exception as e:                                                # noqa: BLE001
        pytest.skip(f"g4see engine module not importable: {e!r}")
    if not ok:
        pytest.skip("G4SEE not available (SIMWS_G4SEE_PREFIX / <envs_root>/g4see)")


@pytest.fixture(scope="module")
def rhub(tmp_path_factory):
    from simws.hub.jobs import Hub
    return Hub(tmp_path_factory.mktemp("chain_see_acceptance") / "store", mode="inprocess")


def _report(label, measured, reference, diff, gate, unit=""):
    ok = diff <= gate
    print(f"\n[{label}] measured {measured:.7g}{unit}, reference {reference:.7g}{unit}, difference {diff:.3e} "
          f"against gate {gate:.3e}: {'PASS' if ok else 'FAIL'}")
    return ok


def test_R40_conversion():
    S = _S()
    q = S.charge_C(1e6)
    ref = 1e6 / 3.66 * 1.602176634e-19
    a = _report("R40 charge_C(1e6) (relative)", q, ref, abs(q / ref - 1), 1e-12, " C")
    print(f"[R40] contract's printed value {REF_R40_C:.6g} C, measured {q:.6g} C")
    worst = 0.0
    for x in (1.0, 1e3, 1e6, 1.61e6, 5e6, 1e9):
        worst = max(worst, abs(S.edep_eV(S.charge_C(x)) / x - 1))
    xs = np.array([1.0, 1e3, 1e6, 1.61e6, 5e6, 1e9])
    worst = max(worst, float(np.max(np.abs(np.asarray(S.edep_eV(S.charge_C(xs)), float) / xs - 1))))
    b = _report("R40 edep_eV(charge_C(x)) (relative, worst)", worst, 0.0, worst, 1e-12)
    assert a and b and f"{q:.6g}" == f"{REF_R40_C:.6g}"


def test_R41_pulse():
    S = _S()
    Q = 2e-15
    p = S.double_exponential(Q, TAU_R, TAU_F)
    t, v = _pts(p)
    integ = _trapz(t, v)
    i = int(np.argmax(v))
    a = _report("R41 trapezoidal integral (relative)", integ, Q, abs(integ / Q - 1), 1e-12, " C")
    b = _report("R41 peak value (relative)", v[i], REF_R41_PEAK_A, abs(v[i] / REF_R41_PEAK_A - 1), 1e-2, " A")
    c = _report("R41 peak time (relative)", t[i], REF_R41_PEAK_T, abs(t[i] / REF_R41_PEAK_T - 1), 2e-2, " s")
    print(f"[R41] {t.size} points")
    assert a and b and c


def _vmin(r, node="n"):
    return float(np.min(r.q(f"circuit.v.{node}").array()))


def test_R42_charge_reaches_node(rhub):
    _need_ngspice()
    S = _S()
    p = S.inject(rc_node(10e-15, 1e4), "n", S.double_exponential(2e-15, TAU_R, TAU_F))
    r = rhub.submit(CIRCUIT, params={"problem": p.model_dump(mode="json")})
    assert r.status == "ok", (r.status, r.message)
    vmin = _vmin(r)
    d = 1.0 - vmin
    print(f"\n[R42] min v(n) = {vmin:.7f} V (closed form {REF_R42_VMIN} V); droop {d:.7f} V "
          f"(closed form {REF_R42_DROOP} V), relative {d / REF_R42_DROOP - 1:+.3e}")
    assert _report("R42 droop", d, REF_R42_DROOP, abs(d - REF_R42_DROOP), 0.01 * REF_R42_DROOP, " V")


def test_R43_critical_charge(rhub):
    _need_ngspice()
    S = _S()
    out = S.critical_charge(rhub, rc_node(10e-15, 1e4), "n", 0.5, TAU_R, TAU_F, 1e-15, 20e-15)
    qc = out["qcrit_C"]
    width = (out["q_hi_C"] - out["q_lo_C"]) / out["q_hi_C"]
    print(f"\n[R43] qcrit {qc:.7e} C, bracket [{out['q_lo_C']:.7e}, {out['q_hi_C']:.7e}] C, {out['runs']} runs, "
          f"relative to the closed form {qc / REF_R43_QCRIT - 1:+.3e}")
    a = _report("R43 qcrit (relative)", qc, REF_R43_QCRIT, abs(qc / REF_R43_QCRIT - 1), 1e-2, " C")
    b = _report("R43 (q_hi - q_lo) / q_hi", width, 0.0, width, 1e-3)
    assert out["runs"] == len(out["result_ids"])
    assert a and b


def test_checker_charge_polarity_real_engine(rhub):
    """Checker's own check (not an A10.5 case): the "charge" polarity on node n held at 0 V through R; the closed
    form of A10.4 is the same with u = v; same 1 % as R43."""
    _need_ngspice()
    S = _S()
    out = S.critical_charge(rhub, rc_low(10e-15, 1e4), "n", 0.5, TAU_R, TAU_F, 1e-15, 20e-15, polarity="charge",
                            rel_tol=1e-2)
    qc = out["qcrit_C"]
    print(f"\n[checker charge polarity] qcrit {qc:.6e} C vs closed form {REF_R43_QCRIT:.6e} C "
          f"({qc / REF_R43_QCRIT - 1:+.3e}), {out['runs']} runs")
    assert abs(qc / REF_R43_QCRIT - 1) <= 1e-2
    r = rhub.submit(CIRCUIT, params={"problem": S.inject(rc_low(10e-15, 1e4), "n",
                                                          S.double_exponential(2e-15, TAU_R, TAU_F),
                                                          polarity="charge").model_dump(mode="json")})
    assert r.status == "ok", r.message
    vmax = float(np.max(r.q("circuit.v.n").array()))
    print(f"[checker charge polarity] max v(n) for 2 fC {vmax:.7f} V (closed form {REF_R42_DROOP})")
    assert abs(vmax - REF_R42_DROOP) <= 0.01 * REF_R42_DROOP


def test_R44_end_to_end(rhub):
    _need_ngspice()
    _need_g4see()
    from simws.spec.library import see_si_slab
    S = _S()
    p = see_si_slab("alpha", 5.0, sv_thickness_um=10.0)
    rs = rhub.submit(SEE, params={"problem": p.model_dump(mode="json")})
    assert rs.status == "ok", (rs.status, rs.message)
    mean = rs.q("see.edep_mean").value if any(q.name == "see.edep_mean" for q in rs.quantities) else float("nan")
    print(f"\n[R44] see.g4see: {p.statistics.n_events} alphas, mean deposit {mean / 1e6:.5f} MeV "
          f"= {S.charge_C(mean) * 1e15:.2f} fC (contract: about 1.61 MeV, 70.6 fC)")
    cases = [((60e-15, 1e5), 10e-15, 100e-15, "ge"), ((200e-15, 1e5), 50e-15, 300e-15, "le")]
    oks = []
    for (C, R), ql, qh, kind in cases:
        qd = S.critical_charge(rhub, rc_node(C, R), "n", 0.5, TAU_R, TAU_F, ql, qh)
        ref = REF_R44_QCRIT[(C, R)]
        print(f"\n[R44 rc_node({C * 1e15:.0f} fF, {R / 1e3:.0f} kOhm)] qcrit {qd['qcrit_C']:.7e} C, closed form "
              f"{ref:.7e} C (relative {qd['qcrit_C'] / ref - 1:+.3e}, reported, not gated), {qd['runs']} runs")
        up = S.upset_probability(rs, qd)
        print(f"[R44 rc_node({C * 1e15:.0f} fF)] E_crit {up['E_crit_eV'] / 1e6:.5f} MeV, P {up['P']:.6f} "
              f"[P_low {up['P_low']:.6f}, P_high {up['P_high']:.6f}], sigma {up['sigma']:.3e}, N {up['n_events']}, "
              f"flags {up['flags']}")
        if kind == "ge":
            a = _report("R44 P (60 fF) >= 0.999", up["P"], 0.999, max(0.0, 0.999 - up["P"]), 0.0)
        else:
            a = _report("R44 P (200 fF) <= 0.001", up["P"], 0.001, max(0.0, up["P"] - 0.001), 0.0)
        b = up["P_low"] <= up["P"] <= up["P_high"]
        print(f"[R44 rc_node({C * 1e15:.0f} fF)] P_low <= P <= P_high: {'PASS' if b else 'FAIL'}")
        oks.append(a and b)
    assert all(oks), oks
