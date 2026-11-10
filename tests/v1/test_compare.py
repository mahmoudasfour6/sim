"""simws.compare (contract section 6.3). Expected worst values are computed by hand in each test."""
import math

import pytest

from simws.hub.schema import Axis, Quantity


def Q(value, unit="V", axis=None, name="y", flags=()):
    axes = [] if axis is None else [Axis(name=axis[0], values=axis[1], unit=axis[2])]
    return Quantity(name=name, value=value, unit=unit, axes=axes, flags=list(flags))


def _cmp(*a, **kw):
    from simws.compare import compare
    return compare(*a, **kw)


# ------------------------------------------------------------------------------------------------ scalars
def test_scalar_relative():
    c = _cmp(Q(1.1, name="ya"), Q(1.0, name="yb"), tolerance=0.15)
    assert c.metric == "relative" and c.kind == "engine_pair"
    assert c.quantity_a == "ya" and c.quantity_b == "yb" and c.a == "" and c.b == ""
    assert c.n_points == 1 and c.worst_at is None and c.window is None
    assert c.worst == pytest.approx(abs(1.1 / 1.0 - 1), rel=1e-12)
    assert c.passed is True and c.inconclusive is False and c.tolerance == 0.15
    c2 = _cmp(Q(1.1), Q(1.0), tolerance=0.05)
    assert c2.passed is False and c2.inconclusive is False


def test_tolerance_bound_included():
    c = _cmp(Q(1.5), Q(1.0), tolerance=0.5)                  # |1.5/1 - 1| = 0.5 exactly in binary
    assert c.worst == 0.5 and c.passed is True


def test_b_converted_to_a_unit():
    c = _cmp(Q(1000.0, unit="mV"), Q(1.0, unit="V"), tolerance=1e-12)
    assert c.worst == pytest.approx(0.0, abs=1e-12) and c.passed
    c2 = _cmp(Q(1.0, unit="A/m**2"), Q(1e-4, unit="A/cm**2"), tolerance=1e-12)
    assert c2.worst == pytest.approx(0.0, abs=1e-12)


def test_different_dimension_refused():
    with pytest.raises(ValueError):
        _cmp(Q(1.0, unit="V"), Q(1.0, unit="A"), tolerance=0.1)


def test_ratio_metric():
    assert _cmp(Q(2.0), Q(1.0), tolerance=2.0, metric="ratio").worst == pytest.approx(2.0)
    assert _cmp(Q(0.5), Q(1.0), tolerance=2.0, metric="ratio").worst == pytest.approx(2.0)
    c = _cmp(Q(-1.0), Q(1.0), tolerance=10.0, metric="ratio")
    assert c.worst == math.inf and c.passed is False
    assert _cmp(Q(0.0), Q(1.0), tolerance=10.0, metric="ratio").worst == math.inf
    assert _cmp(Q(1.0), Q(0.0), tolerance=10.0, metric="ratio").worst == math.inf
    assert _cmp(Q(-2.0), Q(-1.0), tolerance=2.0, metric="ratio").worst == pytest.approx(2.0)


def test_kind_measurement_when_either_side_measured():
    assert _cmp(Q(1.0), Q(1.0, flags=["measured"]), tolerance=0.1).kind == "measurement"
    assert _cmp(Q(1.0, flags=["measured"]), Q(1.0), tolerance=0.1).kind == "measurement"
    assert _cmp(Q(1.0, flags=["sketch"]), Q(1.0, flags=["unverified"]), tolerance=0.1).kind == "engine_pair"


# ------------------------------------------------------------------------------------------------ one axis
A = Q([1.0, 2.0, 4.0, 8.0], axis=("V", [0.0, 1.0, 2.0, 3.0], "V"))
B = Q([1.0, 2.0, 3.0], axis=("V", [0.5, 1.5, 2.5], "V"))
# a's points inside b's range [0.5, 2.5]: x = 1, 2. b (linear) at 1 -> 1.5, at 2 -> 2.5.
# relative: |2/1.5 - 1| = 1/3 at x = 1, |4/2.5 - 1| = 0.6 at x = 2 -> worst 0.6 at 2.


def test_axis_linear_interpolation_and_range():
    # A2.1: the default is now "auto" (log for this all-positive b), so linear is forced for the hand values
    c = _cmp(A, B, tolerance=0.7, interpolation="linear")
    assert c.n_points == 2 and c.interpolation == "linear"
    assert c.worst == pytest.approx(0.6, rel=1e-12) and c.worst_at == pytest.approx(2.0)
    assert c.passed is True
    assert _cmp(A, B, tolerance=0.5, interpolation="linear").passed is False


def test_axis_bounds_included():
    b = Q([1.0, 3.0], axis=("V", [1.0, 2.0], "V"))           # a at 1 and 2 lie exactly on b's bounds
    c = _cmp(A, b, tolerance=1.0)
    # x=1: |2/1 - 1| = 1; x=2: |4/3 - 1| = 1/3
    assert c.n_points == 2 and c.worst == pytest.approx(1.0) and c.worst_at == pytest.approx(1.0)


def test_b_axis_converted_to_a_axis_unit():
    b_mv = Q([1.0, 2.0, 3.0], axis=("V", [500.0, 1500.0, 2500.0], "mV"))
    c = _cmp(A, b_mv, tolerance=0.7, interpolation="linear")
    assert c.n_points == 2 and c.worst == pytest.approx(0.6, rel=1e-12) and c.worst_at == pytest.approx(2.0)


def test_b_values_converted_with_axis():
    b_mv = Q([1000.0, 2000.0, 3000.0], unit="mV", axis=("V", [0.5, 1.5, 2.5], "V"))
    c = _cmp(A, b_mv, tolerance=0.7, interpolation="linear")
    assert c.worst == pytest.approx(0.6, rel=1e-12)


def test_axis_dimension_mismatch_refused():
    b = Q([1.0, 2.0, 3.0], axis=("T", [0.5, 1.5, 2.5], "K"))
    with pytest.raises(ValueError):
        _cmp(A, b, tolerance=0.7)


def test_window():
    c = _cmp(A, B, tolerance=0.5, window=(0.0, 1.5), interpolation="linear")         # only x = 1 left
    assert c.n_points == 1 and c.worst == pytest.approx(1 / 3, rel=1e-12) and c.worst_at == pytest.approx(1.0)
    assert c.passed is True and tuple(c.window) == (0.0, 1.5)
    c2 = _cmp(A, B, tolerance=0.5, window=(2.0, 2.0), interpolation="linear")        # bounds included -> x = 2
    assert c2.n_points == 1 and c2.worst == pytest.approx(0.6, rel=1e-12)


def test_min_points_inconclusive():
    c = _cmp(A, B, tolerance=10.0, min_points=3)
    assert c.n_points == 2 and c.inconclusive is True and c.passed is False


def test_no_overlap():
    b = Q([1.0, 2.0], axis=("V", [10.0, 11.0], "V"))
    c = _cmp(A, b, tolerance=10.0)
    assert c.n_points == 0 and c.worst is None and c.worst_at is None
    assert c.inconclusive is True and c.passed is False


def test_log_interpolation():
    a = Q([10.0], axis=("V", [2.0], "V"))
    b = Q([1.0, 100.0], axis=("V", [1.0, 3.0], "V"))
    # linear in ln|y|: exp((ln 1 + ln 100)/2) = 10 -> exact; linear: (1 + 100)/2 = 50.5 -> |10/50.5 - 1|
    assert _cmp(a, b, tolerance=1e-12, interpolation="log").worst == pytest.approx(0.0, abs=1e-12)
    assert _cmp(a, b, tolerance=1.0, interpolation="linear").worst == pytest.approx(abs(10 / 50.5 - 1), rel=1e-12)
    bn = Q([-1.0, -100.0], axis=("V", [1.0, 3.0], "V"))
    an = Q([-10.0], axis=("V", [2.0], "V"))
    assert _cmp(an, bn, tolerance=1e-12, interpolation="log").worst == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("bvals", [[0.0, 100.0], [-1.0, 100.0]])
def test_log_interpolation_refuses_zero_or_mixed_sign(bvals):
    a = Q([10.0], axis=("V", [2.0], "V"))
    with pytest.raises(ValueError):
        _cmp(a, Q(bvals, axis=("V", [1.0, 3.0], "V")), tolerance=1.0, interpolation="log")


def test_ratio_on_axis_with_sign_change():
    a = Q([1.0, -1.0], axis=("V", [0.0, 1.0], "V"))
    b = Q([1.0, 1.0], axis=("V", [0.0, 1.0], "V"))
    c = _cmp(a, b, tolerance=10.0, metric="ratio")
    assert c.worst == math.inf and c.worst_at == pytest.approx(1.0) and c.passed is False


# ------------------------------------------------------------------------------------------------ stored results
def _sketch(hub, scale, qname="y", xs=(0.0, 1.0, 2.0)):
    def model(scale, qname, xs):
        return {qname: Quantity(name=qname, value=[scale * (1.0 + x) for x in xs], unit="V",
                                axes=[Axis(name="t", unit="s", values=list(xs))])}
    r = hub.sketch(model, params={"scale": scale, "qname": qname, "xs": list(xs)}, name="cmp_model")
    assert r.status == "ok", r.message
    return r


def test_compare_results_and_record(hub):
    from simws.compare import compare_results, record, records_for
    ra, rb = _sketch(hub, 1.1), _sketch(hub, 1.0)
    c = compare_results(hub, ra.id, "y", rb.id, tolerance=0.2)          # name_b defaults to name_a
    assert c.a == ra.id and c.b == rb.id and c.quantity_a == "y" and c.quantity_b == "y"
    assert c.kind == "engine_pair"
    assert c.n_points == 3 and c.worst == pytest.approx(0.1, rel=1e-9) and c.passed
    n_events = len(hub.graph.events())
    record(hub, c)
    e = hub.graph.edges_from(ra.id, "checked_by")
    assert [d for d, _, _ in e] == [rb.id]
    assert e[0][2]["comparisons"] == [c.model_dump()] or e[0][2]["comparisons"] == [c.model_dump(mode="json")]
    assert any(ev["type"] == "compared" for ev in hub.graph.events()[n_events:])
    for rid in (ra.id, rb.id):
        recs = records_for(hub, rid)
        assert len(recs) == 1 and recs[0]["a"] == ra.id and recs[0]["b"] == rb.id and recs[0]["tolerance"] == 0.2
    # same pair + same quantities: replaced
    record(hub, compare_results(hub, ra.id, "y", rb.id, "y", tolerance=0.05))
    recs = records_for(hub, ra.id)
    assert len(recs) == 1 and recs[0]["tolerance"] == 0.05 and recs[0]["passed"] is False


def test_record_keeps_other_quantity_pairs(hub):
    from simws.compare import compare_results, record, records_for
    ra = _sketch(hub, 1.0, qname="y")
    rb = _sketch(hub, 1.0, qname="z")
    record(hub, compare_results(hub, ra.id, "y", rb.id, "z", tolerance=0.1))
    rc = _sketch(hub, 2.0, qname="y")
    record(hub, compare_results(hub, ra.id, "y", rc.id, "y", tolerance=0.1))
    assert len(records_for(hub, ra.id)) == 2
    assert len(records_for(hub, rb.id)) == 1 and len(records_for(hub, rc.id)) == 1
    record(hub, compare_results(hub, ra.id, "y", rb.id, "z", tolerance=0.3))
    recs = records_for(hub, ra.id)
    assert len(recs) == 2 and sorted(r["tolerance"] for r in recs) == [0.1, 0.3]


def test_record_measurement_edge(hub):
    from simws import measured
    from simws.compare import compare_results, record, records_for
    ra = _sketch(hub, 1.0)
    m = measured.ingest(hub, "cmp_ds", ("t", [0.0, 1.0, 2.0], "s"), {"y": ([1.0, 2.0, 3.0], "V")},
                        {"source": "test fixture, not a measurement (tests/v1/test_compare.py)"})
    c = compare_results(hub, ra.id, "y", m.id, tolerance=1e-9)
    assert c.kind == "measurement" and c.passed and c.worst == pytest.approx(0.0, abs=1e-12)
    record(hub, c)
    assert [d for d, _, _ in hub.graph.edges_from(ra.id, "validated_against")] == [m.id]
    assert hub.graph.edges_from(ra.id, "checked_by") == []
    assert len(records_for(hub, m.id)) == 1


def test_record_needs_both_ids(hub):
    from simws.compare import record
    c = _cmp(Q(1.0), Q(1.0), tolerance=0.1)                  # bare quantities: a == b == ""
    with pytest.raises(Exception):
        record(hub, c)


def test_records_for_unknown_result_is_empty(hub):
    from simws.compare import records_for
    assert records_for(hub, "f" * 32) == []


# ================================================================================================ amendment A2
# ------------------------------------------------------------------------------------------------ A2.1 interpolation
def _diode_grid(lo, hi, step):
    n = int(round((hi - lo) / step))
    return [round(lo + k * step, 10) for k in range(n + 1)]


I0, VT = 1e-14, 0.02585                                     # A, V (A2.1)


def _diode(V):
    return [I0 * (math.exp(v / VT) - 1.0) for v in V]


def test_a2_1_reference_case():
    """A2.1 reference case. Expected values derived here: at a midpoint V between b's nodes V -/+ h (h = 0.05 V),
    linear interpolation of b gives I0*(e^(V/Vt)*cosh(h/Vt) - 1), so |a/b - 1| = 1 - (x - 1)/(x*cosh(h/Vt) - 1)
    with x = e^(V/Vt); this grows as x falls, so the worst point is the lowest midpoint, 0.15 V. For large x it
    tends to |1/cosh(h/Vt) - 1|. Tolerances fixed before running: 1e-9 relative on the derived worst; log and auto
    below 0.01 (stated in A2.1)."""
    import numpy as np
    Va, Vb = _diode_grid(0.10, 0.60, 0.05), _diode_grid(0.10, 0.60, 0.10)
    assert len(Va) == 11 and len(Vb) == 6
    a = Q(_diode(Va), unit="A", axis=("V", Va, "V"))
    b = Q(_diode(Vb), unit="A", axis=("V", Vb, "V"))
    h = 0.05
    mids = [v for v in Va if v not in Vb]
    rel = {v: 1.0 - (math.exp(v / VT) - 1.0) / (math.exp(v / VT) * math.cosh(h / VT) - 1.0) for v in mids}
    expected_at = max(rel, key=rel.get)
    expected = rel[expected_at]
    # the same number by a direct linear interpolation (independent of the closed form)
    direct = np.abs(np.asarray(_diode(Va)) / np.interp(Va, Vb, _diode(Vb)) - 1.0)
    assert direct.max() == pytest.approx(expected, rel=1e-9) and Va[int(direct.argmax())] == expected_at == 0.15
    # A2.1 states 0.7175 and the large-current limit |1/cosh(0.05/Vt) - 1|
    assert round(expected, 4) == 0.7175
    assert abs(expected - abs(1.0 / math.cosh(h / VT) - 1.0)) < 1e-3

    lin = _cmp(a, b, tolerance=1.0, interpolation="linear")
    assert lin.interpolation == "linear" and lin.n_points == 11
    assert lin.worst == pytest.approx(expected, rel=1e-9) and lin.worst_at == pytest.approx(0.15)
    log = _cmp(a, b, tolerance=0.01, interpolation="log")
    assert log.interpolation == "log" and log.n_points == 11 and log.worst < 0.01 and log.passed
    auto = _cmp(a, b, tolerance=0.01, interpolation="auto")
    assert auto.interpolation == "log" and auto.worst < 0.01 and auto.passed
    default = _cmp(a, b, tolerance=0.01)
    assert default.interpolation == "log" and default.worst == auto.worst


def test_a2_1_auto_is_the_default():
    import inspect
    from simws.compare import compare
    assert inspect.signature(compare).parameters["interpolation"].default == "auto"


@pytest.mark.parametrize("avals,bvals", [([10.0], [1.0, 100.0]), ([-10.0], [-1.0, -100.0])])
def test_a2_1_auto_picks_log_for_single_signed_nonzero_reference(avals, bvals):
    a, b = Q(avals, axis=("V", [2.0], "V")), Q(bvals, axis=("V", [1.0, 3.0], "V"))
    c = _cmp(a, b, tolerance=1e-12)                          # log: exp((ln 1 + ln 100)/2) = 10 exactly
    assert c.interpolation == "log" and c.worst == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("bvals,b_mid", [([0.0, 100.0], 50.0), ([-1.0, 100.0], 49.5)])
def test_a2_1_auto_picks_linear_for_zero_or_sign_change(bvals, b_mid):
    a = Q([10.0], axis=("V", [2.0], "V"))
    b = Q(bvals, axis=("V", [1.0, 3.0], "V"))
    c = _cmp(a, b, tolerance=1.0)                            # no ValueError: auto falls back to linear
    assert c.interpolation == "linear"
    assert c.worst == pytest.approx(abs(10.0 / b_mid - 1.0), rel=1e-12)


def test_a2_1_interpolation_recorded():
    assert _cmp(Q(1.1), Q(1.0), tolerance=0.2).interpolation == "none"           # two scalars
    a, b = Q([10.0], axis=("V", [2.0], "V")), Q([1.0, 100.0], axis=("V", [1.0, 3.0], "V"))
    assert _cmp(a, b, tolerance=1.0, interpolation="linear").interpolation == "linear"
    assert _cmp(a, b, tolerance=1.0, interpolation="log").interpolation == "log"
    from simws.compare import Comparison
    assert "interpolation" in Comparison.model_fields


@pytest.mark.parametrize("bvals", [[0.0, 100.0], [-1.0, 100.0]])
def test_a2_1_forced_log_still_refuses(bvals):
    a = Q([10.0], axis=("V", [2.0], "V"))
    with pytest.raises(ValueError):
        _cmp(a, Q(bvals, axis=("V", [1.0, 3.0], "V")), tolerance=1.0, interpolation="log")


# ------------------------------------------------------------------------------------------------ A2.2 measurement_label
def QL(value, label, measured=True):
    return Quantity(name="y", value=value, unit="V", flags=["measured"] if measured else [], label=label)


@pytest.mark.parametrize("label,expected", [("calibration", "calibration"), ("validation", "validation"),
                                            ("none", None)])
def test_a2_2_measurement_label_from_measured_b(label, expected):
    c = _cmp(Q(1.0), QL(1.0, label), tolerance=0.1)
    assert c.kind == "measurement" and c.measurement_label == expected


@pytest.mark.parametrize("label,expected", [("calibration", "calibration"), ("validation", "validation"),
                                            ("none", None)])
def test_a2_2_measurement_label_from_measured_a(label, expected):
    c = _cmp(QL(1.0, label), Q(1.0), tolerance=0.1)
    assert c.kind == "measurement" and c.measurement_label == expected


def test_a2_2_b_label_wins_when_both_measured():
    assert _cmp(QL(1.0, "validation"), QL(1.0, "calibration"), tolerance=0.1).measurement_label == "calibration"
    assert _cmp(QL(1.0, "calibration"), QL(1.0, "validation"), tolerance=0.1).measurement_label == "validation"


def test_a2_2_engine_pair_has_no_label():
    c = _cmp(QL(1.0, "validation", measured=False), QL(1.0, "calibration", measured=False), tolerance=0.1)
    assert c.kind == "engine_pair" and c.measurement_label is None
    from simws.compare import Comparison
    assert Comparison.model_fields["measurement_label"].default is None


def test_a2_2_compare_results_carries_ingest_label(hub):
    from simws import measured
    from simws.compare import compare_results
    ra = _sketch(hub, 1.0)
    src = {"source": "test fixture, not a measurement (tests/v1/test_compare.py)"}
    cal = measured.ingest(hub, "cmp_cal", ("t", [0.0, 1.0, 2.0], "s"), {"y": ([1.0, 2.0, 3.0], "V")}, src,
                          label="calibration")
    val = measured.ingest(hub, "cmp_val", ("t", [0.0, 1.0, 2.0], "s"), {"y": ([1.0, 2.0, 3.0], "V")}, src)
    assert compare_results(hub, ra.id, "y", cal.id, tolerance=1e-9).measurement_label == "calibration"
    assert compare_results(hub, ra.id, "y", val.id, tolerance=1e-9).measurement_label == "validation"
    assert compare_results(hub, cal.id, "y", ra.id, tolerance=1e-9).measurement_label == "calibration"   # a side


# ------------------------------------------------------------------------------------------------ A2.3 engines, independent
def test_a2_3_bare_quantities_defaults():
    from simws.compare import Comparison
    c = _cmp(Q(1.0), Q(1.0), tolerance=0.1)
    assert c.engine_a == "" and c.engine_b == "" and c.independent is False
    f = Comparison.model_fields
    assert f["engine_a"].default == "" and f["engine_b"].default == "" and f["independent"].default is False


def test_a2_3_bare_measurement_quantities_defaults():
    """A2.3: "compare() on bare quantities leaves them at the defaults" (independent False); the "True for kind
    measurement" rule is stated for compare_results, which has stored results to look at."""
    c = _cmp(Q(1.0), Q(1.0, flags=["measured"]), tolerance=0.1)
    assert c.kind == "measurement"
    assert c.engine_a == "" and c.engine_b == "" and c.independent is False


def test_a2_3_different_engines_independent(hub, fake_adapters):
    from simws.compare import compare_results
    ra, rb = fake_adapters(hub, "fake.alpha", 1.0), fake_adapters(hub, "fake.beta", 1.0)
    c = compare_results(hub, ra.id, "y", rb.id, tolerance=1e-9)
    assert c.kind == "engine_pair" and c.passed
    assert c.engine_a == hub.store.job(ra.id).key.engine == "fake_alpha"
    assert c.engine_b == hub.store.job(rb.id).key.engine == "fake_beta"
    assert c.independent is True


def test_a2_3_same_engine_not_independent(hub, fake_adapters):
    from simws.compare import compare_results
    ra, rb = fake_adapters(hub, "fake.alpha", 1.0), fake_adapters(hub, "fake.alpha_twin", 1.0)
    assert ra.id != rb.id
    c = compare_results(hub, ra.id, "y", rb.id, tolerance=1e-9)
    assert c.engine_a == c.engine_b == "fake_alpha" and c.independent is False and c.passed


def test_a2_3_sketch_not_independent(hub, fake_adapters):
    from simws.compare import compare_results
    ra = fake_adapters(hub, "fake.alpha", 1.0)
    s = _sketch(hub, 1.0)
    for c in (compare_results(hub, ra.id, "y", s.id, tolerance=1e-9),
              compare_results(hub, s.id, "y", ra.id, tolerance=1e-9)):
        assert {c.engine_a, c.engine_b} == {"fake_alpha", hub.store.job(s.id).key.engine}
        assert c.engine_a != c.engine_b and c.independent is False          # different engines, but a sketch
    s2 = _sketch(hub, 1.1)
    c = compare_results(hub, s.id, "y", s2.id, tolerance=1.0)
    assert c.engine_a == c.engine_b == "python" and c.independent is False


def test_a2_3_measurement_independent(hub):
    from simws import measured
    from simws.compare import compare_results
    s = _sketch(hub, 1.0)
    m = measured.ingest(hub, "cmp_ind", ("t", [0.0, 1.0, 2.0], "s"), {"y": ([1.0, 2.0, 3.0], "V")},
                        {"source": "test fixture, not a measurement (tests/v1/test_compare.py)"})
    c = compare_results(hub, s.id, "y", m.id, tolerance=1e-9)
    assert c.kind == "measurement" and c.independent is True
    assert c.engine_a == "python" and c.engine_b == "measurement"


def test_a2_new_fields_stored_by_record(hub, fake_adapters):
    from simws.compare import compare_results, record, records_for
    ra, rb = fake_adapters(hub, "fake.alpha", 1.0), fake_adapters(hub, "fake.beta", 1.0)
    c = compare_results(hub, ra.id, "y", rb.id, tolerance=1e-9)
    record(hub, c)
    (rec,) = records_for(hub, ra.id)
    assert rec["independent"] is True and rec["engine_a"] == "fake_alpha" and rec["engine_b"] == "fake_beta"
    assert rec["interpolation"] == "log" and rec["measurement_label"] is None
