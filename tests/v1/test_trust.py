"""simws.hub.trust (contract section 6.1): every rule, on hand-built envelopes and comparison dicts."""
import pytest

from simws.hub.schema import Quantity, ResultEnvelope

RID = "a" * 32
OTHER = "b" * 32
THIRD = "c" * 32


def _res(status="ok", flags=(), qflags=(), verification=None, rid=RID):
    return ResultEnvelope(id=rid, job_id="j" * 32, status=status, flags=list(flags),
                          quantities=[Quantity(name="x", value=1.0, unit="V", flags=list(qflags))],
                          verification=verification or {})


_DEFAULT = object()


def _cmp(kind, a=RID, b=OTHER, passed=True, inconclusive=False, label=_DEFAULT, independent=True):
    """A dict shaped like Comparison.model_dump() (contract 6.3 + A2). By default a measurement comparison is
    against "validation" data and an engine pair is between two different engines (independent), so the pre-A2
    expectations of the rule tests below still hold; the A2 tests set label / independent explicitly."""
    if label is _DEFAULT:
        label = "validation" if kind == "measurement" else None
    return {"kind": kind, "a": a, "b": b, "quantity_a": "x", "quantity_b": "x", "metric": "relative",
            "tolerance": 0.1, "worst": 0.01 if passed else (None if inconclusive else 0.5),
            "worst_at": None, "n_points": 0 if inconclusive else 1, "window": None, "passed": passed,
            "inconclusive": inconclusive, "note": "", "interpolation": "none", "measurement_label": label,
            "engine_a": "engine_x", "engine_b": "measurement" if kind == "measurement" else "engine_y",
            "independent": independent}


def _assess(res, comparisons=()):
    from simws.hub.trust import assess
    return assess(res, comparisons)


def test_levels_and_model():
    from simws.hub.trust import LEVELS, Trust
    assert LEVELS == ("sketch", "unverified", "verified", "validated", "measured")
    t = Trust(level="verified", in_range=True, reasons=[])
    assert t.level == "verified"


# ---------------------------------------------------------------------------------------------- rule 1
@pytest.mark.parametrize("status", ["engine_error", "not_converged", "invalid_input", "check_failed"])
def test_rule1_failed_status(status):
    t = _assess(_res(status=status, flags=["measured"]), [_cmp("measurement")])
    assert t.level == "unverified"
    assert f"status:{status}" in t.reasons


# ---------------------------------------------------------------------------------------------- rules 2, 3
def test_rule2_measured():
    assert _assess(_res(flags=["measured"])).level == "measured"
    assert _assess(_res(flags=["measured", "sketch"])).level == "measured"          # rule 2 before rule 3
    assert _assess(_res(flags=["measured"]), [_cmp("engine_pair")]).level == "measured"


def test_rule3_sketch():
    assert _assess(_res(flags=["sketch"])).level == "sketch"
    assert _assess(_res(flags=["sketch"]), [_cmp("measurement")]).level == "sketch"    # rule 3 before rule 4


# ---------------------------------------------------------------------------------------------- rule 4
def test_rule4_validated_by_measurement_comparison():
    t = _assess(_res(), [_cmp("measurement", a=RID, b=OTHER)])
    assert t.level == "validated"
    assert any(OTHER in s for s in t.reasons), t.reasons


def test_rule4_result_on_b_side():
    """The non-measured side may be either a or b of the comparison."""
    t = _assess(_res(), [_cmp("measurement", a=OTHER, b=RID)])
    assert t.level == "validated"
    assert any(OTHER in s for s in t.reasons), t.reasons


def test_rule4_beats_rule5():
    t = _assess(_res(verification={"code": {"kind": "code", "passed": True}}),
                [_cmp("engine_pair"), _cmp("measurement", b=THIRD)])
    assert t.level == "validated"


def test_comparisons_not_involving_result_ignored():
    t = _assess(_res(), [_cmp("measurement", a=OTHER, b=THIRD), _cmp("engine_pair", a=OTHER, b=THIRD)])
    assert t.level == "unverified"


# ---------------------------------------------------------------------------------------------- rule 5
def test_rule5_engine_pair():
    assert _assess(_res(), [_cmp("engine_pair")]).level == "verified"
    assert _assess(_res(), [_cmp("engine_pair", a=OTHER, b=RID)]).level == "verified"


def test_rule5_code_verification():
    assert _assess(_res(verification={"mms": {"kind": "code", "passed": True}})).level == "verified"
    assert _assess(_res(verification={"mms": {"kind": "code", "passed": False}})).level == "unverified"
    assert _assess(_res(verification={"mms": "passed"})).level == "unverified"          # not a dict


# ---------------------------------------------------------------------------------------------- rule 6
def test_rule6_unverified_default():
    t = _assess(_res())
    assert t.level == "unverified" and t.in_range is True


def test_rule6_solution_verification_adds_reason_only():
    t = _assess(_res(verification={"mesh": {"kind": "solution", "passed": True, "error": 0.01}}))
    assert t.level == "unverified"
    assert "numerical error quantified" in t.reasons


# ---------------------------------------------------------------------------------------------- failed comparisons
def test_failed_measurement_comparison_adds_reason():
    t = _assess(_res(), [_cmp("measurement", passed=False)])
    assert t.level == "unverified"
    assert f"failed measurement comparison with {OTHER}" in t.reasons


def test_failed_comparison_never_lowers_or_raises():
    t = _assess(_res(), [_cmp("engine_pair"), _cmp("measurement", b=THIRD, passed=False)])
    assert t.level == "verified"
    assert f"failed measurement comparison with {THIRD}" in t.reasons


def test_inconclusive_comparison_adds_reason():
    t = _assess(_res(), [_cmp("measurement", passed=False, inconclusive=True)])
    assert t.level == "unverified"
    assert any(OTHER in s for s in t.reasons), t.reasons


def test_failed_engine_pair_adds_reason():
    t = _assess(_res(), [_cmp("engine_pair", passed=False)])
    assert t.level == "unverified"
    assert any(OTHER in s for s in t.reasons), t.reasons


# ---------------------------------------------------------------------------------------------- in_range
@pytest.mark.parametrize("where,flag", [
    ("result", "extrapolated"),
    ("result", "extrapolated:temperature_K: 77 outside tested [250, 400]"),
    ("result", "cryo_unvalidated"),
    ("quantity", "extrapolated"),
    ("quantity", "cryo_unvalidated:T"),
])
def test_in_range_false_on_flags(where, flag):
    res = _res(flags=[flag]) if where == "result" else _res(qflags=[flag])
    t = _assess(res)
    assert t.in_range is False
    assert flag in t.reasons


def test_in_range_true_for_other_flags():
    t = _assess(_res(flags=["runner:local", "not_extrapolated"], qflags=["unverified"]))
    assert t.in_range is True


def test_in_range_independent_of_level():
    t = _assess(_res(flags=["extrapolated:temperature_K"]), [_cmp("engine_pair")])
    assert t.level == "verified" and t.in_range is False


# ---------------------------------------------------------------------------------------------- assess_stored
def test_assess_stored_uses_recorded_comparisons(hub):
    """A stored adapter result (materials.intrinsic, pure Python) is unverified until a passing measurement
    comparison is recorded. The 'measured' column is a copy of the model output, labelled as test data in its
    source: it exercises the bookkeeping, it is not evidence of anything."""
    from simws import compare, measured
    from simws.hub.trust import assess_stored
    r = hub.submit("materials.intrinsic", params={"material": "silicon", "profile": None, "T": ([250, 300, 350], "K")})
    assert r.status == "ok"
    assert assess_stored(hub, r.id).level == "unverified"
    eg = r.q("mat.Eg")
    m = measured.ingest(hub, "trust_ds", ("T", eg.axes[0].values, eg.axes[0].unit), {"Eg": (eg.value, eg.unit)},
                        {"source": "copy of the model output, test fixture only (tests/v1/test_trust.py)"})
    assert assess_stored(hub, m.id).level == "measured"
    c = compare.compare_results(hub, r.id, "mat.Eg", m.id, "Eg", tolerance=1e-9)
    assert c.kind == "measurement" and c.passed
    assert c.measurement_label == "validation" and c.independent is True       # ingest's default label (A2.2)
    compare.record(hub, c)
    t = assess_stored(hub, r.id)
    assert t.level == "validated"
    assert any(m.id in s for s in t.reasons)
    assert t == _assess(hub.result(r.id), compare.records_for(hub, r.id))
    assert assess_stored(hub, m.id).level == "measured"      # a measurement comparison never changes the data side
    s = hub.sketch(lambda k: {"y": (k, "V")}, params={"k": 1.0}, name="trust_sketch")
    assert assess_stored(hub, s.id).level == "sketch"


def test_failed_measurement_blocks_validated():
    """Lead, contract amendment A1.10: a failed measurement comparison blocks "validated" even when another
    measurement comparison passed; an inconclusive one does not block."""
    from simws.hub.schema import ResultEnvelope
    from simws.hub.trust import assess
    res = ResultEnvelope(id="r" * 32, job_id="j" * 32, status="ok")
    base = dict(a=res.id, quantity_a="q", quantity_b="q", metric="ratio", tolerance=1.3, worst_at=None, n_points=5,
                window=None, note="", kind="measurement",
                # A2.2 / A2.3 fields: validation data, so that A1.10 is what is tested here
                interpolation="log", measurement_label="validation", engine_a="devsim", engine_b="measurement",
                independent=True)
    ok = dict(base, b="m1", worst=1.1, passed=True, inconclusive=False)
    bad = dict(base, b="m2", worst=11.2, passed=False, inconclusive=False)
    thin = dict(base, b="m3", worst=None, n_points=0, passed=False, inconclusive=True)
    assert assess(res, [ok]).level == "validated"
    t = assess(res, [ok, bad])
    assert t.level == "unverified" and any("not validated" in r for r in t.reasons)
    assert assess(res, [ok, thin]).level == "validated"


# ================================================================================================ amendment A2
def _cal_reason(other):
    return f"matches calibration data {other} (fitted to it: not validation)"


def _unlabelled_reason(other):
    return f"matches unlabelled measured data {other} (not counted as validation)"


def _same_engine_reason(other):
    return f"same-engine or sketch comparison with {other} (consistency only: not verification)"


# ---------------------------------------------------------------------------------------------- A2.2 rule 4
def test_a2_2_validation_label_validates():
    t = _assess(_res(), [_cmp("measurement", label="validation")])
    assert t.level == "validated"


def test_a2_2_calibration_only_not_validated():
    t = _assess(_res(), [_cmp("measurement", label="calibration")])
    assert t.level == "unverified"
    assert _cal_reason(OTHER) in t.reasons, t.reasons
    t = _assess(_res(), [_cmp("measurement", a=OTHER, b=RID, label="calibration")])      # result on the b side
    assert t.level == "unverified" and _cal_reason(OTHER) in t.reasons, t.reasons


def test_a2_2_unlabelled_not_validated():
    t = _assess(_res(), [_cmp("measurement", label=None)])
    assert t.level == "unverified"
    assert _unlabelled_reason(OTHER) in t.reasons, t.reasons


def test_a2_2_calibration_falls_back_to_rule5():
    t = _assess(_res(), [_cmp("measurement", label="calibration"), _cmp("engine_pair", b=THIRD)])
    assert t.level == "verified" and _cal_reason(OTHER) in t.reasons


def test_a2_2_failed_calibration_blocks_validation():
    """A1.10 with any label: a failed calibration comparison blocks "validated" even with a passed validation one."""
    t = _assess(_res(), [_cmp("measurement", label="validation"),
                         _cmp("measurement", b=THIRD, label="calibration", passed=False)])
    assert t.level == "unverified"
    t = _assess(_res(), [_cmp("measurement", label="validation"),
                         _cmp("measurement", b=THIRD, label=None, passed=False)])
    assert t.level == "unverified"


def test_a2_2_inconclusive_calibration_does_not_block():
    t = _assess(_res(), [_cmp("measurement", label="validation"),
                         _cmp("measurement", b=THIRD, label="calibration", passed=False, inconclusive=True)])
    assert t.level == "validated"


def test_a2_2_pre_a2_measurement_dict_without_label_not_validated():
    """A dict without the measurement_label key reads as the field's default, None (A2.2): not validation."""
    d = _cmp("measurement")
    del d["measurement_label"]
    assert _assess(_res(), [d]).level == "unverified"


def test_a2_2_end_to_end_calibration_then_validation(hub):
    """measured.ingest(label="calibration") -> compare_results -> record -> assess_stored: not validated. A second
    dataset ingested as validation data then validates. Both 'measured' columns are copies of the model output,
    labelled as test data in their source: bookkeeping only, not evidence."""
    from simws import compare, measured
    from simws.hub.trust import assess_stored
    r = hub.submit("materials.intrinsic", params={"material": "silicon", "profile": None, "T": ([250, 300, 350], "K")})
    assert r.status == "ok"
    eg = r.q("mat.Eg")
    axis = ("T", eg.axes[0].values, eg.axes[0].unit)
    src = {"source": "copy of the model output, test fixture only (tests/v1/test_trust.py)"}
    cal = measured.ingest(hub, "trust_cal", axis, {"Eg": (eg.value, eg.unit)}, src, label="calibration")
    c = compare.compare_results(hub, r.id, "mat.Eg", cal.id, "Eg", tolerance=1e-9)
    assert c.kind == "measurement" and c.passed and c.measurement_label == "calibration"
    compare.record(hub, c)
    t = assess_stored(hub, r.id)
    assert t.level == "unverified" and _cal_reason(cal.id) in t.reasons, t.reasons
    val = measured.ingest(hub, "trust_val", axis, {"Eg": (eg.value, eg.unit)}, src, label="validation")
    cv = compare.compare_results(hub, r.id, "mat.Eg", val.id, "Eg", tolerance=1e-9)
    assert cv.passed and cv.measurement_label == "validation"
    compare.record(hub, cv)
    t = assess_stored(hub, r.id)
    assert t.level == "validated" and _cal_reason(cal.id) in t.reasons


# ---------------------------------------------------------------------------------------------- A2.3 rule 5
def test_a2_3_independent_engine_pair_verifies():
    assert _assess(_res(), [_cmp("engine_pair", independent=True)]).level == "verified"


def test_a2_3_not_independent_engine_pair_does_not_verify():
    t = _assess(_res(), [_cmp("engine_pair", independent=False)])
    assert t.level == "unverified"
    assert _same_engine_reason(OTHER) in t.reasons, t.reasons
    t = _assess(_res(), [_cmp("engine_pair", a=OTHER, b=RID, independent=False)])
    assert t.level == "unverified" and _same_engine_reason(OTHER) in t.reasons, t.reasons


def test_a2_3_pre_a2_engine_pair_dict_not_verified():
    d = _cmp("engine_pair")
    for k in ("independent", "engine_a", "engine_b", "interpolation", "measurement_label"):
        del d[k]
    assert _assess(_res(), [d]).level == "unverified"


def test_a2_3_code_verification_still_verifies_beside_same_engine_pair():
    t = _assess(_res(verification={"mms": {"kind": "code", "passed": True}}),
                [_cmp("engine_pair", independent=False)])
    assert t.level == "verified"


def test_a2_3_end_to_end(hub, fake_adapters):
    """Two stored results from different (fake) engines verify each other; two from the same engine do not."""
    from simws import compare
    from simws.hub.trust import assess_stored
    ra = fake_adapters(hub, "fake.alpha", 1.0)
    rt = fake_adapters(hub, "fake.alpha_twin", 1.0)
    compare.record(hub, compare.compare_results(hub, ra.id, "y", rt.id, tolerance=1e-9))
    t = assess_stored(hub, ra.id)
    assert t.level == "unverified" and _same_engine_reason(rt.id) in t.reasons, t.reasons
    assert assess_stored(hub, rt.id).level == "unverified"
    rb = fake_adapters(hub, "fake.beta", 1.0)
    compare.record(hub, compare.compare_results(hub, ra.id, "y", rb.id, tolerance=1e-9))
    assert assess_stored(hub, ra.id).level == "verified"
    assert assess_stored(hub, rb.id).level == "verified"
    assert assess_stored(hub, rt.id).level == "unverified"
