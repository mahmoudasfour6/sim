"""simws.study (contract section 6.5): full-factorial sweeps over dotted paths, refusals without submission, trust
from the stored results, and engine comparison. DEVSIM runs are tiny (201 nodes, 2 bias points)."""
import pytest

from simws.spec.library import pn_diode_1d

pytest.importorskip("devsim")


def _base():
    return pn_diode_1d(nodes=201, V=[0.3, 0.4])


def _device_jobs(hub, adapter):
    return hub.graph.db.execute("SELECT COUNT(*) FROM jobs WHERE adapter=?", (adapter,)).fetchone()[0]


def test_full_factorial_order_and_points(hub):
    from simws.hub.trust import LEVELS
    from simws.study import Study, StudyResult
    s = Study(hub, _base(), engines=["devsim"])
    res = s.sweep({"models.temperature_K": [280.0, 300.0], "device.mesh.scale": [1.0, 0.5]})
    assert isinstance(res, StudyResult)
    assert [p["changes"] for p in res.points] == [
        {"models.temperature_K": 280.0, "device.mesh.scale": 1.0},
        {"models.temperature_K": 280.0, "device.mesh.scale": 0.5},
        {"models.temperature_K": 300.0, "device.mesh.scale": 1.0},
        {"models.temperature_K": 300.0, "device.mesh.scale": 0.5},
    ]
    for p in res.points:
        assert p["engine"] == "devsim" and p["status"] == "ok" and p["reasons"] == []
        assert p["result_id"] and hub.store.has(p["result_id"])
        assert p["trust"] in LEVELS and p["in_range"] is True
        r = hub.result(p["result_id"])
        expected = _base().with_changes(p["changes"]).problem_id()
        assert r.diagnostics["problem_id"] == expected
    assert len({p["result_id"] for p in res.points}) == 4
    assert _device_jobs(hub, "device.devsim") == 4


def test_submission_and_fidelity(hub):
    from simws.study import Study
    res = Study(hub, _base(), engines=["devsim"]).sweep({"models.temperature_K": [300.0]})
    p = res.points[0]
    job = hub.store.job(p["result_id"])
    assert job.fidelity == "screening" and job.key.adapter == "device.devsim"
    direct = hub.submit("device.devsim", params={"problem": _base().model_dump(mode="json")})
    assert direct.id == p["result_id"]                     # the study submitted exactly this problem
    assert direct.diagnostics.get("cache") == "hit"


def test_refused_engine_submits_nothing(hub):
    from simws import engines
    from simws.study import Study
    res = Study(hub, _base(), engines=["devsim", "charon"]).sweep({"models.temperature_K": [300.0]})
    by = {p["engine"]: p for p in res.points}
    assert set(by) == {"devsim", "charon"} and len(res.points) == 2
    ch = by["charon"]
    assert ch["status"] == "refused" and ch["result_id"] is None
    assert ch["reasons"] == engines.unsupported("charon", _base()) and ch["reasons"]
    assert _device_jobs(hub, "device.charon") == 0
    assert by["devsim"]["status"] == "ok"


def test_empty_over_and_default_engine(hub):
    from simws.engines.planner import plan
    from simws.study import Study
    res = Study(hub, _base()).sweep({})
    assert plan(_base()).primary == "devsim"
    assert len(res.points) == 1
    p = res.points[0]
    assert p["engine"] == "devsim" and p["changes"] == {} and p["status"] == "ok"


def test_trust_and_in_range_from_store(hub):
    from simws.hub.trust import assess_stored
    from simws.study import Study
    res = Study(hub, _base(), engines=["devsim"]).sweep({"models.temperature_K": [240.0, 300.0]})
    assert [p["in_range"] for p in res.points] == [False, True]          # DEVSIM tested range [250, 400] K
    for p in res.points:
        t = assess_stored(hub, p["result_id"])
        assert p["trust"] == t.level and p["in_range"] == t.in_range


def test_quantity_and_table(hub):
    from simws.study import Study
    res = Study(hub, _base(), engines=["devsim", "charon"]).sweep({"models.temperature_K": [280.0, 300.0]})
    qs = res.quantity("dc.I.anode")
    assert len(qs) == 2                                      # refused points have no quantity
    # the contract types the first element as `dict` without saying which: accept the changes or the point dict
    changes = [c.get("changes", c) for c, _ in qs]
    assert changes == [{"models.temperature_K": 280.0}, {"models.temperature_K": 300.0}]
    for ch, q in qs:
        assert q.name == "dc.I.anode" and q.unit == "A/m**2" and len(q.value) == 2
    assert len(res.quantity("dc.I.anode", engine="devsim")) == 2
    assert res.quantity("dc.I.anode", engine="charon") == []
    # a hotter diode carries more forward current at the same bias (n_i^2 grows ~ exp(-Eg/kT))
    assert qs[1][1].value[0] > qs[0][1].value[0]
    t = res.table("dc.I.anode")
    assert isinstance(t, str) and t


def test_compare_engines(hub):
    from simws.compare import records_for
    from simws.study import Study
    res = Study(hub, _base(), engines=["devsim", "charon"]).sweep({"models.temperature_K": [280.0, 300.0]})
    assert res.compare_engines("dc.I.anode", "devsim", "charon", tolerance=0.02) == []   # charon refused everywhere
    # devsim against itself: both sides ok at every combination, identical quantities -> passed. A2.1: the default
    # interpolation is now "auto" (log for these single-signed currents), and exp(ln y) need not return y bit for
    # bit, so "worst == 0.0" became "worst <= 1e-12" (the tolerance passed in), fixed before running.
    cs = res.compare_engines("dc.I.anode", "devsim", "devsim", tolerance=1e-12, record=False)
    assert len(cs) == 2 and all(c.passed and c.worst <= 1e-12 for c in cs)
    assert all(records_for(hub, p["result_id"]) == [] for p in res.points if p["result_id"])


def test_compare_engine_with_itself_verifies_nothing(hub):
    """A2.3: Study.compare_engines(name, a, a, ...) still runs and records, and verifies nothing."""
    from simws.compare import records_for
    from simws.hub.trust import assess_stored
    from simws.study import Study
    res = Study(hub, _base(), engines=["devsim"]).sweep({"models.temperature_K": [280.0, 300.0]})
    before = {p["result_id"]: assess_stored(hub, p["result_id"]).level for p in res.points}
    assert set(before.values()) == {"unverified"}
    cs = res.compare_engines("dc.I.anode", "devsim", "devsim", tolerance=1e-12)          # record=True (default)
    assert len(cs) == 2
    for c in cs:
        assert c.passed and c.kind == "engine_pair"
        assert c.engine_a == c.engine_b == "devsim" and c.independent is False
    for p in res.points:
        rid = p["result_id"]
        recs = records_for(hub, rid)
        assert len(recs) == 1 and recs[0]["independent"] is False and recs[0]["passed"] is True
        t = assess_stored(hub, rid)
        assert t.level == "unverified"
        assert f"same-engine or sketch comparison with {rid} (consistency only: not verification)" in t.reasons, \
            t.reasons
