"""Unit tests for simws 0.8: citations, overnight queue (incl. crash recovery), retention pruning, disk report."""
import datetime as dt

import pytest

pytest.importorskip("pydantic")


def test_every_adapter_citation_is_registered():
    from simws.hub import adapter as A
    from simws.hub.citations import REGISTRY, bibliography
    for n in A.names():
        assert not [k for k in A.manifest(n).citations if k not in REGISTRY], n
    b = bibliography(["devsim", "hamann2013"], fmt="bibtex")
    assert "@article{devsim," in b and "UNCHECKED" in b
    with pytest.raises(KeyError):
        bibliography(["nope2099"])


def test_queue_order_run_and_recovery(tmp_path):
    from simws.hub.jobs import Hub
    from simws.queue import Queue, parse_until
    q = Queue(tmp_path)
    P = {"material": "silicon", "profile": None, "T": [[300.0], "K"]}
    a = q.add("materials.intrinsic", P, priority="overnight")
    b = q.add("materials.intrinsic", {**P, "T": [[350.0], "K"]}, priority="interactive")
    c = q.add("materials.intrinsic", {**P, "material": "unobtainium"}, priority="batch")
    assert q.next()[0] == b
    q.db.execute("UPDATE queue SET status='running', pid=999999 WHERE id=?", (a,))   # a dead worker's leftover
    hub = Hub(tmp_path, mode="inprocess")
    logs = []
    st = q.run(hub=hub, log=logs.append)
    assert st == {"ok": 2, "failed": 1, "recovered": 1, "remaining": 0}
    rows = {r[0]: r for r in q.status()}
    assert rows[a][3] == "done" and rows[c][4] == "invalid_input"
    assert [l.split()[1] for l in logs if l.startswith("queue: #")] == [f"#{b}", f"#{c}", f"#{a}"]
    with pytest.raises(ValueError):
        q.add("materials.intrinsic", P, priority="urgent")
    now = dt.datetime(2026, 11, 2, 23, 0)
    assert parse_until("07:00", now) == dt.datetime(2026, 11, 3, 7, 0)
    assert parse_until("23:30", now) == dt.datetime(2026, 11, 2, 23, 30)


def test_prune_after_validation(tmp_path):
    from simws.hub import maintain
    from simws.hub.jobs import Hub
    from simws.hub.schema import FieldRef, ResultEnvelope
    hub = Hub(tmp_path, mode="inprocess")
    # a committed producer with one until_validated field
    job = hub.make_job("materials.intrinsic", {"material": "silicon", "profile": None, "T": ([300.0], "K")})
    stg = hub.store.staging(job.id)
    (stg / "job.json").write_text(job.model_dump_json())
    (stg / "data" / "field.npy").write_bytes(b"z" * 4096)
    env = ResultEnvelope(job_id=job.id, status="ok", fields=[FieldRef(name="f", path="field.npy", mesh_id="m", unit="V")])
    with hub.store.lock():
        env = hub.store.finalize(stg, env)
        hub.graph.commit_result(job, env)
    assert maintain.prune_all(hub.store, hub.graph)["pruned"] == []          # nobody consumed it yet
    hub.submit("materials.intrinsic", params={"material": "silicon", "profile": None, "T": ([310.0], "K")},
               inputs={"upstream": env.id})
    p = maintain.prune_all(hub.store, hub.graph)
    assert p == {"pruned": [env.id], "bytes_freed": 4096}
    assert hub.graph.events()[-1]["type"] == "pruned"
    rep = maintain.report(hub.store, hub.graph)
    assert rep["jobs"]["ok"] >= 2 and "disk_used_fraction" in rep
