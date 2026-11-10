"""simws.measured (contract section 6.2): measured data stored as ordinary hub results. The numbers below are
hand-made test fixtures (their `source` says so), never presented as real measurements."""
import re

import pytest

SRC = "test fixture, not a measurement (tests/v1/test_measured.py)"
AXIS = ("V", [0.1, 0.2, 0.3], "V")
COLS = {"J": ([1.0, 2.0, 4.0], "A/m**2"), "T_dev": ([300.0, 301.0, 302.0], "K")}


def _walk(v):
    yield v
    if isinstance(v, dict):
        for x in v.values():
            yield from _walk(x)
    elif isinstance(v, list):
        for x in v:
            yield from _walk(x)


def _ingest(hub, **kw):
    from simws import measured
    args = dict(name="diode_iv", axis=AXIS, columns=COLS, meta={"source": SRC})
    args.update(kw)
    return measured.ingest(hub, args.pop("name"), args.pop("axis"), args.pop("columns"), args.pop("meta"), **args)


def test_ingest_envelope(hub):
    r = _ingest(hub, sigma={"J": [0.1, 0.2, 0.4]})
    assert r.status == "ok" and len(r.id) == 32
    assert "measured" in r.flags and "digitized" not in r.flags
    assert sorted(q.name for q in r.quantities) == ["J", "T_dev"]
    j = r.q("J")
    assert j.value == [1.0, 2.0, 4.0] and j.unit == "A/m**2"
    assert len(j.axes) == 1 and j.axes[0].name == "V" and j.axes[0].values == [0.1, 0.2, 0.3]
    assert j.axes[0].unit == "V"
    assert "measured" in j.flags
    assert j.label == "validation" and j.source == SRC
    assert j.uncertainty.kind == "sigma" and list(j.uncertainty.sigma) == [0.1, 0.2, 0.4]
    assert r.q("T_dev").uncertainty.kind == "none"
    assert r.provenance["measured"] == {"source": SRC}


def test_ingest_job_key_and_graph(hub):
    r = _ingest(hub)
    job = hub.store.job(r.id)
    k = job.key
    assert (k.adapter, k.engine, k.engine_version, k.env_lock) == ("measured.diode_iv", "measurement", "n/a",
                                                                     "measured")
    vals = list(_walk(k.params))                    # the contract fixes the content of params, not the key names
    assert {"source": SRC} in vals, k.params                                   # meta
    assert any(isinstance(v, str) and re.search(r"[0-9a-f]{64}", v) for v in vals), k.params   # SHA-256
    assert hub.graph.result_for_job(r.job_id) == r.id
    assert hub.store.has(r.id)
    assert hub.result(r.id).q("J").value == [1.0, 2.0, 4.0]
    rows = hub.graph.db.execute("SELECT id FROM nodes WHERE type='Dataset' AND name=?", ("diode_iv",)).fetchall()
    assert len(rows) == 1
    ds = rows[0][0]
    linked = [d for d, t, _ in hub.graph.edges_from(ds, "produced")] + \
             [s for s, t, _ in hub.graph.edges_to(ds, "produced")]
    assert r.id in linked, (ds, linked)


def test_digitized_flag_and_label(hub):
    r = _ingest(hub, meta={"source": SRC, "digitized": True}, label="calibration")
    assert "measured" in r.flags and "digitized" in r.flags
    assert all(q.label == "calibration" for q in r.quantities)
    assert r.provenance["measured"] == {"source": SRC, "digitized": True}


def test_same_data_is_one_job(hub):
    a = _ingest(hub)
    b = _ingest(hub)
    assert a.id == b.id and a.job_id == b.job_id
    n = hub.graph.db.execute("SELECT COUNT(*) FROM jobs WHERE adapter='measured.diode_iv'").fetchone()[0]
    assert n == 1


@pytest.mark.parametrize("change", [
    dict(columns={"J": ([1.0, 2.0, 4.5], "A/m**2")}),
    dict(axis=("V", [0.1, 0.2, 0.31], "V")),
    dict(meta={"source": SRC + " (other)"}),
])
def test_different_data_or_meta_is_another_job(hub, change):
    a = _ingest(hub)
    b = _ingest(hub, **change)
    assert b.id != a.id and b.job_id != a.job_id


def test_source_required(hub):
    with pytest.raises(ValueError):
        _ingest(hub, meta={})
    with pytest.raises(ValueError):
        _ingest(hub, meta={"citation": "x"})


def test_lengths_must_agree(hub):
    with pytest.raises(ValueError):
        _ingest(hub, columns={"J": ([1.0, 2.0], "A/m**2")})
    with pytest.raises(ValueError):
        _ingest(hub, axis=("V", [0.1, 0.2], "V"))


def test_scalars_without_axis(hub):
    """axis None: scalars. Whether a scalar column is stored as a number or a one-element list is not fixed by the
    contract; both are accepted here."""
    r = _ingest(hub, name="vbi", axis=None, columns={"Vbi": ([0.72], "V")})
    q = r.q("Vbi")
    assert q.axes == []
    assert q.value == 0.72 or q.value == [0.72]
    assert "measured" in r.flags


def test_load_csv(tmp_path):
    from simws.measured import load_csv
    f = tmp_path / "iv.csv"
    f.write_text("V,J\n0.1,1.0\n0.2,2.5\n0.3,4.0\n")
    x, y = load_csv(f, skip_header=1)
    assert list(x) == [0.1, 0.2, 0.3] and list(y) == [1.0, 2.5, 4.0]
    g = tmp_path / "iv.txt"
    g.write_text("# digitized\n# a;b;c\n1;10;100\n2;20;200\n")
    x, y = load_csv(g, x_col=0, y_col=2, skip_header=2, delimiter=";")
    assert list(x) == [1.0, 2.0] and list(y) == [100.0, 200.0]
