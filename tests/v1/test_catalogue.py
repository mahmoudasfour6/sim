"""simws.catalogue (contract section 6.4): a flat, filterable list of committed results."""
import pytest

from simws.spec.library import pn_diode_1d

ROW_KEYS = {"result_id", "job_id", "adapter", "engine", "status", "fidelity", "finished", "flags", "quantities",
            "temperature_K", "trust", "in_range"}
SRC = "test fixture, not a measurement (tests/v1/test_catalogue.py)"


def _params(p):
    return {"problem": p.model_dump(mode="json")}


@pytest.fixture
def filled(hub):
    """A hub with: two tiny DEVSIM diode runs (300 K, 260 K), one measured dataset, two sketches (T as a number and
    as a quantity), and one failed job."""
    pytest.importorskip("devsim")
    from simws import measured
    out = {}
    out["d300"] = hub.submit("device.devsim", params=_params(pn_diode_1d(nodes=201, V=[0.3, 0.4])))
    out["d260"] = hub.submit("device.devsim", params=_params(pn_diode_1d(nodes=201, T=260.0, V=[0.3, 0.4])))
    out["meas"] = measured.ingest(hub, "cat_ds", ("V", [0.3, 0.4], "V"), {"J": ([1.0, 2.0], "A/m**2")},
                                  {"source": SRC})
    out["sk_num"] = hub.sketch(lambda T: {"s.val": (1.0, "V")}, params={"T": 77.0}, name="cat_num")
    out["sk_q"] = hub.sketch(lambda T: {"s.val": (2.0, "V")}, params={"T": (0.25, "kK")}, name="cat_q")
    out["fail"] = hub.submit("materials.intrinsic", params={"material": "unobtainium", "profile": None,
                                                            "T": ([300], "K")})
    assert out["d300"].status == "ok" and out["d260"].status == "ok", (out["d300"].message, out["d260"].message)
    assert out["fail"].status == "invalid_input"
    return out


def _ids(rows):
    return {r["result_id"] for r in rows}


def test_row_shape(hub, filled):
    from simws.catalogue import rows
    rs = rows(hub)
    assert rs and all(set(r) == ROW_KEYS for r in rs)
    by = {r["result_id"]: r for r in rs}
    d = by[filled["d300"].id]
    assert d["adapter"] == "device.devsim" and d["engine"] == "devsim" and d["status"] == "ok"
    assert d["job_id"] == filled["d300"].job_id and d["fidelity"] == "screening"
    assert sorted(d["quantities"]) == ["dc.I.anode", "dc.I.cathode", "dc.kirchhoff_residual"]      # contract A1.8
    assert d["temperature_K"] == 300.0 and d["in_range"] is True
    m = by[filled["meas"].id]
    assert m["engine"] == "measurement" and m["trust"] == "measured" and m["temperature_K"] is None
    assert "measured" in m["flags"]
    assert by[filled["sk_num"].id]["trust"] == "sketch"


def test_default_status_ok_only(hub, filled):
    from simws.catalogue import rows
    assert filled["fail"].id not in _ids(rows(hub))
    assert all(r["status"] == "ok" for r in rows(hub))
    every = rows(hub, status=None)
    assert filled["fail"].id in _ids(every)
    assert _ids(rows(hub, status="invalid_input")) == {filled["fail"].id}


def test_adapter_pattern(hub, filled):
    from simws.catalogue import rows
    assert _ids(rows(hub, adapter="device.*")) == {filled["d300"].id, filled["d260"].id}
    assert _ids(rows(hub, adapter="measured.*")) == {filled["meas"].id}
    assert _ids(rows(hub, adapter="device.devsim")) == {filled["d300"].id, filled["d260"].id}
    assert rows(hub, adapter="device.charon") == []


def test_quantity_and_flag_substring(hub, filled):
    from simws.catalogue import rows
    assert _ids(rows(hub, quantity="dc.I")) == {filled["d300"].id, filled["d260"].id}
    assert _ids(rows(hub, quantity="I.cath")) == {filled["d300"].id, filled["d260"].id}
    assert _ids(rows(hub, quantity="s.val")) == {filled["sk_num"].id, filled["sk_q"].id}
    assert _ids(rows(hub, flag="measur")) == {filled["meas"].id}
    assert _ids(rows(hub, flag="sketch")) == {filled["sk_num"].id, filled["sk_q"].id}


def test_temperature_from_problem_number_and_quantity(hub, filled):
    from simws.catalogue import rows
    by = {r["result_id"]: r for r in rows(hub)}
    assert by[filled["d260"].id]["temperature_K"] == 260.0
    assert by[filled["sk_num"].id]["temperature_K"] == 77.0                    # params["T"] a number
    assert by[filled["sk_q"].id]["temperature_K"] == pytest.approx(250.0)      # canonical quantity, 0.25 kK
    assert _ids(rows(hub, temperature=(290.0, 310.0))) == {filled["d300"].id}
    assert _ids(rows(hub, temperature=(250.0, 300.0))) == {filled["d300"].id, filled["d260"].id, filled["sk_q"].id}
    assert _ids(rows(hub, temperature=(0.0, 1000.0))) == {filled["d300"].id, filled["d260"].id,
                                                           filled["sk_num"].id, filled["sk_q"].id}   # None never


def test_trust_filter(hub, filled):
    from simws.catalogue import rows
    assert _ids(rows(hub, trust="measured")) == {filled["meas"].id}
    assert _ids(rows(hub, trust="sketch")) == {filled["sk_num"].id, filled["sk_q"].id}
    assert filled["meas"].id not in _ids(rows(hub, trust="unverified"))


def test_fidelity_filter(hub, filled):
    from simws.catalogue import rows
    assert len(rows(hub, fidelity="screening")) == len(rows(hub))
    assert rows(hub, fidelity="paper") == []


def test_newest_first(hub, filled):
    from simws.catalogue import rows
    fin = [r["finished"] for r in rows(hub, status=None)]
    assert fin == sorted(fin, reverse=True)


def test_combined_filters(hub, filled):
    from simws.catalogue import rows
    assert _ids(rows(hub, adapter="device.*", temperature=(255.0, 265.0))) == {filled["d260"].id}
    assert rows(hub, adapter="measured.*", quantity="dc.I") == []


def test_table(hub, filled):
    from simws.catalogue import rows, table
    t = table(rows(hub))
    assert isinstance(t, str) and "device.devsim" in t and "measured.cat_ds" in t
    assert isinstance(table([]), str)


def test_empty_hub(hub):
    from simws.catalogue import rows
    assert rows(hub) == []
