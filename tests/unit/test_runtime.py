"""0.9c parallel runtime: Dask (or a process pool) runs hub jobs with the same keys, cache and commits."""
import numpy as np
import pytest

from simws.hub.runtime import Runtime


@pytest.mark.parametrize("backend", ["processes", "dask"])
def test_runtime_runs_jobs_once(tmp_path, backend):
    if backend == "dask":
        pytest.importorskip("dask.distributed")
    Ts = [[250.0, 300.0], [300.0, 350.0], [250.0, 300.0]]          # the third repeats the first
    params = [{"material": "silicon", "T": (np.array(t), "K")} for t in Ts]
    # inprocess: the workers' own interpreter runs the adapter, so the test does not depend on the env layout
    with Runtime(tmp_path / "store", n_workers=2, backend=backend, mode="inprocess") as rt:
        res = rt.map("materials.intrinsic", params)
        again = rt.map("materials.intrinsic", params[:1])
    assert all(r.status == "ok" for r in res)
    assert res[0].id == res[2].id != res[1].id
    assert again[0].id == res[0].id and again[0].diagnostics.get("cache") == "hit"


def test_model_form_attach(tmp_path):
    from simws.hub import model_form as MF
    from simws.hub.schema import Quantity, ResultEnvelope
    from simws.hub.uncertainty import budget
    MF.record(tmp_path, {"devsim.current": 0.01}, source="test")
    q = Quantity(name="hbt.IC", value=[1.0, 2.0], unit="A", uncertainty=budget([1.0, 2.0], numerical=[0.03, 0.04]))
    other = Quantity(name="hbt.VBE", value=[0.7], unit="V", uncertainty=budget([0.7], numerical=[0.001]))
    r = ResultEnvelope(job_id="x", status="ok", quantities=[q, other])
    assert MF.attach(r, tmp_path) == ["devsim.current"]
    u = r.quantities[0].uncertainty
    assert u.components["model_form"] == [0.01, 0.02]
    assert abs(u.sigma[1] - (0.04 ** 2 + 0.02 ** 2) ** 0.5) < 1e-15
    assert "model_form" not in r.quantities[1].uncertainty.components
