"""Unit tests for simws 0.9a: the design-review fixes to the hub."""
import pytest

pytest.importorskip("pydantic")

from simws.hub import adapter as A  # noqa: E402
from simws.hub import jobs as J  # noqa: E402

P = {"material": "silicon", "profile": None}


@pytest.fixture
def hub(tmp_path):
    return J.Hub(tmp_path / "store", mode="inprocess")


def test_float_noise_does_not_change_the_job(hub):
    a = hub.make_job("materials.intrinsic", {**P, "T": ([0.1 + 0.2], "kK")})
    b = hub.make_job("materials.intrinsic", {**P, "T": ([0.3], "kK")})
    assert a.id == b.id
    c = hub.make_job("materials.intrinsic", {**P, "T": ([300.000001], "K")})
    assert c.id != a.id                                     # real changes still count


def test_env_lock_hashes_only_physics_packages(tmp_path):
    env = tmp_path / "envs"; env.mkdir()
    base = ["@EXPLICIT", "https://conda.anaconda.org/conda-forge/linux-64/numpy-1.26.4-py312h1234_0.conda#abc",
            "https://conda.anaconda.org/conda-forge/linux-64/matplotlib-base-3.8.0-py312h00_0.conda#def"]
    (env / "tcad.explicit.txt").write_text("\n".join(base))
    (env / "tcad.pip.txt").write_text("devsim==2.11.2\npydantic==2.14.2\n")
    l0 = J.env_lock("tcad", root=tmp_path)
    (env / "tcad.explicit.txt").write_text("\n".join(base).replace("matplotlib-base-3.8.0", "matplotlib-base-3.9.1"))
    (env / "tcad.pip.txt").write_text("devsim==2.11.2\npydantic==2.15.0\n")
    assert J.env_lock("tcad", root=tmp_path) == l0           # plotting and validation libraries: same lock
    (env / "tcad.pip.txt").write_text("devsim==2.11.3\npydantic==2.15.0\n")
    assert J.env_lock("tcad", root=tmp_path) != l0           # engine version: new lock
    assert J.env_lock("nope", root=tmp_path) == "unlocked:nope"


def test_physics_version_not_code_version_enters_the_key(hub, monkeypatch):
    m = A.manifest("materials.intrinsic")
    k0 = hub.make_job("materials.intrinsic", {**P, "T": ([300.0], "K")}).id
    monkeypatch.setattr(m, "version", "1.0.1")                # log-text fix
    monkeypatch.setattr(m, "physics_version", "1.0")
    assert hub.make_job("materials.intrinsic", {**P, "T": ([300.0], "K")}).id == k0
    monkeypatch.setattr(m, "physics_version", "1.1")          # model change
    assert hub.make_job("materials.intrinsic", {**P, "T": ([300.0], "K")}).id != k0


def test_store_refuses_windows_and_network_drives(tmp_path, monkeypatch):
    from simws.hub import store as S
    with pytest.raises(S.StoreLocationError):
        S.check_location("/mnt/c/Users/me/store")
    monkeypatch.setattr(S, "filesystem_type", lambda p: "nfs4")
    with pytest.raises(S.StoreLocationError):
        S.check_location(tmp_path)
    monkeypatch.setenv("SIMWS_STORE_ALLOW_UNSAFE", "1")
    S.check_location(tmp_path)


def test_batched_members_one_job(hub):
    from simws.hub import uncertainty as U
    members = {U.member_id(material=m): {"material": m} for m in ("silicon", "sic_4h")}
    rs = hub.submit_members("materials.intrinsic", members, params={**P, "T": ([300.0], "K")})
    assert len(rs) == 1 and rs[0].status == "ok" and rs[0].diagnostics["batched"]
    ens = U.Ensemble.from_quantities([q for q in rs[0].quantities if q.name == "mat.Eg"])
    assert set(ens.members) == set(members)
    assert ens.members["material=sic_4h"][0] > 3.0 > ens.members["material=silicon"][0]
    bad = hub.submit_members("materials.intrinsic", {**members, "material=x": {"material": "unobtainium"}},
                             params={**P, "T": ([300.0], "K")})
    assert bad[0].status == "invalid_input" and "material=x" in bad[0].message


def test_sketch_mode(hub):
    from simws.hub import gatekeeper
    from simws.hub.schema import Manifest

    def debye(N_cm3, T_K):
        import math
        L = math.sqrt(11.7 * 8.854e-14 * 0.025852 * T_K / 300 / (1.602e-19 * N_cm3))
        return {"x.debye": (L * 1e-2, "m")}

    r = hub.sketch(debye, {"N_cm3": 1e16, "T_K": 300.0})
    assert r.status == "ok" and "sketch" in r.flags and r.q("x.debye").to("nm") == pytest.approx(41.1, rel=0.01)
    assert hub.sketch(debye, {"N_cm3": 1e16, "T_K": 300.0}).diagnostics.get("cache") == "hit"
    job = hub.store.job(r.id)
    job.fidelity = "paper"
    rep = gatekeeper.check(job, r, Manifest(name="sketch.debye", version="x", env="local", engine="python"))
    assert any("sketch" in e for e in rep.errors)


def test_coupling_registry_rule():
    from simws.hub import coupling as C
    assert C.check() == []
    bad = [C.Coupling("a", "b", "q", "two_way", "store")]
    assert any("inside one adapter" in p for p in C.check(bad))


def test_relink_after_key_change(hub, monkeypatch):
    from simws.hub import relink as R
    rs = [hub.submit("materials.intrinsic", params={**P, "T": ([t], "K")}) for t in (250.0, 300.0, 350.0)]
    assert all(r.status == "ok" for r in rs)
    monkeypatch.setattr(J, "env_lock", lambda env, root=None: "upgraded-env")
    stale = R.stale_jobs(hub)
    assert len(stale["materials.intrinsic"]) == 3
    rep = R.relink(hub, canaries=1, log=lambda *a: None)
    assert rep["materials.intrinsic"]["reproduced"] and rep["materials.intrinsic"]["linked"] == 2
    again = hub.submit("materials.intrinsic", params={**P, "T": ([350.0], "K")})
    assert again.diagnostics.get("cache") == "hit" and again.id == rs[2].id
    assert R.stale_jobs(hub) == {}
