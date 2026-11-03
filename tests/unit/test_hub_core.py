"""Unit tests for simws 0.8: the hub core contract (canonical IDs, schemas, conventions, store, graph, gatekeeper,
job submission). No engine needed: the materials.intrinsic adapter is pure Python."""
import json

import numpy as np
import pytest

pytest.importorskip("pint")
pytest.importorskip("pydantic")
pytest.importorskip("networkx")

from simws import conventions as cv  # noqa: E402
from simws.hub import canonical as C  # noqa: E402
from simws.hub.schema import (JobEnvelope, JobKey, Manifest, OutputSpec, Quantity, ResultEnvelope,  # noqa: E402
                              Uncertainty)

pydantic = pytest.importorskip("pydantic")


# ------------------------------------------------------------------------------------------------ canonical
def test_canonical_units_and_exceptions():
    assert C.canonical_quantity(1.0, "V")["dims"] == C.canonical_quantity(1.0, "kg*m**2/A/s**3")["dims"]
    assert C.canonical_quantity(2.0, "mV")["value"] == pytest.approx(2e-3)
    assert C.canonical_unit("J") == "eV"
    assert C.canonical_unit("MeV*cm**2/g") == "MeV*cm**2/mg"
    assert C.canonical_quantity(1e15, "cm**-3")["value"] == pytest.approx(1e21)


def test_job_id_independent_of_unit_spelling_and_key_order():
    a = {"T": C.canonical_quantity(300, "K"), "N": C.canonical_quantity(1e17, "cm**-3"), "m": "Si"}
    b = {"m": "Si", "N": C.canonical_quantity(1e23, "m**-3"), "T": C.canonical_quantity(300.0, "kelvin")}
    assert C.job_id(a) == C.job_id(b)
    assert len(C.job_id(a)) == 32
    assert C.job_id({"x": 1.0}) != C.job_id({"x": 1.0 + 1e-12})


def test_canonical_refuses_nan_and_paths():
    from pathlib import Path
    with pytest.raises(C.CanonicalError):
        C.canonical_json({"x": float("nan")})
    with pytest.raises(C.CanonicalError):
        C.canonical_json({"p": Path("/tmp/x")})
    assert C.canonical_json({"a": np.arange(3), "b": np.float64(0.5)}) == '{"a":[0,1,2],"b":0.5}'


# ------------------------------------------------------------------------------------------------ schema
def test_schema_refuses_unknown_fields_and_bad_units():
    with pytest.raises(pydantic.ValidationError):
        Quantity(name="x", value=1.0, unit="V", colour="red")
    with pytest.raises(Exception):
        Quantity(name="x", value=1.0, unit="not_a_unit")
    with pytest.raises(pydantic.ValidationError):
        Uncertainty(kind="sigma")
    q = Quantity(name="x", value=np.array([1.0, 2.0]), unit="mV")
    assert q.to("V") == pytest.approx([1e-3, 2e-3])
    r = ResultEnvelope.model_validate_json(ResultEnvelope(job_id="j", status="ok", quantities=[q]).model_dump_json())
    assert r.q("x").value == [1.0, 2.0]


# ------------------------------------------------------------------------------------------------ conventions
@pytest.mark.parametrize("axis", ["stress", "terminal_current", "band_offset", "doping", "strain_shear"])
def test_conventions_round_trip(axis):
    x = np.array([1.0, -2.0, 3.0, 0.4, -0.5, 0.6])
    alt = [a for a in cv.ALTERNATIVES[axis] if a != cv.CANONICAL[axis]][0]
    assert np.allclose(cv.from_canonical(axis, cv.to_canonical(axis, x, alt), alt), x)


def test_conventions_specific_values():
    assert cv.convert("stress", [1.0], "compressive+")[0] == -1.0
    assert np.allclose(cv.convert("strain_shear", [0, 0, 0, 0.1, 0.2, 0.3], "tensor"), [0, 0, 0, 0.2, 0.4, 0.6])
    v = cv.convert("voigt_order", [1, 2, 3, 4, 5, 6], "xx,yy,zz,xy,xz,yz")      # LAMMPS -> Voigt
    assert list(v) == [1, 2, 3, 6, 5, 4]
    # trap level: Et-Ei = 0 at midgap-ish Ei; Ec-Et of a level 0.3 eV below Ec
    assert cv.convert("trap_level", 0.3, "Ec-Et", Eg=1.12, Ei_minus_Ev=0.56) == pytest.approx(0.26)
    assert cv.convert("dose", 100.0, "rad(SiO2)") == pytest.approx(1.0)
    with pytest.raises(cv.ConventionError):
        cv.convert("dose", 1.0, "rad")
    with pytest.raises(cv.ConventionError):
        cv.convert("dose", 1.0, "Gy(Si)", "Gy(SiO2)")
    assert cv.convert("per_cell", 2.0, "per_atom", n_atoms=8) == 16.0


def test_kpoint_round_trip():
    a = 10.26
    cell = 0.5 * a * np.array([[-1, 0, 1], [0, 1, 1], [-1, 1, 0]])
    k = np.array([[0.5, 0.5, 0.5], [0.0, 0.5, 0.5]])
    t = cv.convert("kpoints", k, "crystal", "tpiba", cell=cell, alat=a)
    assert np.allclose(cv.convert("kpoints", t, "tpiba", "crystal", cell=cell, alat=a), k)


def test_frames_and_bond_rotation():
    R = cv.rotation_from_wafer([0, 0, 1], [1, 1, 0])
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.allclose(cv.rotate_vector(R, [0, 0, 1]), [0, -1, 0])        # wafer normal -> -y (depth is +y)
    C11, C12, C44 = 165.7, 63.9, 79.6
    Cc = np.zeros((6, 6)); Cc[:3, :3] = C12; np.fill_diagonal(Cc[:3, :3], C11); Cc[3:, 3:] = np.eye(3) * C44
    Cr = cv.rotate_stiffness(R, Cc)
    # stiffness along <110>: C11' = (C11 + C12 + 2 C44) / 2
    assert Cr[0, 0] == pytest.approx((C11 + C12 + 2 * C44) / 2)
    # tensor rotation agrees with the Bond matrix on a strain
    eps = np.array([1e-3, -2e-4, 5e-4, 1e-4, -3e-4, 2e-4])
    lhs = cv.tensor_to_voigt(cv.rotate_tensor(R, cv.voigt_to_tensor(eps)))
    M = cv.bond_matrix(R)
    assert np.allclose(np.linalg.inv(M).T @ eps, lhs)                 # engineering strain rotates with M^-T
    sig = Cc @ eps                                                     # and stress with M: sigma' = C' eps'
    assert np.allclose(M @ sig, Cr @ lhs)


# ------------------------------------------------------------------------------------------------ store / graph / jobs
@pytest.fixture
def hub(tmp_path):
    from simws.hub.jobs import Hub
    return Hub(tmp_path / "store", mode="inprocess")


P = {"material": "silicon", "profile": None, "T": ([250, 300, 350], "K")}


def test_submit_commit_and_cache(hub):
    r = hub.submit("materials.intrinsic", params=P)
    assert r.status == "ok" and len(r.id) == 32
    assert hub.store.has(r.id)
    ni = r.q("mat.n_i").to("cm**-3")
    assert ni[1] == pytest.approx(9.65e9, rel=0.02)
    assert "unverified" in r.q("mat.n_i").flags                       # screening, no uncertainty
    r2 = hub.submit("materials.intrinsic", params={**P, "T": ([0.25, 0.3, 0.35], "kK")})
    assert r2.id == r.id and r2.diagnostics.get("cache") == "hit"    # unit spelling does not change the key
    job = hub.store.job(r.id)
    assert job.key.env_lock and job.key.conventions_version == cv.VERSION


def test_extrapolation_flagged_not_rejected(hub):
    r = hub.submit("materials.intrinsic", params={**P, "T": ([77, 300], "K")})
    assert r.status == "ok"
    assert "extrapolated" in r.flags and "extrapolated" in r.q("mat.n_i").flags


def test_failure_category_and_ledger(hub):
    r = hub.submit("materials.intrinsic", params={**P, "material": "unobtainium"})
    assert r.status == "invalid_input"
    ev = hub.graph.events()
    assert ev[-1]["type"] == "failed" and ev[-1]["data"]["status"] == "invalid_input"
    assert hub.graph.result_for_job(r.job_id) is None                 # failures are never served from cache


def test_paper_fidelity_requires_uncertainty(hub):
    r = hub.submit("materials.intrinsic", params=P, fidelity="paper")
    assert r.status == "check_failed" and "uncertainty" in r.message


def test_graph_lineage_staleness_and_networkx(hub):
    a = hub.submit("materials.intrinsic", params=P)
    b = hub.submit("materials.intrinsic", params={**P, "T": ([300], "K")}, inputs={"upstream": f"{a.id}:mat.Eg"})
    assert a.id in hub.graph.ancestors(f"{b.id}:mat.n_i")
    assert not hub.graph.is_stale(f"{b.id}:mat.n_i")
    c = hub.submit("materials.intrinsic", params={**P, "T": ([250, 300, 351], "K")}, supersedes=a.id)
    assert c.status == "ok" and hub.graph.is_stale(f"{b.id}:mat.n_i")
    G = hub.graph.to_networkx()
    assert G.nodes[a.id]["type"] == "Result"
    assert any(d["type"] == "depends_on" for _, _, d in G.edges(data=True))


def test_gatekeeper_catches_missing_output_wrong_unit_and_convention():
    from simws.hub import gatekeeper
    m = Manifest(name="t", version="1", env="x", engine="y",
                 outputs=[OutputSpec(name="a", unit="V"), OutputSpec(name="b", unit="A", constraint="positive")])
    job = JobEnvelope(id="j", key=JobKey(adapter="t", adapter_version="1", engine="y", engine_version="0", env_lock="z"),
                      env="x")
    res = ResultEnvelope(job_id="j", status="ok", quantities=[
        Quantity(name="a", value=1.0, unit="m"),
        Quantity(name="c", value=1.0, unit="Pa", conventions={"stress": "compressive+"})])
    rep = gatekeeper.check(job, res, m)
    text = " ".join(rep.errors)
    assert "'b' missing" in text and "wrong dimension" in text and "canonical is 'tensile+'" in text
    res2, _ = gatekeeper.apply(job, res, m)
    assert res2.status == "check_failed"


def test_prune_until_validated(tmp_path):
    from simws.hub.schema import FieldRef
    from simws.hub.store import Store
    s = Store(tmp_path)
    st = s.staging("job1")
    (st / "data" / "big.npy").write_bytes(b"x" * 1000)
    (st / "data" / "tmp.npy").write_bytes(b"y" * 10)
    env = ResultEnvelope(job_id="job1", status="ok", fields=[
        FieldRef(name="psi", path="big.npy", mesh_id="m", unit="V"),
        FieldRef(name="t", path="tmp.npy", mesh_id="m", unit="V", retention="scratch")])
    env = s.finalize(st, env)
    assert [f.name for f in env.fields] == ["psi"]
    assert not (s.object_dir(env.id) / "data" / "tmp.npy").exists()
    assert s.prune(env.id, validated=False) == 0
    assert s.prune(env.id, validated=True) == 1000
    assert json.loads((s.object_dir(env.id) / "envelope.json").read_text())["id"] == env.id


def test_hbt_gummel_single_bias_point(hub):
    """A one-point sweep is a valid request (the upgrade gate uses VBE = 0.6 V alone); found on the first install."""
    pytest.importorskip("devsim")
    r = hub.submit("devsim.hbt_gummel", params={"case": {"ge_peak": 0.15}, "VBE": ([0.6], "V"), "intrinsic_jn": True})
    assert r.status == "ok", r.message
    assert len(r.q("hbt.IC").value) == 1 and r.q("hbt.Jn_intrinsic").value[0] > 0
