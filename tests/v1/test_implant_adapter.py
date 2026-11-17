"""Contract amendment A7.5 - A7.7: the hub adapter implant.rustbca (manifest, quantities, error bars, flags,
verification record, self-checks), the acceptance checks R11 (hub part) to R16, and the study layer on an implant
problem. Written from docs/ARCHITECTURE_V1.md (A7) only.

The adapter arithmetic is tested with a FAKE engine run (hand-made depth lists, monkeypatched over the rustbca engine
module's run) and, where libRustBCA is installed, against the real engine's ImplantResult for the same problem. All
expected numbers are recomputed here with numpy from the formulas of A7.5. Reference values of R16 are the table of
A7.7 (Gibbons, Johnson and Mylroie 1975 as reproduced by Chapman); the tolerances are those of A7.7."""
import math

import numpy as np
import pytest

ADAPTER = "implant.rustbca"
UM = 1e-6                                     # m per um
OUTPUTS = ["implant.Rp", "implant.dRp", "implant.lateral_straggle", "implant.fraction_implanted",
           "implant.fraction_reflected", "implant.fraction_transmitted", "implant.profile"]


def _implant(*a, **k):
    from simws.spec.library import implant_si
    return implant_si(*a, **k)


def _params(p):
    return {"problem": p.model_dump(mode="json")}


def _as_problem(problem):
    from simws.spec import ImplantProblem
    return problem if isinstance(problem, ImplantProblem) else ImplantProblem.model_validate(problem)


# ------------------------------------------------------------------------------------------------ expected values
def _std_sigma(v):
    """(s, sigma_s) of A7.5: s with k-1; sigma = (s/2) sqrt((m4/s^4 - (k-3)/(k-1)) / k)."""
    v = np.asarray(v, float)
    k = v.size
    s = v.std(ddof=1)
    m4 = np.mean((v - v.mean()) ** 4)
    return s, (s / 2.0) * math.sqrt((m4 / s ** 4 - (k - 3) / (k - 1)) / k)


def expected(problem, res):
    """name -> (value, sigma, unit) from an ImplantResult and its problem, per the table of A7.5."""
    N = res.n_ions
    x = np.asarray(res.depth_um, float) * UM
    lat = np.asarray(res.lateral_um, float) * UM
    k = x.size
    out = {}
    s = x.std(ddof=1)
    out["implant.Rp"] = (x.mean(), s / math.sqrt(k), "m")
    out["implant.dRp"] = (*_std_sigma(x), "m")
    out["implant.lateral_straggle"] = (*_std_sigma(lat), "m")
    for name, c in (("implant.fraction_implanted", k), ("implant.fraction_reflected", res.n_reflected),
                    ("implant.fraction_transmitted", res.n_transmitted)):
        p = (c + 1) / (N + 2)
        out[name] = (c / N, math.sqrt(p * (1 - p) / N), "dimensionless")
    b_um = problem.depth_bin_um
    nb = math.ceil(problem.target.thickness_um / b_um)
    edges = np.arange(nb + 1) * b_um
    counts = np.histogram(np.asarray(res.depth_um, float), bins=edges)[0].astype(float)
    b = b_um * UM
    prof = counts / (N * b)
    psig = np.sqrt(np.maximum(counts, 1.0)) / (N * b)
    centres = (np.arange(nb) + 0.5) * b
    out["implant.profile"] = (prof, psig, "1/m", centres, counts)
    if problem.beam.dose_cm2 is not None:
        dose_m2 = problem.beam.dose_cm2 * 1e4                       # cm^-2 -> m^-2
        out["implant.concentration"] = (dose_m2 * prof, dose_m2 * psig, "1/m**3", centres, counts)
    return out


def halves(res):
    x = np.asarray(res.depth_um, float)
    k = x.size
    h1, h2 = x[: k // 2], x[k // 2:]
    a, b = h1.mean(), h2.mean()
    se = math.sqrt(h1.var(ddof=1) / h1.size + h2.var(ddof=1) / h2.size)
    return a, b, (a - b) / se


def _sigma(q):
    assert q.uncertainty.kind == "sigma", (q.name, q.uncertainty.kind)
    return q.uncertainty.sigma


def check_against(r, problem, res, profile_tol_ions=0):
    """Every quantity of hub result r equals the A7.5 formula evaluated on ImplantResult res."""
    exp = expected(problem, res)
    names = sorted(q.name for q in r.quantities)
    assert names == sorted(exp), names
    for name, e in exp.items():
        q = r.q(name)
        assert q.unit == e[2], (name, q.unit)
        sig = _sigma(q)
        assert "statistical" in q.uncertainty.components, (name, q.uncertainty.components)
        assert np.allclose(q.uncertainty.components["statistical"], sig, rtol=0, atol=0), name
        if name in ("implant.profile", "implant.concentration"):
            assert len(q.axes) == 1
            ax = q.axes[0]
            assert ax.name == "depth" and ax.unit == "m"
            assert np.allclose(ax.values, e[3], rtol=1e-12, atol=0)
            if profile_tol_ions == 0:
                assert np.allclose(q.value, e[0], rtol=1e-9, atol=0), name
                assert np.allclose(sig, e[1], rtol=1e-9, atol=0), name
            else:            # real engine: allow an ion at a bin edge to fall on the other side
                Nb = res.n_ions * problem.depth_bin_um * UM
                got = np.asarray(q.value) * Nb
                if name == "implant.concentration":
                    got = got / (problem.beam.dose_cm2 * 1e4)
                assert np.abs(np.rint(got) - e[4]).sum() <= profile_tol_ions, name
        else:
            assert q.axes == [], name
            assert isinstance(q.value, (int, float)), (name, q.value)
            assert q.value == pytest.approx(e[0], rel=1e-9, abs=0), name
            assert sig == pytest.approx(e[1], rel=1e-9, abs=0), name


# ------------------------------------------------------------------------------------------------ fake engine
def _fake_result(problem, k, n_ref, n_trans, *, seed=0, mean=0.06, sd=0.025, n_ions=None, split=None,
                 flags=("implant:amorphous_target (fake engine for tests)",)):
    """A hand-made ImplantResult (test data, not physics): k depths placed well inside their bins (so bin-edge
    conventions cannot matter), drawn around `mean` um. With `split`, the first k//2 depths are shifted by -split
    and the rest by +split (a split-halves failure)."""
    from simws.engines.base import ImplantResult
    rng = np.random.default_rng(seed)
    T = problem.target.thickness_um
    b = problem.depth_bin_um
    x = rng.normal(mean, sd, k)
    if split is not None:
        x[: k // 2] -= split
        x[k // 2:] += split
    x = np.clip(x, 0.5 * b, T - 0.5 * b)
    idx = np.floor(x / b)
    x = (idx + 0.5 + rng.uniform(-0.4, 0.4, k)) * b
    lat = rng.normal(0.0, 0.02, k)
    N = problem.statistics.n_ions if n_ions is None else n_ions
    return ImplantResult(engine="rustbca", problem_id=problem.problem_id(), n_ions=N, n_reflected=n_ref,
                         n_transmitted=n_trans, depth_um=x.tolist(), lateral_um=lat.tolist(), flags=list(flags))


@pytest.fixture
def fake_engine(monkeypatch):
    """install(make) replaces the rustbca engine module's run() (and available()) by a fake: make(problem) returns
    the ImplantResult. The fake refuses what the real capability refuses, as A7.4 says run() does. Returns the list
    of problems the fake was called with."""
    from simws import engines
    from simws.engines.base import Unsupported
    mod = engines.module("rustbca")
    real_unsupported = mod.unsupported
    calls = []

    def install(make):
        def run(problem, workdir=None):
            p = _as_problem(problem)
            reasons = real_unsupported(p)
            if reasons:
                raise Unsupported("rustbca", reasons)
            calls.append(p)
            return make(p)
        monkeypatch.setattr(mod, "run", run)
        monkeypatch.setattr(mod, "available", lambda: True)
        return calls
    return install


# ------------------------------------------------------------------------------------------------ manifest / resolve
def test_manifest():
    from simws.hub import adapter as A
    from simws.hub import citations
    m = A.manifest(ADAPTER)
    assert ADAPTER in A.names()
    assert m.name == ADAPTER and m.version == "1.0" and m.env == "tcad" and m.engine == "rustbca"
    assert m.inputs == ["problem"]
    assert "drobny2021" in m.citations
    assert citations.unknown(["drobny2021"]) == []
    assert sorted(o.name for o in m.outputs) == sorted(OUTPUTS)
    assert "implant.concentration" not in [o.name for o in m.outputs]
    from simws.units import Q_
    units = {"implant.Rp": "m", "implant.dRp": "m", "implant.lateral_straggle": "m",
             "implant.fraction_implanted": "dimensionless", "implant.fraction_reflected": "dimensionless",
             "implant.fraction_transmitted": "dimensionless", "implant.profile": "1/m"}
    for o in m.outputs:
        assert Q_(1.0, o.unit).dimensionality == Q_(1.0, units[o.name]).dimensionality, o


def test_adapter_classes():
    from simws.adapters import implant as I
    from simws.hub import adapter as A
    assert issubclass(I.ImplantRustbca, I.ImplantAdapter)
    assert isinstance(A.get(ADAPTER), I.ImplantRustbca)


def test_resolve_fills_defaults_and_accepts_objects():
    from simws.hub import adapter as A
    from simws.spec import ImplantProblem
    ad = A.get(ADAPTER)
    p = _implant("P", 50, n_ions=2000)
    full = p.model_dump(mode="json")
    assert ad.resolve({"problem": p})["problem"] == full
    assert ad.resolve({"problem": full})["problem"] == full
    sparse = {k: v for k, v in full.items() if k not in ("models", "spec_version", "depth_bin_um")}
    out = ad.resolve({"problem": sparse})["problem"]
    assert out == ImplantProblem.model_validate(sparse).model_dump(mode="json")
    assert "models" in out and "depth_bin_um" in out


def test_invalid_description_raised_from_submit(hub):
    bad = _implant("P").model_dump(mode="json")
    bad["beam"]["energy_keV"] = -5.0
    with pytest.raises(Exception):                 # A1.4 / A7.5: raised from submit (type not pinned here)
        hub.submit(ADAPTER, params={"problem": bad})
    assert hub.graph.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


# ------------------------------------------------------------------------------------------------ fake-engine arithmetic
def test_values_and_sigmas_fake(hub, fake_engine):
    p = _implant("P", 50, n_ions=2000)
    res = _fake_result(p, 1960, 40, 0, seed=1)
    calls = fake_engine(lambda q: res)
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    assert len(calls) == 1 and calls[0].problem_id() == p.problem_id()
    check_against(r, p, res)
    assert "implant.concentration" not in [q.name for q in r.quantities]
    assert "budget:statistical_only" in r.flags
    assert "implant:amorphous_target (fake engine for tests)" in r.flags          # engine flags pass through
    assert not any(f.startswith("implant:transmitted:") for f in r.flags), r.flags
    assert not any(f.startswith("extrapolated:") for f in r.flags), r.flags
    # the profile integrates to the implanted fraction (A7.5)
    prof = r.q("implant.profile")
    assert np.sum(prof.array()) * p.depth_bin_um * UM == pytest.approx(r.q("implant.fraction_implanted").value,
                                                                       rel=1e-12)
    assert len(prof.value) == math.ceil(p.target.thickness_um / p.depth_bin_um) == 400
    # cache
    r2 = hub.submit(ADAPTER, params=_params(p))
    assert r2.id == r.id and r2.diagnostics.get("cache") == "hit"
    job = hub.store.job(r.id)
    assert job.key.adapter == ADAPTER and job.key.engine == "rustbca"


def test_verification_record_fake(hub, fake_engine):
    p = _implant("P", 50, n_ions=2001)
    res = _fake_result(p, 1977, 24, 0, seed=2)             # odd k: halves of 988 and 989
    fake_engine(lambda q: res)
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    v = r.verification["statistics"]
    a, b, z = halves(res)
    assert abs(z) < 3.0                                     # the test data are one distribution
    assert v["kind"] == "solution" and v["method"] == "split halves"
    assert v["n_implanted"] == 1977
    assert v["rp_halves_um"][0] == pytest.approx(a, rel=1e-12)
    assert v["rp_halves_um"][1] == pytest.approx(b, rel=1e-12)
    # A7.5 fixes "difference over its standard error" but not its sign nor the variance estimator: |z| to 1 %
    assert abs(abs(v["z"]) - abs(z)) <= 0.01 * abs(z) + 0.01, (v["z"], z)
    assert v["passed"] is True


def test_split_halves_failure_is_check_failed(hub, fake_engine):
    p = _implant("P", 50, n_ions=2000)
    res = _fake_result(p, 2000, 0, 0, seed=3, split=0.02)
    a, b, z = halves(res)
    assert abs(z) > 4.0
    fake_engine(lambda q: res)
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "check_failed", (r.status, r.message)
    v = r.verification.get("statistics")
    if v is not None:
        assert v["passed"] is False and abs(v["z"]) > 4.0


def test_counts_must_add_up(hub, fake_engine):
    p = _implant("P", 50, n_ions=2000)
    res = _fake_result(p, 1900, 50, 0, seed=4)              # 1950 != 2000
    fake_engine(lambda q: res)
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "check_failed", (r.status, r.message)


def test_fewer_than_100_implanted_is_outside_validity(hub, fake_engine):
    p = _implant("P", 50, n_ions=2000)
    res = _fake_result(p, 99, 1901, 0, seed=5)
    fake_engine(lambda q: res)
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "outside_validity", (r.status, r.message)
    assert "stopped in the target" in r.message


def test_exactly_100_implanted_is_valid(hub, fake_engine):
    p = _implant("P", 50, n_ions=2000)
    res = _fake_result(p, 100, 1900, 0, seed=6)
    fake_engine(lambda q: res)
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", (r.status, r.message)
    check_against(r, p, res)


def test_transmitted_flag_and_fractions(hub, fake_engine):
    p = _implant("P", 50, n_ions=2000)
    res = _fake_result(p, 1900, 30, 70, seed=7)
    fake_engine(lambda q: res)
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    assert any(f.startswith("implant:transmitted:") for f in r.flags), r.flags
    check_against(r, p, res)
    total = sum(r.q(n).value for n in ("implant.fraction_implanted", "implant.fraction_reflected",
                                       "implant.fraction_transmitted"))
    assert total == pytest.approx(1.0, rel=1e-12)


def test_concentration_only_with_dose(hub, fake_engine):
    p = _implant("P", 50, n_ions=2000, dose_cm2=1e15)
    res = _fake_result(p, 1980, 20, 0, seed=8)
    fake_engine(lambda q: res)
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    check_against(r, p, res)               # includes implant.concentration = dose x profile, sigma likewise
    c, prof = r.q("implant.concentration"), r.q("implant.profile")
    assert np.allclose(c.array(), 1e15 * 1e4 * prof.array(), rtol=1e-12, atol=0)
    # integral of the concentration over depth = dose x implanted fraction (in m^-2)
    integral = np.sum(c.array()) * p.depth_bin_um * UM
    assert integral == pytest.approx(1e19 * r.q("implant.fraction_implanted").value, rel=1e-12)


def test_extrapolated_flags(hub, fake_engine):
    from simws import engines
    p = _implant("As", 50, n_ions=2000)                     # 0.67 keV/amu, below the tested 1.6
    s = engines.capability("rustbca").outside_tested_range(p)
    assert s and s[0].startswith("energy_keV_per_amu:")
    fake_engine(lambda q: _fake_result(q, 1990, 10, 0, seed=9, mean=0.03, sd=0.01))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    for x in s:
        assert "extrapolated:" + x in r.flags, (x, r.flags)


def test_extrapolated_target_flag(hub, fake_engine):
    from simws import engines
    from simws.spec import ImplantProblem
    d = _implant("P", 50, n_ions=2000).model_dump(mode="json")
    d["target"]["atoms"].append({"symbol": "Ge", "Z": 32, "mass_amu": 72.63, "cutoff_eV": 1.5})
    d["target"]["layers"][0]["density_cm3"] = {"Si": 4.0e22, "Ge": 1.0e22}
    p = ImplantProblem.model_validate(d)
    s = engines.capability("rustbca").outside_tested_range(p)
    assert any(x.startswith("target:") for x in s), s
    fake_engine(lambda q: _fake_result(q, 1990, 10, 0, seed=10))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    for x in s:
        assert "extrapolated:" + x in r.flags, (x, r.flags)


def test_unsupported_is_invalid_input(hub, fake_engine):
    calls = fake_engine(lambda q: pytest.fail("the engine must not run an unsupported problem"))
    p = _implant("P", 50, n_ions=2000).with_changes({"models.recoils": True})
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "invalid_input", (r.status, r.message)
    assert "unsupported by rustbca: " in r.message
    assert "recoils:" in r.message
    assert r.quantities == []
    r = hub.submit(ADAPTER, params=_params(_implant("B", 300.0, n_ions=2000)))
    assert r.status == "invalid_input" and "max_keV_per_amu:" in r.message, r.message
    assert calls == []


def test_engine_missing_is_engine_error(hub, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "libRustBCA", None)
    r = hub.submit(ADAPTER, params=_params(_implant("P", 50, n_ions=100)))
    assert r.status == "engine_error", (r.status, r.message)
    assert "not installed" in r.message.lower(), r.message
    assert hub.graph.result_for_job(r.job_id) is None


def test_paper_fidelity_needs_10000_implanted(hub, fake_engine):
    p = _implant("P", 50, n_ions=5000)
    fake_engine(lambda q: _fake_result(q, q.statistics.n_ions - 100, 100, 0, seed=11))
    r = hub.submit(ADAPTER, params=_params(p), fidelity="paper")
    assert r.status == "check_failed", (r.status, r.message)
    assert "paper fidelity needs at least 10000 implanted ions" in r.message
    rs = hub.submit(ADAPTER, params=_params(p))              # the same run at screening fidelity is fine
    assert rs.status == "ok", rs.message


def test_paper_fidelity_with_enough_ions(hub, fake_engine):
    """With k >= 10000 the paper-fidelity rule does not fire; every quantity carries a sigma, so (inferred, not
    stated in A7) the gatekeeper accepts the result."""
    p = _implant("P", 50, n_ions=10100)
    fake_engine(lambda q: _fake_result(q, 10000, 100, 0, seed=12))
    r = hub.submit(ADAPTER, params=_params(p), fidelity="paper")
    assert "paper fidelity needs at least 10000 implanted ions" not in r.message
    assert r.status == "ok", (r.status, r.message)


def test_trust_unverified(hub, fake_engine):
    """A7.6: an implant result stays unverified (no second implant engine)."""
    from simws.hub.trust import assess_stored
    p = _implant("P", 50, n_ions=2000)
    fake_engine(lambda q: _fake_result(q, 1990, 10, 0, seed=13))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok"
    assert assess_stored(hub, r.id).level == "unverified"


# ------------------------------------------------------------------------------------------------ real engine
def _engine_result(p, tmp_path):
    from simws import engines
    return engines.module("rustbca").run(p, tmp_path)


def test_real_engine_matches_formulas(hub, tmp_path):
    pytest.importorskip("libRustBCA")
    p = _implant("P", 50, n_ions=2000, dose_cm2=5e14)
    res = _engine_result(p, tmp_path / "direct")
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    check_against(r, p, res, profile_tol_ions=2)
    assert any(f.startswith("implant:amorphous_target") for f in r.flags), r.flags
    assert "budget:statistical_only" in r.flags
    v = r.verification["statistics"]
    a, b, z = halves(res)
    assert v["n_implanted"] == len(res.depth_um)
    assert v["rp_halves_um"] == pytest.approx([a, b], rel=1e-12)
    assert v["passed"] is (abs(v["z"]) <= 4.0)
    d = r.diagnostics
    print(f"\n[adapter P 50 keV n=2000] Rp={r.q('implant.Rp').value:.6e} m  z={v['z']:.3f}  diag keys={sorted(d)}")


# ------------------------------------------------------------------------------------------------ acceptance R11-R16
@pytest.fixture(scope="module")
def rhub(tmp_path_factory):
    """One hub for the acceptance runs, so a library problem used by several checks runs once (cache)."""
    from simws.hub.jobs import Hub
    return Hub(tmp_path_factory.mktemp("implant_acceptance") / "store", mode="inprocess")


def _ok(hub, p):
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", (r.status, r.message)
    N = p.statistics.n_ions
    counts = [round(r.q(n).value * N) for n in ("implant.fraction_implanted", "implant.fraction_reflected",
                                                 "implant.fraction_transmitted")]
    assert sum(counts) == N                                   # R11: in every run
    return r


def _vs(r, name):
    q = r.q(name)
    return float(q.value), float(q.uncertainty.sigma)


def _within_4_sigma(ra, rb, name, label):
    (a, sa), (b, sb) = _vs(ra, name), _vs(rb, name)
    lim = 4.0 * math.sqrt(sa ** 2 + sb ** 2)
    print(f"\n[{label}] {name}: {a:.6e} +- {sa:.2e} vs {b:.6e} +- {sb:.2e} m; |diff|={abs(a - b):.3e}, "
          f"4 sigma={lim:.3e}, |diff|/combined sigma={abs(a - b) / (lim / 4):.2f}")
    return abs(a - b) <= lim


def test_R11_hub(hub):
    pytest.importorskip("libRustBCA")
    thin = hub.submit(ADAPTER, params=_params(_implant("P", 100, thickness_um=0.05, n_ions=2000)))
    # R11 says only "has transmitted ions and the adapter flags it"; it does not fix the status
    print(f"\n[R11] 0.05 um: status={thin.status} message={thin.message!r} flags={thin.flags}")
    assert any(f.startswith("implant:transmitted:") for f in thin.flags), (thin.status, thin.flags)
    vthin = hub.submit(ADAPTER, params=_params(_implant("P", 100, thickness_um=0.005, n_ions=2000)))
    assert vthin.status == "outside_validity", (vthin.status, vthin.message)
    assert "stopped in the target" in vthin.message


def test_R12_seeds(rhub):
    pytest.importorskip("libRustBCA")
    r1 = _ok(rhub, _implant("P", 50))
    r2 = _ok(rhub, _implant("P", 50, seed=2))
    res = {n: _within_4_sigma(r1, r2, n, "R12 seed 1 vs 2") for n in
           ("implant.Rp", "implant.dRp", "implant.lateral_straggle")}
    assert all(res.values()), res


def test_R13_layers(rhub):
    pytest.importorskip("libRustBCA")
    from simws.spec import ImplantProblem
    p = _implant("P", 50)
    d = p.model_dump(mode="json")
    lay = d["target"]["layers"][0]
    d["target"]["layers"] = [dict(lay, name="si_top", thickness_um=0.03), dict(lay, name="si", thickness_um=1.97)]
    p2 = ImplantProblem.model_validate(d)
    assert p2.target.thickness_um == pytest.approx(2.0, rel=1e-15)
    r1, r2 = _ok(rhub, p), _ok(rhub, p2)
    res = {n: _within_4_sigma(r2, r1, n, "R13 two layers vs one") for n in ("implant.Rp", "implant.dRp")}
    assert all(res.values()), res


def test_R14_ion_modelling_choice(rhub):
    pytest.importorskip("libRustBCA")
    p = _implant("P", 50)
    p2 = p.with_changes({"beam.ion.cutoff_eV": 5.0, "beam.ion.surface_binding_eV": 3.0})
    r1, r2 = _ok(rhub, p), _ok(rhub, p2)
    res = {n: _within_4_sigma(r2, r1, n, "R14 Ec 5 / Es 3 vs default") for n in ("implant.Rp", "implant.dRp")}
    assert all(res.values()), res


def test_R15_trends(rhub):
    pytest.importorskip("libRustBCA")
    rp = {}
    for ion, e, tilt in (("P", 100.0, 0.0), ("P", 50.0, 0.0), ("B", 50.0, 0.0), ("As", 50.0, 0.0),
                         ("P", 50.0, 60.0)):
        rp[(ion, e, tilt)] = _ok(rhub, _implant(ion, e, tilt_deg=tilt)).q("implant.Rp").value
    ratio = rp[("P", 50.0, 60.0)] / rp[("P", 50.0, 0.0)]
    print(f"\n[R15] Rp(P100)={rp[('P', 100.0, 0.0)]:.5e}  Rp(P50)={rp[('P', 50.0, 0.0)]:.5e}  "
          f"Rp(B50)={rp[('B', 50.0, 0.0)]:.5e}  Rp(As50)={rp[('As', 50.0, 0.0)]:.5e}  "
          f"Rp(P50, 60 deg)={rp[('P', 50.0, 60.0)]:.5e} m; tilt ratio={ratio:.4f}")
    assert rp[("P", 100.0, 0.0)] > rp[("P", 50.0, 0.0)]
    assert rp[("B", 50.0, 0.0)] > rp[("P", 50.0, 0.0)] > rp[("As", 50.0, 0.0)]
    assert 0.45 <= ratio <= 0.60


R16_GATED = [("B", 50.0, 0.1608, 0.0504), ("B", 100.0, 0.2994, 0.0710), ("P", 50.0, 0.0607, 0.0256)]
# Marked by the lead after the checker's report (contract A7.9): the two boron cases are a recorded FAIL of R16
# (Rp +34 % and +36 %, dRp +43 % and +51 % against a 25 % gate). The gate is unchanged; strict, so a pass is noticed.
_R16_BORON_FAIL = pytest.mark.xfail(strict=True, reason="recorded FAIL of R16 for boron (contract A7.9): RustBCA's "
                                    "Lindhard-Scharff stopping is too weak for boron in silicon, ranges ~35 % too deep")
_R16_PARAMS = [pytest.param(*c, id=f"{c[0]}{int(c[1])}keV", marks=[_R16_BORON_FAIL] if c[0] == "B" else [])
               for c in R16_GATED]


@pytest.mark.parametrize("ion,E,rp_ref,drp_ref", _R16_PARAMS)
def test_R16_tables(rhub, ion, E, rp_ref, drp_ref):
    pytest.importorskip("libRustBCA")
    r = _ok(rhub, _implant(ion, E))
    ok = {}
    for name, ref in (("implant.Rp", rp_ref), ("implant.dRp", drp_ref)):
        v, s = _vs(r, name)
        v_um, s_um = v / UM, s / UM
        rel = v_um / ref - 1
        ok[name] = abs(v_um - ref) <= 0.25 * ref
        print(f"\n[R16 {ion} {E:g} keV] {name}: {v_um:.5f} +- {s_um:.5f} um, table {ref} um, "
              f"relative difference {rel:+.4f} ({'within' if ok[name] else 'OUTSIDE'} 25 %)")
    assert all(ok.values()), ok


def test_R16_P100_reported_not_gated(rhub):
    pytest.importorskip("libRustBCA")
    r = _ok(rhub, _implant("P", 100.0))
    for name, ref in (("implant.Rp", 0.1238), ("implant.dRp", 0.0456)):
        v, s = _vs(r, name)
        print(f"\n[R16 P 100 keV, no gate] {name}: {v / UM:.5f} +- {s / UM:.5f} um, table {ref} um, "
              f"relative difference {v / UM / ref - 1:+.4f}")


# ------------------------------------------------------------------------------------------------ study
def test_study_sweep_through_implant_adapter(hub):
    pytest.importorskip("libRustBCA")
    from simws.study import Study
    p = _implant("P", 100, n_ions=2000)
    res = Study(hub, p).sweep({"beam.energy_keV": [50.0, 100.0]})
    assert [pt["changes"] for pt in res.points] == [{"beam.energy_keV": 50.0}, {"beam.energy_keV": 100.0}]
    for pt in res.points:
        assert pt["engine"] == "rustbca" and pt["status"] == "ok", pt
        job = hub.store.job(pt["result_id"])
        assert job.key.adapter == ADAPTER and job.fidelity == "screening"
        assert pt["trust"] == "unverified" and pt["in_range"] is True
        r = hub.result(pt["result_id"])
        direct = hub.submit(ADAPTER, params=_params(p.with_changes(pt["changes"])))
        assert direct.id == r.id                               # the study submitted exactly this problem
    t = res.table("implant.Rp")
    assert isinstance(t, str) and "implant.Rp" in t
    qs = res.quantity("implant.Rp")
    assert len(qs) == 2 and qs[1][1].value > qs[0][1].value
    print("\n" + t)


def test_study_refuses_device_engine_on_implant_problem(hub):
    from simws.study import Study
    res = Study(hub, _implant("P", 100, n_ions=2000), engines=["devsim"]).sweep({})
    (pt,) = res.points
    assert pt["status"] == "refused" and pt["result_id"] is None
    assert len(pt["reasons"]) == 1 and pt["reasons"][0].startswith("problem kind:")
    assert hub.graph.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


# ------------------------------------------------------------------------------------------------ contract A7.9
# Added by the lead with amendment A7.9 (changes made because of the boron FAIL).
def test_A79_boron_result_is_flagged_and_out_of_range(rhub):
    pytest.importorskip("libRustBCA")
    from simws.hub import trust
    r = _ok(rhub, _implant("B", 50.0, n_ions=2000))
    assert any(f.startswith("implant:known_failure:") for f in r.flags), r.flags
    assert any(f.startswith("extrapolated:ion:") for f in r.flags), r.flags
    t = trust.assess_stored(rhub, r.id)
    assert t.level == "unverified" and t.in_range is False
    rp = _ok(rhub, _implant("P", 50.0, n_ions=2000))
    assert not any(f.startswith(("implant:known_failure:", "extrapolated:")) for f in rp.flags), rp.flags
    assert trust.assess_stored(rhub, rp.id).in_range is True
