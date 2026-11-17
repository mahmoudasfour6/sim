"""Contract amendment A8.5 - A8.7: the hub adapter see.g4see (manifest, quantities, error bars, flags, self-checks,
trust), the study layer on a SEE problem, and the acceptance checks R21 to R28. Written from
docs/ARCHITECTURE_V1.md (A8, with the shared conventions of A7) only.

The adapter arithmetic is tested with a FAKE engine run (hand-made SeeResult objects, monkeypatched over the g4see
engine module's run() and available()) and, where G4SEE is installed, against the real engine's SeeResult for the
same problem. Expected numbers are recomputed here with numpy from the formulas of A8.5.

Reference values of R22 to R25 are recomputed here by the method of A8.7 (RK4, step 0.01 um, NIST total stopping
power interpolated linearly in log S against log E, rho = 2.33 g/cm3) from the rows of the NIST ASTAR / PSTAR silicon
tables copied below; the acceptance gates use the contract's numbers, with the contract's tolerances."""
import bisect
import math

import numpy as np
import pytest

ADAPTER = "see.g4see"
EV = 1e6                                         # eV per MeV
OUTPUTS = ["see.edep_spectrum", "see.p_above", "see.fraction_underflow", "see.fraction_overflow"]
UNITS = {"see.edep_spectrum": "1/eV", "see.p_above": "dimensionless", "see.fraction_underflow": "dimensionless",
         "see.fraction_overflow": "dimensionless", "see.edep_mean": "eV", "see.edep_std": "eV"}


def _slab(*a, **k):
    from simws.spec.library import see_si_slab
    return see_si_slab(*a, **k)


def _params(p):
    return {"problem": p.model_dump(mode="json")}


def _as_problem(problem):
    from simws.spec import SeeProblem
    return problem if isinstance(problem, SeeProblem) else SeeProblem.model_validate(problem)


def ref_edges(h):
    i = np.arange(h.bins + 1, dtype=float)
    if h.scale == "lin":
        return h.low_MeV + i * (h.high_MeV - h.low_MeV) / h.bins
    lo, hi = math.log10(h.low_MeV), math.log10(h.high_MeV)
    return 10.0 ** (lo + i * (hi - lo) / h.bins)


def _g4():
    from simws import engines
    return engines.module("g4see")


def _need_engine():
    try:
        ok = _g4().available()
    except Exception as e:                                      # noqa: BLE001
        pytest.skip(f"g4see engine module not importable: {e!r}")
    if not ok:
        pytest.skip("G4SEE not available (SIMWS_G4SEE_PREFIX / <envs_root>/g4see)")


# ------------------------------------------------------------------------------------------------ expected values
def _binom_sigma(k, N):
    p = (k + 1.0) / (N + 2.0)
    return np.sqrt(p * (1.0 - p) / N)


def expected(problem, res):
    """name -> dict(value, sigma, unit, axis, components) from a SeeResult and its problem (table of A8.5)."""
    N = float(res.n_events)
    e = np.asarray(res.edges_MeV, float) * EV
    c = np.asarray(res.counts, float)
    w = np.diff(e)
    m = 0.5 * (e[1:] + e[:-1]) if problem.histogram.scale == "lin" else np.sqrt(e[1:] * e[:-1])
    out = {}
    sig = np.sqrt(np.maximum(c, 1.0)) / (N * w)
    out["see.edep_spectrum"] = dict(value=c / (N * w), sigma=sig, axis=("edep", "eV", m),
                                    components={"statistical": sig})
    k = np.cumsum(c[::-1])[::-1] + res.overflow
    sig = _binom_sigma(k, N)
    out["see.p_above"] = dict(value=k / N, sigma=sig, axis=("threshold", "eV", e[:-1]),
                              components={"statistical": sig})
    for name, cnt in (("see.fraction_underflow", res.underflow), ("see.fraction_overflow", res.overflow)):
        sig = float(_binom_sigma(cnt, N))
        out[name] = dict(value=cnt / N, sigma=sig, axis=None, components={"statistical": sig})
    if res.underflow == 0 and res.overflow == 0:
        mean = float(np.sum(c * m) / N)
        s = math.sqrt(float(np.sum(c * (m - mean) ** 2)) / (N - 1))
        b = float(np.max(w[c > 0])) / math.sqrt(12.0)
        stat = math.sqrt(s ** 2 / N)
        out["see.edep_mean"] = dict(value=mean, sigma=math.sqrt(s ** 2 / N + b ** 2), axis=None,
                                    components={"statistical": stat, "binning": b})
        m4 = float(np.sum(c * (m - mean) ** 4)) / N
        ssig = (s / 2.0) * math.sqrt((m4 / s ** 4 - (N - 3) / (N - 1)) / N)
        out["see.edep_std"] = dict(value=s, sigma=ssig, axis=None, components={"statistical": ssig})
    return out


def _sigma(q):
    assert q.uncertainty.kind == "sigma", (q.name, q.uncertainty.kind)
    return q.uncertainty.sigma


def check_against(r, problem, res):
    """Every quantity of hub result r equals the A8.5 formula evaluated on SeeResult res."""
    exp = expected(problem, res)
    names = sorted(q.name for q in r.quantities)
    assert names == sorted(exp), names
    for name, e in exp.items():
        q = r.q(name)
        assert q.unit == UNITS[name], (name, q.unit)
        sig = _sigma(q)
        comps = q.uncertainty.components
        assert sorted(comps) == sorted(e["components"]), (name, sorted(comps))
        for cn, cv in e["components"].items():
            assert np.allclose(np.asarray(comps[cn], float), cv, rtol=1e-9, atol=0), (name, cn)
        if e["axis"] is not None:
            assert len(q.axes) == 1, name
            ax = q.axes[0]
            assert (ax.name, ax.unit) == e["axis"][:2], (name, ax.name, ax.unit)
            assert np.allclose(ax.values, e["axis"][2], rtol=1e-12, atol=0), name
            assert np.allclose(np.asarray(q.value, float), e["value"], rtol=1e-9, atol=0), name
            assert np.allclose(np.asarray(sig, float), e["sigma"], rtol=1e-9, atol=0), name
        else:
            assert q.axes == [], name
            assert isinstance(q.value, (int, float)), (name, q.value)
            assert q.value == pytest.approx(e["value"], rel=1e-9, abs=0), name
            assert float(sig) == pytest.approx(e["sigma"], rel=1e-9, abs=0), name


# ------------------------------------------------------------------------------------------------ fake engine
def _fake_result(problem, *, seed=0, under=0.0, over=0.0, centre=0.3, width=0.04, n_events=None, extra=0.0,
                 flags=("see:em_only (fake engine for tests)",)):
    """A hand-made SeeResult (test data, not physics): N - under - over events spread over the bins around
    `centre` (a fraction of the histogram range), plus `extra` (to break the bookkeeping on purpose)."""
    from simws.engines.base import SeeResult
    rng = np.random.default_rng(seed)
    N = problem.statistics.n_events if n_events is None else n_events
    h = problem.histogram
    pdf = np.exp(-0.5 * ((np.arange(h.bins) / h.bins - centre) / width) ** 2)
    n_in = int(round(N - under - over))
    counts = rng.multinomial(n_in, pdf / pdf.sum()).astype(float)
    counts[int(np.argmax(counts))] += extra
    return SeeResult(engine="g4see", problem_id=problem.problem_id(), n_events=N, edges_MeV=ref_edges(h).tolist(),
                     counts=counts.tolist(), underflow=float(under), overflow=float(over), flags=list(flags))


@pytest.fixture
def fake_engine(monkeypatch):
    """install(make) replaces the g4see engine module's run() and available() by a fake: make(problem) returns the
    SeeResult. The fake refuses what the real capability refuses, as A8.4 says run() does. Returns the list of
    problems the fake was called with."""
    from simws.engines.base import Unsupported
    mod = _g4()
    real_unsupported = mod.unsupported
    calls = []

    def install(make):
        def run(problem, workdir=None, *a, **k):
            p = _as_problem(problem)
            reasons = real_unsupported(p)
            if reasons:
                raise Unsupported("g4see", reasons)
            calls.append(p)
            return make(p)
        monkeypatch.setattr(mod, "run", run)
        monkeypatch.setattr(mod, "available", lambda: True)
        return calls
    return install


# ------------------------------------------------------------------------------------------------ manifest
def test_manifest():
    from simws.hub import adapter as A
    from simws.hub import citations
    from simws.units import Q_
    m = A.manifest(ADAPTER)
    assert ADAPTER in A.names()
    assert m.name == ADAPTER and m.version == "1.0" and m.env == "tcad" and m.engine == "g4see"
    assert m.inputs == ["problem"]
    assert "lucsanyi2022" in m.citations
    assert citations.unknown(["lucsanyi2022"]) == []
    assert sorted(o.name for o in m.outputs) == sorted(OUTPUTS)
    for o in m.outputs:
        assert Q_(1.0, o.unit).dimensionality == Q_(1.0, UNITS[o.name]).dimensionality, o


def test_adapter_classes():
    from simws.adapters import see as S
    from simws.hub import adapter as A
    assert issubclass(S.SeeG4see, S.SeeAdapter)
    assert isinstance(A.get(ADAPTER), S.SeeG4see)


def test_invalid_description_raised_from_submit(hub):
    bad = _slab(n_events=1000).model_dump(mode="json")
    bad["geometry"]["sv_thickness_um"] = 500.0                   # beyond the bulk
    with pytest.raises(Exception):                                 # A7.5 / A8.5: raised from submit
        hub.submit(ADAPTER, params={"problem": bad})
    bad = _slab(n_events=1000).model_dump(mode="json")
    bad["source"]["ion_Z"] = 2                                     # forbidden for an alpha
    with pytest.raises(Exception):
        hub.submit(ADAPTER, params={"problem": bad})
    assert hub.graph.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


# ------------------------------------------------------------------------------------------------ fake-engine arithmetic
def test_values_and_sigmas_lin(hub, fake_engine):
    p = _slab(n_events=10000, bins=200)
    res = _fake_result(p, seed=1)
    calls = fake_engine(lambda q: res)
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    assert len(calls) == 1 and calls[0].problem_id() == p.problem_id()
    check_against(r, p, res)
    assert "budget:statistical_and_binning" in r.flags
    assert "see:em_only (fake engine for tests)" in r.flags        # the engine's flags reach the result
    assert not any(f.startswith("see:mean_undefined:") for f in r.flags), r.flags
    assert not any(f.startswith("extrapolated:") for f in r.flags), r.flags
    # p_above at threshold 0 is 1 when underflow is 0 (R21); the spectrum integrates to the in-range fraction
    assert r.q("see.p_above").array()[0] == pytest.approx(1.0, rel=1e-12)
    e = ref_edges(p.histogram) * EV
    assert np.sum(r.q("see.edep_spectrum").array() * np.diff(e)) == pytest.approx(1.0, rel=1e-12)
    job = hub.store.job(r.id)
    assert job.key.adapter == ADAPTER and job.key.engine == "g4see"


def test_values_and_sigmas_log(hub, fake_engine):
    p = _slab(n_events=5000, bins=50).with_changes({"histogram.scale": "log", "histogram.low_MeV": 0.01})
    res = _fake_result(p, seed=2, centre=0.6, width=0.1)
    fake_engine(lambda q: res)
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    check_against(r, p, res)
    # geometric centres, in eV
    e = ref_edges(p.histogram) * EV
    assert np.allclose(r.q("see.edep_spectrum").axes[0].values, np.sqrt(e[1:] * e[:-1]), rtol=1e-12)
    assert np.allclose(r.q("see.p_above").axes[0].values, e[:-1], rtol=1e-12)


def test_mean_binning_term_uses_widest_filled_bin(hub, fake_engine):
    """Log bins have different widths: b comes from the widest bin that holds an event, not from the widest bin."""
    p = _slab(n_events=1000, bins=20).with_changes({"histogram.scale": "log", "histogram.low_MeV": 0.01})
    from simws.engines.base import SeeResult
    counts = np.zeros(20)
    counts[3], counts[5] = 600.0, 400.0                            # the widest bins (top) stay empty
    res = SeeResult(engine="g4see", problem_id=p.problem_id(), n_events=1000, edges_MeV=ref_edges(p.histogram).tolist(),
                    counts=counts.tolist(), underflow=0.0, overflow=0.0)
    fake_engine(lambda q: res)
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    check_against(r, p, res)
    e = ref_edges(p.histogram) * EV
    b = (e[6] - e[5]) / math.sqrt(12.0)
    assert float(np.asarray(r.q("see.edep_mean").uncertainty.components["binning"])) == pytest.approx(b, rel=1e-9)


def test_mean_undefined_with_overflow(hub, fake_engine):
    p = _slab(n_events=2000, bins=100)
    res = _fake_result(p, seed=3, over=7.0)
    fake_engine(lambda q: res)
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    check_against(r, p, res)                                       # no edep_mean, no edep_std
    names = [q.name for q in r.quantities]
    assert "see.edep_mean" not in names and "see.edep_std" not in names
    fl = [f for f in r.flags if f.startswith("see:mean_undefined:")]
    assert len(fl) == 1 and "7" in fl[0], r.flags
    assert r.q("see.fraction_overflow").value == pytest.approx(7.0 / 2000, rel=1e-12)
    assert r.q("see.p_above").array()[0] == pytest.approx(1.0, rel=1e-12)          # overflow counts as above


def test_mean_undefined_with_underflow(hub, fake_engine):
    p = _slab(n_events=3000, bins=100).with_changes({"histogram.low_MeV": 0.5})
    res = _fake_result(p, seed=4, under=123.0, over=4.0)
    fake_engine(lambda q: res)
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    check_against(r, p, res)
    fl = [f for f in r.flags if f.startswith("see:mean_undefined:")]
    # "says how many events fell outside the histogram": the total, or the two parts
    assert len(fl) == 1 and ("127" in fl[0] or ("123" in fl[0] and "4" in fl[0])), r.flags
    assert r.q("see.p_above").array()[0] == pytest.approx((3000 - 123) / 3000, rel=1e-12)
    assert r.q("see.fraction_underflow").value == pytest.approx(123 / 3000, rel=1e-12)


def test_counts_must_add_up(hub, fake_engine):
    p = _slab(n_events=10000, bins=100)
    fake_engine(lambda q: _fake_result(q, seed=5, extra=-1.0))      # one event missing
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "check_failed", (r.status, r.message)


def test_counts_tolerance_1e6(hub, fake_engine):
    """A difference of at most 1e-6 N passes (here 0.005 events with N = 10000)."""
    p = _slab(n_events=10000, bins=100, seed=11)
    fake_engine(lambda q: _fake_result(q, seed=6, extra=0.005))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", (r.status, r.message)
    p2 = _slab(n_events=10000, bins=100, seed=12)
    fake_engine(lambda q: _fake_result(q, seed=6, extra=0.02))
    r2 = hub.submit(ADAPTER, params=_params(p2))
    assert r2.status == "check_failed", (r2.status, r2.message)


def test_counts_too_many(hub, fake_engine):
    p = _slab(n_events=1000, bins=100)
    fake_engine(lambda q: _fake_result(q, seed=7, extra=3.0))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "check_failed", (r.status, r.message)


def test_paper_fidelity_needs_10000_events(hub, fake_engine):
    p = _slab(n_events=5000, bins=100)
    fake_engine(lambda q: _fake_result(q, seed=8))
    r = hub.submit(ADAPTER, params=_params(p), fidelity="paper")
    assert r.status == "check_failed", (r.status, r.message)
    rs = hub.submit(ADAPTER, params=_params(p))                    # the same run at screening fidelity is fine
    assert rs.status == "ok", rs.message


def test_paper_fidelity_with_enough_events(hub, fake_engine):
    """N = 10000 does not trip the paper-fidelity self-check; every quantity carries a sigma, so (inferred, A8 does
    not say it) the result is accepted."""
    p = _slab(n_events=10000, bins=100)
    fake_engine(lambda q: _fake_result(q, seed=9))
    r = hub.submit(ADAPTER, params=_params(p), fidelity="paper")
    assert r.status == "ok", (r.status, r.message)


def test_extrapolated_flags(hub, fake_engine):
    from simws import engines
    p = _slab("neutron", 14.0, hadronic="ftfp_bert_hp", n_events=2000, bins=100,
              beol=[("ox", "G4_SILICON_DIOXIDE", 1.0)])
    s = engines.capability("g4see").outside_tested_range(p)
    assert {x.split(":", 1)[0] for x in s} == {"particle", "energy_MeV", "hadronic", "materials"}, s
    fake_engine(lambda q: _fake_result(q, seed=10, flags=()))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    for x in s:
        assert "extrapolated:" + x in r.flags, (x, r.flags)
    assert not any(f.startswith("see:em_only") for f in r.flags), r.flags


def test_em_flag_from_engine_passes_through(hub, fake_engine):
    p = _slab(n_events=1000, bins=100, seed=21)
    fake_engine(lambda q: _fake_result(q, seed=11, flags=("see:em_only: test text",)))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    assert "see:em_only: test text" in r.flags


def test_unsupported_is_invalid_input(hub, fake_engine):
    calls = fake_engine(lambda q: pytest.fail("the engine must not run an unsupported problem"))
    p = _slab(n_events=1000).with_changes({"geometry.beol": [{"name": f"l{i}", "material": "G4_Si",
                                                             "thickness_um": 0.1} for i in range(21)]})
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "invalid_input", (r.status, r.message)
    assert "unsupported by g4see: " in r.message
    assert "max_beol_layers:" in r.message
    assert r.quantities == []
    assert calls == []


def test_engine_missing_is_engine_error(hub, monkeypatch, tmp_path):
    monkeypatch.setenv("SIMWS_G4SEE_PREFIX", str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()
    r = hub.submit(ADAPTER, params=_params(_slab(n_events=100)))
    assert r.status == "engine_error", (r.status, r.message)
    assert "not installed" in r.message.lower(), r.message


def test_trust_unverified(hub, fake_engine):
    """A8.6: a SEE result stays unverified."""
    from simws.hub.trust import assess_stored
    p = _slab(n_events=1000, bins=100)
    fake_engine(lambda q: _fake_result(q, seed=12))
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok"
    assert assess_stored(hub, r.id).level == "unverified"


# ------------------------------------------------------------------------------------------------ study (A7.3 rule)
def test_study_submits_to_see_adapter(hub, fake_engine):
    from simws.study import Study
    fake_engine(lambda q: _fake_result(q, seed=13))
    p = _slab(n_events=1000, bins=100)
    res = Study(hub, p).sweep({"source.energy_MeV": [5.0, 5.5]})
    assert [pt["changes"] for pt in res.points] == [{"source.energy_MeV": 5.0}, {"source.energy_MeV": 5.5}]
    for pt in res.points:
        assert pt["engine"] == "g4see" and pt["status"] == "ok", pt
        job = hub.store.job(pt["result_id"])
        assert job.key.adapter == ADAPTER
        assert pt["trust"] == "unverified"


def test_study_refuses_device_engine_on_see_problem(hub):
    from simws.study import Study
    res = Study(hub, _slab(n_events=1000), engines=["devsim"]).sweep({})
    (pt,) = res.points
    assert pt["status"] == "refused" and pt["result_id"] is None
    assert len(pt["reasons"]) == 1 and pt["reasons"][0].startswith("problem kind:")
    assert hub.graph.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


# ------------------------------------------------------------------------------------------------ real engine
def test_real_engine_matches_formulas(hub, tmp_path):
    _need_engine()
    p = _slab(n_events=1000)
    res = _g4().run(p, tmp_path / "direct")
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", r.message
    check_against(r, p, res)
    assert any(f.startswith("see:em_only") for f in r.flags), r.flags
    assert "budget:statistical_and_binning" in r.flags


# ------------------------------------------------------------------------------------------------ reference (A8.7)
# NIST Standard Reference Database 124, ASTAR and PSTAR, silicon, default energy grid (retrieved 17 Nov 2026 by the
# lead; https://physics.nist.gov/PhysRefData/Star/Text/). Columns: kinetic energy (MeV), total stopping power
# (electronic + nuclear, MeV cm2/g). Only the rows that bracket the energies of R22 to R25 are copied (alpha 1 to
# 10 MeV, proton 5 to 20 MeV); the interpolation uses only the two bracketing rows, so the subset gives the same
# numbers as the full table.
ASTAR_SI = [
    (1.000E+00, 1.296E+03), (1.250E+00, 1.219E+03), (1.500E+00, 1.148E+03), (1.750E+00, 1.082E+03),
    (2.000E+00, 1.024E+03), (2.250E+00, 9.715E+02), (2.500E+00, 9.248E+02), (2.750E+00, 8.822E+02),
    (3.000E+00, 8.432E+02), (3.500E+00, 7.738E+02), (4.000E+00, 7.137E+02), (4.500E+00, 6.618E+02),
    (5.000E+00, 6.179E+02), (5.500E+00, 5.802E+02), (6.000E+00, 5.474E+02), (6.500E+00, 5.186E+02),
    (7.000E+00, 4.931E+02), (7.500E+00, 4.703E+02), (8.000E+00, 4.498E+02), (8.500E+00, 4.312E+02),
    (9.000E+00, 4.143E+02), (9.500E+00, 3.989E+02), (1.000E+01, 3.847E+02),
]
PSTAR_SI = [
    (5.000E+00, 5.828E+01), (5.500E+00, 5.431E+01), (6.000E+00, 5.091E+01), (6.500E+00, 4.795E+01),
    (7.000E+00, 4.536E+01), (7.500E+00, 4.306E+01), (8.000E+00, 4.101E+01), (8.500E+00, 3.916E+01),
    (9.000E+00, 3.750E+01), (9.500E+00, 3.598E+01), (1.000E+01, 3.459E+01), (1.250E+01, 2.913E+01),
    (1.500E+01, 2.529E+01), (1.750E+01, 2.243E+01), (2.000E+01, 2.020E+01),
]
RHO_SI = 2.33                                    # g/cm3 (A8.7)
CONTRACT_REF = {"R22": 1.6137, "R23": 0.8327, "R24": 0.8558, "R25": 1.6137}


def _stopping(table):
    lE = [math.log(e) for e, _ in table]
    lS = [math.log(s) for _, s in table]

    def S(E):
        x = math.log(E)
        assert lE[0] <= x <= lE[-1], f"energy {E} MeV outside the copied table rows"
        i = min(max(bisect.bisect_right(lE, x) - 1, 0), len(lE) - 2)
        t = (x - lE[i]) / (lE[i + 1] - lE[i])
        return math.exp(lS[i] + t * (lS[i + 1] - lS[i]))
    return S


def energy_after(E0, path_um, table, h=0.01):
    """RK4 integration of dE/dx = -S(E) rho along path_um of silicon, step h um."""
    S = _stopping(table)

    def f(E):
        return -S(E) * RHO_SI * 1e-4             # MeV per um

    n = int(round(path_um / h))
    E = E0
    for _ in range(n):
        k1 = f(E)
        k2 = f(E + 0.5 * h * k1)
        k3 = f(E + 0.5 * h * k2)
        k4 = f(E + h * k3)
        E += h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return E


def reference_values():
    a10 = energy_after(5.0, 10.0, ASTAR_SI)
    a5 = energy_after(5.0, 5.0, ASTAR_SI)
    return {"R22": 5.0 - a10,
            "R23": 10.0 - energy_after(10.0, 100.0, PSTAR_SI),
            "R24": a5 - energy_after(a5, 5.0, ASTAR_SI),
            "R25": 5.0 - a10}


def test_reference_values_recomputed():
    """The checker's own computation by the method of A8.7 agrees with the contract's numbers within 0.1 %."""
    ref = reference_values()
    for k, v in ref.items():
        rel = v / CONTRACT_REF[k] - 1
        print(f"\n[reference {k}] recomputed {v:.5f} MeV, contract {CONTRACT_REF[k]} MeV, relative {rel:+.5%}")
        assert abs(rel) <= 1e-3, (k, v, CONTRACT_REF[k])


# ------------------------------------------------------------------------------------------------ acceptance R21-R28
@pytest.fixture(scope="module")
def rhub(tmp_path_factory):
    """One hub for the acceptance runs, so a problem used by several checks runs once (cache)."""
    from simws.hub.jobs import Hub
    return Hub(tmp_path_factory.mktemp("see_acceptance") / "store", mode="inprocess")


def _ok(hub, p):
    r = hub.submit(ADAPTER, params=_params(p))
    assert r.status == "ok", (r.status, r.message)
    # R21: underflow + overflow + sum of counts = N, and p_above at threshold 0 is 1 when underflow is 0
    N = p.statistics.n_events
    pa = r.q("see.p_above").array()
    fu = r.q("see.fraction_underflow").value
    assert (pa[0] + fu) * N == pytest.approx(N, rel=0, abs=1e-6 * N)
    if fu == 0:
        assert pa[0] == pytest.approx(1.0, rel=0, abs=1e-12)
    return r


def _vs(r, name):
    q = r.q(name)
    return float(q.value), float(q.uncertainty.sigma)


def _gate(r, ref_MeV, tol, label):
    v, s = _vs(r, "see.edep_mean")
    v_MeV, s_MeV = v / EV, s / EV
    rel = v_MeV / ref_MeV - 1
    ok = abs(rel) <= tol
    comps = {k: float(np.asarray(x)) / EV for k, x in r.q("see.edep_mean").uncertainty.components.items()}
    print(f"\n[{label}] see.edep_mean = {v_MeV:.5f} +- {s_MeV:.5f} MeV (components {comps}), reference "
          f"{ref_MeV} MeV, relative difference {rel:+.4%} ({'within' if ok else 'OUTSIDE'} {tol:.0%})")
    return ok


def test_R21_R20_hub(rhub):
    _need_engine()
    _ok(rhub, _slab(n_events=1000))
    _ok(rhub, _slab(n_events=1000, seed=2))


def test_R22_alpha_partial(rhub):
    _need_engine()
    r = _ok(rhub, _slab("alpha", 5.0, sv_thickness_um=10.0))
    assert _gate(r, CONTRACT_REF["R22"], 0.03, "R22 alpha 5 MeV, SV 0-10 um")


def test_R23_proton_partial(rhub):
    _need_engine()
    r = _ok(rhub, _slab("proton", 10.0, sv_thickness_um=100.0, bulk_thickness_um=300.0))
    assert _gate(r, CONTRACT_REF["R23"], 0.03, "R23 proton 10 MeV, SV 0-100 um, bulk 300 um")


def test_R24_alpha_through_layers(rhub):
    _need_engine()
    p = _slab("alpha", 5.0, beol=[("l1", "G4_Si", 1.0), ("l2", "G4_Si", 2.0)], sv_depth_um=2.0, sv_thickness_um=5.0)
    r = _ok(rhub, p)
    assert _gate(r, CONTRACT_REF["R24"], 0.03, "R24 alpha 5 MeV, BEOL 1+2 um Si, SV 2-7 um")


def test_R25_tilt(rhub):
    _need_engine()
    r = _ok(rhub, _slab("alpha", 5.0, tilt_deg=60.0, sv_thickness_um=5.0))
    assert _gate(r, CONTRACT_REF["R25"], 0.03, "R25 alpha 5 MeV at 60 deg, SV 0-5 um")


def test_R26_full_absorption(rhub):
    _need_engine()
    p = _slab("alpha", 5.0, sv_thickness_um=50.0)
    r = _ok(rhub, p)
    ok_mean = _gate(r, 5.0, 0.01, "R26 alpha 5 MeV, SV 0-50 um")
    q = r.q("see.p_above")
    thr = np.asarray(q.axes[0].values, float)
    i = int(np.argmin(np.abs(thr - 4.75 * EV)))
    pa, ps = float(q.array()[i]), float(np.asarray(q.uncertainty.sigma)[i])
    print(f"[R26] see.p_above at threshold {thr[i] / EV:.6f} MeV = {pa:.5f} +- {ps:.5f} (gate >= 0.99)")
    assert ok_mean and pa >= 0.99, (ok_mean, pa)


def test_R27_honest_error_bars(rhub):
    _need_engine()
    base = dict(sv_thickness_um=100.0, bulk_thickness_um=300.0)
    r1 = _ok(rhub, _slab("proton", 10.0, **base))
    r2 = _ok(rhub, _slab("proton", 10.0, seed=2, **base))
    (a, sa), (b, sb) = _vs(r1, "see.edep_mean"), _vs(r2, "see.edep_mean")
    comb = math.sqrt(sa ** 2 + sb ** 2)
    print(f"\n[R27] see.edep_mean seed 1 {a / EV:.6f} +- {sa / EV:.6f} MeV, seed 2 {b / EV:.6f} +- {sb / EV:.6f} MeV; "
          f"|diff| = {abs(a - b) / EV:.6f} MeV = {abs(a - b) / comb:.2f} combined sigma (gate 4); relative "
          f"{b / a - 1:+.4%}")
    assert abs(a - b) <= 4.0 * comb


def test_R28_hadronic_smoke(rhub):
    _need_engine()
    p = _slab("neutron", 14.0, sv_thickness_um=100.0, bulk_thickness_um=300.0, hadronic="ftfp_bert_hp",
              n_events=2000, high_MeV=20.0)
    assert p.histogram.low_MeV == 0.0 and p.histogram.high_MeV == 20.0
    r = _ok(rhub, p)
    q = r.q("see.p_above")
    thr = np.asarray(q.axes[0].values, float)
    i = int(np.argmin(np.abs(thr - 0.1 * EV)))
    k = float(q.array()[i]) * 2000
    nz = int(np.count_nonzero(r.q("see.edep_spectrum").array()[1:]))
    mean = r.q("see.edep_mean").value / EV if "see.edep_mean" in [x.name for x in r.quantities] else None
    print(f"\n[R28] neutron 14 MeV, FTFP_BERT_HP, 2000 events: status {r.status}; events above "
          f"{thr[i] / EV:.4f} MeV = {k:.0f}; non-empty bins above the first = {nz}; mean = {mean} MeV; "
          f"flags = {r.flags}")
    assert not any(f.startswith("see:em_only") for f in r.flags), r.flags
    assert k >= 1 - 1e-9
