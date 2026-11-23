"""Contract amendment A11 (docs/ARCHITECTURE_V1.md): the electron energy-balance device engine "devsim_eb".

Written from the contract text alone (A11 and the sections it builds on: 4 to 6, A1 to A4), without reading the
implementation (simws/engines/devsim_eb_engine.py, simws/tcad/energy_balance.py, the A11 parts of the registry,
compare, the device adapter and the library). Tolerances are the contract's (A11.5), fixed before any run; every
acceptance measurement is printed with its reference and gate (run pytest with -s).

Readings of the contract that the tests rely on (where the text is not explicit):
- "the A3 floor" (R50) is the A3.2 magnitude floor, 1e-2 A/m**2 = 1e-6 A/cm**2 (the 0.9a cross-check floor). The
  stricter adapter noise floor of section 5 / A4.4 (1e-4 A/m**2 = 1e-8 A/cm**2) is printed, not gated.
- R51 "the current": every contact current of the two-terminal bar (left and right) is gated.
- R52 field: |E| at the node nearest the middle, central difference of potential_V over x_um (um -> cm);
  n = n_cm3 at that node; current density = |dc.I.right| from device.devsim_eb (A/m**2 -> A/cm**2, x 1e-4).
- R54: compare_results(a = devsim result, b = devsim_eb result as reference); the contract does not fix the order.
"""
import inspect
import math

import numpy as np
import pytest

from simws.spec.library import npn_hbt_2d, pn_diode_1d

Q_E = 1.602176634e-19          # C (exact, SI 2019)
T0 = 300.0
MU0 = 1000.0                   # cm^2/Vs: n_bar_1d's default mu_n (A11.5 R52)
VSAT_CANALI = 1.07e7           # cm/s, A11.0 (simws.tcad.physics.saturation_velocity, electrons, 300 K)
A3_FLOOR_A_CM2 = 1e-6          # A3.2: 1e-2 A/m**2 = 1e-6 A/cm**2
ADAPTER_FLOOR_A_CM2 = 1e-8     # section 5 / A4.4: 1e-4 A/m**2 (printed only)
FLAG = "physics:energy_balance_electrons"
EB_MODEL = "baccarani_wordeman_1985_energy_balance"
PROFILE_KEYS = {"x_um", "Tn_K", "n_cm3", "potential_V"}


def _n_bar(**kw):
    from simws.spec.library import n_bar_1d
    return n_bar_1d(**kw)


def _eb():
    from simws import engines
    return engines.module("devsim_eb")


def _params(p, **extra):
    return {"problem": p.model_dump(mode="json"), **extra}


def _need_devsim():
    pytest.importorskip("devsim")


@pytest.fixture(scope="module")
def mhub(tmp_path_factory):
    from simws.hub.jobs import Hub
    return Hub(tmp_path_factory.mktemp("a11") / "store", mode="inprocess")


# ================================================================================================ A11.3 registry
def test_registry_entry_family_adapter_code():
    from simws import engines
    assert "devsim_eb" in engines.ENGINES
    assert engines.ENGINES["devsim_eb"] == "simws.engines.devsim_eb_engine"
    assert "devsim_eb" in engines.names() and engines.names() == sorted(engines.names())
    assert engines.family("devsim_eb") == "device"
    assert engines.adapter_name("devsim_eb") == "device.devsim_eb"
    assert engines.code("devsim_eb") == "devsim"
    mod = engines.module("devsim_eb")
    assert mod.__name__ == "simws.engines.devsim_eb_engine"
    assert getattr(mod, "CODE") == "devsim"
    assert getattr(mod, "FAMILY", "device") == "device"


def test_code_of_every_other_engine_is_its_name():
    """A11.4: CODE is the engine's name when absent; only devsim_eb sets another code (A11.3)."""
    from simws import engines
    others = [n for n in engines.names() if n not in ("devsim_eb", "g4implant", "channeling", "devsim_anneal")]   # amended by the lead (A13.3, A15.5, A16.3)
    assert engines.code("g4implant") == "geant4"
    assert engines.code("channeling") == "published_fits"   # amended by the lead (A15.5)
    assert engines.code("devsim_anneal") == "devsim"   # amended by the lead (A16.3)
    assert {"devsim", "charon"} <= set(others)
    for n in others:
        assert engines.code(n) == n, (n, engines.code(n))


def test_module_surface():
    mod = _eb()
    for attr in ("CAPABILITY", "unsupported", "available", "run", "profiles"):
        assert hasattr(mod, attr), attr
    assert isinstance(mod.available(), bool)
    try:
        import devsim  # noqa: F401
    except Exception:  # noqa: BLE001
        return
    assert mod.available() is True


def test_capability_exact():
    from simws import engines
    c = engines.capability("devsim_eb")
    assert c is _eb().CAPABILITY or c == _eb().CAPABILITY
    assert c.engine == "devsim_eb"
    assert c.env == "tcad" and c.in_process is True
    assert c.dimensions == [1]
    assert c.materials == ["silicon"]
    assert c.grading_elements == []                                  # "no grading"
    assert c.contact_kinds == ["ohmic"]
    assert c.statistics == ["boltzmann"]
    assert c.mobility == ["constant", "masetti"]
    assert c.velocity_saturation == [False]
    assert c.srh == ["off", "constant", "scharfetter"]
    assert c.auger == [False, True]
    assert c.band_gap_narrowing == ["off", "slotboom"]
    assert c.incomplete_ionization == ["off"]
    assert c.impact_ionization == ["off"]
    assert c.analyses == ["equilibrium", "dc_sweep"]
    assert tuple(c.tested_temperature_K) == (300.0, 300.0)
    assert c.mesh_study is True


def _small_signal(p):
    from simws.spec.schema import Problem
    d = p.model_dump(mode="json")
    d["analysis"] = {"kind": "small_signal", "bias": [{"contact": "anode", "value_V": 0.5}],
                     "ports": ["anode", "cathode"], "frequencies_Hz": [1e6]}
    return Problem.model_validate(d)


REFUSALS = [
    ("2d", lambda: npn_hbt_2d(VBE=[0.6]), ["dimensions: 2 not in [1]", "grading_elements: 'Ge' not in []"]),
    ("velocity_saturation", lambda: pn_diode_1d(V=[0.3]).with_changes({"models.velocity_saturation": True}),
     ["velocity_saturation: True not in [False]"]),
    ("fermi_dirac", lambda: pn_diode_1d(V=[0.3]).with_changes({"models.statistics": "fermi_dirac"}),
     ["statistics: 'fermi_dirac' not in ['boltzmann']"]),
    ("klaassen", lambda: pn_diode_1d(V=[0.3]).with_changes({"models.mobility": "klaassen",
                                                             "models.mu_n_cm2_Vs": None,
                                                             "models.mu_p_cm2_Vs": None}),
     ["mobility: 'klaassen' not in ['constant', 'masetti']"]),
    ("small_signal", lambda: _small_signal(pn_diode_1d()),
     ["analyses: 'small_signal' not in ['equilibrium', 'dc_sweep']"]),
    ("altermatt", lambda: pn_diode_1d().with_changes({"models.incomplete_ionization": "altermatt"}),
     ["incomplete_ionization: 'altermatt' not in ['off']"]),
    ("constant_energy", lambda: pn_diode_1d().with_changes({"models.incomplete_ionization": "constant_energy",
                                                             "models.ii_energy_eV": 0.045,
                                                             "models.ii_critical_doping_cm3": 1e18}),
     ["incomplete_ionization: 'constant_energy' not in ['off']"]),
    ("impact", lambda: pn_diode_1d().with_changes({"models.impact_ionization": "van_overstraeten"}),
     ["impact_ionization: 'van_overstraeten' not in ['off']"]),
    ("schottky", lambda: pn_diode_1d().with_changes({"device.contacts.0.kind": "schottky",
                                                      "device.contacts.0.barrier_eV": 0.7}),
     ["contact_kinds: 'schottky' not in ['ohmic']"]),
    ("sic", lambda: pn_diode_1d().with_changes({"device.regions.0.material": "sic_4h"}),
     ["materials: 'sic_4h' not in ['silicon']"]),
]


@pytest.mark.parametrize("label,make,reasons", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_refusal_reasons(label, make, reasons):
    from simws import engines
    p = make()
    got = engines.unsupported("devsim_eb", p)
    for r in reasons:
        assert r in got, (label, got)
    cap = engines.capability("devsim_eb").unsupported(p)
    assert all(s in got for s in cap), (cap, got)


@pytest.mark.parametrize("change", [
    {},
    {"models.mobility": "masetti", "models.mu_n_cm2_Vs": None, "models.mu_p_cm2_Vs": None},
    {"models.srh": "scharfetter", "models.tau_n_s": None, "models.tau_p_s": None},
    {"models.srh": "off", "models.tau_n_s": None, "models.tau_p_s": None},
    {"models.auger": True},
    {"models.band_gap_narrowing": "slotboom"},
])
def test_supported_variants(change):
    from simws import engines
    assert engines.unsupported("devsim_eb", pn_diode_1d(V=[0.3]).with_changes(change)) == []
    assert engines.unsupported("devsim_eb", pn_diode_1d().with_changes(change)) == []


def test_temperature_outside_tested_range_is_not_a_refusal():
    from simws import engines
    c = engines.capability("devsim_eb")
    p = _n_bar(T=310.0)
    assert engines.unsupported("devsim_eb", p) == []
    assert c.outside_tested_range(_n_bar()) == []
    assert c.outside_tested_range(p) == ["temperature_K: 310 outside tested [300, 300]"]


# ------------------------------------------------------------------------------------------------ planner (4.3)
def test_planner_lists_devsim_eb_as_checker_of_1d_silicon():
    """Section 4.3 (not amended by A11): checkers are the other supporting engines, sorted."""
    from simws.engines.planner import plan
    for p in (_n_bar(V=[0.5]), _n_bar(), pn_diode_1d(V=[0.3])):
        pl = plan(p)
        assert pl.primary == "devsim", pl.summary()
        assert pl.checkers == ["devsim_eb"], pl.checkers               # charon refuses these (Boltzmann, ...)
        assert "devsim_eb" not in pl.refused
        assert pl.warnings == {}                                         # 300 K: inside [300, 300], bounds included
    pe = plan(_n_bar(V=[0.5]), prefer=("devsim_eb",))
    assert pe.primary == "devsim_eb" and pe.checkers == ["devsim"]


def test_planner_refuses_and_warns():
    from simws.engines.planner import plan
    ph = plan(npn_hbt_2d(VBE=[0.6]))
    assert "dimensions: 2 not in [1]" in ph.refused["devsim_eb"]
    assert ph.primary == "devsim" and "devsim_eb" not in ph.checkers
    pc = plan(pn_diode_1d(V=[0.3], T=250.0))
    assert "devsim_eb" in pc.checkers
    assert len(pc.warnings.get("devsim_eb", [])) == 1
    assert pc.warnings["devsim_eb"][0].startswith("temperature_K:")
    assert "devsim" not in pc.warnings                                  # 250 K is inside DEVSIM's [250, 400]


# ------------------------------------------------------------------------------------------------ n_bar_1d
def test_n_bar_1d_signature_defaults():
    from simws.spec.library import n_bar_1d
    sig = inspect.signature(n_bar_1d)
    expected = {"N_D_cm3": 1e16, "length_um": 2.0, "mu_n": 1000.0, "mu_p": 400.0, "nodes": 2001, "T": 300.0,
                "V": None}
    assert list(sig.parameters) == list(expected), list(sig.parameters)
    for k, v in expected.items():
        assert sig.parameters[k].default == v, (k, sig.parameters[k].default)


def _check_bar(p, N_D, L, mu_n, mu_p, nodes, T, V):
    from simws import engines
    from simws.spec.schema import DCSweep, Equilibrium
    d = p.device
    assert d.dimension == 1
    assert len(d.regions) == 1 and d.regions[0].material == "silicon"
    lo, hi = d.regions[0].box.x
    assert hi - lo == pytest.approx(L, rel=1e-12)
    # contacts "left" (x_min) and "right" (x_max), ohmic
    assert sorted(c.name for c in d.contacts) == ["left", "right"]
    assert d.contact("left").side == "x_min" and d.contact("right").side == "x_max"
    assert all(c.kind == "ohmic" for c in d.contacts)
    # uniform n-type: donors N_D everywhere, no acceptors, no grading
    xs = np.linspace(lo, hi, 4001)
    nd, na = d.doping_at(xs)
    assert np.all(nd == N_D) and np.all(na == 0.0)
    assert d.grading == []
    # uniform mesh with `nodes` nodes
    h = L / (nodes - 1)
    lines = d.mesh.x_lines
    assert lines[0][0] == pytest.approx(lo, abs=1e-15) and lines[-1][0] == pytest.approx(hi, rel=1e-12)
    for _, s in lines:
        assert s * d.mesh.scale == pytest.approx(h, rel=1e-9), (lines, d.mesh.scale, h)
    # models
    m = p.models
    assert m.temperature_K == T
    assert m.mobility == "constant" and m.mu_n_cm2_Vs == mu_n and m.mu_p_cm2_Vs == mu_p
    assert m.srh == "off" and m.auger is False and m.band_gap_narrowing == "off"
    assert m.statistics == "boltzmann"
    # analysis
    if V is None:
        assert isinstance(p.analysis, Equilibrium)
    else:
        a = p.analysis
        assert isinstance(a, DCSweep)
        assert a.contact == "right" and a.values_V == [float(v) for v in V] and a.step_V == 0.05
        assert a.tracking == [] and a.fixed_V == {}
    # numerics as pn_diode_1d
    assert p.numerics == pn_diode_1d().numerics
    # both DEVSIM engines take it (R51 runs it on both)
    assert engines.unsupported("devsim_eb", p) == []
    assert engines.unsupported("devsim", p) == []


def test_n_bar_1d_defaults():
    _check_bar(_n_bar(), 1e16, 2.0, 1000.0, 400.0, 2001, 300.0, None)
    _check_bar(_n_bar(V=[0.04]), 1e16, 2.0, 1000.0, 400.0, 2001, 300.0, [0.04])


def test_n_bar_1d_parameters():
    _check_bar(_n_bar(N_D_cm3=3e15, length_um=1.5, mu_n=700.0, mu_p=250.0, nodes=301, T=310.0, V=[0.1, 0.2]),
               3e15, 1.5, 700.0, 250.0, 301, 310.0, [0.1, 0.2])


# ================================================================================================ A11.3 adapter
def test_adapter_manifest():
    from simws.hub import adapter as A
    assert "device.devsim_eb" in A.names()
    m = A.manifest("device.devsim_eb")
    md = A.manifest("device.devsim")
    assert m.name == "device.devsim_eb" and m.version == "1.0"
    assert m.engine == "devsim_eb" and m.env == "tcad"
    assert m.engine_version == md.engine_version
    assert m.inputs == md.inputs == ["problem"]
    assert [o.name for o in m.outputs] == [o.name for o in md.outputs] == ["dc.kirchhoff_residual"]
    assert m.native_conventions == md.native_conventions == {"terminal_current": "into_device+"}


def test_adapter_unsupported_is_invalid_input(hub):
    p = pn_diode_1d(V=[0.3]).with_changes({"models.velocity_saturation": True})
    r = hub.submit("device.devsim_eb", params=_params(p))
    assert r.status == "invalid_input"
    assert "unsupported by devsim_eb" in r.message, r.message
    assert "velocity_saturation: True not in [False]" in r.message, r.message


# ================================================================================================ A11.3 engine behaviour
@pytest.fixture(scope="module")
def small_sweep():
    """n_bar_1d on 201 nodes swept to 1 V (5 kV/cm: mu_0 E / v_s ~ 0.47, so Tn ~ 1.1 T in the bar)."""
    _need_devsim()
    p = _n_bar(nodes=201, V=[0.5, 1.0])
    return p, _eb().run(p), _eb().profiles(p)


def test_engine_result_flag_and_diagnostics(small_sweep):
    from simws.tcad.physics import saturation_velocity
    from simws.materials import load
    p, r, prof = small_sweep
    assert r.engine == "devsim_eb"
    assert r.problem_id == p.problem_id()
    assert r.analysis == "dc_sweep" and r.sweep_contact == "right" and r.sweep_V == [0.5, 1.0]
    assert r.current_unit == "A/cm**2" and set(r.currents) == {"left", "right"}
    assert FLAG in r.flags, r.flags
    d = r.diagnostics
    for k in ("eb_vsat_n_cm_s", "eb_Tn_max_K", "eb_model"):
        assert k in d, (k, sorted(d))
    assert d["eb_model"] == EB_MODEL
    vs = saturation_velocity(load("silicon"), T0)[0]
    assert vs == pytest.approx(VSAT_CANALI, rel=1e-12)                # A11.0
    assert d["eb_vsat_n_cm_s"] == pytest.approx(vs, rel=1e-12)
    Tn = np.asarray(prof["Tn_K"], float)
    assert d["eb_Tn_max_K"] == pytest.approx(Tn.max(), rel=1e-6)       # largest Tn at the last point
    assert d["eb_Tn_max_K"] > T0 * 1.01
    print(f"\n[A11.3] eb_vsat_n_cm_s={d['eb_vsat_n_cm_s']:.6g} eb_Tn_max_K={d['eb_Tn_max_K']:.6g} "
          f"(profiles max {Tn.max():.6g}) at 1 V on 201 nodes")


def test_profiles_keys_shapes(small_sweep):
    p, r, prof = small_sweep
    assert isinstance(prof, dict) and set(prof) == PROFILE_KEYS, sorted(prof)
    arrs = {k: np.asarray(v, float) for k, v in prof.items()}
    for k, a in arrs.items():
        assert a.ndim == 1 and a.shape == (201,), (k, a.shape)
        assert np.all(np.isfinite(a)), k
    x = arrs["x_um"]
    assert np.all(np.diff(x) > 0)                                       # increasing x
    assert x[0] == pytest.approx(0.0, abs=1e-9) and x[-1] == pytest.approx(2.0, rel=1e-9)   # micrometres
    assert np.allclose(np.diff(x), 0.01, rtol=1e-6)                     # uniform 10 nm mesh
    n = arrs["n_cm3"]
    assert n[100] == pytest.approx(1e16, rel=0.05)                      # cm^-3, uniform n-type bar
    psi = arrs["potential_V"]
    assert psi[-1] - psi[0] == pytest.approx(1.0, rel=0.05)             # volts: the last point is V = 1.0
    Tn = arrs["Tn_K"]
    assert Tn[0] == pytest.approx(T0, rel=1e-6) and Tn[-1] == pytest.approx(T0, rel=1e-6)   # A11.2 ohmic: Tn = T
    assert Tn[100] > T0 * 1.05


def test_equilibrium_result_carries_flag():
    _need_devsim()
    r = _eb().run(_n_bar(nodes=201))
    assert r.analysis == "equilibrium" and r.sweep_V == [0.0]
    assert FLAG in r.flags
    assert r.diagnostics["eb_model"] == EB_MODEL
    assert r.diagnostics["eb_Tn_max_K"] == pytest.approx(T0, rel=1e-6)


def test_hub_result_same_quantities_as_devsim(hub):
    _need_devsim()
    p = _n_bar(nodes=201, V=[0.2, 0.4])
    re_ = hub.submit("device.devsim_eb", params=_params(p))
    rd = hub.submit("device.devsim", params=_params(p))
    assert re_.status == "ok", re_.message
    assert rd.status == "ok", rd.message
    assert sorted(q.name for q in re_.quantities) == sorted(q.name for q in rd.quantities) == \
        ["dc.I.left", "dc.I.right", "dc.kirchhoff_residual"]
    for name in ("dc.I.left", "dc.I.right"):
        qe, qd = re_.q(name), rd.q(name)
        assert qe.unit == qd.unit == "A/m**2"
        assert qe.conventions == {"terminal_current": "into_device+"}
        assert [(a.name, a.unit, list(a.values)) for a in qe.axes] == [("V_right", "V", [0.2, 0.4])]
    assert FLAG in re_.flags and FLAG not in rd.flags
    d = re_.diagnostics
    assert d["engine"] == "devsim_eb" and d["problem_id"] == p.problem_id() and d["temperature_K"] == 300.0
    for k in ("eb_vsat_n_cm_s", "eb_Tn_max_K", "eb_model"):
        assert k in d
    assert hub.store.job(re_.id).key.engine == "devsim_eb"
    r2 = hub.submit("device.devsim_eb", params=_params(p))
    assert r2.id == re_.id and r2.diagnostics.get("cache") == "hit"


def test_hub_outside_tested_temperature_flagged(hub):
    _need_devsim()
    r = hub.submit("device.devsim_eb", params=_params(_n_bar(nodes=201, T=310.0)))
    assert r.status == "ok", r.message
    assert "extrapolated:temperature_K: 310 outside tested [300, 300]" in r.flags, r.flags


# ================================================================================================ A11.4 independence
def _fake_run(engine, flags=()):
    """An engine run() returning a fixed EngineResult (test values only), so no engine runs (pattern of the A2.4
    test in test_device_adapter.py)."""
    def run(problem, workdir=None):
        from simws.engines.base import EngineResult
        from simws.spec.schema import Problem
        p = problem if isinstance(problem, Problem) else Problem.model_validate(problem)
        V = list(p.analysis.values_V)
        ia = [1.0 + k for k in range(len(V))]
        return EngineResult(engine=engine, problem_id=p.problem_id(), analysis="dc_sweep", sweep_contact="anode",
                            sweep_V=V, currents={"anode": ia, "cathode": [-x for x in ia]},
                            current_unit="A/cm**2", flags=list(flags))
    return run


@pytest.fixture(scope="module")
def eb_pair(mhub):
    """The same small problem through device.devsim and device.devsim_eb (real runs)."""
    _need_devsim()
    p = _n_bar(nodes=201, V=[0.05, 0.1])
    rd = mhub.submit("device.devsim", params=_params(p))
    re_ = mhub.submit("device.devsim_eb", params=_params(p))
    assert rd.status == "ok", rd.message
    assert re_.status == "ok", re_.message
    return rd, re_


def test_a11_4_devsim_vs_devsim_eb_not_independent(mhub, eb_pair):
    from simws.compare import compare_results
    rd, re_ = eb_pair
    for a, b in ((rd, re_), (re_, rd)):
        c = compare_results(mhub, a.id, "dc.I.right", b.id, tolerance=1.0)
        assert c.kind == "engine_pair"
        assert c.passed, c
        assert c.independent is False, c
        # A2.3 (unchanged): the engines are filled from the stored job keys
        assert c.engine_a == mhub.store.job(a.id).key.engine
        assert c.engine_b == mhub.store.job(b.id).key.engine
        assert {c.engine_a, c.engine_b} == {"devsim", "devsim_eb"}


def test_a11_4_passed_same_code_comparison_does_not_verify(tmp_path):
    """A passed devsim / devsim_eb comparison, recorded, leaves both results unverified with the A2.3 reason."""
    _need_devsim()
    from simws.compare import compare_results, record
    from simws.hub.jobs import Hub
    from simws.hub.trust import assess_stored
    h = Hub(tmp_path / "store", mode="inprocess")
    p = _n_bar(nodes=201, V=[0.05])
    rd = h.submit("device.devsim", params=_params(p))
    re_ = h.submit("device.devsim_eb", params=_params(p))
    assert rd.status == "ok" and re_.status == "ok"
    before = (assess_stored(h, rd.id).level, assess_stored(h, re_.id).level)
    c = compare_results(h, rd.id, "dc.I.right", re_.id, tolerance=1.0)
    assert c.passed and c.independent is False
    record(h, c)
    td, te = assess_stored(h, rd.id), assess_stored(h, re_.id)
    assert (td.level, te.level) == before == ("unverified", "unverified")
    assert f"same-engine or sketch comparison with {re_.id} (consistency only: not verification)" in td.reasons
    assert f"same-engine or sketch comparison with {rd.id} (consistency only: not verification)" in te.reasons


def test_a11_4_devsim_vs_charon_unchanged(hub, monkeypatch, charon_problem):
    """devsim and charon have different codes: still independent (stand-in engine runs, Charon is not installed)."""
    from simws import engines
    from simws.compare import compare_results
    assert engines.code("devsim") != engines.code("charon")
    for e in ("devsim", "charon"):
        monkeypatch.setattr(engines.module(e), "run", _fake_run(e))
    p = charon_problem(V=(0.1, 0.2))
    rd = hub.submit("device.devsim", params=_params(p))
    rc = hub.submit("device.charon", params=_params(p))
    assert rd.status == "ok", rd.message
    assert rc.status == "ok", rc.message
    for a, b in ((rd, rc), (rc, rd)):
        c = compare_results(hub, a.id, "dc.I.anode", b.id, tolerance=1e-9)
        assert c.kind == "engine_pair" and c.passed
        assert {c.engine_a, c.engine_b} == {"devsim", "charon"}
        assert c.independent is True


def test_a11_4_unregistered_engine_names_unchanged(hub, fake_adapters):
    """Engines outside the registry (fake adapters, sketches) keep the A2.3 behaviour."""
    from simws.compare import compare_results
    ra, rb = fake_adapters(hub, "fake.alpha", 1.0), fake_adapters(hub, "fake.beta", 1.0)
    rt = fake_adapters(hub, "fake.alpha_twin", 1.0)
    assert compare_results(hub, ra.id, "y", rb.id, tolerance=1e-9).independent is True
    assert compare_results(hub, ra.id, "y", rt.id, tolerance=1e-9).independent is False


# ================================================================================================ A11.5 acceptance
def _max_abs_current(r):
    return max(abs(v) for vals in r.currents.values() for v in vals)


@pytest.mark.parametrize("name,make", [("n_bar_1d", lambda: _n_bar()), ("pn_diode_1d", lambda: pn_diode_1d())])
def test_R50_equilibrium(name, make):
    _need_devsim()
    p = make()
    E = _eb()
    prof = E.profiles(p)
    Tn = np.asarray(prof["Tn_K"], float)
    dev = float(np.max(np.abs(Tn - T0)) / T0)
    r = E.run(p)
    imax = _max_abs_current(r)
    print(f"\n[R50 {name}] max|Tn-T|/T = {dev:.3e} (gate <= 1e-6); max|I| = {imax:.3e} {r.current_unit} "
          f"(gate < A3 floor {A3_FLOOR_A_CM2:g} A/cm**2; adapter floor {ADAPTER_FLOOR_A_CM2:g} A/cm**2: "
          f"{'below' if imax < ADAPTER_FLOOR_A_CM2 else 'NOT below'}); nodes {len(Tn)}")
    assert r.analysis == "equilibrium" and r.current_unit == "A/cm**2"
    assert len(Tn) == len(np.asarray(prof["x_um"]))
    assert dev <= 1e-6
    for c, vals in r.currents.items():
        for v in vals:
            assert abs(v) < A3_FLOOR_A_CM2, (c, v)


def test_R51_low_field_limit():
    _need_devsim()
    from simws import engines
    p = _n_bar(V=[0.04])
    re_ = engines.module("devsim_eb").run(p)
    rd = engines.module("devsim").run(p)
    assert re_.current_unit == rd.current_unit == "A/cm**2"
    worst = 0.0
    for c in ("left", "right"):
        ie, idd = re_.currents[c][0], rd.currents[c][0]
        rel = abs(ie / idd - 1.0)
        worst = max(worst, rel)
        print(f"\n[R51 {c}] devsim_eb {ie:.9e} vs devsim {idd:.9e} A/cm**2: rel {rel:.3e} (gate <= 1e-3)")
    assert worst <= 1e-3


@pytest.fixture(scope="module", params=[2.0, 6.0, 20.0], ids=["2V", "6V", "20V"])
def r52_case(request, mhub):
    _need_devsim()
    V = request.param
    p = _n_bar(V=[V])
    r = mhub.submit("device.devsim_eb", params=_params(p))
    assert r.status == "ok", r.message
    prof = _eb().profiles(p)
    return V, p, r, prof


# Marked by the lead after the checker's report (contract A11.6): R52 at 6 V and 20 V is a recorded FAIL. The bar is
# not homogeneous there (space-charge-limited: n = 1.06 and 1.46 N_D, the field changes by 64 % and 142 % across
# it), so the closed form R52 compares with does not apply; the gate is unchanged. 2 V passes. Strict xfail.
def _r52_xfail(request):
    if request.node.callspec.params.get("r52_case") in (6.0, 20.0):
        request.applymarker(pytest.mark.xfail(strict=True, reason="recorded FAIL of R52 at 6 V and 20 V (A11.6): "
                                              "the bar is space-charge limited, not homogeneous"))


def test_R52_hot_electrons_homogeneous(r52_case, request):
    _r52_xfail(request)
    V, p, r, prof = r52_case
    x = np.asarray(prof["x_um"], float)
    psi = np.asarray(prof["potential_V"], float)
    n = np.asarray(prof["n_cm3"], float)
    Tn = np.asarray(prof["Tn_K"], float)
    i = int(np.argmin(np.abs(x - 0.5 * (x[0] + x[-1]))))
    E = abs((psi[i + 1] - psi[i - 1]) / ((x[i + 1] - x[i - 1]) * 1e-4))          # V/cm (x in um)
    vs = float(r.diagnostics["eb_vsat_n_cm_s"])
    g = math.sqrt(1.0 + (MU0 * E / vs) ** 2)
    Tn_ref = T0 * g
    J_ref = Q_E * n[i] * MU0 * E / g                                              # A/cm^2
    q = r.q("dc.I.right")
    assert q.unit == "A/m**2"
    val = q.value[0] if isinstance(q.value, list) else float(np.asarray(q.value).ravel()[0])
    J = abs(val) * 1e-4                                                            # A/m^2 -> A/cm^2
    rT = abs(Tn[i] / Tn_ref - 1.0)
    rJ = abs(J / J_ref - 1.0)
    print(f"\n[R52 V={V:g}] node {i} x={x[i]:.4f} um, E={E:.6e} V/cm, n={n[i]:.6e} cm^-3, v_s={vs:.6g} cm/s, "
          f"mu0E/vs={MU0 * E / vs:.4f}\n"
          f"    Tn={Tn[i]:.6f} K vs ref {Tn_ref:.6f} K: rel {rT:.3e} (gate <= 1e-2)\n"
          f"    J={J:.6e} A/cm^2 (dc.I.right {val:.6e} A/m^2) vs ref {J_ref:.6e}: rel {rJ:.3e} (gate <= 1e-2)\n"
          f"    eb_Tn_max_K={r.diagnostics.get('eb_Tn_max_K')}")
    assert val > 0                                       # current flows into the device at the positive contact
    assert rT <= 1e-2
    assert rJ <= 1e-2


@pytest.fixture(scope="module")
def r53_result(mhub):
    _need_devsim()
    r = mhub.submit("device.devsim_eb", params=_params(_n_bar(V=[6.0])), fidelity="paper")
    return r


@pytest.mark.slow                # the paper-fidelity mesh study takes about 250 s here
def test_R53_mesh_error_bar(r53_result):
    r = r53_result
    assert r.status == "ok", r.message
    q = r.q("dc.I.right")
    assert q.uncertainty.kind == "sigma"
    sig = np.asarray(q.uncertainty.sigma, float).ravel()
    val = np.asarray(q.value, float).ravel()
    assert sig.shape == val.shape == (1,)
    rel = float(sig[0] / abs(val[0]))
    print(f"\n[R53] dc.I.right = {val[0]:.6e} A/m^2, sigma = {sig[0]:.3e} A/m^2: rel {rel:.3e} (gate < 1e-2); "
          f"mesh record passed={r.verification.get('mesh', {}).get('passed')}")
    assert rel < 1e-2


@pytest.mark.slow                # the paper-fidelity mesh study takes about 250 s here
def test_R53_mesh_record_as_device_devsim(r53_result):
    """'same ... mesh study' (A11.3): the A4.5 record and flag on the paper-fidelity result."""
    r = r53_result
    assert r.status == "ok", r.message
    rec = r.verification["mesh"]
    assert rec["kind"] == "solution" and rec["levels"] == 3
    assert rec["ratio"] == pytest.approx(math.sqrt(2.0), rel=1e-12)
    assert set(rec["quantities"]) == {"dc.I.left", "dc.I.right"}
    assert "budget:numerical_only" in r.flags
    assert FLAG in r.flags
    assert r.q("dc.I.right").uncertainty.components.keys() == {"numerical"}


R54_V = [round(0.30 + 0.05 * k, 2) for k in range(7)]


def test_R54_different_physics_check(mhub):
    _need_devsim()
    from simws.compare import compare_results, record, records_for
    from simws.hub.trust import assess_stored
    assert R54_V == [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    p = pn_diode_1d(V=R54_V)
    rd = mhub.submit("device.devsim", params=_params(p))
    re_ = mhub.submit("device.devsim_eb", params=_params(p))
    assert rd.status == "ok", rd.message
    assert re_.status == "ok", re_.message
    ia, ie = rd.q("dc.I.anode").array(), re_.q("dc.I.anode").array()
    rels = np.abs(ia / ie - 1.0)
    k = int(np.argmax(rels))
    print("\n[R54] V, devsim dc.I.anode, devsim_eb dc.I.anode (A/m^2), rel diff:")
    for v, a, b, d in zip(R54_V, ia, ie, rels):
        print(f"    {v:.2f}  {a:.6e}  {b:.6e}  {d:.3e}")
    level_before = assess_stored(mhub, rd.id).level
    c = compare_results(mhub, rd.id, "dc.I.anode", re_.id, tolerance=0.01)
    record(mhub, c)
    level_after = assess_stored(mhub, rd.id).level
    print(f"[R54] largest relative difference {rels[k]:.4e} at {R54_V[k]:.2f} V (reported, not gated); "
          f"compare_results worst={c.worst} at {c.worst_at} passed={c.passed} independent={c.independent}; "
          f"trust of devsim result {level_before} -> {level_after}; devsim_eb Tn max "
          f"{re_.diagnostics.get('eb_Tn_max_K')} K")
    assert c.worst == pytest.approx(float(rels.max()), rel=1e-6)
    assert c.independent is False
    assert any(rec["a"] == rd.id and rec["b"] == re_.id for rec in records_for(mhub, rd.id))
    assert level_after == level_before
