"""Contract amendment A13: a second implant engine on different code, Geant4 ("g4implant"). Engine, capability,
flags, adapter, and the acceptance checks R60 to R67 (R66 reported). Written from docs/ARCHITECTURE_V1.md (sections
2, 6, A2, A7, A12, A13) only; the implementation files and tools/g4implant/g4implant.cc were not read.

The program is called through the public names of the engine module (build_input, run_program, parse_output) only
where the contract asks for something the hub does not give: the `# dedx_e` header lines (R63) and the 1 eV physics
cutoff (R66, "run through the program directly"). Tests that need the program are skipped where
<SIMWS_G4SEE_PREFIX>/bin/g4implant is not installed; RustBCA comparisons also need libRustBCA."""
import math
import os
import re
from pathlib import Path

import numpy as np
import pytest

ADAPTER = "implant.g4implant"
RB = "implant.rustbca"
UM = 1e-6
ISOTOPE = {"B": (5, 11.009305), "P": (15, 30.973762), "As": (33, 74.921595)}   # A7.2 table
CUTOFF_FLAG = ("implant:geant4_nuclear_cutoff: collisions transferring less than 10 eV are ignored "
               "(G4ScreenedNuclearRecoil physics cutoff, A13.3)")
EMPIRICAL_FLAG = ("implant:empirical_stopping: Geant4 G4IonParametrisedLossModel (its ion data, scaled with the "
                  "square root of the energy below the lowest table energy; A13)")
R16 = {("B", 50.0): (0.1608, 0.0504), ("B", 100.0): (0.2994, 0.0710), ("P", 50.0): (0.0607, 0.0256)}


def _implant(*a, **k):
    from simws.spec.library import implant_si
    return implant_si(*a, **k)


def _emp(p):
    return p.with_changes({"models.electronic_stopping": "empirical"})


def _params(p):
    return {"problem": p.model_dump(mode="json")}


def _mod():
    from simws import engines
    return engines.module("g4implant")


def _need_g4():
    pre = os.environ.get("SIMWS_G4SEE_PREFIX")
    exe = Path(pre) / "bin" / "g4implant" if pre else None
    if not ((exe is not None and exe.exists()) or _mod().available()):
        pytest.skip("g4implant is not installed (<SIMWS_G4SEE_PREFIX>/bin/g4implant)")


def _need_rb():
    pytest.importorskip("libRustBCA")


def _run(p, folder):
    from simws.engines.base import ImplantResult
    r = _mod().run(p, folder)
    assert isinstance(r, ImplantResult)
    assert len(r.depth_um) + r.n_reflected + r.n_transmitted == r.n_ions == p.statistics.n_ions   # R61
    assert len(r.lateral_um) == len(r.depth_um)
    return r


# ------------------------------------------------------------------------------------------------ registry, capability
def test_registry_and_family():
    from simws import engines
    assert engines.ENGINES["g4implant"] == "simws.engines.g4implant_engine"
    assert engines.family("g4implant") == "implant"
    assert engines.adapter_name("g4implant") == ADAPTER
    assert _mod().FAMILY == "implant"
    for attr in ("CAPABILITY", "unsupported", "available", "run"):
        assert hasattr(_mod(), attr), attr


def test_capability_fixed_fields():
    from simws import engines
    from simws.engines.base import ImplantCapability
    c = engines.capability("g4implant")
    assert isinstance(c, ImplantCapability)
    assert c.engine == "g4implant" and c.env == "g4see" and c.in_process is False
    assert c.structures == ["amorphous"]
    assert c.potentials == ["zbl", "moliere"]
    assert c.electronic_stopping == ["lindhard_scharff", "empirical"]
    assert c.recoils == [False, True]
    assert c.max_layers == 20 and c.max_keV_per_amu == 25.0
    print(f"\n[A13.3/A13.6] tested_keV_per_amu={c.tested_keV_per_amu} tested_target_symbols={c.tested_target_symbols} "
          f"tested_ion_symbols={c.tested_ion_symbols} by_stopping={c.tested_ion_symbols_by_stopping}")
    print(f"[A13.3] notes: {c.notes!r}")
    assert "bind" in c.notes.lower()                   # A13.3: binding energies not used, stated in its notes


def test_refusals():
    from simws import engines
    r = engines.unsupported("g4implant", _implant("P", 50).with_changes({"models.potential": "kr_c"}))
    assert len(r) == 1 and r[0].startswith("potentials:"), r
    r = engines.unsupported("g4implant", _implant("B", 300.0))
    assert any(s.startswith("max_keV_per_amu:") for s in r), r
    for p in (_implant("P", 50), _emp(_implant("B", 50)), _emp(_implant("P", 50)),
              _implant("P", 50).with_changes({"models.potential": "moliere"})):
        assert engines.unsupported("g4implant", p) == [], p.models


def test_manifest():
    from simws.hub import adapter as A
    from simws.hub import citations
    from simws.adapters import implant as I
    m = A.manifest(ADAPTER)
    assert m.name == ADAPTER and m.engine == "g4implant" and m.engine_version == "11.2.2"
    assert m.version == "1.0" and m.inputs == ["problem"]
    assert "geant4_2016" in m.citations and "mendenhall2005" in m.citations
    assert citations.unknown(["geant4_2016", "mendenhall2005"]) == []
    assert issubclass(I.ImplantG4implant, I.ImplantAdapter)
    rb = A.manifest(RB)
    assert sorted(o.name for o in m.outputs) == sorted(o.name for o in rb.outputs)       # same quantities as A7.5


# Amended by the lead: A13.4 was corrected before any acceptance run (adapter env "tcad", as G4SEE's adapter); the
# checker had read the text before the correction.
def test_manifest_env():
    from simws.hub import adapter as A
    assert A.manifest(ADAPTER).env == "tcad", A.manifest(ADAPTER).env


def test_not_installed(monkeypatch, tmp_path, hub):
    from simws.engines.base import EngineNotAvailable
    monkeypatch.setenv("SIMWS_G4SEE_PREFIX", str(tmp_path / "no_env"))
    assert _mod().available() is False
    with pytest.raises(EngineNotAvailable) as ei:
        _mod().run(_implant("P", 50, n_ions=100), tmp_path / "w")
    assert "install_g4implant" in str(ei.value), str(ei.value)
    r = hub.submit(ADAPTER, params=_params(_implant("P", 50, n_ions=100)))
    assert r.status == "engine_error" and "not installed" in r.message.lower(), (r.status, r.message)


def test_unsupported_through_hub(hub):
    r = hub.submit(ADAPTER, params=_params(_implant("P", 50, n_ions=100).with_changes({"models.potential": "kr_c"})))
    assert r.status == "invalid_input", (r.status, r.message)
    assert "unsupported by g4implant: " in r.message and "potentials:" in r.message, r.message


# ------------------------------------------------------------------------------------------------ flags, diagnostics
def test_flags_and_diagnostics(tmp_path, hub):
    _need_g4()
    p = _implant("P", 50, n_ions=2000)
    r = _run(p, tmp_path / "a")
    assert any(f.startswith("implant:amorphous_target") for f in r.flags), r.flags
    assert CUTOFF_FLAG in r.flags
    assert EMPIRICAL_FLAG not in r.flags
    print(f"\n[A13.3] diagnostics: {r.diagnostics}")
    vals = list(r.diagnostics.values())
    assert any(isinstance(v, str) and "geant4" in v.lower() for v in vals)          # version line
    assert any(isinstance(v, (int, float)) and v == 10.0 for v in vals)              # physics cutoff, eV
    # recoil cutoff per nucleon = the ion's cutoff_eV / its mass (A13.3); which mass is not fixed: 1e-3
    exp_rc = p.beam.ion.cutoff_eV / p.beam.ion.mass_amu
    assert any(isinstance(v, float) and abs(v / exp_rc - 1) < 1e-3 for v in vals), (exp_rc, vals)
    re_ = _run(_emp(p), tmp_path / "b")
    assert EMPIRICAL_FLAG in re_.flags and CUTOFF_FLAG in re_.flags
    h = hub.submit(ADAPTER, params=_params(p))
    assert h.status == "ok", h.message
    assert CUTOFF_FLAG in h.flags and "budget:statistical_only" in h.flags
    from simws import engines
    for x in engines.capability("g4implant").outside_tested_range(p):
        assert "extrapolated:" + x in h.flags, (x, h.flags)
    assert "statistics" in h.verification and h.verification["statistics"]["method"] == "split halves"


# ------------------------------------------------------------------------------------------------ R60, R61, R62
def test_R60_repeatability(tmp_path):
    _need_g4()
    p = _implant("P", 50, n_ions=2000)
    a, b = _run(p, tmp_path / "a"), _run(p, tmp_path / "b")
    assert a.depth_um == b.depth_um
    c = _run(_implant("P", 50, n_ions=2000, seed=2), tmp_path / "c")
    assert c.depth_um != a.depth_um
    print(f"\n[R60] seed 1 twice identical ({len(a.depth_um)} implanted); seed 2 differs; "
          f"mean depth {np.mean(a.depth_um) * 1e3:.2f} / {np.mean(c.depth_um) * 1e3:.2f} nm")


# Marked after the run (section 2 of the contract): with seed 1 the program aborts (Geant4 c2_exception "Failed root
# find" in G4ScreenedNuclearRecoil, "A 1.00728", i.e. a collision with a hydrogen nucleus) for this thin target; seed
# 2 runs. Kept as a recorded FAIL of R61.
_R61_FAIL = pytest.mark.xfail(strict=True, reason="recorded FAIL A13 R61: g4implant aborts (Geant4 'Failed root find' "
                              "in G4ScreenedNuclearRecoil) on the 0.05 um target, P 100 keV, 2000 ions, seed 1")


# amended by the lead (A13.7): the abort was a program defect (atoms leaving the target collided in the vacuum
# world); fixed, so the recorded-FAIL mark is removed and R61 runs again
def test_R61_counting_thin_target(hub, tmp_path):
    _need_g4()
    p = _implant("P", 100, thickness_um=0.05, n_ions=2000)
    r = _run(p, tmp_path / "thin")
    assert r.n_transmitted > 0
    h = hub.submit(ADAPTER, params=_params(p))
    print(f"\n[R61] 0.05 um: implanted {len(r.depth_um)} reflected {r.n_reflected} transmitted {r.n_transmitted}; "
          f"hub status {h.status}")
    assert any(f.startswith("implant:transmitted:") for f in h.flags), (h.status, h.message, h.flags)


def test_R61_counting_thin_target_seed2_reported(hub, tmp_path):
    """Not a contract check: the same thin target with seed 2, to show the counting rule where the program runs."""
    _need_g4()
    p = _implant("P", 100, thickness_um=0.05, n_ions=2000, seed=2)
    r = _run(p, tmp_path / "thin2")
    h = hub.submit(ADAPTER, params=_params(p))
    print(f"\n[R61 seed 2] implanted {len(r.depth_um)} reflected {r.n_reflected} transmitted {r.n_transmitted}; "
          f"hub status {h.status}: {h.message}")
    assert r.n_transmitted > 0
    assert any(f.startswith("implant:transmitted:") for f in h.flags), h.flags


@pytest.fixture(scope="module")
def rhub(tmp_path_factory):
    from simws.hub.jobs import Hub
    return Hub(tmp_path_factory.mktemp("a13_acceptance") / "store", mode="inprocess")


def _ok(hub, adapter, p):
    r = hub.submit(adapter, params=_params(p))
    assert r.status == "ok", (adapter, r.status, r.message)
    N = p.statistics.n_ions
    counts = [round(r.q(n).value * N) for n in ("implant.fraction_implanted", "implant.fraction_reflected",
                                                 "implant.fraction_transmitted")]
    assert sum(counts) == N                                   # R61: in every run
    return r


def _vs(r, name):
    q = r.q(name)
    return float(q.value), float(q.uncertainty.sigma)


@pytest.mark.slow
def test_R62_seeds(rhub):
    _need_g4()
    r1, r2 = _ok(rhub, ADAPTER, _implant("P", 50)), _ok(rhub, ADAPTER, _implant("P", 50, seed=2))
    res = {}
    for n in ("implant.Rp", "implant.dRp", "implant.lateral_straggle"):
        (a, sa), (b, sb) = _vs(r1, n), _vs(r2, n)
        comb = math.sqrt(sa ** 2 + sb ** 2)
        res[n] = abs(a - b) <= 4 * comb
        print(f"\n[R62] {n}: {a * 1e9:.3f} +- {sa * 1e9:.3f} vs {b * 1e9:.3f} +- {sb * 1e9:.3f} nm; "
              f"|diff|/sigma = {abs(a - b) / comb:.2f}")
    assert all(res.values()), res


# ------------------------------------------------------------------------------------------------ R63
_DEDX = re.compile(r"^#\s*dedx_e\s+(\S+)\s+keV\s+(\S+)\s+MeV_cm2_g\s*$")


def _dedx_lines(problem, energies, folder):
    m = _mod()
    txt = m.run_program(m.build_input(problem, dedx_keV=tuple(energies)), folder)
    out = {}
    for line in txt.splitlines():
        mm = _DEDX.match(line.strip())
        if mm:
            out[float(mm.group(1))] = float(mm.group(2))
    return out


@pytest.mark.parametrize("ion", ["B", "P", "As"])
def test_R63_ls_formula_in_program(ion, tmp_path):
    _need_g4()
    from simws.radiation import empirical_stopping as es
    Z1, M1 = ISOTOPE[ion]
    p = _implant(ion, 50, n_ions=100)
    got = _dedx_lines(p, (10.0, 50.0, 100.0), tmp_path / "ls")
    assert sorted(got) == [10.0, 50.0, 100.0], got
    emp = _dedx_lines(_emp(p), (10.0, 50.0, 100.0), tmp_path / "emp")
    for E in (10.0, 50.0, 100.0):
        ref = es.lindhard_scharff_MeV_cm2_mg(Z1, M1, E * 1e3)
        val = got[E] / 1000.0                                          # MeV cm^2/g -> MeV cm^2/mg
        rel = val / ref - 1
        print(f"\n[R63 {ion} {E:g} keV] program {val:.6e} MeV cm2/mg, formula {ref:.6e}, rel {rel:+.2e}; "
              f"empirical/LS = {emp.get(E, float('nan')) / got[E]:.4f}")
        assert abs(rel) <= 1e-3


# ------------------------------------------------------------------------------------------------ R64
def _table(r, ion, E, label):
    rp_ref, drp_ref = R16[(ion, E)]
    ok = {}
    for name, ref in (("implant.Rp", rp_ref), ("implant.dRp", drp_ref)):
        v, s = _vs(r, name)
        ok[name] = abs(v / UM - ref) <= 0.25 * ref
        print(f"\n[{label} {ion} {E:g} keV] {name}: {v * 1e9:.2f} +- {s * 1e9:.2f} nm, table {ref * 1e3:.1f} nm, "
              f"{v / UM / ref - 1:+.4f} ({'within' if ok[name] else 'OUTSIDE'} 25 %)")
    return ok


_CASES = [pytest.param(ion, E, id=f"{ion}{int(E)}keV") for ion, E in R16]
# Marked after the runs (section 2 of the contract); strict, so a change is noticed.
_CRASH = ("g4implant aborts: Geant4 c2_exception 'Failed root find' in G4ScreenedNuclearRecoil for a collision with "
          "A = 1.00728 (hydrogen, presumably the vacuum world), engine_error")
_R64_FAIL = {   # re-marked by the lead after the program defect was fixed and R64 rerun (A13.7)
    ("B", 50.0): pytest.mark.xfail(strict=True, reason="recorded FAIL A13 R64 B 50 keV empirical: Rp 201.92 nm "
                                   "against 160.8 (+25.6 %), dRp 64.72 against 50.4 (+28.4 %), gate 25 %"),
    ("B", 100.0): pytest.mark.xfail(strict=True, reason="recorded FAIL A13 R64 B 100 keV empirical: dRp 89.35 nm "
                                    "against 71.0 (+25.8 %, gate 25 %); Rp 362.00 nm +20.9 % within"),
    ("P", 50.0): pytest.mark.xfail(strict=True, reason="recorded FAIL A13 R64 P 50 keV empirical: Rp 79.90 nm "
                                   "against 60.7 (+31.6 %), dRp 33.87 against 25.6 (+32.3 %), gate 25 %"),
}
_R64_CASES = [pytest.param(ion, E, id=f"{ion}{int(E)}keV", marks=[_R64_FAIL[(ion, E)]] if (ion, E) in _R64_FAIL else []) for ion, E in R16]
_R65_FAIL = pytest.mark.xfail(strict=True, reason="recorded FAIL A13 R65 P 50 keV: Geant4 Rp 80.60 nm against RustBCA "
                              "71.34 nm (+12.97 %, gate 10 %), dRp 34.17 against 29.47 nm (+15.93 %, gate 15 %)")
_R65_CASES = [pytest.param(ion, E, id=f"{ion}{int(E)}keV", marks=[_R65_FAIL] if ion == "P" else [])
              for ion, E in R16]


@pytest.mark.slow
@pytest.mark.parametrize("ion,E", _R64_CASES)
def test_R64_tables_empirical(rhub, ion, E):
    _need_g4()
    r = _ok(rhub, ADAPTER, _emp(_implant(ion, E)))
    assert EMPIRICAL_FLAG in r.flags
    ok = _table(r, ion, E, "R64 empirical")
    assert all(ok.values()), ok


@pytest.mark.slow
@pytest.mark.parametrize("ion,E", _CASES)
def test_R64_tables_ls_reported(rhub, ion, E):
    _need_g4()
    _table(_ok(rhub, ADAPTER, _implant(ion, E)), ion, E, "R64 lindhard_scharff, no gate")


# ------------------------------------------------------------------------------------------------ R65, R67
R65_TOL = {"implant.Rp": 0.10, "implant.dRp": 0.15}


def _r65(rhub, ion, E, record=True):
    from simws import compare
    p = _implant(ion, E)
    g, b = _ok(rhub, ADAPTER, p), _ok(rhub, RB, p)
    out = {}
    for name, tol in R65_TOL.items():
        c = compare.compare_results(rhub, g.id, name, b.id, tolerance=tol, metric="relative")
        if record:
            compare.record(rhub, c)
        (vg, sg), (vb, sb) = _vs(g, name), _vs(b, name)
        print(f"\n[R65 {ion} {E:g} keV] {name}: Geant4 {vg * 1e9:.2f} +- {sg * 1e9:.2f} nm, RustBCA {vb * 1e9:.2f} +- "
              f"{sb * 1e9:.2f} nm, relative {vg / vb - 1:+.4f} (tolerance {tol}); worst={c.worst:.4f} passed={c.passed} "
              f"independent={c.independent} engines={c.engine_a}/{c.engine_b}")
        out[name] = c
    return g, b, out


@pytest.mark.slow
@pytest.mark.parametrize("ion,E", _R65_CASES)
def test_R65_cross_code(rhub, ion, E):
    _need_g4()
    _need_rb()
    g, b, cmps = _r65(rhub, ion, E)
    for name, c in cmps.items():
        assert c.kind == "engine_pair" and c.metric == "relative"
        assert c.independent is True, name
    assert all(c.passed for c in cmps.values()), {n: (c.worst, c.passed) for n, c in cmps.items()}


@pytest.mark.slow
@pytest.mark.parametrize("ion,E", _CASES)
def test_R67_trust_follows_recorded_comparison(rhub, ion, E):
    """A RustBCA result whose R65 comparison passed and was recorded is "verified" (A2.3 rule 5); one whose
    comparison failed stays "unverified". Per section 6.1 a single passed independent comparison suffices."""
    _need_g4()
    _need_rb()
    from simws.hub.trust import assess_stored
    g, b, cmps = _r65(rhub, ion, E)                   # recorded (replaces the same pair and quantities)
    t = assess_stored(rhub, b.id)
    passed = [n for n, c in cmps.items() if c.passed and c.independent]
    print(f"\n[R67 {ion} {E:g} keV] passed comparisons: {passed}; RustBCA trust {t.level} (in_range {t.in_range}); "
          f"Geant4 trust {assess_stored(rhub, g.id).level}; reasons {t.reasons}")
    # amended by the lead (A13.7): a result flagged as a known failure (boron in Lindhard-Scharff mode, A7.9) is
    # never above "unverified", and one failed comparison blocks "verified"
    known = any(":known_failure:" in f for f in b.flags)
    expect = "verified" if (len(passed) == len(cmps) and not known) else "unverified"
    assert t.level == expect, (t.level, t.reasons)
    if known and passed:
        assert any("known-failure" in r for r in t.reasons), t.reasons
    if 0 < len(passed) < len(cmps):
        print(f"[R67 {ion} {E:g} keV] AMBIGUITY: some but not all R65 quantities passed; the trust rule verifies on any")


def test_R67_failed_comparison_stays_unverified(hub):
    """The negative half of R67 with real engine results on a fresh hub: a recorded engine-pair comparison that
    fails (RustBCA P 50 keV against Geant4 B 50 keV, R65's 10 % on Rp) leaves the RustBCA result unverified."""
    _need_g4()
    _need_rb()
    from simws import compare
    from simws.hub.trust import assess_stored
    b = _ok(hub, RB, _implant("P", 50, n_ions=2000))
    g = _ok(hub, ADAPTER, _implant("B", 50, n_ions=2000))
    c = compare.compare_results(hub, g.id, "implant.Rp", b.id, tolerance=0.10, metric="relative")
    compare.record(hub, c)
    assert c.independent is True and c.passed is False
    t = assess_stored(hub, b.id)
    print(f"\n[R67 negative] worst={c.worst:.3f}; RustBCA trust {t.level}; reasons {t.reasons}")
    assert t.level == "unverified"


# ------------------------------------------------------------------------------------------------ R66 (reported)
@pytest.mark.slow
@pytest.mark.parametrize("E", [50.0, 100.0])   # amended by the lead (A13.7): the crash mark removed after the fix
def test_R66_empirical_vs_empirical_reported(rhub, E):
    _need_g4()
    _need_rb()
    p = _emp(_implant("B", E))
    g, b = _ok(rhub, ADAPTER, p), _ok(rhub, RB, p)
    for name in ("implant.Rp", "implant.dRp"):
        (vg, sg), (vb, sb) = _vs(g, name), _vs(b, name)
        ref = R16[("B", E)][0 if name == "implant.Rp" else 1]
        print(f"\n[R66 B {E:g} keV empirical] {name}: Geant4 {vg * 1e9:.2f} +- {sg * 1e9:.2f} nm, RustBCA "
              f"{vb * 1e9:.2f} +- {sb * 1e9:.2f} nm, Geant4/RustBCA - 1 = {vg / vb - 1:+.4f}; vs table: "
              f"{vg / UM / ref - 1:+.4f} / {vb / UM / ref - 1:+.4f}")


def _program_depths(problem, folder, cutoff=None):
    """End points of the implanted ions (status 0) from the program, optionally with the physics cutoff changed in
    the keyword input file."""
    m = _mod()
    text = m.build_input(problem)
    if cutoff is not None:
        new, n = re.subn(r"(?m)^physics_cutoff_eV\s+\S+\s*$", f"physics_cutoff_eV {cutoff}", text)
        assert n == 1, "the keyword input has no single physics_cutoff_eV line"
        text = new
    o = m.parse_output(m.run_program(text, folder))
    x = np.asarray(o["x"], float)
    s = np.asarray(o["status"])
    return x[s == 0], o


@pytest.mark.slow
def test_R66_physics_cutoff_1eV_reported(rhub, tmp_path):
    _need_g4()
    p = _implant("B", 50.0)
    r = _ok(rhub, ADAPTER, p)
    x10, _ = _program_depths(p, tmp_path / "c10")
    # the direct path reproduces the adapter at the fixed 10 eV (so the 1 eV numbers below are comparable)
    assert np.mean(x10) * UM == pytest.approx(r.q("implant.Rp").value, rel=1e-9)
    x1, o1 = _program_depths(p, tmp_path / "c1", cutoff=1.0)
    rp_ref, drp_ref = R16[("B", 50.0)]
    print(f"\n[R66 B 50 keV LS] physics cutoff 10 eV: Rp {np.mean(x10) * 1e3:.2f} nm dRp {np.std(x10, ddof=1) * 1e3:.2f} nm "
          f"(k={x10.size}); 1 eV: Rp {np.mean(x1) * 1e3:.2f} nm dRp {np.std(x1, ddof=1) * 1e3:.2f} nm (k={x1.size}); "
          f"1 eV / 10 eV - 1: Rp {np.mean(x1) / np.mean(x10) - 1:+.4f}, dRp "
          f"{np.std(x1, ddof=1) / np.std(x10, ddof=1) - 1:+.4f}; table {rp_ref * 1e3:.1f} / {drp_ref * 1e3:.1f} nm")
