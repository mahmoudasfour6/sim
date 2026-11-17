"""Contract amendment A8.3 / A8.4: SeeResult, SeeCapability, the registry with three families, the planner, and the
G4SEE translation (build_macro, edges, parse_histogram, availability, run's plumbing with a stand-in executable, and
R20 / R21 on the real engine). Written from docs/ARCHITECTURE_V1.md (A8, with the shared conventions of A7) only.

Histogram files are written here exactly as G4SEE v1.0 writes them (src/Scoring.cc, PrintHistogramToFile: width 15,
scientific with the set precision, zeros as a fixed "0"), with the precision 8 of A8.9 (which replaced the 10 of
A8.4: with 10 a number takes 16 characters and runs into its neighbour). Engine runs need G4SEE (SIMWS_G4SEE_PREFIX or <envs_root>/g4see) and are
skipped without it."""
import math
import os

import numpy as np
import pydantic
import pytest

from simws.spec.library import implant_si, npn_hbt_2d, pn_diode_1d


def _slab(*a, **k):
    from simws.spec.library import see_si_slab
    return see_si_slab(*a, **k)


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


def _scap(**over):
    """A hand-made SEE capability allowing everything; `over` restricts single fields."""
    from simws.engines.base import SeeCapability
    d = dict(engine="toy", env="toyenv", description="test sheet", in_process=False,
             particles=["proton", "alpha", "neutron", "e-", "gamma", "ion"], em=["option4", "option3", "livermore"],
             hadronic=["none", "ftfp_bert_hp"], max_beol_layers=20, tested_particles=["alpha", "proton"],
             tested_energy_MeV=(5.0, 10.0), tested_materials=["G4_Si"], tested_em=["option4"],
             tested_hadronic=["none"])
    d.update(over)
    return SeeCapability(**d)


def _with_beol(p, n, material="G4_Si"):
    return p.with_changes({"geometry.beol": [{"name": f"l{i}", "material": material, "thickness_um": 0.5}
                                             for i in range(n)]})


# ------------------------------------------------------------------------------------------------ G4SEE file format
def g4see_number(x, prec):
    """One histogram number as G4SEE prints it: setw(15) scientific(prec), or a fixed '0' for a zero count."""
    return f"{x:15.{prec}e}"


def write_g4see_histogram(path, edges, counts, underflow, overflow, prec=8, quantity="Edep"):
    """Writes a histogram file the way G4SEE v1.0 does (Scoring::PrintHistogramToFile)."""
    def cnt(c):
        return f"{0.0:15.0f}" if c == 0 else g4see_number(c, prec)
    s = f"{quantity:>15}{'Counts':>15}\n"
    s += f"{'underflow':>15}" + cnt(underflow) + "\n"
    s += f"{'overflow':>15}" + cnt(overflow) + "\n"
    for e, c in zip(edges[:-1], counts):
        s += g4see_number(e, prec) + cnt(c) + "\n"
    s += g4see_number(edges[-1], prec)                       # the upper edge only, no newline (as G4SEE)
    with open(path, "w") as f:
        f.write(s)
    return path


def ref_edges(scale, bins, low, high):
    i = np.arange(bins + 1, dtype=float)
    if scale == "lin":
        return low + i * (high - low) / bins
    return 10.0 ** (math.log10(low) + i * (math.log10(high) - math.log10(low)) / bins)


def _parse(path, problem):
    """parse_histogram(path, problem). A8.4 says it 'reads Edep_0_hist_t0.out' without fixing whether `path` is the
    file or the folder holding it; the file path is tried first, then its folder (contract ambiguity, reported)."""
    mod = _g4()
    try:
        return mod.parse_histogram(path, problem)
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return mod.parse_histogram(os.path.dirname(path), problem)


def _hist_problem(scale="lin", bins=20, low=0.0, high=5.25):
    return _slab(bins=bins).with_changes({"histogram.scale": scale, "histogram.low_MeV": low,
                                          "histogram.high_MeV": high})


# ------------------------------------------------------------------------------------------------ SeeResult
def test_see_result_model():
    import json

    from simws.engines.base import SeeResult
    r = SeeResult(engine="toy", problem_id="0" * 32, n_events=10, edges_MeV=[0.0, 1.0, 2.0], counts=[4.0, 5.0],
                  underflow=0.0, overflow=1.0)
    assert r.flags == [] and r.diagnostics == {}
    json.loads(r.model_dump_json())
    with pytest.raises(pydantic.ValidationError):
        SeeResult(engine="toy", problem_id="x", n_events=1, edges_MeV=[0.0, 1.0], counts=[1.0], underflow=0.0,
                  overflow=0.0, colour="red")


# ------------------------------------------------------------------------------------------------ SeeCapability
def test_scap_refuses_unknown_fields_and_defaults():
    with pytest.raises(pydantic.ValidationError):
        _scap(gpus=4)
    assert _scap().notes == ""


def test_scap_supported_is_empty():
    c = _scap()
    for p in (_slab(), _slab("proton", 10.0), _slab("neutron", 14.0, hadronic="ftfp_bert_hp"),
              _slab("ion", 100.0, ion_Z=8, ion_A=16), _with_beol(_slab(), 20)):
        assert c.unsupported(p) == []


@pytest.mark.parametrize("field,over,change", [
    ("particles", {"particles": ["proton"]}, {}),                                   # alpha refused
    ("particles", {"particles": ["alpha", "proton"]}, {"source.particle": "gamma"}),
    ("em", {"em": ["option4"]}, {"physics.em": "livermore"}),
    ("em", {"em": ["option3"]}, {}),
    ("hadronic", {"hadronic": ["none"]}, {"physics.hadronic": "ftfp_bert_hp"}),
    ("hadronic", {"hadronic": []}, {}),
])
def test_scap_one_reason_per_field(field, over, change):
    p = _slab().with_changes(change)
    r = _scap(**over).unsupported(p)
    assert len(r) == 1, r
    assert r[0].startswith(field + ":"), r


def test_scap_max_beol_layers():
    p = _with_beol(_slab(), 3)
    r = _scap(max_beol_layers=2).unsupported(p)
    assert len(r) == 1 and r[0].startswith("max_beol_layers:"), r
    assert _scap(max_beol_layers=3).unsupported(p) == []
    assert _scap(max_beol_layers=0).unsupported(_slab()) == []


def test_scap_reasons_add_up():
    p = _with_beol(_slab("neutron", 14.0, hadronic="ftfp_bert_hp"), 3).with_changes({"physics.em": "option3"})
    c = _scap(particles=["alpha"], em=["option4"], hadronic=["none"], max_beol_layers=1)
    assert sorted(s.split(":", 1)[0] for s in c.unsupported(p)) == ["em", "hadronic", "max_beol_layers",
                                                                      "particles"]


def _prefixes(strings):
    return sorted({s.split(":", 1)[0] for s in strings})


def test_scap_outside_tested_range_inside():
    c = _scap()
    for p in (_slab(), _slab("proton", 10.0), _slab("alpha", 7.5), _slab("proton", 5.0),
              _slab(beol=[("a", "G4_Si", 1.0)])):
        assert c.outside_tested_range(p) == [], p.source


@pytest.mark.parametrize("E,inside", [(5.0, True), (10.0, True), (4.999, False), (10.001, False), (14.0, False),
                                      (0.5, False)])
def test_scap_energy_bounds_included(E, inside):
    r = _scap().outside_tested_range(_slab("alpha", E))
    if inside:
        assert r == [], r
    else:
        assert len(r) == 1 and r[0].startswith("energy_MeV:"), r


@pytest.mark.parametrize("particle", ["neutron", "gamma", "e-"])
def test_scap_particle_not_tested(particle):
    r = _scap().outside_tested_range(_slab(particle, 5.0))
    assert len(r) == 1 and r[0].startswith("particle:"), r


def test_scap_ion_not_tested():
    r = _scap().outside_tested_range(_slab("ion", 7.0, ion_Z=2, ion_A=4))
    assert _prefixes(r) == ["particle"], r


def test_scap_materials():
    c = _scap()
    bulk = _slab().with_changes({"geometry.bulk_material": "G4_SILICON_CARBIDE"})
    r = c.outside_tested_range(bulk)
    assert r and _prefixes(r) == ["materials"], r
    beol = _slab(beol=[("si", "G4_Si", 1.0), ("ox", "G4_SILICON_DIOXIDE", 1.0)])
    r = c.outside_tested_range(beol)
    assert r and _prefixes(r) == ["materials"], r
    assert c.outside_tested_range(_slab(beol=[("si", "G4_Si", 1.0)])) == []


def test_scap_em_and_hadronic_not_tested():
    c = _scap()
    for em in ("option3", "livermore"):
        r = c.outside_tested_range(_slab().with_changes({"physics.em": em}))
        assert len(r) == 1 and r[0].startswith("em:"), r
    r = c.outside_tested_range(_slab(hadronic="ftfp_bert_hp"))
    assert len(r) == 1 and r[0].startswith("hadronic:"), r


def test_scap_outside_all_together():
    p = _slab("neutron", 14.0, hadronic="ftfp_bert_hp", beol=[("ox", "G4_SILICON_DIOXIDE", 1.0)]).with_changes(
        {"physics.em": "livermore"})
    assert _prefixes(_scap().outside_tested_range(p)) == ["em", "energy_MeV", "hadronic", "materials", "particle"]


# ------------------------------------------------------------------------------------------------ registry
def test_registry_has_g4see():
    from simws import engines
    assert engines.ENGINES["g4see"] == "simws.engines.g4see_engine"
    for n, m in (("devsim", "simws.engines.devsim_engine"), ("charon", "simws.engines.charon_engine"),
                 ("rustbca", "simws.engines.rustbca_engine")):
        assert engines.ENGINES[n] == m
    assert engines.names() == sorted(engines.ENGINES) and "g4see" in engines.names()


def test_family_and_adapter_name():
    from simws import engines
    assert engines.family("g4see") == "see"
    assert engines.adapter_name("g4see") == "see.g4see"
    assert engines.module("g4see").FAMILY == "see"
    assert engines.family("rustbca") == "implant" and engines.adapter_name("rustbca") == "implant.rustbca"
    for n in ("devsim", "charon"):
        assert engines.family(n) == "device" and engines.adapter_name(n) == f"device.{n}"


def test_g4see_module_interface():
    from simws import engines
    from simws.engines.base import SeeCapability
    mod = _g4()
    for attr in ("available", "unsupported", "build_macro", "edges", "parse_histogram", "run"):
        assert callable(getattr(mod, attr, None)), attr
    assert isinstance(engines.capability("g4see"), SeeCapability)
    assert isinstance(mod.available(), bool)
    for p in (_slab(), _slab("proton", 10.0), _slab("neutron", 14.0, hadronic="ftfp_bert_hp")):
        assert engines.unsupported("g4see", p) == mod.unsupported(p) == []


def test_problem_kind_refusal_all_directions(monkeypatch):
    """A7.3 for three families: exactly one 'problem kind:' reason across families, without calling the engine
    module's own check (replaced here by functions that fail when called)."""
    from simws import engines

    def boom(problem):
        raise AssertionError("the engine module's own check must not be called across families")

    for n in ("devsim", "charon", "rustbca", "g4see"):
        monkeypatch.setattr(engines.module(n), "unsupported", boom)
    see = _slab()
    for n in ("devsim", "charon", "rustbca"):
        r = engines.unsupported(n, see)
        assert len(r) == 1 and r[0].startswith("problem kind:"), (n, r)
    for other in (pn_diode_1d(), npn_hbt_2d(), implant_si()):
        r = engines.unsupported("g4see", other)
        assert len(r) == 1 and r[0].startswith("problem kind:"), r
    # the older pairs still refuse each other
    r = engines.unsupported("rustbca", pn_diode_1d())
    assert len(r) == 1 and r[0].startswith("problem kind:"), r
    r = engines.unsupported("devsim", implant_si())
    assert len(r) == 1 and r[0].startswith("problem kind:"), r


def test_same_family_uses_module_check():
    from simws import engines
    r = engines.unsupported("g4see", _with_beol(_slab(), 21))
    assert any(s.startswith("max_beol_layers:") for s in r), r


# ------------------------------------------------------------------------------------------------ planner
def test_plan_see_problem():
    from simws.engines.planner import plan
    p = _slab()
    pl = plan(p)
    assert pl.problem_id == p.problem_id()
    assert pl.primary == "g4see" and pl.checkers == []
    assert set(pl.refused) == {"devsim", "charon", "rustbca"}
    for n in ("devsim", "charon", "rustbca"):
        assert len(pl.refused[n]) == 1 and pl.refused[n][0].startswith("problem kind:")
    assert pl.warnings == {}
    assert "g4see" in pl.summary()


def test_plan_see_outside_tested_warns():
    from simws.engines.planner import plan
    pl = plan(_slab("neutron", 14.0, hadronic="ftfp_bert_hp"))
    assert pl.primary == "g4see"
    assert list(pl.warnings) == ["g4see"]
    assert _prefixes(pl.warnings["g4see"]) == ["energy_MeV", "hadronic", "particle"]


def test_plan_see_refused_by_capability():
    from simws.engines.planner import plan
    pl = plan(_with_beol(_slab(), 21))
    assert pl.primary is None
    assert any(s.startswith("max_beol_layers:") for s in pl.refused["g4see"])


def test_plan_other_families_refuse_g4see(charon_problem):
    from simws.engines.planner import plan
    for p in (pn_diode_1d(V=[0.1]), npn_hbt_2d(), charon_problem()):
        pl = plan(p)
        assert pl.primary == "devsim"
        assert "g4see" in pl.refused and "g4see" not in pl.checkers
        assert len(pl.refused["g4see"]) == 1 and pl.refused["g4see"][0].startswith("problem kind:")
    pl = plan(implant_si("P"))
    assert pl.primary == "rustbca" and "g4see" in pl.refused
    assert len(pl.refused["g4see"]) == 1 and pl.refused["g4see"][0].startswith("problem kind:")


# ------------------------------------------------------------------------------------------------ capability sheet
def test_g4see_capability_exact():
    from simws import engines
    c = engines.capability("g4see")
    assert c.engine == "g4see" and c.env == "g4see" and c.in_process is False
    assert c.particles == ["proton", "alpha", "neutron", "e-", "gamma", "ion"]
    assert c.em == ["option4", "option3", "livermore"]
    assert c.hadronic == ["none", "ftfp_bert_hp"]
    assert c.max_beol_layers == 20
    assert c.tested_particles == ["alpha", "proton"]
    assert tuple(c.tested_energy_MeV) == (5.0, 10.0)
    assert c.tested_materials == ["G4_Si"]
    assert c.tested_em == ["option4"]
    assert c.tested_hadronic == ["none"]


def test_g4see_refuses_only_too_many_layers():
    from simws import engines
    assert engines.unsupported("g4see", _with_beol(_slab(), 20)) == []
    r = engines.unsupported("g4see", _with_beol(_slab(), 21))
    assert len(r) == 1 and r[0].startswith("max_beol_layers:"), r
    for p in (_slab("gamma", 1.0), _slab("e-", 2.0), _slab("ion", 100.0, ion_Z=8, ion_A=16),
              _slab().with_changes({"physics.em": "livermore"}), _slab(hadronic="ftfp_bert_hp")):
        assert engines.unsupported("g4see", p) == []


def test_g4see_acceptance_cases_inside_tested_range():
    """R22 to R25 lie inside the tested range (A8.4: 'the cases of R22 to R25'); R28 does not."""
    from simws import engines
    c = engines.capability("g4see")
    for p in (_slab("alpha", 5.0), _slab("proton", 10.0, sv_thickness_um=100.0, bulk_thickness_um=300.0),
              _slab("alpha", 5.0, beol=[("l1", "G4_Si", 1.0), ("l2", "G4_Si", 2.0)], sv_depth_um=2.0,
                    sv_thickness_um=5.0),
              _slab("alpha", 5.0, tilt_deg=60.0, sv_thickness_um=5.0)):
        assert c.outside_tested_range(p) == []
    r28 = _slab("neutron", 14.0, sv_thickness_um=100.0, bulk_thickness_um=300.0, hadronic="ftfp_bert_hp",
                n_events=2000, high_MeV=20.0)
    assert _prefixes(c.outside_tested_range(r28)) == ["energy_MeV", "hadronic", "particle"]


# ------------------------------------------------------------------------------------------------ build_macro
def _commands(macro):
    """The macro's command lines (comments and blank lines dropped), each split into tokens."""
    out = []
    for line in macro.splitlines():
        s = line.strip()
        if s.startswith("/"):
            out.append(s.split())
    return out


def _find(cmds, name):
    return [i for i, c in enumerate(cmds) if c[0] == name]


def _one(cmds, name):
    idx = _find(cmds, name)
    assert len(idx) == 1, (name, idx)
    return idx[0], cmds[idx[0]]


def _num(tok, expected, rel=1e-9):
    """A number in the macro equals the expected value. The contract does not fix the printed format; values that
    come straight from the description are tested to 1e-9 relative, derived ones (sin, cos, tan) with rel=1e-5,
    which any print of 6 or more significant digits meets."""
    v = float(tok)
    assert v == pytest.approx(expected, rel=rel, abs=1e-12), (tok, expected)


def _macro(p, monkeypatch=None, tmp_path=None):
    return _g4().build_macro(p)


def test_build_macro_is_pure(monkeypatch, tmp_path):
    """build_macro needs no Geant4: it works with SIMWS_G4SEE_PREFIX pointing to an empty folder and without
    starting any process."""
    import subprocess
    monkeypatch.setenv("SIMWS_G4SEE_PREFIX", str(tmp_path))

    def no_process(*a, **k):
        raise AssertionError("build_macro must not start a process")
    monkeypatch.setattr(subprocess, "run", no_process)
    monkeypatch.setattr(subprocess, "Popen", no_process)
    m = _g4().build_macro(_slab())
    assert isinstance(m, str) and "/run/beamOn" in m


def test_build_macro_default_library_problem():
    p = _slab()
    cmds = _commands(_macro(p))
    i_seed, c = _one(cmds, "/random/setSeeds")
    assert [int(t) for t in c[1:3]] == [3, 4] and len(c) == 3
    i_bulk, c = _one(cmds, "/SEE/geometry/Bulk")
    assert c[1] == "G4_Si" and c[3] == "um" and c[5] == "um" and len(c) == 6
    _num(c[2], 1000.0)
    _num(c[4], 100.0)
    i_sv, c = _one(cmds, "/SEE/geometry/SV")
    assert len(c) == 10 and c[4] == "um" and c[7] == "um" and c[9] == "um"
    _num(c[1], 0.0)
    _num(c[2], 0.0)
    _num(c[3], -0.0)
    _num(c[5], 1000.0)
    _num(c[6], 1000.0)
    _num(c[8], 10.0)
    assert _find(cmds, "/SEE/geometry/BEOL/addLayer") == []
    assert _find(cmds, "/SEE/physics/addPhysics") == []           # option4 + none: G4SEE adds option4 itself
    i_init, _ = _one(cmds, "/run/initialize")
    i_part, c = _one(cmds, "/gps/particle")
    assert c[1:] == ["alpha"]
    assert _find(cmds, "/gps/ion") == []
    i_et, c = _one(cmds, "/gps/ene/type")
    assert c[1:] == ["Mono"]
    i_em, c = _one(cmds, "/gps/ene/mono")
    assert len(c) == 3 and c[2] == "MeV"
    _num(c[1], 5.0)
    i_pt, c = _one(cmds, "/gps/pos/type")
    assert c[1:] == ["Point"]
    i_dir, c = _one(cmds, "/gps/direction")
    assert len(c) == 4
    _num(c[1], 0.0)
    _num(c[2], 0.0)
    _num(c[3], -1.0)
    i_cen, c = _one(cmds, "/gps/pos/centre")
    assert len(c) == 5 and c[4] == "um"
    _num(c[1], 0.0)
    _num(c[2], 0.0)
    _num(c[3], 1.0)                                              # no BEOL: 1 um above the bulk
    i_sc, c = _one(cmds, "/SEE/scoring/standard/Edep")
    assert c[1:] == ["0"]
    i_h, c = _one(cmds, "/SEE/scoring/setHistogram")
    assert len(c) == 6 and c[1] == "lin" and int(c[2]) == 2000 and c[5] == "MeV"
    _num(c[3], 0.0)
    _num(c[4], 5.25)
    i_prec, c = _one(cmds, "/SEE/scoring/setPrecision")
    assert c[1:] == ["8"]                                        # A8.9 (was 10 in A8.4)
    i_beam, c = _one(cmds, "/run/beamOn")
    assert c[1:] == ["10000"]
    # order of sections
    gps = [i_part, i_et, i_em, i_pt, i_dir, i_cen]
    scoring = [i_sc, i_h, i_prec]
    assert i_seed < i_bulk < i_sv < i_init < min(gps)
    assert max(gps) < min(scoring) and max(scoring) < i_beam
    assert i_beam == max(i for i, c in enumerate(cmds) if c[0] in ("/run/beamOn",))


@pytest.mark.parametrize("seed", [0, 1, 2, 17, 123456])
def test_build_macro_seeds(seed):
    c = _one(_commands(_macro(_slab(seed=seed))), "/random/setSeeds")[1]
    assert [int(t) for t in c[1:3]] == [2 * seed + 1, 2 * seed + 2]


def test_build_macro_geometry_values():
    p = _slab("proton", 10.0, sv_thickness_um=100.0, sv_depth_um=3.5, bulk_thickness_um=300.0, width_um=750.0)
    p = p.with_changes({"geometry.sv_width_um": [125.0, 250.0], "geometry.bulk_material": "G4_SILICON_CARBIDE"})
    cmds = _commands(_macro(p))
    c = _one(cmds, "/SEE/geometry/Bulk")[1]
    assert c[1] == "G4_SILICON_CARBIDE"
    _num(c[2], 750.0)
    _num(c[4], 300.0)
    c = _one(cmds, "/SEE/geometry/SV")[1]
    _num(c[1], 0.0)
    _num(c[2], 0.0)
    _num(c[3], -3.5)
    _num(c[5], 125.0)
    _num(c[6], 250.0)
    _num(c[8], 100.0)
    c = _one(cmds, "/gps/ene/mono")[1]
    _num(c[1], 10.0)
    assert _one(cmds, "/gps/particle")[1][1:] == ["proton"]


def test_build_macro_beol_reverse_order_and_source_height():
    beol = [("m1", "G4_Cu", 0.5), ("ild", "G4_SILICON_DIOXIDE", 1.25), ("pass", "G4_Si", 2.0)]
    p = _slab(beol=beol, width_um=640.0)
    cmds = _commands(_macro(p))
    idx = _find(cmds, "/SEE/geometry/BEOL/addLayer")
    assert len(idx) == 3
    rows = [cmds[i] for i in idx]
    for row, (name, mat, t) in zip(rows, list(reversed(beol))):    # reverse: the layer next to the bulk first
        assert len(row) == 7 and row[1] == mat and row[3] == "um" and row[5] == "um" and row[6] == name, row
        _num(row[2], 640.0)                                          # the bulk width
        _num(row[4], t)
    i_sv = _one(cmds, "/SEE/geometry/SV")[0]
    i_init = _one(cmds, "/run/initialize")[0]
    assert i_sv < min(idx) and max(idx) < i_init
    c = _one(cmds, "/gps/pos/centre")[1]
    _num(c[1], 0.0)
    _num(c[2], 0.0)
    _num(c[3], 0.5 + 1.25 + 2.0 + 1.0)                              # total BEOL + 1 um


@pytest.mark.parametrize("tilt", [0.0, 7.0, 30.0, 60.0, 85.0])
def test_build_macro_direction_and_position(tilt):
    p = _slab(tilt_deg=tilt, beol=[("a", "G4_Si", 1.0), ("b", "G4_Si", 2.0)])
    cmds = _commands(_macro(p))
    t = math.radians(tilt)
    d = _one(cmds, "/gps/direction")[1]
    _num(d[1], math.sin(t), rel=1e-5)
    _num(d[2], 0.0)
    _num(d[3], -math.cos(t), rel=1e-5)
    c = _one(cmds, "/gps/pos/centre")[1]
    assert c[4] == "um"
    x0, y0, z0 = float(c[1]), float(c[2]), float(c[3])
    _num(c[1], -math.tan(t), rel=1e-5)
    _num(c[2], 0.0)
    _num(c[3], 4.0)
    # the beam enters the top of the stack (z = 3 um) at x = 0
    s = (z0 - 3.0) / math.cos(t)
    x_entry = x0 + s * math.sin(t)
    assert abs(x_entry) < 1e-4 * max(1.0, abs(x0)), x_entry


@pytest.mark.parametrize("em,modules", [("option4", set()), ("option3", {"G4EmStandardPhysics_option3"}),
                                        ("livermore", {"G4EmLivermorePhysics"})])
@pytest.mark.parametrize("had", ["none", "ftfp_bert_hp"])
def test_build_macro_physics(em, modules, had):
    p = _slab(hadronic=had).with_changes({"physics.em": em})
    cmds = _commands(_macro(p))
    idx = _find(cmds, "/SEE/physics/addPhysics")
    got = [cmds[i][1] for i in idx]
    for i in idx:
        assert len(cmds[i]) == 2, cmds[i]
    expected = set(modules)
    if had == "ftfp_bert_hp":
        expected |= {"G4HadronElasticPhysicsHP", "G4HadronPhysicsFTFP_BERT_HP", "G4IonElasticPhysics",
                     "G4IonPhysics"}
    assert sorted(got) == sorted(expected), got                    # each once, nothing else
    if idx:
        i_init = _one(cmds, "/run/initialize")[0]
        last_geo = max(_find(cmds, "/SEE/geometry/SV") + _find(cmds, "/SEE/geometry/Bulk"))
        assert last_geo < min(idx) and max(idx) < i_init


def test_build_macro_ion_and_particles():
    cmds = _commands(_macro(_slab("ion", 160.0, ion_Z=8, ion_A=16)))
    i_p, c = _one(cmds, "/gps/particle")
    assert c[1:] == ["ion"]
    i_i, c = _one(cmds, "/gps/ion")
    assert [int(t) for t in c[1:3]] == [8, 16]
    assert i_p < i_i
    _num(_one(cmds, "/gps/ene/mono")[1][1], 160.0)
    for particle in ("proton", "alpha", "neutron", "e-", "gamma"):
        cmds = _commands(_macro(_slab(particle, 2.0)))
        assert _one(cmds, "/gps/particle")[1][1:] == [particle]
        assert _find(cmds, "/gps/ion") == []


def test_build_macro_histogram_and_events():
    p = _slab(n_events=1234, bins=500).with_changes({"histogram.scale": "log", "histogram.low_MeV": 0.001,
                                                     "histogram.high_MeV": 12.5})
    cmds = _commands(_macro(p))
    c = _one(cmds, "/SEE/scoring/setHistogram")[1]
    assert c[1] == "log" and int(c[2]) == 500 and c[5] == "MeV"
    _num(c[3], 0.001)
    _num(c[4], 12.5)
    assert _one(cmds, "/SEE/scoring/setPrecision")[1][1:] == ["8"]               # A8.9
    assert _one(cmds, "/SEE/scoring/standard/Edep")[1][1:] == ["0"]
    assert _one(cmds, "/run/beamOn")[1][1:] == ["1234"]


def test_build_macro_deterministic():
    p = _slab(beol=[("a", "G4_Cu", 1.0)], tilt_deg=30.0)
    assert _macro(p) == _macro(p.model_copy(deep=True))


# ------------------------------------------------------------------------------------------------ edges
@pytest.mark.parametrize("scale,bins,low,high", [("lin", 2000, 0.0, 5.25), ("lin", 10, 0.5, 3.0),
                                                 ("lin", 777, 0.0, 10.5), ("log", 100, 0.001, 10.0),
                                                 ("log", 2000, 1e-4, 20.0), ("log", 13, 0.3, 0.9)])
def test_edges_formulas(scale, bins, low, high):
    p = _hist_problem(scale, bins, low, high)
    e = np.asarray(_g4().edges(p), float)
    ref = ref_edges(scale, bins, low, high)
    assert e.shape == (bins + 1,)
    assert np.allclose(e, ref, rtol=1e-12, atol=1e-15)
    assert np.all(np.diff(e) > 0)
    assert e[0] == pytest.approx(low, rel=1e-12, abs=1e-15) and e[-1] == pytest.approx(high, rel=1e-12)


def test_edges_library_default():
    e = np.asarray(_g4().edges(_slab()), float)
    assert len(e) == 2001 and np.allclose(np.diff(e), 5.25 / 2000, rtol=1e-9)


# ------------------------------------------------------------------------------------------------ parse_histogram
@pytest.mark.parametrize("scale,bins,low,high", [("lin", 20, 0.0, 5.25), ("lin", 37, 0.25, 3.0),
                                                 ("log", 25, 0.001, 10.0), ("lin", 2000, 0.0, 5.25)])
def test_parse_histogram_g4see_format(tmp_path, scale, bins, low, high):
    p = _hist_problem(scale, bins, low, high)
    rng = np.random.default_rng(bins)
    counts = rng.integers(0, 50000, bins).astype(float)
    counts[::3] = 0.0                                              # zeros are printed as "0"
    counts[1] = 123456.0                                          # above 9999 (the reason for setPrecision)
    under, over = (0.0, 3.0) if low == 0.0 else (17.0, 2.0)
    path = write_g4see_histogram(str(tmp_path / "Edep_0_hist_t0.out"), ref_edges(scale, bins, low, high),
                                 counts, under, over)
    c, u, o = _parse(path, p)
    assert np.asarray(c, float).shape == (bins,)
    assert np.array_equal(np.asarray(c, float), counts)
    assert float(u) == under and float(o) == over


def test_parse_histogram_trial_shape(tmp_path):
    """A file in the shape G4SEE writes when every count is zero (header, underflow, overflow, bins, upper edge)."""
    p = _hist_problem("lin", 10, 0.0, 1.0)
    path = write_g4see_histogram(str(tmp_path / "Edep_0_hist_t0.out"), ref_edges("lin", 10, 0.0, 1.0),
                                 np.zeros(10), 0.0, 0.0)
    text = open(path).read().splitlines()
    assert text[0].split() == ["Edep", "Counts"] and text[1].split() == ["underflow", "0"]
    assert len(text) == 1 + 2 + 10 + 1
    c, u, o = _parse(path, p)
    assert list(np.asarray(c, float)) == [0.0] * 10 and float(u) == 0.0 and float(o) == 0.0


def test_parse_histogram_fractional_counts(tmp_path):
    """Counts are floats in SeeResult; G4SEE prints them in scientific notation."""
    p = _hist_problem("lin", 10, 0.0, 1.0)
    counts = np.array([0.0, 1.5, 2.25, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 99999.0])
    path = write_g4see_histogram(str(tmp_path / "Edep_0_hist_t0.out"), ref_edges("lin", 10, 0.0, 1.0),
                                 counts, 0.5, 0.0)
    c, u, o = _parse(path, p)
    assert np.allclose(np.asarray(c, float), counts, rtol=1e-12, atol=0) and float(u) == 0.5


@pytest.mark.parametrize("delta", [-1, 1])
def test_parse_histogram_wrong_bin_count(tmp_path, delta):
    p = _hist_problem("lin", 20, 0.0, 5.25)
    n = 20 + delta
    e = ref_edges("lin", n, 0.0, 5.25)
    path = write_g4see_histogram(str(tmp_path / "Edep_0_hist_t0.out"), e, np.ones(n), 0.0, 0.0)
    with pytest.raises(ValueError):
        _parse(path, p)


def test_parse_histogram_bin_count_same_range_other_problem(tmp_path):
    """A file written for 2000 bins does not parse for a problem with 1000 bins."""
    e = ref_edges("lin", 2000, 0.0, 5.25)
    path = write_g4see_histogram(str(tmp_path / "Edep_0_hist_t0.out"), e, np.ones(2000), 0.0, 0.0)
    with pytest.raises(ValueError):
        _parse(path, _slab(bins=1000))
    c, _, _ = _parse(path, _slab())
    assert len(c) == 2000


@pytest.mark.parametrize("scale,low", [("lin", 0.0), ("log", 0.001)])
@pytest.mark.parametrize("k", [1, 7, 20])
def test_parse_histogram_edge_tolerance(tmp_path, scale, low, k):
    """An edge off by 1e-5 relative raises; off by 2e-7 relative it parses (tolerance 1e-6 relative). k = 20 is the
    upper edge (the last line), itself a printed bin edge."""
    p = _hist_problem(scale, 20, low, 5.25)
    e = ref_edges(scale, 20, low, 5.25)
    bad = e.copy()
    bad[k] *= 1 + 1e-5
    path = write_g4see_histogram(str(tmp_path / "Edep_0_hist_t0.out"), bad, np.ones(20), 0.0, 0.0)
    with pytest.raises(ValueError):
        _parse(path, p)
    ok = e.copy()
    ok[k] *= 1 + 2e-7
    path = write_g4see_histogram(str(tmp_path / "Edep_0_hist_t0.out"), ok, np.ones(20), 0.0, 0.0)
    c, _, _ = _parse(path, p)
    assert np.array_equal(np.asarray(c, float), np.ones(20))


def test_parse_histogram_zero_edge_absolute(tmp_path):
    """An edge of 0 is compared with 1e-12 MeV absolute."""
    p = _hist_problem("lin", 20, 0.0, 5.25)
    e = ref_edges("lin", 20, 0.0, 5.25)
    bad = e.copy()
    bad[0] = 1e-9
    path = write_g4see_histogram(str(tmp_path / "Edep_0_hist_t0.out"), bad, np.ones(20), 0.0, 0.0)
    with pytest.raises(ValueError):
        _parse(path, p)
    ok = e.copy()
    ok[0] = 1e-13
    path = write_g4see_histogram(str(tmp_path / "Edep_0_hist_t0.out"), ok, np.ones(20), 0.0, 0.0)
    _parse(path, p)


def test_parse_histogram_lin_file_for_log_problem(tmp_path):
    lin = ref_edges("lin", 20, 0.001, 5.25)
    path = write_g4see_histogram(str(tmp_path / "Edep_0_hist_t0.out"), lin, np.ones(20), 0.0, 0.0)
    with pytest.raises(ValueError):
        _parse(path, _hist_problem("log", 20, 0.001, 5.25))


# ------------------------------------------------------------------------------------------------ availability
def _fake_prefix(root, exe_body=None, mode=0o755, datasets=True):
    """A stand-in conda prefix: bin/g4see (a shell script) and bin/geant4-config (prints one dataset line)."""
    b = root / "bin"
    b.mkdir(parents=True, exist_ok=True)
    data = root / "share" / "Geant4" / "data" / "G4FAKEDATA1.0"
    data.mkdir(parents=True, exist_ok=True)
    cfg = b / "geant4-config"
    line = f'echo "G4FAKEDATA G4FAKEDATA_DIR {data}"' if datasets else ":"
    cfg.write_text(f"#!/bin/sh\nif [ \"$1\" = \"--datasets\" ]; then {line}; fi\nexit 0\n")
    cfg.chmod(0o755)
    if exe_body is not None:
        exe = b / "g4see"
        exe.write_text(exe_body)
        exe.chmod(mode)
    return root, data


def test_not_available_with_empty_prefix(monkeypatch, tmp_path):
    from simws.engines.base import EngineNotAvailable
    monkeypatch.setenv("SIMWS_G4SEE_PREFIX", str(tmp_path))
    mod = _g4()
    assert mod.available() is False
    with pytest.raises(EngineNotAvailable):
        mod.run(_slab(n_events=100), tmp_path / "work")


def test_not_available_when_not_executable(monkeypatch, tmp_path):
    root, _ = _fake_prefix(tmp_path / "pre", exe_body="#!/bin/sh\nexit 0\n", mode=0o644)
    monkeypatch.setenv("SIMWS_G4SEE_PREFIX", str(root))
    assert _g4().available() is False


def test_available_with_executable(monkeypatch, tmp_path):
    root, _ = _fake_prefix(tmp_path / "pre", exe_body="#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("SIMWS_G4SEE_PREFIX", str(root))
    assert _g4().available() is True


def test_run_refuses_unsupported(monkeypatch, tmp_path):
    """run raises Unsupported (with the reasons) for a problem the capability refuses; a stand-in executable is
    installed so that availability is not the reason, and it must not be started."""
    from simws.engines.base import Unsupported
    marker = tmp_path / "started"
    root, _ = _fake_prefix(tmp_path / "pre", exe_body=f"#!/bin/sh\ntouch {marker}\nexit 0\n")
    monkeypatch.setenv("SIMWS_G4SEE_PREFIX", str(root))
    with pytest.raises(Unsupported) as ei:
        _g4().run(_with_beol(_slab(n_events=100), 21), tmp_path / "work")
    assert any(s.startswith("max_beol_layers:") for s in ei.value.reasons), ei.value.reasons
    assert not marker.exists()


# ------------------------------------------------------------------------------------------------ run plumbing (stand-in)
def _recording_exe(hist_src, rec):
    """A stand-in g4see: records its arguments, folder and the dataset variable, checks that out/ exists, then copies
    a prepared histogram to <-o folder>/Edep_0_hist_t0.out."""
    return ("#!/bin/sh\n"
            f'echo "$@" > {rec}/args.txt\n'
            f"pwd > {rec}/cwd.txt\n"
            f'echo "$G4FAKEDATA_DIR" > {rec}/env.txt\n'
            f"cp run.mac {rec}/seen.mac 2>/dev/null\n"
            'echo "fake g4see running"\n'
            '[ -d "$3" ] || { echo "no output folder $3"; exit 3; }\n'
            f'cp {hist_src} "$3/Edep_0_hist_t0.out"\n'
            "exit 0\n")


@pytest.mark.parametrize("had", ["none", "ftfp_bert_hp"])
def test_run_with_stand_in_executable(monkeypatch, tmp_path, had):
    from simws.engines.base import SeeResult
    p = _slab(n_events=500, bins=20, hadronic=had)
    e = ref_edges("lin", 20, 0.0, 5.25)
    counts = np.zeros(20)
    counts[6] = 300.0
    counts[7] = 199.0
    hist = write_g4see_histogram(str(tmp_path / "prepared.out"), e, counts, 0.0, 1.0)
    rec = tmp_path / "rec"
    rec.mkdir()
    root, data = _fake_prefix(tmp_path / "pre", exe_body=_recording_exe(hist, rec))
    monkeypatch.setenv("SIMWS_G4SEE_PREFIX", str(root))
    work = tmp_path / "work"
    work.mkdir()
    r = _g4().run(p, work)
    assert isinstance(r, SeeResult)
    assert r.engine == "g4see" and r.problem_id == p.problem_id() and r.n_events == 500
    assert np.allclose(r.edges_MeV, e, rtol=1e-12, atol=1e-15) and len(r.edges_MeV) == 21
    assert np.array_equal(np.asarray(r.counts, float), counts)
    assert r.underflow == 0.0 and r.overflow == 1.0
    folder = work / f"g4see_{p.problem_id()[:12]}"
    assert (folder / "run.mac").read_text() == _g4().build_macro(p)
    assert (folder / "out").is_dir()
    assert (folder / "g4see.log").exists() and "fake g4see running" in (folder / "g4see.log").read_text()
    assert (rec / "args.txt").read_text().split() == ["run.mac", "-o", "out"]
    assert os.path.realpath((rec / "cwd.txt").read_text().strip()) == os.path.realpath(folder)
    assert (rec / "env.txt").read_text().strip() == str(data)      # Geant4 data variables from dataset_env
    em_only = [f for f in r.flags if f.startswith("see:em_only")]
    if had == "none":
        assert len(em_only) == 1, r.flags
    else:
        assert em_only == [], r.flags


def test_run_failure_raises_with_log_tail(monkeypatch, tmp_path):
    body = "#!/bin/sh\n" + "".join(f'echo "L{i:03d} line"\n' for i in range(1, 31)) + "exit 2\n"
    root, _ = _fake_prefix(tmp_path / "pre", exe_body=body)
    monkeypatch.setenv("SIMWS_G4SEE_PREFIX", str(root))
    work = tmp_path / "work"
    work.mkdir()
    p = _slab(n_events=100)
    with pytest.raises(RuntimeError) as ei:
        _g4().run(p, work)
    msg = str(ei.value)
    for i in range(11, 31):
        assert f"L{i:03d}" in msg, (i, msg)
    for i in (1, 5, 9):
        assert f"L{i:03d}" not in msg, (i, msg)
    log = (work / f"g4see_{p.problem_id()[:12]}" / "g4see.log").read_text()
    assert "L001" in log and "L030" in log


# ------------------------------------------------------------------------------------------------ real engine
def _run(p, workdir):
    from simws.engines.base import SeeResult
    r = _g4().run(p, workdir)
    assert isinstance(r, SeeResult)
    N = p.statistics.n_events
    # R21: underflow + overflow + sum of counts = N
    assert r.underflow + r.overflow + float(np.sum(r.counts)) == pytest.approx(N, rel=0, abs=1e-6 * N)
    assert r.n_events == N and len(r.counts) == p.histogram.bins and len(r.edges_MeV) == p.histogram.bins + 1
    assert np.allclose(r.edges_MeV, ref_edges(p.histogram.scale, p.histogram.bins, p.histogram.low_MeV,
                                              p.histogram.high_MeV), rtol=1e-12, atol=1e-15)
    return r


def test_engine_run_basic(tmp_path):
    _need_engine()
    p = _slab(n_events=1000)
    r = _run(p, tmp_path)
    assert r.engine == "g4see" and r.problem_id == p.problem_id()
    assert any(f.startswith("see:em_only") for f in r.flags), r.flags
    assert (tmp_path / f"g4see_{p.problem_id()[:12]}" / "run.mac").exists()
    assert (tmp_path / f"g4see_{p.problem_id()[:12]}" / "g4see.log").exists()
    c = np.asarray(r.counts)
    m = 0.5 * (np.asarray(r.edges_MeV[1:]) + np.asarray(r.edges_MeV[:-1]))
    print(f"\n[engine alpha 5 MeV, SV 10 um, n=1000] under={r.underflow} over={r.overflow} "
          f"mean={np.sum(c * m) / 1000:.5f} MeV")


def test_R20_repeatability(tmp_path):
    _need_engine()
    p = _slab(n_events=1000)
    a = _run(p, tmp_path / "a")
    b = _run(p, tmp_path / "b")
    assert list(a.counts) == list(b.counts)
    assert (a.underflow, a.overflow) == (b.underflow, b.overflow)
    c = _run(_slab(n_events=1000, seed=2), tmp_path / "c")
    assert list(c.counts) != list(a.counts)
    print(f"\n[R20] seed 1 twice identical; seed 2 differs in {int(np.sum(np.asarray(a.counts) != np.asarray(c.counts)))}"
          " bins")


def test_engine_log_histogram(tmp_path):
    _need_engine()
    p = _slab(n_events=500, bins=200).with_changes({"histogram.scale": "log", "histogram.low_MeV": 0.01})
    r = _run(p, tmp_path)
    assert float(np.sum(r.counts)) > 0
