"""Contract amendment A7.3 / A7.4: ImplantResult, ImplantCapability, the registry with problem families, the
planner for both families, the RustBCA translation (build_input, capability, flags, availability) and R10 / R11 on
the engine. Written from docs/ARCHITECTURE_V1.md (A7) only. Engine runs need libRustBCA (skipped without it); never
with recoils on (memory, A7.0c)."""
import math
import sys

import numpy as np
import pydantic
import pytest

from simws.spec.library import npn_hbt_2d, pn_diode_1d


def _implant(*a, **k):
    from simws.spec.library import implant_si
    return implant_si(*a, **k)


def _icap(**over):
    """A hand-made implant capability allowing everything; `over` restricts single fields."""
    from simws.engines.base import ImplantCapability
    d = dict(engine="toy", env="toyenv", description="test sheet", in_process=True, structures=["amorphous"],
             potentials=["zbl", "kr_c", "moliere"], electronic_stopping=["lindhard_scharff"], recoils=[False, True],
             max_layers=20, max_keV_per_amu=25.0, tested_keV_per_amu=(1.6, 9.1), tested_target_symbols=["Si"])
    d.update(over)
    return ImplantCapability(**d)


def _two_layer(p):
    from simws.spec import ImplantProblem
    d = p.model_dump(mode="json")
    lay = d["target"]["layers"][0]
    d["target"]["layers"] = [dict(lay, name="top", thickness_um=0.03), dict(lay, name="bulk", thickness_um=1.97)]
    return ImplantProblem.model_validate(d)


# ------------------------------------------------------------------------------------------------ ImplantResult
def test_implant_result_model():
    import json

    from simws.engines.base import ImplantResult
    r = ImplantResult(engine="toy", problem_id="0" * 32, n_ions=3, n_reflected=1, n_transmitted=0,
                      depth_um=[0.1, 0.2], lateral_um=[0.0, -0.01])
    assert r.flags == [] and r.diagnostics == {}
    json.loads(r.model_dump_json())
    with pytest.raises(pydantic.ValidationError):
        ImplantResult(engine="toy", problem_id="x", n_ions=1, n_reflected=0, n_transmitted=0, depth_um=[0.1],
                      lateral_um=[0.0], colour="red")


# ------------------------------------------------------------------------------------------------ ImplantCapability
def test_icap_refuses_unknown_fields_and_defaults():
    with pytest.raises(pydantic.ValidationError):
        _icap(gpus=4)
    assert _icap().notes == ""


def test_icap_supported_is_empty():
    c = _icap()
    for p in (_implant("P"), _implant("B", 50.0), _implant("As", 100.0), _implant("P", 50, tilt_deg=60.0)):
        assert c.unsupported(p) == []


def _with_layers(p, n):
    from simws.spec import ImplantProblem
    d = p.model_dump(mode="json")
    lay = d["target"]["layers"][0]
    d["target"]["layers"] = [dict(lay, name=f"l{i}", thickness_um=0.1) for i in range(n)]
    return ImplantProblem.model_validate(d)


@pytest.mark.parametrize("field,over,change", [
    ("structures", {"structures": []}, {}),
    ("potentials", {"potentials": ["zbl"]}, {"models.potential": "kr_c"}),
    ("electronic_stopping", {"electronic_stopping": []}, {}),
    ("recoils", {"recoils": [False]}, {"models.recoils": True}),
    ("max_keV_per_amu", {"max_keV_per_amu": 1.0}, {}),            # P at 100 keV: 3.2 keV/amu
])
def test_icap_one_reason_per_field(field, over, change):
    p = _implant("P").with_changes(change)
    r = _icap(**over).unsupported(p)
    assert len(r) == 1, r
    assert r[0].startswith(field + ":"), r


def test_icap_max_layers():
    p = _with_layers(_implant("P"), 3)
    r = _icap(max_layers=2).unsupported(p)
    assert len(r) == 1 and r[0].startswith("max_layers:"), r
    assert _icap(max_layers=3).unsupported(p) == []


def test_icap_max_keV_per_amu_boundary():
    """energy / ion mass above the limit is refused; at the limit it is not ("above")."""
    from simws.spec import Atom
    p = _implant("P").model_copy()
    ion = Atom(symbol="X", Z=10, mass_amu=10.0, cutoff_eV=1.5)
    p = p.with_changes({"beam.ion": ion.model_dump(mode="json"), "beam.energy_keV": 250.0})     # 25.0 exactly
    assert _icap(max_keV_per_amu=25.0).unsupported(p) == []
    r = _icap(max_keV_per_amu=25.0).unsupported(p.with_changes({"beam.energy_keV": 251.0}))
    assert len(r) == 1 and r[0].startswith("max_keV_per_amu:"), r


def test_icap_reasons_add_up():
    p = _with_layers(_implant("P"), 3).with_changes({"models.potential": "moliere", "models.recoils": True})
    c = _icap(structures=[], potentials=["zbl"], electronic_stopping=[], recoils=[False], max_layers=1,
              max_keV_per_amu=1.0)
    fields = sorted(s.split(":", 1)[0] for s in c.unsupported(p))
    assert fields == sorted(["structures", "potentials", "electronic_stopping", "recoils", "max_layers",
                             "max_keV_per_amu"])


def test_icap_outside_tested_range():
    from simws.spec import Atom
    c = _icap(tested_keV_per_amu=(1.6, 9.1), tested_target_symbols=["Si"])
    ion = Atom(symbol="X", Z=10, mass_amu=10.0, cutoff_eV=1.5).model_dump(mode="json")
    base = _implant("P").with_changes({"beam.ion": ion})
    for e in (16.0, 50.0, 91.0):                                     # 1.6 and 9.1 keV/amu: bounds included
        assert c.outside_tested_range(base.with_changes({"beam.energy_keV": e})) == [], e
    for e in (15.9, 91.1, 200.0):
        r = c.outside_tested_range(base.with_changes({"beam.energy_keV": e}))
        assert len(r) == 1 and r[0].startswith("energy_keV_per_amu:"), (e, r)
    # library cases (A7.7): As at 50 keV = 0.67 keV/amu is outside; B, P at 50 and 100 keV inside
    assert c.outside_tested_range(_implant("As", 50.0))[0].startswith("energy_keV_per_amu:")
    for ion_, e in (("B", 50.0), ("B", 100.0), ("P", 50.0), ("P", 100.0)):
        assert c.outside_tested_range(_implant(ion_, e)) == [], (ion_, e)


def test_icap_outside_tested_target():
    from simws.spec import ImplantProblem
    c = _icap()
    d = _implant("P").model_dump(mode="json")
    d["target"]["atoms"].append({"symbol": "Ge", "Z": 32, "mass_amu": 72.63, "cutoff_eV": 1.5})
    d["target"]["layers"] = [{"name": "sige", "thickness_um": 2.0, "density_cm3": {"Si": 4e22, "Ge": 1e22}}]
    r = c.outside_tested_range(ImplantProblem.model_validate(d))
    assert len(r) == 1 and r[0].startswith("target:"), r
    d["beam"]["energy_keV"] = 20.0                                    # 0.65 keV/amu: both strings
    r = c.outside_tested_range(ImplantProblem.model_validate(d))
    assert sorted(s.split(":", 1)[0] for s in r) == ["energy_keV_per_amu", "target"], r


# ------------------------------------------------------------------------------------------------ registry
def test_registry_has_rustbca():
    from simws import engines
    assert engines.ENGINES["rustbca"] == "simws.engines.rustbca_engine"
    assert engines.ENGINES["devsim"] == "simws.engines.devsim_engine"
    assert engines.ENGINES["charon"] == "simws.engines.charon_engine"
    assert engines.names() == sorted(engines.ENGINES) and "rustbca" in engines.names()


def test_family_and_adapter_name():
    from simws import engines
    assert engines.family("rustbca") == "implant"
    assert engines.family("devsim") == "device" and engines.family("charon") == "device"
    assert engines.adapter_name("devsim") == "device.devsim"
    assert engines.adapter_name("charon") == "device.charon"
    assert engines.adapter_name("rustbca") == "implant.rustbca"
    assert engines.module("rustbca").FAMILY == "implant"
    # FAMILY "device when absent"
    for n in ("devsim", "charon"):
        assert getattr(engines.module(n), "FAMILY", "device") == "device"


def test_rustbca_module_interface():
    from simws import engines
    from simws.engines.base import ImplantCapability
    mod = engines.module("rustbca")
    for attr in ("CAPABILITY", "unsupported", "available", "run", "build_input"):
        assert hasattr(mod, attr), attr
    assert isinstance(engines.capability("rustbca"), ImplantCapability)
    assert isinstance(mod.available(), bool)
    for p in (_implant("P"), _implant("B", 100.0), _implant("As", 50.0)):
        assert engines.unsupported("rustbca", p) == mod.unsupported(p) == []


def test_problem_kind_refusal_both_directions(monkeypatch):
    """engines.unsupported returns exactly one 'problem kind:' reason across families, without calling the engine
    module's own check (the module checks are replaced by functions that fail if called)."""
    from simws import engines

    def boom(problem):
        raise AssertionError("the engine module's own check must not be called across families")

    for n in ("devsim", "charon", "rustbca"):
        monkeypatch.setattr(engines.module(n), "unsupported", boom)
    imp = _implant("P")
    for n in ("devsim", "charon"):
        r = engines.unsupported(n, imp)
        assert len(r) == 1 and r[0].startswith("problem kind:"), (n, r)
    for dev in (pn_diode_1d(), npn_hbt_2d()):
        r = engines.unsupported("rustbca", dev)
        assert len(r) == 1 and r[0].startswith("problem kind:"), r


def test_same_family_still_uses_module_check():
    from simws import engines
    imp = _implant("P").with_changes({"models.recoils": True})
    r = engines.unsupported("rustbca", imp)
    assert any(s.startswith("recoils:") for s in r), r
    assert engines.unsupported("devsim", pn_diode_1d()) == []


# ------------------------------------------------------------------------------------------------ planner
def test_plan_implant_problem():
    from simws.engines.planner import plan
    p = _implant("P")
    pl = plan(p)
    assert pl.problem_id == p.problem_id()
    assert pl.primary == "rustbca" and pl.checkers == []
    assert set(pl.refused) == {"devsim", "charon", "g4see", "ngspice", "devsim_eb"}   # amended by the lead (A8.3, A9.3, A11.3)
    for n in ("devsim", "charon", "g4see", "ngspice", "devsim_eb"):
        assert len(pl.refused[n]) == 1 and pl.refused[n][0].startswith("problem kind:")
    assert pl.warnings == {}
    assert "rustbca" in pl.summary()


def test_plan_implant_outside_tested_warns():
    from simws.engines.planner import plan
    pl = plan(_implant("As", 50.0))
    assert pl.primary == "rustbca"
    assert list(pl.warnings) == ["rustbca"]
    assert pl.warnings["rustbca"][0].startswith("energy_keV_per_amu:")


def test_plan_device_problem_refuses_rustbca(charon_problem):
    from simws.engines.planner import plan
    for p in (pn_diode_1d(V=[0.1]), npn_hbt_2d(), charon_problem()):
        pl = plan(p)
        assert pl.primary == "devsim"
        assert "rustbca" in pl.refused and "rustbca" not in pl.checkers
        assert len(pl.refused["rustbca"]) == 1 and pl.refused["rustbca"][0].startswith("problem kind:")
    assert plan(charon_problem()).checkers == ["charon"]


def test_plan_implant_refused_by_capability():
    from simws.engines.planner import plan
    pl = plan(_implant("B", 300.0))                          # 27.2 keV/amu > 25
    assert pl.primary is None
    assert any(s.startswith("max_keV_per_amu:") for s in pl.refused["rustbca"])


# ------------------------------------------------------------------------------------------------ capability sheet
def test_rustbca_capability_exact():
    from simws import engines
    c = engines.capability("rustbca")
    assert c.engine == "rustbca" and c.in_process is True
    assert c.structures == ["amorphous"]
    assert c.potentials == ["zbl", "kr_c", "moliere"]
    assert c.electronic_stopping == ["lindhard_scharff"]
    assert c.recoils == [False]
    assert c.max_layers == 20
    assert c.max_keV_per_amu == 25.0
    # amended by the lead (contract A7.9): only what passed R16 counts as tested, after the boron FAIL
    assert tuple(c.tested_keV_per_amu) == (1.6, 3.3)
    assert c.tested_target_symbols == ["Si"]
    assert c.tested_ion_symbols == ["P"] and c.env == "tcad"


def test_A79_tested_ions():
    """Added by the lead with amendment A7.9."""
    from simws import engines
    assert _icap().tested_ion_symbols == []                          # default: any ion
    assert _icap().outside_tested_range(_implant("B", 50.0)) == []
    r = _icap(tested_ion_symbols=["P"]).outside_tested_range(_implant("B", 50.0))
    assert len(r) == 1 and r[0].startswith("ion:"), r
    assert _icap(tested_ion_symbols=["P"]).outside_tested_range(_implant("P", 50.0)) == []
    c = engines.capability("rustbca")
    assert c.outside_tested_range(_implant("P", 50.0)) == [] and c.outside_tested_range(_implant("P", 100.0)) == []
    assert sorted(s.split(":", 1)[0] for s in c.outside_tested_range(_implant("B", 100.0))) == ["energy_keV_per_amu", "ion"]
    assert sorted(s.split(":", 1)[0] for s in c.outside_tested_range(_implant("As", 50.0))) == ["energy_keV_per_amu", "ion"]


def test_rustbca_refuses_recoils_and_fast_beams():
    from simws import engines
    r = engines.unsupported("rustbca", _implant("P").with_changes({"models.recoils": True}))
    assert len(r) == 1 and r[0].startswith("recoils:"), r
    r = engines.unsupported("rustbca", _implant("B", 300.0))
    assert len(r) == 1 and r[0].startswith("max_keV_per_amu:"), r
    assert engines.unsupported("rustbca", _with_layers(_implant("P"), 20)) == []
    r = engines.unsupported("rustbca", _with_layers(_implant("P"), 21))
    assert any(s.startswith("max_layers:") for s in r), r


# ------------------------------------------------------------------------------------------------ build_input
def test_build_input_is_pure(monkeypatch):
    """build_input does not import the engine: it works with libRustBCA made unimportable."""
    from simws import engines
    monkeypatch.setitem(sys.modules, "libRustBCA", None)
    d = engines.module("rustbca").build_input(_implant("P"))
    assert isinstance(d, dict)


@pytest.mark.parametrize("pot,name", [("zbl", "ZBL"), ("kr_c", "KR_C"), ("moliere", "MOLIERE")])
def test_build_input_options(pot, name):
    from simws import engines
    p = _implant("P", 50.0, seed=7).with_changes({"models.potential": pot})
    o = engines.module("rustbca").build_input(p)["options"]
    assert o["track_recoils"] is False
    assert o["electronic_stopping_mode"] == "LOW_ENERGY_NONLOCAL"
    assert o["interaction_potential"] == [[name]]
    assert o["scattering_integral"] == [["MENDENHALL_WELLER"]]
    assert o["mean_free_path_model"] == "LIQUID"
    assert o["seed"] == 7
    assert o["num_chunks"] == 1


@pytest.mark.parametrize("ion,E,tilt", [("P", 100.0, 0.0), ("B", 50.0, 7.0), ("As", 80.0, 60.0)])
def test_build_input_particles(ion, E, tilt):
    from simws import engines
    p = _implant(ion, E, tilt_deg=tilt, n_ions=1234)
    pp = engines.module("rustbca").build_input(p)["particle_parameters"]
    a = p.beam.ion
    assert pp["length_unit"] == "MICRON" and pp["energy_unit"] == "EV" and pp["mass_unit"] == "AMU"
    assert list(pp["N"]) == [1234]
    assert list(pp["Z"]) == [a.Z]
    assert np.allclose(pp["m"], [a.mass_amu], rtol=1e-15) and len(pp["m"]) == 1
    assert np.allclose(pp["Ec"], [a.cutoff_eV], rtol=1e-15) and len(pp["Ec"]) == 1
    assert np.allclose(pp["Es"], [a.surface_binding_eV], rtol=1e-15) and len(pp["Es"]) == 1
    assert np.allclose(pp["E"], [E * 1000.0], rtol=1e-15) and len(pp["E"]) == 1
    assert np.asarray(pp["pos"], float).tolist() == [[0.0, 0.0, 0.0]]
    t = math.radians(tilt)
    dirs = np.asarray(pp["dir"], float)
    assert dirs.shape == (1, 3)
    assert np.allclose(dirs[0], [math.cos(t), math.sin(t), 0.0], rtol=1e-12, atol=1e-15)


def test_build_input_geometry_and_materials():
    """Two species, two layers, one species absent from a layer: densities per layer and per entry of
    target.atoms (in that order), atoms per um^3 = cm^-3 x 1e-12, 0 for an absent species."""
    from simws import engines
    from simws.spec import ImplantProblem
    d = _implant("P").model_dump(mode="json")
    d["target"]["atoms"] = [{"symbol": "O", "Z": 8, "mass_amu": 15.999, "cutoff_eV": 2.0,
                             "surface_binding_eV": 2.6, "bulk_binding_eV": 0.5},
                            {"symbol": "Si", "Z": 14, "mass_amu": 28.085, "cutoff_eV": 1.5,
                             "surface_binding_eV": 4.72}]
    d["target"]["layers"] = [{"name": "oxide", "thickness_um": 0.01, "density_cm3": {"Si": 2.3e22, "O": 4.6e22}},
                             {"name": "si", "thickness_um": 1.5, "density_cm3": {"Si": 4.994e22}}]
    p = ImplantProblem.model_validate(d)
    inp = engines.module("rustbca").build_input(p)
    g = inp["geometry_input"]
    assert g["length_unit"] == "MICRON"
    assert np.allclose(g["layer_thicknesses"], [0.01, 1.5], rtol=1e-15) and len(g["layer_thicknesses"]) == 2
    dens = np.asarray(g["densities"], float)
    expected = np.array([[4.6e22 * 1e-12, 2.3e22 * 1e-12], [0.0, 4.994e22 * 1e-12]])
    assert dens.shape == (2, 2)
    assert np.allclose(dens, expected, rtol=1e-12, atol=0.0)
    esc = np.asarray(g["electronic_stopping_correction_factors"], float)
    assert esc.size >= 1 and np.all(esc == 1.0)
    m = inp["material_parameters"]
    assert m["energy_unit"] == "EV" and m["mass_unit"] == "AMU"
    assert list(m["Z"]) == [8, 14]
    assert np.allclose(m["m"], [15.999, 28.085], rtol=1e-15)
    assert np.allclose(m["Ec"], [2.0, 1.5], rtol=1e-15)
    assert np.allclose(m["Es"], [2.6, 4.72], rtol=1e-15)
    assert np.allclose(m["Eb"], [0.5, 0.0], rtol=1e-15)


def test_build_input_library_silicon():
    from simws import engines
    inp = engines.module("rustbca").build_input(_implant("P", thickness_um=2.0))
    g = inp["geometry_input"]
    assert np.allclose(g["layer_thicknesses"], [2.0]) and np.allclose(g["densities"], [[4.994e10]], rtol=1e-12)
    m = inp["material_parameters"]
    assert list(m["Z"]) == [14] and np.allclose(m["Es"], [4.72]) and np.allclose(m["Eb"], [0.0])


# ------------------------------------------------------------------------------------------------ availability
def test_not_available_without_libRustBCA(monkeypatch, tmp_path):
    from simws import engines
    from simws.engines.base import EngineNotAvailable
    mod = engines.module("rustbca")
    monkeypatch.setitem(sys.modules, "libRustBCA", None)
    assert mod.available() is False
    with pytest.raises(EngineNotAvailable):
        mod.run(_implant("P", 50, n_ions=100), tmp_path)


def test_run_refuses_unsupported(tmp_path):
    """run raises Unsupported (with the reasons) when the capability refuses the problem. A fast beam is used, not
    recoils (a wrongly accepted recoil run could exhaust the memory of this machine)."""
    from simws import engines
    from simws.engines.base import EngineNotAvailable, Unsupported
    try:
        with pytest.raises(Unsupported) as ei:
            engines.module("rustbca").run(_implant("B", 300.0, n_ions=100), tmp_path)
    except EngineNotAvailable:
        pytest.fail("EngineNotAvailable raised for an unsupported problem")
    assert any(s.startswith("max_keV_per_amu:") for s in ei.value.reasons), ei.value.reasons


# ------------------------------------------------------------------------------------------------ engine runs
def _run(p, tmp_path):
    from simws import engines
    from simws.engines.base import ImplantResult
    r = engines.module("rustbca").run(p, tmp_path)
    assert isinstance(r, ImplantResult)
    # R11: in every run implanted + reflected + transmitted = N
    assert len(r.depth_um) + r.n_reflected + r.n_transmitted == r.n_ions == p.statistics.n_ions
    assert len(r.lateral_um) == len(r.depth_um)
    return r


def test_engine_run_basic(tmp_path):
    pytest.importorskip("libRustBCA")
    from simws import engines
    assert engines.module("rustbca").available() is True
    p = _implant("P", 50, n_ions=2000)
    r = _run(p, tmp_path)
    assert r.engine == "rustbca" and r.problem_id == p.problem_id()
    assert any(f.startswith("implant:amorphous_target") for f in r.flags), r.flags
    x = np.asarray(r.depth_um)
    assert len(x) > 1000                                   # most ions stop in a 2 um target at 50 keV
    assert np.all(x >= 0.0) and np.all(x <= p.target.thickness_um)
    lat = np.asarray(r.lateral_um)
    assert np.std(lat) > 0.0
    print(f"\n[engine P 50 keV n=2000] k={len(x)} refl={r.n_reflected} trans={r.n_transmitted} "
          f"mean depth={x.mean():.5f} um")


def test_R10_repeatability(tmp_path, monkeypatch):
    pytest.importorskip("libRustBCA")
    monkeypatch.delenv("SIMWS_THREADS", raising=False)
    p = _implant("P", 50, n_ions=2000)
    a = _run(p, tmp_path / "a")
    b = _run(p, tmp_path / "b")
    assert a.depth_um == b.depth_um
    c = _run(_implant("P", 50, n_ions=2000, seed=2), tmp_path / "c")
    assert c.depth_um != a.depth_um
    monkeypatch.setenv("SIMWS_THREADS", "1")
    d = _run(p, tmp_path / "d")
    assert d.depth_um == a.depth_um


def test_R11_engine_counts(tmp_path):
    pytest.importorskip("libRustBCA")
    thin = _run(_implant("P", 100, thickness_um=0.05, n_ions=2000), tmp_path / "thin")
    assert thin.n_transmitted > 0
    print(f"\n[R11 0.05 um] k={len(thin.depth_um)} refl={thin.n_reflected} trans={thin.n_transmitted}")
    vthin = _run(_implant("P", 100, thickness_um=0.005, n_ions=2000), tmp_path / "vthin")
    assert len(vthin.depth_um) < 100
    print(f"[R11 0.005 um] k={len(vthin.depth_um)} refl={vthin.n_reflected} trans={vthin.n_transmitted}")
    two = _run(_two_layer(_implant("P", 50, n_ions=2000)), tmp_path / "two")
    assert len(two.depth_um) > 0
