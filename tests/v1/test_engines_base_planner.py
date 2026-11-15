"""simws.engines.base, the engine registry and the planner (contract sections 4.1-4.3, capability sheets of 4.4
and 4.5). No engine run: capability is about the description, not about what is installed."""
import json

import pydantic
import pytest

from simws.spec.library import npn_hbt_2d, pn_diode_1d


def _cap(**over):
    """A hand-made capability that allows everything the spec can express; `over` restricts single fields."""
    from simws.engines.base import Capability
    d = dict(engine="toy", env="toyenv", description="test sheet", in_process=True, dimensions=[1, 2], max_regions=4,
             materials=["silicon"], grading_elements=["Ge"], contact_kinds=["ohmic", "schottky"],
             statistics=["boltzmann", "fermi_dirac"], mobility=["constant", "masetti", "klaassen"],
             velocity_saturation=[False, True], srh=["off", "constant", "scharfetter"], auger=[False, True],
             band_gap_narrowing=["off", "slotboom"], incomplete_ionization=["off", "altermatt", "constant_energy"],
             impact_ionization=["off", "van_overstraeten"], analyses=["equilibrium", "dc_sweep"],
             tested_temperature_K=(250.0, 400.0))
    d.update(over)
    return Capability(**d)


# ------------------------------------------------------------------------------------------------ base types
def test_exceptions():
    from simws.engines.base import EngineNotAvailable, Unsupported
    assert issubclass(EngineNotAvailable, RuntimeError)
    assert issubclass(Unsupported, ValueError)
    e = Unsupported("toy", ["a: x", "b: y"])
    assert e.reasons == ["a: x", "b: y"]
    with pytest.raises(ValueError):
        raise e


def test_engine_result_model():
    from simws.engines.base import EngineResult
    r = EngineResult(engine="toy", problem_id="0" * 32, analysis="dc_sweep", sweep_contact="anode",
                     sweep_V=[0.1, 0.2], currents={"anode": [1.0, 2.0], "cathode": [-1.0, -2.0]},
                     current_unit="A/cm**2")
    assert r.flags == [] and r.diagnostics == {}
    json.loads(r.model_dump_json())
    eq = EngineResult(engine="toy", problem_id="0" * 32, analysis="equilibrium", sweep_V=[0.0],
                      currents={"anode": [0.0]}, current_unit="A/cm")
    assert eq.sweep_contact is None
    with pytest.raises(pydantic.ValidationError):
        EngineResult(engine="toy", problem_id="x", analysis="transient", sweep_V=[0.0], currents={},
                     current_unit="A/cm")
    with pytest.raises(pydantic.ValidationError):
        EngineResult(engine="toy", problem_id="x", analysis="equilibrium", sweep_V=[0.0], currents={},
                     current_unit="A")
    with pytest.raises(pydantic.ValidationError):
        EngineResult(engine="toy", problem_id="x", analysis="equilibrium", sweep_V=[0.0], currents={},
                     current_unit="A/cm", colour="red")


def test_capability_refuses_unknown_fields():
    with pytest.raises(pydantic.ValidationError):
        _cap(gpus=4)
    assert _cap().notes == ""
    assert _cap().mesh_study is False                               # contract A4.2: the new field defaults to False


# ------------------------------------------------------------------------------------------------ Capability.unsupported
def test_capability_supported_is_empty():
    c = _cap()
    assert c.unsupported(pn_diode_1d()) == []
    assert c.unsupported(npn_hbt_2d(VBE=[0.6])) == []


def test_capability_reason_strings_from_contract():
    """The example strings of contract 4.1, produced by the matching restrictions."""
    p1 = pn_diode_1d().with_changes({"models.mobility": "klaassen", "models.mu_n_cm2_Vs": None,
                                     "models.mu_p_cm2_Vs": None})
    assert _cap(mobility=["constant"]).unsupported(p1) == ["mobility: 'klaassen' not in ['constant']"]
    assert _cap(dimensions=[1]).unsupported(npn_hbt_2d()) == ["dimensions: 2 not in [1]"]
    assert _cap(grading_elements=[]).unsupported(npn_hbt_2d()) == ["grading_elements: 'Ge' not in []"]
    p2 = pn_diode_1d().with_changes({"device.regions.0.material": "sic_4h"})
    assert _cap(materials=["silicon"]).unsupported(p2) == ["materials: 'sic_4h' not in ['silicon']"]
    p3 = pn_diode_1d().with_changes({"device.contacts.0.kind": "schottky", "device.contacts.0.barrier_eV": 0.7})
    assert _cap(contact_kinds=["ohmic"]).unsupported(p3) == ["contact_kinds: 'schottky' not in ['ohmic']"]
    two = pn_diode_1d().model_dump(mode="json")
    two["device"]["regions"] = [{"name": "si", "material": "silicon", "box": {"x": [0.0, 2.0]}},
                                {"name": "si2", "material": "silicon", "box": {"x": [2.0, 4.0]}}]
    from simws.spec.schema import Problem
    p4 = Problem.model_validate(two)
    assert _cap(max_regions=1).unsupported(p4) == ["max_regions: 2 > 1"]
    r = _cap(analyses=["equilibrium"]).unsupported(pn_diode_1d(V=[0.1]))
    assert len(r) == 1 and r[0].startswith("analyses: 'dc_sweep' not in ")


@pytest.mark.parametrize("field,allowed,change,value", [
    ("statistics", ["fermi_dirac"], {}, "boltzmann"),
    ("velocity_saturation", [False], {"models.velocity_saturation": True}, True),
    ("srh", ["off", "constant"], {"models.srh": "scharfetter", "models.tau_n_s": None, "models.tau_p_s": None},
     "scharfetter"),
    ("auger", [False], {"models.auger": True}, True),
    ("band_gap_narrowing", ["off"], {"models.band_gap_narrowing": "slotboom"}, "slotboom"),
    ("incomplete_ionization", ["constant_energy"], {}, "off"),
    ("impact_ionization", ["off"], {"models.impact_ionization": "van_overstraeten"}, "van_overstraeten"),
    ("analyses", ["dc_sweep"], {}, "equilibrium"),
])
def test_capability_one_reason_per_field_starting_with_field_name(field, allowed, change, value):
    p = pn_diode_1d().with_changes(change)
    r = _cap(**{field: allowed}).unsupported(p)
    assert len(r) == 1, r
    assert r[0].startswith(field + ":"), r
    assert repr(value) in r[0], r


def test_capability_reasons_add_up_one_per_violated_field():
    c = _cap(dimensions=[1], grading_elements=[], mobility=["constant"], auger=[False],
             band_gap_narrowing=["off"], analyses=["dc_sweep"])
    r = c.unsupported(npn_hbt_2d())        # 2D, Ge, masetti, auger on, slotboom, equilibrium: six fields
    fields = sorted(s.split(":", 1)[0] for s in r)
    assert fields == sorted(["dimensions", "grading_elements", "mobility", "auger", "band_gap_narrowing",
                             "analyses"])


def test_capability_temperature_is_not_a_refusal():
    c = _cap()
    assert c.unsupported(pn_diode_1d(T=77.0)) == []


def test_outside_tested_range():
    c = _cap(tested_temperature_K=(250.0, 400.0))
    assert c.outside_tested_range(pn_diode_1d(T=300.0)) == []
    assert c.outside_tested_range(pn_diode_1d(T=250.0)) == []          # bounds included
    assert c.outside_tested_range(pn_diode_1d(T=400.0)) == []
    for T in (77.0, 249.9, 400.1):
        r = c.outside_tested_range(pn_diode_1d(T=T))
        assert len(r) == 1 and r[0].startswith("temperature_K:") and "outside tested" in r[0], r
        assert f"{T:g}" in r[0], r
    assert "250" in c.outside_tested_range(pn_diode_1d(T=77.0))[0]
    assert "400" in c.outside_tested_range(pn_diode_1d(T=77.0))[0]


# ------------------------------------------------------------------------------------------------ registry
def test_registry():
    from simws import engines
    from simws.engines.base import Capability
    assert engines.ENGINES == {"devsim": "simws.engines.devsim_engine", "charon": "simws.engines.charon_engine"}
    assert engines.names() == ["charon", "devsim"]
    for n in engines.names():
        mod = engines.module(n)
        assert isinstance(engines.capability(n), Capability)
        assert engines.capability(n) is mod.CAPABILITY or engines.capability(n) == mod.CAPABILITY
        assert engines.capability(n).engine == n
        for attr in ("CAPABILITY", "unsupported", "available", "run"):
            assert hasattr(mod, attr), (n, attr)
        assert isinstance(mod.available(), bool)
        p = npn_hbt_2d()
        assert engines.unsupported(n, p) == mod.unsupported(p)
    for attr in ("prepare", "collect"):
        assert hasattr(engines.module("charon"), attr)
    with pytest.raises(KeyError) as ei:
        engines.module("comsol")
    assert "devsim" in str(ei.value) and "charon" in str(ei.value)


def test_engine_unsupported_includes_capability_reasons():
    from simws import engines
    for n in engines.names():
        p = npn_hbt_2d()
        cap_r = engines.capability(n).unsupported(p)
        mod_r = engines.unsupported(n, p)
        assert all(s in mod_r for s in cap_r), (n, cap_r, mod_r)


def test_devsim_capability_exact():
    from simws import engines
    c = engines.capability("devsim")
    assert c.engine == "devsim" and c.env == "tcad" and c.in_process is True
    assert c.dimensions == [1, 2] and c.max_regions == 1
    assert c.materials == ["silicon"] and c.grading_elements == ["Ge"] and c.contact_kinds == ["ohmic"]
    assert c.statistics == ["boltzmann", "fermi_dirac"]
    assert c.mobility == ["constant", "masetti", "klaassen"]
    assert c.velocity_saturation == [False, True]
    assert c.srh == ["off", "constant", "scharfetter"]
    assert c.auger == [False, True]
    assert c.band_gap_narrowing == ["off", "slotboom"]
    assert c.incomplete_ionization == ["off", "altermatt", "constant_energy"]
    assert c.impact_ionization == ["off", "van_overstraeten"]
    assert c.analyses == ["equilibrium", "dc_sweep"]
    assert tuple(c.tested_temperature_K) == (250.0, 400.0)
    assert c.mesh_study is True                                     # contract A4.2


def test_charon_capability_exact():
    from simws import engines
    c = engines.capability("charon")
    assert c.engine == "charon" and c.env == "tcad" and c.in_process is False
    assert c.dimensions == [1] and c.max_regions == 1
    assert c.materials == ["silicon"] and c.grading_elements == [] and c.contact_kinds == ["ohmic"]
    assert c.statistics == ["fermi_dirac"] and c.mobility == ["constant"]
    assert c.velocity_saturation == [False] and c.srh == ["off", "constant"] and c.auger == [False]
    assert c.band_gap_narrowing == ["off"] and c.incomplete_ionization == ["constant_energy"]
    assert c.impact_ionization == ["off"] and c.analyses == ["dc_sweep"]
    assert tuple(c.tested_temperature_K) == (77.0, 300.0)
    assert c.mesh_study is False                                    # contract A4.2: Charon's meshes are fixed files


def test_library_devices_supported_by_devsim():
    from simws import engines
    for p in (pn_diode_1d(), pn_diode_1d(V=[0.1, 0.2]), npn_hbt_2d(), npn_hbt_2d(VBE=[0.6]),
              npn_hbt_2d(ge_peak=0.0)):
        assert engines.unsupported("devsim", p) == [], p.device.name


# ------------------------------------------------------------------------------------------------ planner
def test_plan_hbt_devsim_only():
    from simws.engines.planner import Plan, plan
    p = npn_hbt_2d(VBE=[0.5, 0.6])
    pl = plan(p)
    assert isinstance(pl, Plan)
    assert pl.problem_id == p.problem_id()
    assert pl.primary == "devsim" and pl.checkers == []
    assert "devsim" not in pl.refused
    assert pl.refused["charon"], pl.refused
    assert "dimensions: 2 not in [1]" in pl.refused["charon"]
    assert "grading_elements: 'Ge' not in []" in pl.refused["charon"]
    assert pl.warnings == {}
    s = pl.summary()
    assert isinstance(s, str) and "devsim" in s and "charon" in s


def test_plan_77K_warns_not_refuses_devsim():
    from simws.engines.planner import plan
    p = pn_diode_1d(T=77.0, V=[0.5])
    pl = plan(p)
    assert pl.primary == "devsim"
    assert "devsim" not in pl.refused
    assert list(pl.warnings) == ["devsim"]
    assert len(pl.warnings["devsim"]) == 1 and pl.warnings["devsim"][0].startswith("temperature_K:")
    assert "charon" in pl.refused                      # Boltzmann, no incomplete ionization, 4 um long


def test_plan_both_engines_prefer_and_registry_order(charon_problem):
    from simws.engines.planner import plan
    p = charon_problem()
    pl = plan(p)
    assert pl.refused == {}
    assert pl.primary == "devsim" and pl.checkers == ["charon"]
    assert pl.warnings == {}                            # 300 K is inside both tested ranges
    pc = plan(p, prefer=("charon",))
    assert pc.primary == "charon" and pc.checkers == ["devsim"]
    pn = plan(p, prefer=())                             # no preference: registry (sorted) order
    assert pn.primary == "charon" and pn.checkers == ["devsim"]


def test_plan_warnings_only_non_empty(charon_problem):
    from simws.engines.planner import plan
    pl = plan(charon_problem(T=77.0))                   # inside Charon's [77, 300], outside DEVSIM's [250, 400]
    assert pl.refused == {}
    assert list(pl.warnings) == ["devsim"]


def test_plan_engine_subset():
    from simws.engines.planner import plan
    pl = plan(npn_hbt_2d(), engines=["charon"])
    assert pl.primary is None and pl.checkers == []
    assert list(pl.refused) == ["charon"]
    pd = plan(npn_hbt_2d(), engines=["devsim"])
    assert pd.primary == "devsim" and pd.refused == {}


def test_plan_does_not_depend_on_installation(charon_problem):
    """Charon is not installed here, yet a plan names it as a checker (a plan is about capability)."""
    from simws import engines
    from simws.engines.planner import plan
    assert "charon" in plan(charon_problem()).checkers
    assert isinstance(engines.module("charon").available(), bool)
