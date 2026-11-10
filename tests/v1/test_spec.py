"""simws.spec (frozen description): validation rules, interval rule, reference evaluators, problem_id, with_changes,
and the library entries against the legacy constants they claim to mirror. No engine needed."""
import numpy as np
import pydantic
import pytest

from simws.spec.library import npn_hbt_2d, pn_diode_1d
from simws.spec.schema import (SPEC_VERSION, Box, Contact, DCSweep, DeviceSpec, DopingTerm, Equilibrium, GradeTerm,
                               MeshHints, ModelSpec, Numerics, Problem, Region, Track)

VE = pydantic.ValidationError


def _dev1d(**kw):
    d = dict(name="d", dimension=1, regions=[Region(name="si", box=Box(x=(0.0, 1.0)))],
             contacts=[Contact(name="a", region="si", side="x_min"), Contact(name="c", region="si", side="x_max")],
             mesh=MeshHints(x_lines=[(0.0, 0.01), (1.0, 0.01)]))
    d.update(kw)
    return DeviceSpec(**d)


# ------------------------------------------------------------------------------------------------ unknown fields
@pytest.mark.parametrize("build", [
    lambda: Box(x=(0, 1), z=(0, 1)),
    lambda: Region(name="r", box=Box(x=(0, 1)), colour="red"),
    lambda: Contact(name="a", region="si", side="x_min", resistance=1.0),
    lambda: DopingTerm(species="donor", concentration_cm3=1e16, profile="gauss"),
    lambda: GradeTerm(start_um=0, stop_um=1, x_start=0, x_stop=0.1, curvature=1),
    lambda: MeshHints(x_lines=[(0, 1), (1, 1)], z_lines=None),
    lambda: ModelSpec(colour="red"),
    lambda: Numerics(tolerance=1e-3),
    lambda: Equilibrium(time=1.0),
    lambda: DCSweep(contact="a", values_V=[0.1], ramp="fast"),
    lambda: Track(contact="c", gain=1.0),
    lambda: _dev1d(temperature=300.0),
])
def test_unknown_fields_refused(build):
    with pytest.raises(VE):
        build()


def test_unknown_field_refused_on_problem_and_nested_dict():
    d = pn_diode_1d().model_dump(mode="json")
    Problem.model_validate(d)                                   # the dump itself is valid
    with pytest.raises(VE):
        Problem.model_validate({**d, "solver": "newton"})
    d2 = pn_diode_1d().model_dump(mode="json")
    d2["models"]["mobilty"] = "constant"                        # a typo must not pass silently
    with pytest.raises(VE):
        Problem.model_validate(d2)


# ------------------------------------------------------------------------------------------------ interval rule
def test_interval_rule_default_excludes_upper():
    b = Box(x=(1.0, 2.0))
    assert list(b.contains([0.999, 1.0, 1.5, 1.999, 2.0, 2.001])) == [False, True, True, True, False, False]


def test_interval_rule_include_upper():
    b = Box(x=(1.0, 2.0))
    assert list(b.contains([0.999, 1.0, 2.0, 2.001], include_upper=True)) == [False, True, True, False]


def test_interval_rule_unbounded_sides():
    assert list(Box(x=(None, 1.0)).contains([-1e30, 0.0, 1.0])) == [True, True, False]
    assert list(Box(x=(None, 1.0)).contains([1.0], include_upper=True)) == [True]
    assert list(Box(x=(1.0, None)).contains([0.999, 1.0, 1e30])) == [False, True, True]
    assert list(Box().contains([-5.0, 0.0, 5.0])) == [True, True, True]


def test_interval_rule_2d_and_missing_y():
    b = Box(x=(0.0, 1.0), y=(0.0, 1.0))
    assert list(b.contains([0.5, 0.5, 1.0], [0.5, 1.0, 0.5])) == [True, False, False]
    assert list(b.contains([0.5, 0.5, 1.0], [0.5, 1.0, 0.5], include_upper=True)) == [True, True, True]
    with pytest.raises(ValueError):
        b.contains([0.5])
    # a box without a y interval is unbounded in y: y values are ignored
    assert list(Box(x=(0.0, 1.0)).contains([0.5], [123.0])) == [True]


def test_box_bounds_must_be_ordered():
    with pytest.raises(VE):
        Box(x=(1.0, 1.0))
    with pytest.raises(VE):
        Box(x=(2.0, 1.0))
    with pytest.raises(VE):
        Box(x=(0.0, 1.0), y=(0.5, 0.5))
    Box(x=(None, 1.0), y=(0.0, None))                       # half-open is fine


# ------------------------------------------------------------------------------------------------ validators
def test_region_must_be_bounded():
    with pytest.raises(VE):
        Region(name="r", box=Box(x=(None, 1.0)))
    with pytest.raises(VE):
        Region(name="r", box=Box(x=(0.0, 1.0), y=(0.0, None)))


def test_contact_validators():
    with pytest.raises(VE):
        Contact(name="s", region="si", side="x_min", kind="schottky")              # no barrier
    with pytest.raises(VE):
        Contact(name="o", region="si", side="x_min", barrier_eV=0.7)              # barrier on ohmic
    with pytest.raises(VE):
        Contact(name="o", region="si", side="y_min", span=(0.5, 0.5))
    with pytest.raises(VE):
        Contact(name="o", region="si", side="z_min")
    Contact(name="s", region="si", side="x_min", kind="schottky", barrier_eV=0.7)


def test_doping_and_grade_validators():
    with pytest.raises(VE):
        DopingTerm(species="donor", concentration_cm3=0.0)
    with pytest.raises(VE):
        DopingTerm(species="donor", concentration_cm3=-1e16)
    with pytest.raises(VE):
        DopingTerm(species="hole", concentration_cm3=1e16)
    with pytest.raises(VE):
        GradeTerm(start_um=0.1, stop_um=0.1, x_start=0, x_stop=0.1)
    with pytest.raises(VE):
        GradeTerm(start_um=0.0, stop_um=0.1, x_start=0, x_stop=1.1)
    with pytest.raises(VE):
        GradeTerm(start_um=0.0, stop_um=0.1, x_start=-0.1, x_stop=0.1)
    with pytest.raises(VE):
        GradeTerm(element="Sn", start_um=0.0, stop_um=0.1, x_start=0, x_stop=0.1)


def test_mesh_validators():
    with pytest.raises(VE):
        MeshHints(x_lines=[(0.0, 0.1)])
    with pytest.raises(VE):
        MeshHints(x_lines=[(1.0, 0.1), (0.0, 0.1)])
    with pytest.raises(VE):
        MeshHints(x_lines=[(0.0, 0.1), (0.0, 0.1), (1.0, 0.1)])
    with pytest.raises(VE):
        MeshHints(x_lines=[(0.0, 0.1), (1.0, 0.0)])
    with pytest.raises(VE):
        MeshHints(x_lines=[(0.0, 0.1), (1.0, 0.1)], scale=0.0)
    with pytest.raises(VE):
        MeshHints(x_lines=[(0.0, 0.1), (1.0, 0.1)], y_lines=[(0.0, 0.1)])


def test_device_validators():
    r = Region(name="si", box=Box(x=(0.0, 1.0)))
    with pytest.raises(VE):
        _dev1d(regions=[])
    with pytest.raises(VE):
        _dev1d(regions=[r, r])
    with pytest.raises(VE):
        _dev1d(contacts=[Contact(name="a", region="si", side="x_min"), Contact(name="a", region="si", side="x_max")])
    with pytest.raises(VE):                                               # unknown region
        _dev1d(contacts=[Contact(name="a", region="ox", side="x_min")])
    with pytest.raises(VE):                                               # 1D contact on y
        _dev1d(contacts=[Contact(name="a", region="si", side="y_min")])
    with pytest.raises(VE):                                               # 1D contact with span
        _dev1d(contacts=[Contact(name="a", region="si", side="x_min", span=(0.0, 1.0))])
    with pytest.raises(VE):                                               # 1D region with y
        _dev1d(regions=[Region(name="si", box=Box(x=(0.0, 1.0), y=(0.0, 1.0)))])
    with pytest.raises(VE):                                               # 1D doping with y
        _dev1d(doping=[DopingTerm(species="donor", concentration_cm3=1e16, box=Box(y=(0.0, 1.0)))])
    with pytest.raises(VE):                                               # 1D grading along y
        _dev1d(grading=[GradeTerm(axis="y", start_um=0, stop_um=1, x_start=0, x_stop=0.1)])
    with pytest.raises(VE):                                               # 1D with y_lines
        _dev1d(mesh=MeshHints(x_lines=[(0, 0.1), (1, 0.1)], y_lines=[(0, 0.1), (1, 0.1)]))
    with pytest.raises(VE):                                               # 2D region without y
        DeviceSpec(name="d", dimension=2, regions=[r], contacts=[],
                   mesh=MeshHints(x_lines=[(0, 0.1), (1, 0.1)], y_lines=[(0, 0.1), (1, 0.1)]))
    with pytest.raises(VE):                                               # 2D without y_lines
        DeviceSpec(name="d", dimension=2, regions=[Region(name="si", box=Box(x=(0, 1), y=(0, 1)))], contacts=[],
                   mesh=MeshHints(x_lines=[(0, 0.1), (1, 0.1)]))
    with pytest.raises(VE):
        _dev1d(dimension=3)
    _dev1d(grading=[GradeTerm(axis="x", start_um=0, stop_um=1, x_start=0, x_stop=0.1)])     # x grading in 1D ok


def test_model_validators():
    with pytest.raises(VE):
        ModelSpec(mobility="constant", mu_n_cm2_Vs=1000.0)                       # needs both
    with pytest.raises(VE):
        ModelSpec(mobility="masetti", mu_n_cm2_Vs=1000.0, mu_p_cm2_Vs=400.0)     # takes neither
    with pytest.raises(VE):
        ModelSpec(srh="constant", tau_n_s=1e-7)
    with pytest.raises(VE):
        ModelSpec(srh="scharfetter", tau_n_s=1e-7, tau_p_s=1e-7)
    with pytest.raises(VE):
        ModelSpec(incomplete_ionization="constant_energy", ii_energy_eV=0.045)
    with pytest.raises(VE):
        ModelSpec(incomplete_ionization="altermatt", ii_energy_eV=0.045, ii_critical_doping_cm3=1e18)
    with pytest.raises(VE):
        ModelSpec(temperature_K=0.0)
    with pytest.raises(VE):
        ModelSpec(mobility="constant", mu_n_cm2_Vs=0.0, mu_p_cm2_Vs=400.0)
    with pytest.raises(VE):
        ModelSpec(statistics="maxwell")
    m = ModelSpec(mobility="constant", mu_n_cm2_Vs=1000.0, mu_p_cm2_Vs=400.0, srh="constant", tau_n_s=1e-7,
                  tau_p_s=2e-7, incomplete_ionization="constant_energy", ii_energy_eV=0.045,
                  ii_critical_doping_cm3=1e18)
    assert m.tau_p_s == 2e-7
    # defaults
    d = ModelSpec()
    assert (d.temperature_K, d.statistics, d.mobility, d.srh, d.auger, d.band_gap_narrowing,
            d.incomplete_ionization, d.impact_ionization) == (300.0, "boltzmann", "masetti", "scharfetter", True,
                                                              "slotboom", "off", "off")


def test_analysis_and_numerics_validators():
    with pytest.raises(VE):
        DCSweep(contact="anode", values_V=[])
    with pytest.raises(VE):
        DCSweep(contact="anode", values_V=[0.1], step_V=0.0)
    with pytest.raises(VE):
        Track(contact="c", step_V=-0.1)
    with pytest.raises(VE):
        Numerics(relative_error=0.0)
    with pytest.raises(VE):
        Numerics(max_iterations=0)
    dev = pn_diode_1d().device
    with pytest.raises(VE):
        Problem(device=dev, analysis=DCSweep(contact="gate", values_V=[0.1]))
    with pytest.raises(VE):
        Problem(device=dev, analysis=DCSweep(contact="anode", values_V=[0.1], tracking=[Track(contact="drain")]))
    with pytest.raises(VE):
        Problem(device=dev, analysis=DCSweep(contact="anode", values_V=[0.1], fixed_V={"drain": 1.0}))
    with pytest.raises(VE):                                   # swept and tracking
        Problem(device=dev, analysis=DCSweep(contact="anode", values_V=[0.1], tracking=[Track(contact="anode")]))
    with pytest.raises(VE):                                   # tracking and fixed
        Problem(device=dev, analysis=DCSweep(contact="anode", values_V=[0.1], tracking=[Track(contact="cathode")],
                                             fixed_V={"cathode": 0.0}))
    p = Problem(device=dev, analysis=DCSweep(contact="anode", values_V=[0.1], fixed_V={"cathode": 0.0}))
    assert p.analysis.kind == "dc_sweep"
    assert Problem(device=dev).analysis.kind == "equilibrium"


def test_analysis_discriminator_from_dict():
    d = pn_diode_1d(V=[0.1, 0.2]).model_dump(mode="json")
    assert d["analysis"]["kind"] == "dc_sweep"
    p = Problem.model_validate(d)
    assert isinstance(p.analysis, DCSweep) and p.analysis.values_V == [0.1, 0.2]
    d["analysis"] = {"kind": "equilibrium"}
    assert isinstance(Problem.model_validate(d).analysis, Equilibrium)
    d["analysis"] = {"kind": "transient"}
    with pytest.raises(VE):
        Problem.model_validate(d)


# ------------------------------------------------------------------------------------------------ evaluators
def test_doping_at_pn_diode_hand_points():
    p = pn_diode_1d()                      # NA 1e17 for x < 1.0 um, ND 1e16 for x >= 1.0 um (default interval rule)
    x = [0.0, 0.5, 0.999999, 1.0, 1.000001, 4.0, -1.0, 10.0]
    nd, na = p.device.doping_at(x)
    assert list(nd) == [0.0, 0.0, 0.0, 1e16, 1e16, 1e16, 0.0, 1e16]
    assert list(na) == [1e17, 1e17, 1e17, 0.0, 0.0, 0.0, 1e17, 0.0]
    nd2, na2 = pn_diode_1d(NA_cm3=3e17, ND_cm3=2e15, junction_um=0.25).device.doping_at([0.2, 0.25])
    assert list(nd2) == [0.0, 2e15] and list(na2) == [3e17, 0.0]
    # shapes are kept
    nd3, na3 = p.device.doping_at(np.array([[0.0, 2.0], [3.0, 0.5]]))
    assert nd3.shape == (2, 2) and na3.shape == (2, 2)


# Hand-evaluated from the npn_hbt_2d defaults (all terms include_upper=True): emitter donor 1e20 on x<=0.5, y<=0.03;
# collector donor 5e16 on y>=0.08; subcollector donor 1e19 on y>=1.0; base acceptor 2e18 on y<=0.08; extrinsic base
# acceptor 1e20 on x>=0.7, y<=0.1.
HBT_POINTS = [
    ((0.25, 0.01), (1e20, 2e18)),
    ((0.5, 0.03), (1e20, 2e18)),              # emitter corner, both bounds included
    ((0.5000001, 0.03), (0.0, 2e18)),
    ((0.25, 0.0300001), (0.0, 2e18)),
    ((0.25, 0.08), (5e16, 2e18)),             # closed intervals overlap at the base-collector boundary
    ((0.25, 0.5), (5e16, 0.0)),
    ((0.25, 1.0), (5e16 + 1e19, 0.0)),
    ((1.0, 0.05), (0.0, 2e18 + 1e20)),
    ((1.0, 0.09), (5e16, 1e20)),
    ((0.7, 0.1), (5e16, 1e20)),               # extrinsic base corner included
    ((0.6999999, 0.1), (5e16, 0.0)),
    ((1.5, 1.5), (5e16 + 1e19, 0.0)),
]


@pytest.mark.parametrize("pt,expected", HBT_POINTS)
def test_doping_at_hbt_hand_points(pt, expected):
    nd, na = npn_hbt_2d().device.doping_at([pt[0]], [pt[1]])
    assert float(nd[0]) == pytest.approx(expected[0], rel=1e-15, abs=0)
    assert float(na[0]) == pytest.approx(expected[1], rel=1e-15, abs=0)


def test_doping_at_2d_needs_y():
    with pytest.raises(ValueError):
        npn_hbt_2d().device.doping_at([0.1])


def test_mole_fraction_at_hbt_hand_points():
    dev = npn_hbt_2d().device            # Ge 0 at y = 0.03 rising linearly to 0.15 at y = 0.08, zero outside
    y = [0.0, 0.02, 0.03, 0.055, 0.08, 0.0801, 1.0]
    got = dev.mole_fraction_at([0.3] * len(y), y)
    exp = [0.0, 0.0, 0.0, 0.075, 0.15, 0.0, 0.0]
    assert np.allclose(got, exp, rtol=1e-12, atol=1e-15)
    # with a non-zero start: 0.05 at y = 0.03, 0.15 at y = 0.08 -> 0.1 at the midpoint
    got2 = npn_hbt_2d(ge_start=0.05).device.mole_fraction_at([0.3, 0.3, 0.3], [0.03, 0.055, 0.08])
    assert np.allclose(got2, [0.05, 0.1, 0.15], rtol=1e-12, atol=0)
    # Si twin: no grading term at all
    assert npn_hbt_2d(ge_peak=0.0).device.grading == []
    assert np.all(npn_hbt_2d(ge_peak=0.0).device.mole_fraction_at([0.3], [0.05]) == 0.0)
    # another element: nothing
    assert np.all(dev.mole_fraction_at([0.3], [0.05], element="C") == 0.0)


def test_mole_fraction_at_x_axis_1d():
    dev = _dev1d(grading=[GradeTerm(axis="x", start_um=0.2, stop_um=0.6, x_start=0.1, x_stop=0.3)])
    got = dev.mole_fraction_at([0.1, 0.2, 0.4, 0.6, 0.7])
    assert np.allclose(got, [0.0, 0.1, 0.2, 0.3, 0.0], rtol=1e-12, atol=0)
    assert np.all(pn_diode_1d().device.mole_fraction_at([0.0, 2.0]) == 0.0)


def test_doping_terms_add():
    dev = _dev1d(doping=[DopingTerm(species="donor", concentration_cm3=1e16, box=Box(x=(0.0, 0.6))),
                         DopingTerm(species="donor", concentration_cm3=2e16, box=Box(x=(0.4, 1.0))),
                         DopingTerm(species="acceptor", concentration_cm3=5e15)])
    nd, na = dev.doping_at([0.2, 0.5, 0.8])
    assert list(nd) == [1e16, 3e16, 2e16] and list(na) == [5e15] * 3


def test_extent_region_contact_lookup():
    dev = npn_hbt_2d().device
    e = dev.extent()
    assert tuple(e.x) == (0.0, 1.5) and tuple(e.y) == (0.0, 1.5)
    assert tuple(pn_diode_1d().device.extent().x) == (0.0, 4.0)
    assert dev.contact("base").span == (0.9, 1.5)
    assert dev.region("si").material == "silicon"
    with pytest.raises(KeyError):
        dev.contact("drain")
    with pytest.raises(KeyError):
        dev.region("ox")


# ------------------------------------------------------------------------------------------------ problem_id
def test_problem_id_stable():
    a, b = pn_diode_1d(V=[0.1, 0.2]), pn_diode_1d(V=[0.1, 0.2])
    pid = a.problem_id()
    assert len(pid) == 32 and all(c in "0123456789abcdef" for c in pid)
    assert pid == b.problem_id() == a.problem_id()
    assert Problem.model_validate(a.model_dump(mode="json")).problem_id() == pid
    assert Problem.model_validate_json(a.model_dump_json()).problem_id() == pid
    assert a.with_changes({}).problem_id() == pid
    assert npn_hbt_2d().problem_id() == npn_hbt_2d().problem_id()
    # explicit defaults are the same problem as omitted defaults
    dev = a.device
    assert Problem(device=dev).problem_id() == Problem(device=dev, models=ModelSpec(), analysis=Equilibrium(),
                                                       numerics=Numerics()).problem_id()


@pytest.mark.parametrize("change", [
    {"models.temperature_K": 301.0},
    {"device.mesh.scale": 0.5},
    {"analysis.values_V": [0.1, 0.3]},
    {"numerics.relative_error": 1e-10},
    {"device.doping.0.concentration_cm3": 2e17},
    {"device.doping.1.include_upper": True},
    {"device.description": "renamed"},
    {"models.auger": True},
])
def test_problem_id_sensitive(change):
    p = pn_diode_1d(V=[0.1, 0.2])
    assert p.with_changes(change).problem_id() != p.problem_id()


def test_problem_id_differs_between_library_devices_and_arguments():
    ids = {pn_diode_1d().problem_id(), pn_diode_1d(V=[0.1]).problem_id(), pn_diode_1d(T=77.0).problem_id(),
           npn_hbt_2d().problem_id(), npn_hbt_2d(VBE=[0.6]).problem_id(), npn_hbt_2d(ge_peak=0.0).problem_id()}
    assert len(ids) == 6


# ------------------------------------------------------------------------------------------------ with_changes
def test_with_changes_returns_validated_copy():
    p = pn_diode_1d(V=[0.1, 0.2])
    q = p.with_changes({"models.temperature_K": 77.0, "device.mesh.scale": 0.5,
                        "device.doping.0.concentration_cm3": 3e17, "analysis.values_V": [0.4]})
    assert isinstance(q, Problem)
    assert q.models.temperature_K == 77.0 and q.device.mesh.scale == 0.5
    assert q.device.doping[0].concentration_cm3 == 3e17 and q.analysis.values_V == [0.4]
    # the original is untouched
    assert p.models.temperature_K == 300.0 and p.device.mesh.scale == 1.0
    assert p.device.doping[0].concentration_cm3 == 1e17 and p.analysis.values_V == [0.1, 0.2]
    # several coupled fields in one call are validated together (pairing rule)
    r = p.with_changes({"models.mobility": "masetti", "models.mu_n_cm2_Vs": None, "models.mu_p_cm2_Vs": None})
    assert r.models.mobility == "masetti"


def test_with_changes_revalidates():
    p = pn_diode_1d()
    with pytest.raises(VE):
        p.with_changes({"models.mobility": "masetti"})               # mu_n/mu_p still set
    with pytest.raises(VE):
        p.with_changes({"models.temperature_K": -1.0})
    with pytest.raises(VE):
        p.with_changes({"device.contacts.0.side": "y_min"})          # 1D contact on y
    with pytest.raises(VE):
        p.with_changes({"device.mesh.scale": 0.0})


def test_with_changes_unknown_path_keyerror():
    p = pn_diode_1d()
    with pytest.raises(KeyError):
        p.with_changes({"models.colour": "red"})
    with pytest.raises(KeyError):
        p.with_changes({"models.temperature": 77.0})
    with pytest.raises(KeyError):
        p.with_changes({"device.nothing.scale": 0.5})


# ------------------------------------------------------------------------------------------------ library vs legacy
def test_pn_diode_mirrors_legacy_constants():
    from simws.validation.engine_pairs import DD
    p = pn_diode_1d()
    dev, m = p.device, p.models
    assert p.spec_version == SPEC_VERSION == "1.0"
    na = [t for t in dev.doping if t.species == "acceptor"]
    nd = [t for t in dev.doping if t.species == "donor"]
    assert len(na) == 1 and len(nd) == 1
    assert na[0].concentration_cm3 == DD["NA"] and nd[0].concentration_cm3 == DD["ND"]
    assert na[0].box.x == (None, DD["xj_um"]) and nd[0].box.x == (DD["xj_um"], None)
    assert dev.region("si").box.x == (0.0, DD["L_um"])
    assert (m.mu_n_cm2_Vs, m.mu_p_cm2_Vs, m.tau_n_s, m.tau_p_s, m.temperature_K) == \
        (DD["mu_n"], DD["mu_p"], DD["tau"], DD["tau"], DD["T"])
    assert (m.mobility, m.srh, m.auger, m.band_gap_narrowing, m.statistics) == \
        ("constant", "constant", False, "off", "boltzmann")
    h = DD["L_um"] / (801 - 1)                                    # legacy n = 801
    assert dev.mesh.x_lines == [(0.0, h), (DD["L_um"], h)]
    assert (p.numerics.relative_error, p.numerics.absolute_error, p.numerics.max_iterations) == (1e-12, 1e10, 80)
    a = pn_diode_1d(V=[0.1, 0.2]).analysis
    assert a.contact == "anode" and a.step_V == 0.05 and a.tracking == [] and a.fixed_V == {}
    assert dev.contact("anode").side == "x_min" and dev.contact("cathode").side == "x_max"


def _legacy_eval(expr, x_cm, y_cm):
    """Evaluate a legacy DEVSIM node expression with numpy (step(0) = 1, as DEVSIM's)."""
    env = {"step": lambda v: np.heaviside(v, 1.0), "x": x_cm, "y": y_cm}
    return eval(expr, {"__builtins__": {}}, env)


def test_hbt_doping_and_grade_match_legacy_expressions():
    """The library HBT must describe the doping and Ge profile of simws.tcad.hbt.doping / ge_profile (non-cryo
    defaults). Evaluated on every mesh-line crossing plus a fine grid; coordinates in cm = um x 1e-4 (as legacy)."""
    import simws.tcad.hbt as H
    case = H.HBTCase()
    net_expr, tot_expr = H.doping(case)
    ge_expr = H.ge_profile(case)
    p = npn_hbt_2d()
    dev = p.device
    xs = sorted({x for x, _ in dev.mesh.x_lines} | set(np.linspace(0, case.width_um, 61).tolist()))
    ys = sorted({y for y, _ in dev.mesh.y_lines} | {case.t_e + case.w_b, case.y_xb} |
                set(np.linspace(0, case.depth_um, 61).tolist()))
    X, Y = np.meshgrid(np.array(xs), np.array(ys))
    X, Y = X.ravel(), Y.ravel()
    nd, na = dev.doping_at(X, Y)
    net_l = _legacy_eval(net_expr, X * H.UM, Y * H.UM)
    tot_l = _legacy_eval(tot_expr, X * H.UM, Y * H.UM)
    assert np.allclose(nd - na, net_l, rtol=1e-12, atol=0)
    assert np.allclose(nd + na, tot_l, rtol=1e-12, atol=0)
    ge_l = np.maximum(_legacy_eval(ge_expr, X * H.UM, Y * H.UM), 0)
    assert np.allclose(dev.mole_fraction_at(X, Y), ge_l, rtol=1e-12, atol=1e-15)


def test_hbt_mirrors_legacy_case():
    import simws.tcad.hbt as H
    c = H.HBTCase()
    p = npn_hbt_2d(VBE=[0.5, 0.6])
    m = p.models
    assert (m.temperature_K, m.tau_n_s, m.tau_p_s, m.mobility, m.statistics, m.incomplete_ionization) == \
        (c.T, c.tau, c.tau, "masetti", "boltzmann", "off")
    assert (m.srh, m.auger, m.band_gap_narrowing) == ("constant", True, "slotboom")
    assert p.device.mesh.scale == c.mesh_scale
    g = p.device.grading
    assert len(g) == 1 and (g[0].start_um, g[0].stop_um, g[0].x_start, g[0].x_stop, g[0].axis, g[0].mode) == \
        (c.t_e, c.t_e + c.w_b, c.ge_start, c.ge_peak, "y", "strained")
    a = p.analysis
    assert a.contact == "base" and a.step_V == 0.05 and a.values_V == [0.5, 0.6]
    assert len(a.tracking) == 1 and a.tracking[0].contact == "collector" and a.tracking[0].step_V == 0.1
    assert a.tracking[0].offset_V == 0.0 and a.fixed_V == {}
    assert (p.numerics.relative_error, p.numerics.absolute_error, p.numerics.max_iterations) == (1e-8, 1e10, 40)
    em, b, col = (p.device.contact(n) for n in ("emitter", "base", "collector"))
    assert (em.side, em.span) == ("y_min", (0.0, c.w_e))
    assert (b.side, b.span) == ("y_min", (c.x_bc, c.width_um))
    assert (col.side, col.span) == ("y_max", None)
