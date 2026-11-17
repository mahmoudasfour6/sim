"""Contract amendment A8.1 / A8.2: the single-event description (simws.spec.see, exported from simws.spec), the
family() helper for the third family and the library problem see_si_slab. Written from docs/ARCHITECTURE_V1.md (A8,
with the shared conventions of A7) only."""
import copy

import pydantic
import pytest


def _base() -> dict:
    """A minimal valid SeeProblem as a dict (test input, built by hand from the A8.1 field list)."""
    return {"source": {"particle": "alpha", "energy_MeV": 5.0},
            "geometry": {"bulk_width_um": 1000.0, "bulk_thickness_um": 100.0, "sv_width_um": [1000.0, 1000.0],
                         "sv_thickness_um": 10.0},
            "histogram": {"high_MeV": 5.25}}


def _validate(d):
    from simws.spec import SeeProblem
    return SeeProblem.model_validate(d)


def _set(d, path, value):
    node = d
    parts = path.split(".")
    for k in parts[:-1]:
        if isinstance(node, list):
            node = node[int(k)]
        else:
            node = node.setdefault(k, {})
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value
    return d


def _del(d, path):
    node = d
    parts = path.split(".")
    for k in parts[:-1]:
        node = node[int(k)] if isinstance(node, list) else node[k]
    del node[parts[-1]]
    return d


BEOL2 = [{"name": "m1", "material": "G4_Cu", "thickness_um": 0.5},
         {"name": "ild", "material": "G4_SILICON_DIOXIDE", "thickness_um": 1.0}]


# ------------------------------------------------------------------------------------------------ exports
NAMES = ("SeeSource", "BeolLayer", "SeeGeometry", "SeePhysics", "EdepHistogram", "SeeStatistics", "SeeProblem")


def test_exports_from_simws_spec():
    import simws.spec as S
    from simws.spec import see
    for name in NAMES + ("family",):
        assert hasattr(S, name), name
    for name in NAMES:
        assert getattr(see, name) is getattr(S, name), name
    # A7.1 unchanged: the implant and device descriptions are still exported
    for name in ("ImplantProblem", "Atom", "Problem"):
        assert hasattr(S, name), name


def test_g4_material_pattern_constant():
    from simws.spec import see
    assert see.G4_MATERIAL == r"^G4_[A-Za-z0-9_]+$"


def test_defaults():
    p = _validate(_base())
    assert p.spec_version == "1.0" and p.kind == "see"
    assert p.source.tilt_deg == 0.0 and p.source.ion_Z is None and p.source.ion_A is None
    g = p.geometry
    assert g.bulk_material == "G4_Si" and g.sv_depth_um == 0.0 and g.beol == []
    assert tuple(g.sv_width_um) == (1000.0, 1000.0)
    assert p.physics.em == "option4" and p.physics.hadronic == "none"
    h = p.histogram
    assert h.scale == "lin" and h.bins == 2000 and h.low_MeV == 0.0 and h.high_MeV == 5.25
    assert p.statistics.n_events == 10000 and p.statistics.seed == 1


def test_classes_construct_directly():
    from simws.spec import (BeolLayer, EdepHistogram, SeeGeometry, SeePhysics, SeeProblem, SeeSource,
                            SeeStatistics)
    p = SeeProblem(source=SeeSource(particle="ion", ion_Z=8, ion_A=16, energy_MeV=100.0, tilt_deg=30.0),
                   geometry=SeeGeometry(bulk_width_um=50.0, bulk_thickness_um=20.0, sv_depth_um=1.0,
                                        sv_width_um=(5.0, 10.0), sv_thickness_um=2.0,
                                        beol=[BeolLayer(name="ox", material="G4_SILICON_DIOXIDE", thickness_um=1.0)]),
                   physics=SeePhysics(em="livermore", hadronic="ftfp_bert_hp"),
                   histogram=EdepHistogram(scale="log", bins=100, low_MeV=0.001, high_MeV=10.0),
                   statistics=SeeStatistics(n_events=100, seed=0))
    assert p.source.ion_Z == 8 and p.source.ion_A == 16 and p.geometry.beol[0].name == "ox"
    assert p.physics.hadronic == "ftfp_bert_hp" and p.histogram.scale == "log" and p.statistics.seed == 0


# ------------------------------------------------------------------------------------------------ validation rules
BAD = [
    # SeeSource
    ("source.particle", "pion"),
    ("source.particle", "electron"),
    ("source.particle", "Proton"),
    ("source.energy_MeV", 0.0),
    ("source.energy_MeV", -1.0),
    ("source.tilt_deg", -0.001),
    ("source.tilt_deg", 90.0),
    ("source.tilt_deg", 135.0),
    ("source.ion_Z", 2),                         # forbidden for particle != "ion"
    ("source.ion_A", 4),                         # forbidden for particle != "ion"
    # BeolLayer (a valid layer is set first by _bad_with_beol)
    ("geometry.beol.0.name", ""),
    ("geometry.beol.0.material", "Cu"),
    ("geometry.beol.0.material", "G4_"),
    ("geometry.beol.0.material", "g4_Cu"),
    ("geometry.beol.0.material", "G4-Cu"),
    ("geometry.beol.0.material", "G4_SILICON DIOXIDE"),
    ("geometry.beol.0.material", "G4_Cu+"),
    ("geometry.beol.0.thickness_um", 0.0),
    ("geometry.beol.0.thickness_um", -0.5),
    # SeeGeometry
    ("geometry.bulk_material", "Si"),
    ("geometry.bulk_material", "G4_"),
    ("geometry.bulk_material", "xG4_Si"),
    ("geometry.bulk_material", "G4_Si "),
    ("geometry.bulk_width_um", 0.0),
    ("geometry.bulk_width_um", -1000.0),
    ("geometry.bulk_thickness_um", 0.0),
    ("geometry.bulk_thickness_um", -100.0),
    ("geometry.sv_depth_um", -0.001),
    ("geometry.sv_width_um", [0.0, 1000.0]),
    ("geometry.sv_width_um", [1000.0, 0.0]),
    ("geometry.sv_width_um", [-1.0, 10.0]),
    ("geometry.sv_width_um", [1000.5, 10.0]),     # > bulk_width_um
    ("geometry.sv_width_um", [10.0, 1001.0]),     # > bulk_width_um
    ("geometry.sv_width_um", [10.0]),             # a pair
    ("geometry.sv_width_um", [10.0, 10.0, 10.0]),
    ("geometry.sv_thickness_um", 0.0),
    ("geometry.sv_thickness_um", -10.0),
    ("geometry.sv_thickness_um", 100.5),          # depth 0 + 100.5 > 100
    ("geometry.sv_depth_um", 90.5),               # 90.5 + 10 > 100
    # SeePhysics
    ("physics.em", "option1"),
    ("physics.em", "penelope"),
    ("physics.hadronic", "qgsp_bic"),
    ("physics.hadronic", "ftfp_bert"),
    # EdepHistogram
    ("histogram.scale", "sqrt"),
    ("histogram.bins", 9),
    ("histogram.bins", 20001),
    ("histogram.low_MeV", -0.001),
    ("histogram.high_MeV", 0.0),                  # not > low 0
    ("histogram.scale", "log"),                   # log needs low > 0 (low is the default 0)
    # SeeStatistics
    ("statistics.n_events", 99),
    ("statistics.n_events", 100_000_001),
    ("statistics.seed", -1),
    # problem
    ("kind", "implant"),
    ("kind", "device"),
]


def _bad_case(path, value):
    d = _base()
    if path.startswith("geometry.beol"):
        d["geometry"]["beol"] = copy.deepcopy(BEOL2)
    return _set(d, path, value)


@pytest.mark.parametrize("path,value", BAD, ids=[f"{p}={v!r}" for p, v in BAD])
def test_each_rule_raises_validation_error(path, value):
    d = _bad_case(path, value)
    with pytest.raises(pydantic.ValidationError):
        _validate(d)
    assert issubclass(pydantic.ValidationError, ValueError)


def test_bad_case_base_is_valid():
    """The BEOL base used by the parametrised cases is itself valid (so a raise comes from the changed field)."""
    d = _base()
    d["geometry"]["beol"] = copy.deepcopy(BEOL2)
    p = _validate(d)
    assert [b.name for b in p.geometry.beol] == ["m1", "ild"]


GOOD = [
    ("source.tilt_deg", 0.0), ("source.tilt_deg", 89.999), ("source.energy_MeV", 1e-6),
    ("source.particle", "proton"), ("source.particle", "neutron"), ("source.particle", "e-"),
    ("source.particle", "gamma"), ("source.particle", "alpha"),
    ("geometry.sv_depth_um", 0.0), ("geometry.sv_depth_um", 90.0),          # 90 + 10 = 100: SV ends at the bottom
    ("geometry.sv_thickness_um", 100.0),                                     # 0 + 100 = 100
    ("geometry.sv_width_um", [1000.0, 1000.0]), ("geometry.sv_width_um", [0.5, 1000.0]),
    ("geometry.bulk_material", "G4_SILICON_DIOXIDE"), ("geometry.bulk_material", "G4_GALLIUM_ARSENIDE"),
    ("geometry.bulk_material", "G4_Si"), ("geometry.bulk_material", "G4_lH2"),
    ("physics.em", "option3"), ("physics.em", "livermore"), ("physics.hadronic", "ftfp_bert_hp"),
    ("histogram.bins", 10), ("histogram.bins", 20000), ("histogram.low_MeV", 0.0),
    ("statistics.n_events", 100), ("statistics.n_events", 100_000_000), ("statistics.seed", 0),
]


@pytest.mark.parametrize("path,value", GOOD, ids=[f"{p}={v!r}" for p, v in GOOD])
def test_boundary_values_accepted(path, value):
    _validate(_set(_base(), path, value))


def test_sv_inside_bulk_with_depth_and_thickness():
    d = _set(_set(_base(), "geometry.sv_depth_um", 2.0), "geometry.sv_thickness_um", 98.0)
    _validate(d)                                                  # 2 + 98 = 100 exactly
    with pytest.raises(pydantic.ValidationError):
        _validate(_set(copy.deepcopy(d), "geometry.sv_thickness_um", 98.5))
    with pytest.raises(pydantic.ValidationError):
        _validate(_set(copy.deepcopy(d), "geometry.bulk_thickness_um", 99.5))


def test_sv_width_limited_by_bulk_width_change():
    d = _set(_base(), "geometry.bulk_width_um", 999.0)          # SV stays 1000 x 1000
    with pytest.raises(pydantic.ValidationError):
        _validate(d)


def test_histogram_high_must_exceed_low():
    for low, high in ((1.0, 1.0), (2.0, 1.0)):
        d = _set(_set(_base(), "histogram.low_MeV", low), "histogram.high_MeV", high)
        with pytest.raises(pydantic.ValidationError):
            _validate(d)
    _validate(_set(_set(_base(), "histogram.low_MeV", 1.0), "histogram.high_MeV", 1.001))


def test_log_histogram_needs_positive_low():
    d = _set(_base(), "histogram.scale", "log")
    with pytest.raises(pydantic.ValidationError):
        _validate(d)                                              # low 0 (default)
    with pytest.raises(pydantic.ValidationError):
        _validate(_set(copy.deepcopy(d), "histogram.low_MeV", 0.0))
    p = _validate(_set(copy.deepcopy(d), "histogram.low_MeV", 1e-3))
    assert p.histogram.scale == "log" and p.histogram.low_MeV == 1e-3
    # a linear histogram may start at 0 or above
    _validate(_set(_base(), "histogram.low_MeV", 0.5))


def test_histogram_high_is_required():
    with pytest.raises(pydantic.ValidationError):
        _validate(_del(_base(), "histogram.high_MeV"))


@pytest.mark.parametrize("path", ["source", "geometry", "histogram", "source.particle", "source.energy_MeV",
                                  "geometry.bulk_width_um", "geometry.bulk_thickness_um", "geometry.sv_width_um",
                                  "geometry.sv_thickness_um"])
def test_required_fields(path):
    with pytest.raises(pydantic.ValidationError):
        _validate(_del(_base(), path))


# ------------------------------------------------------------------------------------------------ ion_Z / ion_A
def _ion(Z, A, particle="ion"):
    d = _base()
    d["source"] = {"particle": particle, "energy_MeV": 100.0}
    if Z is not None:
        d["source"]["ion_Z"] = Z
    if A is not None:
        d["source"]["ion_A"] = A
    d["histogram"]["high_MeV"] = 105.0
    return d


@pytest.mark.parametrize("Z,A", [(8, 16), (1, 1), (2, 4), (118, 294), (26, 56), (92, 238)])
def test_ion_valid(Z, A):
    p = _validate(_ion(Z, A))
    assert p.source.particle == "ion" and p.source.ion_Z == Z and p.source.ion_A == A


@pytest.mark.parametrize("Z,A", [(None, None), (8, None), (None, 16), (0, 16), (119, 300), (-1, 4), (8, 7),
                                 (26, 25), (2, 1)])
def test_ion_invalid(Z, A):
    with pytest.raises(pydantic.ValidationError):
        _validate(_ion(Z, A))


@pytest.mark.parametrize("particle", ["proton", "alpha", "neutron", "e-", "gamma"])
def test_ion_fields_forbidden_for_other_particles(particle):
    for Z, A in ((1, 1), (2, None), (None, 4), (2, 4)):
        with pytest.raises(pydantic.ValidationError):
            _validate(_ion(Z, A, particle=particle))
    p = _validate(_ion(None, None, particle=particle))
    assert p.source.ion_Z is None and p.source.ion_A is None


# ------------------------------------------------------------------------------------------------ BEOL
def test_beol_order_kept_and_names_unique():
    d = _base()
    d["geometry"]["beol"] = copy.deepcopy(BEOL2)
    p = _validate(d)
    assert [(b.name, b.material, b.thickness_um) for b in p.geometry.beol] == \
        [("m1", "G4_Cu", 0.5), ("ild", "G4_SILICON_DIOXIDE", 1.0)]
    d["geometry"]["beol"].append({"name": "m1", "material": "G4_Al", "thickness_um": 0.3})
    with pytest.raises(pydantic.ValidationError):
        _validate(d)
    # the same material twice is fine when the names differ
    d["geometry"]["beol"][-1]["name"] = "m2"
    _validate(d)


@pytest.mark.parametrize("where", ["", "source", "geometry", "geometry.beol.0", "physics", "histogram",
                                   "statistics"])
def test_unknown_fields_forbidden(where):
    d = _base()
    d["geometry"]["beol"] = copy.deepcopy(BEOL2)
    d.setdefault("physics", {})
    d.setdefault("statistics", {})
    node = d
    for k in [w for w in where.split(".") if w]:
        node = node[int(k)] if isinstance(node, list) else node[k]
    node["colour"] = "red"
    with pytest.raises(pydantic.ValidationError):
        _validate(d)


# ------------------------------------------------------------------------------------------------ problem_id
def test_problem_id_stable_and_same_recipe_as_problem():
    from simws.hub.canonical import job_id
    from simws.spec import SeeProblem
    a, b = _validate(_base()), _validate(_base())
    pid = a.problem_id()
    assert isinstance(pid, str) and pid == b.problem_id()
    assert SeeProblem.model_validate(a.model_dump(mode="json")).problem_id() == pid
    assert SeeProblem.model_validate_json(a.model_dump_json()).problem_id() == pid
    assert a.with_changes({}).problem_id() == pid
    assert pid == job_id(a.model_dump(mode="json"))            # same recipe as Problem.problem_id
    assert _validate(a.model_dump(mode="json")).problem_id() == pid


@pytest.mark.parametrize("change", [
    {"source.energy_MeV": 6.0}, {"source.tilt_deg": 30.0}, {"source.particle": "proton"},
    {"geometry.bulk_material": "G4_SILICON_DIOXIDE"}, {"geometry.bulk_width_um": 2000.0},
    {"geometry.bulk_thickness_um": 300.0}, {"geometry.sv_depth_um": 2.0}, {"geometry.sv_width_um": [500.0, 1000.0]},
    {"geometry.sv_thickness_um": 5.0}, {"geometry.beol": BEOL2}, {"physics.em": "option3"},
    {"physics.hadronic": "ftfp_bert_hp"}, {"histogram.bins": 1000}, {"histogram.low_MeV": 0.1},
    {"histogram.high_MeV": 6.0}, {"statistics.n_events": 1000}, {"statistics.seed": 2},
])
def test_problem_id_sensitive(change):
    p = _validate(_base())
    assert p.with_changes(change).problem_id() != p.problem_id()


def test_problem_id_sensitive_to_beol_order_and_ion():
    d = _base()
    d["geometry"]["beol"] = copy.deepcopy(BEOL2)
    p = _validate(d)
    d2 = copy.deepcopy(d)
    d2["geometry"]["beol"] = list(reversed(d2["geometry"]["beol"]))
    assert _validate(d2).problem_id() != p.problem_id()
    a = _validate(_ion(8, 16))
    assert a.problem_id() != _validate(_ion(8, 17)).problem_id()
    assert a.problem_id() != _validate(_ion(9, 17)).problem_id()


def test_with_changes_returns_validated_copy():
    from simws.spec import SeeProblem
    d = _base()
    d["geometry"]["beol"] = copy.deepcopy(BEOL2)
    p = _validate(d)
    q = p.with_changes({"source.energy_MeV": 10.0, "geometry.beol.1.thickness_um": 2.0, "statistics.seed": 3,
                        "histogram.high_MeV": 10.5})
    assert isinstance(q, SeeProblem)
    assert q.source.energy_MeV == 10.0 and q.geometry.beol[1].thickness_um == 2.0 and q.statistics.seed == 3
    assert q.histogram.high_MeV == 10.5
    assert p.source.energy_MeV == 5.0 and p.geometry.beol[1].thickness_um == 1.0 and p.statistics.seed == 1
    for bad in ({"source.energy_MeV": -1.0}, {"source.tilt_deg": 90.0}, {"statistics.n_events": 10},
                {"geometry.sv_thickness_um": 200.0}, {"source.ion_Z": 8}, {"histogram.scale": "log"},
                {"geometry.beol.0.name": "ild"}):
        with pytest.raises(ValueError):
            p.with_changes(bad)
    # changing particle and ion fields together
    r = p.with_changes({"source.particle": "ion", "source.ion_Z": 8, "source.ion_A": 16})
    assert r.source.particle == "ion" and r.source.ion_Z == 8


# ------------------------------------------------------------------------------------------------ family
def test_family():
    from simws.spec import family
    from simws.spec.library import implant_si, npn_hbt_2d, pn_diode_1d
    assert family(_validate(_base())) == "see"
    assert family(implant_si()) == "implant"
    assert family(pn_diode_1d()) == "device"
    assert family(npn_hbt_2d()) == "device"


def test_existing_problem_ids_unchanged_recipe():
    from simws.hub.canonical import job_id
    from simws.spec.library import implant_si, pn_diode_1d
    for p in (pn_diode_1d(), implant_si()):
        assert p.problem_id() == job_id(p.model_dump(mode="json"))


# ------------------------------------------------------------------------------------------------ library (A8.2)
def _slab(*a, **k):
    from simws.spec.library import see_si_slab
    return see_si_slab(*a, **k)


def test_see_si_slab_defaults():
    from simws.spec import SeeProblem
    p = _slab()
    assert isinstance(p, SeeProblem) and p.kind == "see"
    s = p.source
    assert s.particle == "alpha" and s.energy_MeV == 5.0 and s.tilt_deg == 0.0
    assert s.ion_Z is None and s.ion_A is None
    g = p.geometry
    assert g.bulk_material == "G4_Si"
    assert g.bulk_width_um == 1000.0 and g.bulk_thickness_um == 100.0
    assert tuple(g.sv_width_um) == (1000.0, 1000.0)
    assert g.sv_thickness_um == 10.0 and g.sv_depth_um == 0.0
    assert g.beol == []
    assert p.physics.em == "option4" and p.physics.hadronic == "none"
    h = p.histogram
    assert h.scale == "lin" and h.bins == 2000 and h.low_MeV == 0.0
    assert h.high_MeV == pytest.approx(1.05 * 5.0, rel=1e-11)          # A8.9: rounded to 12 digits
    assert p.statistics.n_events == 10000 and p.statistics.seed == 1


def test_see_si_slab_same_as_hand_made():
    """The library default equals the description written out by hand from A8.2."""
    d = _base()
    d["histogram"]["high_MeV"] = 1.05 * 5.0
    assert _slab().problem_id() == _validate(d).problem_id()


def test_see_si_slab_arguments():
    beol = [("m1", "G4_Cu", 0.5), ("ild", "G4_SILICON_DIOXIDE", 1.0), ("pass", "G4_Si", 2.0)]
    p = _slab("proton", 10.0, sv_thickness_um=100.0, sv_depth_um=3.0, bulk_thickness_um=300.0, width_um=500.0,
              tilt_deg=30.0, beol=beol, hadronic="ftfp_bert_hp", n_events=2000, seed=7, bins=500, high_MeV=12.0)
    assert p.source.particle == "proton" and p.source.energy_MeV == 10.0 and p.source.tilt_deg == 30.0
    g = p.geometry
    assert g.bulk_material == "G4_Si" and g.bulk_width_um == 500.0 and g.bulk_thickness_um == 300.0
    assert tuple(g.sv_width_um) == (500.0, 500.0) and g.sv_thickness_um == 100.0 and g.sv_depth_um == 3.0
    assert [(b.name, b.material, b.thickness_um) for b in g.beol] == beol          # beam side first, as given
    assert p.physics.hadronic == "ftfp_bert_hp" and p.physics.em == "option4"
    assert p.statistics.n_events == 2000 and p.statistics.seed == 7
    h = p.histogram
    assert h.scale == "lin" and h.bins == 500 and h.low_MeV == 0.0 and h.high_MeV == 12.0


@pytest.mark.parametrize("E", [5.0, 10.0, 14.0, 0.3])
def test_see_si_slab_high_default(E):
    assert _slab("proton", E).histogram.high_MeV == pytest.approx(1.05 * E, rel=1e-11)


def test_see_si_slab_ion():
    p = _slab("ion", 100.0, ion_Z=8, ion_A=16)
    assert p.source.particle == "ion" and p.source.ion_Z == 8 and p.source.ion_A == 16
    assert p.histogram.high_MeV == pytest.approx(105.0, rel=1e-11)
    with pytest.raises(ValueError):
        _slab("ion", 100.0)                                       # ion without Z, A
    with pytest.raises(ValueError):
        _slab("proton", 10.0, ion_Z=1, ion_A=1)                  # forbidden for a proton


def test_see_si_slab_is_valid_for_the_acceptance_cases():
    """The library problems of A8.7 (R20 to R28) validate with the stated arguments."""
    _slab(n_events=1000)
    _slab(n_events=1000, seed=2)
    _slab("alpha", 5.0, sv_thickness_um=10.0)
    _slab("proton", 10.0, sv_thickness_um=100.0, bulk_thickness_um=300.0)
    p = _slab("alpha", 5.0, beol=[("l1", "G4_Si", 1.0), ("l2", "G4_Si", 2.0)], sv_depth_um=2.0, sv_thickness_um=5.0)
    assert sum(b.thickness_um for b in p.geometry.beol) == 3.0
    _slab("alpha", 5.0, tilt_deg=60.0, sv_thickness_um=5.0)
    _slab("alpha", 5.0, sv_thickness_um=50.0)
    q = _slab("neutron", 14.0, sv_thickness_um=100.0, bulk_thickness_um=300.0, hadronic="ftfp_bert_hp",
              n_events=2000, high_MeV=20.0)
    assert q.histogram.high_MeV == 20.0


def test_see_si_slab_invalid_geometry_raises():
    with pytest.raises(ValueError):
        _slab(sv_thickness_um=100.0, sv_depth_um=1.0)            # SV beyond the bulk
    with pytest.raises(ValueError):
        _slab(beol=[("a", "G4_Si", 1.0), ("a", "G4_Cu", 1.0)])   # duplicate names
    with pytest.raises(ValueError):
        _slab(beol=[("a", "Copper", 1.0)])                       # not a G4 material name
