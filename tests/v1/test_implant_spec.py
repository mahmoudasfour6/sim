"""Contract amendment A7.1 / A7.2: the implant description (simws.spec.implant, exported from simws.spec), the
family() helper and the library problem implant_si. Written from docs/ARCHITECTURE_V1.md (A7) only."""
import copy
import math

import pydantic
import pytest

SI = {"symbol": "Si", "Z": 14, "mass_amu": 28.085, "cutoff_eV": 1.5, "surface_binding_eV": 4.72}
P_ION = {"symbol": "P", "Z": 15, "mass_amu": 30.973762, "cutoff_eV": 1.5}


def _base() -> dict:
    """A minimal valid ImplantProblem as a dict (test input, built by hand from the A7.1 field list)."""
    return {"beam": {"ion": dict(P_ION), "energy_keV": 100.0},
            "target": {"atoms": [dict(SI)],
                       "layers": [{"name": "si", "thickness_um": 2.0, "density_cm3": {"Si": 4.994e22}}]}}


def _validate(d):
    from simws.spec import ImplantProblem
    return ImplantProblem.model_validate(d)


def _set(d, path, value):
    node = d
    parts = path.split(".")
    for k in parts[:-1]:
        node = node[int(k)] if isinstance(node, list) else node[k]
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value
    return d


# ------------------------------------------------------------------------------------------------ exports
def test_exports_from_simws_spec():
    import simws.spec as S
    for name in ("Atom", "Layer", "Target", "Beam", "ImplantModels", "Statistics", "ImplantProblem", "family"):
        assert hasattr(S, name), name
    # A7.1: nothing in the device description changes
    for name in ("SPEC_VERSION", "Problem", "DeviceSpec", "ModelSpec", "Numerics", "SmallSignal", "Bias"):
        assert hasattr(S, name), name
    from simws.spec import implant
    assert implant.ImplantProblem is S.ImplantProblem


def test_defaults():
    p = _validate(_base())
    assert p.spec_version == "1.0" and p.kind == "implant"
    assert p.models.potential == "zbl" and p.models.electronic_stopping == "lindhard_scharff"
    assert p.models.recoils is False
    assert p.statistics.n_ions == 20000 and p.statistics.seed == 1
    assert p.depth_bin_um == 0.005
    assert p.beam.tilt_deg == 0.0 and p.beam.dose_cm2 is None
    assert p.beam.ion.surface_binding_eV == 0.0 and p.beam.ion.bulk_binding_eV == 0.0
    assert p.target.atoms[0].bulk_binding_eV == 0.0
    assert p.target.structure == "amorphous"


def test_target_thickness_property_is_sum_of_layers():
    d = _base()
    d["target"]["layers"] = [{"name": "a", "thickness_um": 0.03, "density_cm3": {"Si": 4.994e22}},
                             {"name": "b", "thickness_um": 1.97, "density_cm3": {"Si": 4.994e22}},
                             {"name": "c", "thickness_um": 0.5, "density_cm3": {"Si": 4.994e22}}]
    p = _validate(d)
    assert p.target.thickness_um == pytest.approx(0.03 + 1.97 + 0.5, rel=1e-15)


def test_classes_construct_directly():
    from simws.spec import Atom, Beam, ImplantModels, ImplantProblem, Layer, Statistics, Target
    si = Atom(**SI)
    t = Target(atoms=[si], layers=[Layer(name="si", thickness_um=1.0, density_cm3={"Si": 5e22})])
    b = Beam(ion=Atom(**P_ION), energy_keV=50.0, tilt_deg=7.0, dose_cm2=1e15)
    p = ImplantProblem(beam=b, target=t, models=ImplantModels(potential="kr_c"), statistics=Statistics(n_ions=100))
    assert p.models.potential == "kr_c" and p.statistics.n_ions == 100 and p.beam.dose_cm2 == 1e15


# ------------------------------------------------------------------------------------------------ validation rules
BAD = [
    # Atom (ion and target atom)
    ("beam.ion.symbol", ""),
    ("beam.ion.Z", 0),
    ("beam.ion.Z", 119),
    ("target.atoms.0.Z", 0),
    ("target.atoms.0.Z", 119),
    ("beam.ion.mass_amu", 0.0),
    ("beam.ion.mass_amu", -1.0),
    ("target.atoms.0.mass_amu", 0.0),
    ("beam.ion.cutoff_eV", 0.0),
    ("target.atoms.0.cutoff_eV", -1.0),
    ("beam.ion.surface_binding_eV", -0.1),
    ("target.atoms.0.bulk_binding_eV", -0.1),
    ("target.atoms.0.symbol", ""),
    # Layer
    ("target.layers.0.thickness_um", 0.0),
    ("target.layers.0.thickness_um", -1.0),
    ("target.layers.0.density_cm3", {}),
    ("target.layers.0.density_cm3", {"Si": 0.0}),
    ("target.layers.0.density_cm3", {"Si": -4.994e22}),
    # Target
    ("target.atoms", []),
    ("target.layers", []),
    ("target.structure", "crystalline"),
    ("target.layers.0.density_cm3", {"Ge": 4.4e22}),                       # symbol not in atoms
    # Beam
    ("beam.energy_keV", 0.0),
    ("beam.energy_keV", -5.0),
    ("beam.tilt_deg", -0.001),
    ("beam.tilt_deg", 90.0),
    ("beam.tilt_deg", 120.0),
    ("beam.dose_cm2", 0.0),
    ("beam.dose_cm2", -1e14),
    # models
    ("models", {"potential": "lenz_jensen"}),
    ("models", {"electronic_stopping": "biersack_haggmark"}),
    # statistics
    ("statistics", {"n_ions": 99}),
    ("statistics", {"n_ions": 2_000_001}),
    ("statistics", {"seed": -1}),
    # problem
    ("kind", "device"),
    ("depth_bin_um", 0.0),
    ("depth_bin_um", -0.005),
]


@pytest.mark.parametrize("path,value", BAD, ids=[f"{p}={v!r}" for p, v in BAD])
def test_each_rule_raises_validation_error(path, value):
    d = _set(_base(), path, value)
    with pytest.raises(pydantic.ValidationError):
        _validate(d)
    assert issubclass(pydantic.ValidationError, ValueError)


def test_duplicate_atom_symbols_refused():
    d = _base()
    d["target"]["atoms"] = [dict(SI), dict(SI)]
    with pytest.raises(pydantic.ValidationError):
        _validate(d)


def test_duplicate_layer_names_refused():
    d = _base()
    lay = d["target"]["layers"][0]
    d["target"]["layers"] = [dict(lay), dict(lay)]
    with pytest.raises(pydantic.ValidationError):
        _validate(d)


GOOD = [
    ("beam.ion.Z", 1), ("beam.ion.Z", 118), ("beam.tilt_deg", 0.0), ("beam.tilt_deg", 89.999),
    ("beam.dose_cm2", None), ("beam.dose_cm2", 1e15), ("beam.ion.surface_binding_eV", 0.0),
    ("target.atoms.0.bulk_binding_eV", 0.0), ("statistics", {"n_ions": 100}), ("statistics", {"n_ions": 2_000_000}),
    ("statistics", {"seed": 0}), ("models", {"potential": "kr_c"}), ("models", {"potential": "moliere"}),
    ("models", {"recoils": True}),                    # allowed by the description; an engine may refuse it
]


@pytest.mark.parametrize("path,value", GOOD, ids=[f"{p}={v!r}" for p, v in GOOD])
def test_boundary_values_accepted(path, value):
    _validate(_set(_base(), path, value))


def test_bins_limit_5000():
    """target thickness / depth_bin_um must not exceed 5000 (binary-exact numbers, so the boundary is exact)."""
    b = 2.0 ** -10
    ok = _base()
    ok["target"]["layers"][0]["thickness_um"] = 5000 * b          # = 4.8828125 exactly: 5000 bins
    ok["depth_bin_um"] = b
    _validate(ok)
    bad = copy.deepcopy(ok)
    bad["target"]["layers"][0]["thickness_um"] = 5.0               # 5120 bins
    with pytest.raises(pydantic.ValidationError):
        _validate(bad)
    # the rule is on the total thickness, not on a single layer
    two = copy.deepcopy(ok)
    two["target"]["layers"] = [{"name": "a", "thickness_um": 2.5, "density_cm3": {"Si": 4.994e22}},
                               {"name": "b", "thickness_um": 2.5, "density_cm3": {"Si": 4.994e22}}]
    with pytest.raises(pydantic.ValidationError):
        _validate(two)


@pytest.mark.parametrize("where", ["", "beam", "beam.ion", "target", "target.atoms.0", "target.layers.0",
                                   "models", "statistics"])
def test_unknown_fields_forbidden(where):
    d = _base()
    d.setdefault("models", {})
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
    from simws.spec import ImplantProblem
    a, b = _validate(_base()), _validate(_base())
    pid = a.problem_id()
    assert isinstance(pid, str) and pid == b.problem_id()
    assert ImplantProblem.model_validate(a.model_dump(mode="json")).problem_id() == pid
    assert ImplantProblem.model_validate_json(a.model_dump_json()).problem_id() == pid
    assert a.with_changes({}).problem_id() == pid
    # "same recipe as Problem.problem_id": the content hash of the JSON dump (contract 3 / schema.Problem)
    assert pid == job_id(a.model_dump(mode="json"))
    # defaults written out give the same id
    full = a.model_dump(mode="json")
    assert _validate(full).problem_id() == pid


@pytest.mark.parametrize("change", [
    {"beam.energy_keV": 50.0}, {"beam.tilt_deg": 7.0}, {"beam.dose_cm2": 1e15}, {"beam.ion.cutoff_eV": 5.0},
    {"statistics.seed": 2}, {"statistics.n_ions": 2000}, {"models.potential": "kr_c"}, {"depth_bin_um": 0.002},
    {"target.layers.0.thickness_um": 1.0}, {"target.layers.0.density_cm3": {"Si": 5.0e22}},
    {"target.atoms.0.surface_binding_eV": 4.0},
])
def test_problem_id_sensitive(change):
    p = _validate(_base())
    assert p.with_changes(change).problem_id() != p.problem_id()


def test_with_changes_returns_validated_copy():
    from simws.spec import ImplantProblem
    p = _validate(_base())
    q = p.with_changes({"beam.energy_keV": 50.0, "target.layers.0.thickness_um": 1.0, "statistics.seed": 3})
    assert isinstance(q, ImplantProblem)
    assert q.beam.energy_keV == 50.0 and q.target.layers[0].thickness_um == 1.0 and q.statistics.seed == 3
    assert p.beam.energy_keV == 100.0 and p.target.layers[0].thickness_um == 2.0 and p.statistics.seed == 1
    with pytest.raises(ValueError):
        p.with_changes({"beam.energy_keV": -1.0})
    with pytest.raises(ValueError):
        p.with_changes({"beam.tilt_deg": 90.0})
    with pytest.raises(ValueError):
        p.with_changes({"statistics.n_ions": 10})


# ------------------------------------------------------------------------------------------------ family
def test_family():
    from simws.spec import family
    from simws.spec.library import npn_hbt_2d, pn_diode_1d
    assert family(_validate(_base())) == "implant"
    assert family(pn_diode_1d()) == "device"
    assert family(npn_hbt_2d()) == "device"


def test_device_description_unchanged():
    """A7.1: no existing problem id changes. The recipe of contract 3 is unchanged (content hash of the dump)."""
    from simws.hub.canonical import job_id
    from simws.spec.library import npn_hbt_2d, pn_diode_1d
    for p in (pn_diode_1d(), npn_hbt_2d()):
        assert p.problem_id() == job_id(p.model_dump(mode="json"))
        assert "kind" not in p.model_dump(mode="json") or p.model_dump(mode="json").get("kind") != "implant"


# ------------------------------------------------------------------------------------------------ library (A7.2)
MASS = {"B": 11.009305, "P": 30.973762, "As": 74.921595}           # A7.2 table (AME2020, rounded)
ZNUM = {"B": 5, "P": 15, "As": 33}                                  # atomic numbers (periodic table)


@pytest.mark.parametrize("ion", ["B", "P", "As"])
def test_implant_si_ions(ion):
    from simws.spec import ImplantProblem
    from simws.spec.library import implant_si
    p = implant_si(ion=ion)
    assert isinstance(p, ImplantProblem) and p.kind == "implant"
    a = p.beam.ion
    assert a.symbol == ion and a.Z == ZNUM[ion] and a.mass_amu == MASS[ion]
    assert a.cutoff_eV == 1.5 and a.surface_binding_eV == 0.0          # modelling choice, A7.2
    assert a.bulk_binding_eV == 0.0


def test_implant_si_defaults_and_target():
    from simws.spec.library import implant_si
    p = implant_si()
    assert p.beam.ion.symbol == "P" and p.beam.energy_keV == 100.0 and p.beam.tilt_deg == 0.0
    assert p.beam.dose_cm2 is None
    assert p.statistics.n_ions == 20000 and p.statistics.seed == 1
    assert p.models.potential == "zbl" and p.models.recoils is False
    t = p.target
    assert t.structure == "amorphous"
    assert len(t.layers) == 1 and t.layers[0].name == "si" and t.layers[0].thickness_um == 2.0
    assert t.thickness_um == 2.0
    assert t.layers[0].density_cm3 == {"Si": 4.994e22}
    assert len(t.atoms) == 1
    si = t.atoms[0]
    assert si.symbol == "Si" and si.Z == 14 and si.mass_amu == 28.085
    assert si.surface_binding_eV == 4.72 and si.cutoff_eV == 1.5 and si.bulk_binding_eV == 0.0


def test_silicon_density_matches_its_stated_source():
    """4.994e22 = 8 atoms per cubic cell of edge 0.5431020511 nm (CODATA 2018), rounded to 4 digits."""
    from simws.spec.library import implant_si
    a_cm = 0.5431020511e-7
    n = 8.0 / a_cm ** 3
    assert abs(implant_si().target.layers[0].density_cm3["Si"] / n - 1) < 1e-4


def test_implant_si_arguments():
    from simws.spec.library import implant_si
    p = implant_si("B", 50.0, tilt_deg=7.0, dose_cm2=1e15, thickness_um=1.0, n_ions=2000, seed=5)
    assert p.beam.ion.symbol == "B" and p.beam.energy_keV == 50.0 and p.beam.tilt_deg == 7.0
    assert p.beam.dose_cm2 == 1e15
    assert p.target.layers[0].thickness_um == 1.0 and p.target.thickness_um == 1.0
    assert p.statistics.n_ions == 2000 and p.statistics.seed == 5
    q = implant_si(ion="As", energy_keV=80.0)
    assert q.beam.ion.symbol == "As" and q.beam.energy_keV == 80.0
    # the same arguments give the same problem
    assert implant_si("P", 50).problem_id() == implant_si(ion="P", energy_keV=50.0).problem_id()
    assert implant_si("P", 50).problem_id() != implant_si("P", 50, seed=2).problem_id()


@pytest.mark.parametrize("bad", ["Sb", "p", "", "Ge"])
def test_implant_si_unknown_ion(bad):
    from simws.spec.library import implant_si
    with pytest.raises(ValueError) as ei:
        implant_si(ion=bad)
    msg = str(ei.value)
    for known in ("B", "P", "As"):
        assert known in msg, msg


def test_implant_si_is_valid_for_every_default_combination():
    """The library problems of A7.7 validate (R10 to R16 arguments)."""
    from simws.spec.library import implant_si
    for ion in ("B", "P", "As"):
        for e in (50.0, 100.0):
            implant_si(ion, e)
    implant_si("P", 50, tilt_deg=60.0)
    implant_si("P", 100, thickness_um=0.05, n_ions=2000)
    implant_si("P", 100, thickness_um=0.005, n_ions=2000)
    assert math.isclose(implant_si("P", 100, thickness_um=0.005).target.thickness_um, 0.005)
