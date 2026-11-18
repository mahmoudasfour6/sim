"""Contract amendment A9.1 / A9.2: the circuit description (simws.spec.circuit, exported from simws.spec), family()
for the fourth family and the library problems SHOCKLEY_VA, rc_lowpass, divider and diode_iv. Written from
docs/ARCHITECTURE_V1.md (A9, with the shared conventions of A7 / A8) only."""
import copy
import re

import pydantic
import pytest

VA_SRC = """`include "disciplines.vams"
module mydiode(a, c);
  inout a, c;
  electrical a, c;
  parameter real is = 1e-14;
  analog I(a, c) <+ is * (limexp(V(a, c) / $vt) - 1);
endmodule
"""


def _base() -> dict:
    """A minimal valid CircuitProblem as a dict (built by hand from the A9.1 field list)."""
    return {"elements": [{"name": "vin", "kind": "vsource", "nodes": ["in", "0"], "value": 1.0},
                         {"name": "r1", "kind": "resistor", "nodes": ["in", "out"], "value": 1e3},
                         {"name": "c1", "kind": "capacitor", "nodes": ["out", "0"], "value": 1e-9}],
            "analysis": {"kind": "op"},
            "probes": ["v(out)"]}


def _full() -> dict:
    """A valid problem using every element kind and model kind (test data)."""
    d = _base()
    d["elements"] += [
        {"name": "l1", "kind": "inductor", "nodes": ["out", "x"], "value": 1e-6},
        {"name": "rx", "kind": "resistor", "nodes": ["x", "0"], "value": 50.0},
        {"name": "i1", "kind": "isource", "nodes": ["0", "x"], "value": 1e-6},
        {"name": "d1", "kind": "diode", "nodes": ["x", "0"], "model": "dmod", "params": {"area": 2.0}},
        {"name": "q1", "kind": "bjt", "nodes": ["out", "x", "0"], "model": "qn"},
        {"name": "q2", "kind": "bjt", "nodes": ["out", "x", "0", "0"], "model": "qp"},
        {"name": "n1", "kind": "va", "nodes": ["x", "0"], "model": "vam"},
        {"name": "x1", "kind": "subckt", "nodes": ["out", "x", "0"], "model": "npn13G2", "params": {"Nx": 1}},
    ]
    d["models"] = [{"name": "dmod", "kind": "diode", "params": {"is": 1e-14}},
                   {"name": "qn", "kind": "npn", "params": {"bf": 100.0}},
                   {"name": "qp", "kind": "pnp"},
                   {"name": "vam", "kind": "va", "params": {"is": 1e-14},
                    "va": {"module": "mydiode", "source": VA_SRC}}]
    d["libraries"] = [{"path": "/nowhere/models.lib", "section": "typ"}]
    return d


def _validate(d):
    from simws.spec import CircuitProblem
    return CircuitProblem.model_validate(d)


def _set(d, path, value):
    node = d
    parts = path.split(".")
    for k in parts[:-1]:
        node = node[int(k)] if isinstance(node, list) else node.setdefault(k, {})
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
    del node[parts[-1] if not isinstance(node, list) else int(parts[-1])]
    return d


def _el(d, name):
    return next(e for e in d["elements"] if e["name"] == name)


def _raises(d):
    with pytest.raises(pydantic.ValidationError):
        _validate(d)


# ------------------------------------------------------------------------------------------------ exports
NAMES = ("Pulse", "Pwl", "Element", "VerilogA", "Model", "Library", "Op", "DcSweep", "AcSweep", "Transient",
         "CircuitNumerics", "CircuitProblem")


def test_exports_from_simws_spec():
    import simws.spec as S
    from simws.spec import circuit
    for name in NAMES + ("family",):
        assert hasattr(S, name), name
    for name in NAMES:
        assert getattr(circuit, name) is getattr(S, name), name
    for name in ("SeeProblem", "ImplantProblem", "Problem"):      # the other families are still exported
        assert hasattr(S, name), name


def test_name_and_node_patterns():
    from simws.spec import circuit
    assert circuit.NAME == r"^[A-Za-z][A-Za-z0-9_]*$"
    assert circuit.NODE == r"^[A-Za-z0-9_]+$"


def test_validation_error_is_value_error():
    assert issubclass(pydantic.ValidationError, ValueError)


def test_base_and_full_are_valid():
    _validate(_base())
    p = _validate(_full())
    assert [e.kind for e in p.elements] == ["vsource", "resistor", "capacitor", "inductor", "resistor", "isource",
                                            "diode", "bjt", "bjt", "va", "subckt"]
    assert [m.kind for m in p.models] == ["diode", "npn", "pnp", "va"]


def test_defaults():
    p = _validate(_base())
    assert p.spec_version == "1.0" and p.kind == "circuit" and p.title == "simws circuit"
    assert p.models == [] and p.libraries == []
    assert p.temperature_C == 27.0 and p.tnom_C == 27.0
    n = p.numerics
    assert (n.reltol, n.abstol_A, n.vntol_V, n.gmin_S) == (1e-6, 1e-15, 1e-9, 1e-15)
    e = p.elements[1]
    assert e.ac is None and e.pulse is None and e.pwl is None and e.model is None and e.params == {}
    assert p.elements[0].value == 1.0
    from simws.spec import CircuitNumerics, Library, Op, Pulse, Transient
    assert Op().kind == "op"
    assert Pulse(v1=0, v2=1, rise_s=1e-9, fall_s=1e-9, width_s=1e-6, period_s=2e-6).delay_s == 0.0
    assert Library(path="x.lib").section is None
    assert Transient(step_s=1e-9, stop_s=1e-6).max_step_s is None
    assert CircuitNumerics().reltol == 1e-6


def test_source_value_optional():
    d = _base()
    del d["elements"][0]["value"]
    assert _validate(d).elements[0].value is None
    d["elements"][0]["value"] = -2.5                            # a DC value of any sign
    assert _validate(d).elements[0].value == -2.5


@pytest.mark.parametrize("analysis,cls", [
    ({"kind": "op"}, "Op"),
    ({"kind": "dc", "source": "vin", "start": 0.0, "stop": 1.0, "step": 0.1}, "DcSweep"),
    ({"kind": "ac", "points_per_decade": 10, "f_start_Hz": 10.0, "f_stop_Hz": 1e6}, "AcSweep"),
    ({"kind": "tran", "step_s": 1e-9, "stop_s": 1e-6}, "Transient"),
])
def test_analysis_chosen_by_kind(analysis, cls):
    import simws.spec as S
    p = _validate(_set(_base(), "analysis", analysis))
    assert isinstance(p.analysis, getattr(S, cls))


def test_classes_construct_directly():
    from simws.spec import (AcSweep, CircuitNumerics, CircuitProblem, Element, Library, Model, Pulse, Pwl,
                            VerilogA)
    p = CircuitProblem(
        title="t",
        elements=[Element(name="vin", kind="vsource", nodes=["in", "0"], ac=1.0,
                          pulse=Pulse(v1=0.0, v2=1.0, delay_s=1e-9, rise_s=1e-12, fall_s=1e-12, width_s=1e-6,
                                      period_s=2e-6)),
                  Element(name="i2", kind="isource", nodes=["0", "in"], pwl=Pwl(points=[(0.0, 0.0), (1e-6, 1e-3)])),
                  Element(name="n1", kind="va", nodes=["in", "0"], model="m1"),
                  Element(name="x1", kind="subckt", nodes=["in"], model="sub")],
        models=[Model(name="m1", kind="va", va=VerilogA(module="mydiode", source=VA_SRC))],
        libraries=[Library(path="~/lib.lib")],
        analysis=AcSweep(points_per_decade=5, f_start_Hz=1.0, f_stop_Hz=1e3),
        probes=["v(in)", "i(vin)"], temperature_C=50.0, tnom_C=25.0,
        numerics=CircuitNumerics(reltol=1e-4, abstol_A=1e-12, vntol_V=1e-6, gmin_S=0.0))
    assert p.elements[0].pulse.delay_s == 1e-9 and p.elements[1].pwl.points[1][1] == 1e-3
    assert p.models[0].va.module == "mydiode" and p.numerics.gmin_S == 0.0


# ------------------------------------------------------------------------------------------------ validation rules
BAD = [
    # Element names / nodes / kinds
    ("elements.1.name", "1r"), ("elements.1.name", ""), ("elements.1.name", "r-1"), ("elements.1.name", "r 1"),
    ("elements.1.name", "_r1"), ("elements.1.name", "r.1"),
    ("elements.1.nodes", ["in", "o-ut"]), ("elements.1.nodes", ["in", ""]), ("elements.1.nodes", ["in", "o ut"]),
    ("elements.1.nodes", ["in", "a.b"]),
    ("elements.1.kind", "transistor"), ("elements.1.kind", "Resistor"), ("elements.1.kind", "mosfet"),
    # node counts (2 for two-terminal kinds)
    ("elements.1.nodes", ["in"]), ("elements.1.nodes", ["in", "out", "0"]), ("elements.1.nodes", []),
    ("elements.2.nodes", ["out"]), ("elements.2.nodes", ["out", "0", "in"]),
    ("elements.0.nodes", ["in"]), ("elements.0.nodes", ["in", "0", "out"]),
    # value rules
    ("elements.1.value", None), ("elements.1.value", 0.0), ("elements.1.value", -1e3),
    ("elements.2.value", 0.0), ("elements.2.value", -1e-9), ("elements.2.value", None),
    # source-only fields on a resistor
    ("elements.1.ac", 1.0),
    ("elements.1.pulse", {"v1": 0, "v2": 1, "rise_s": 1e-9, "fall_s": 1e-9, "width_s": 1e-6, "period_s": 2e-6}),
    ("elements.1.pwl", {"points": [[0.0, 0.0], [1e-6, 1.0]]}),
    ("elements.2.ac", 1.0),
    # model forbidden for non-model kinds
    ("elements.1.model", "dmod"), ("elements.0.model", "dmod"), ("elements.2.model", "dmod"),
    # params values must be numbers
    ("elements.1.params", {"tc1": "hot"}),
    # problem level
    ("kind", "device"), ("kind", "see"), ("elements", []), ("probes", []),
    ("temperature_C", -273.15), ("temperature_C", -300.0), ("tnom_C", -273.15), ("tnom_C", -1000.0),
    # numerics
    ("numerics.reltol", 1e-10), ("numerics.reltol", 0.02), ("numerics.reltol", 0.0), ("numerics.reltol", -1e-6),
    ("numerics.abstol_A", 0.0), ("numerics.abstol_A", -1e-15), ("numerics.vntol_V", 0.0),
    ("numerics.vntol_V", -1e-9), ("numerics.gmin_S", -1e-15),
]


@pytest.mark.parametrize("path,value", BAD, ids=[f"{p}={v!r}" for p, v in BAD])
def test_each_rule_raises_validation_error(path, value):
    _raises(_set(_base(), path, value))


GOOD = [
    ("temperature_C", -273.0), ("temperature_C", 400.0), ("tnom_C", -273.14), ("tnom_C", 125.0),
    ("numerics.reltol", 1e-9), ("numerics.reltol", 1e-2), ("numerics.gmin_S", 0.0), ("numerics.abstol_A", 1e-30),
    ("numerics.vntol_V", 1e-12), ("elements.1.name", "R1_x9"), ("elements.1.name", "r"),
    ("elements.1.nodes", ["IN", "n_1"]), ("elements.1.nodes", ["in", "123"]), ("title", "another title"),
    ("elements.0.ac", 1.0), ("elements.0.ac", 0.5), ("elements.1.params", {"tc1": 1e-3}),
]


@pytest.mark.parametrize("path,value", GOOD, ids=[f"{p}={v!r}" for p, v in GOOD])
def test_boundary_values_accepted(path, value):
    _validate(_set(_base(), path, value))


@pytest.mark.parametrize("path", ["elements", "analysis", "probes", "elements.0.name", "elements.0.kind",
                                  "elements.0.nodes"])
def test_required_fields(path):
    _raises(_del(_base(), path))


# ------------------------------------------------------------------------------------------------ node counts per kind
TWO = ["resistor", "capacitor", "inductor", "vsource", "isource", "diode"]


def _kind_case(kind, nodes):
    d = _full()
    el = {"name": "zz", "kind": kind, "nodes": nodes}
    if kind in ("resistor", "capacitor", "inductor"):
        el["value"] = 1.0
    if kind == "diode":
        el["model"] = "dmod"
    if kind == "bjt":
        el["model"] = "qn"
    if kind == "va":
        el["model"] = "vam"
    if kind == "subckt":
        el["model"] = "sub"
    d["elements"].append(el)
    return d


@pytest.mark.parametrize("kind", TWO)
def test_two_terminal_node_counts(kind):
    _validate(_kind_case(kind, ["out", "0"]))
    for nodes in (["out"], ["out", "0", "x"], []):
        _raises(_kind_case(kind, nodes))


def test_bjt_node_counts():
    _validate(_kind_case("bjt", ["out", "x", "0"]))
    _validate(_kind_case("bjt", ["out", "x", "0", "0"]))
    for nodes in (["out", "x"], ["out", "x", "0", "0", "in"], ["out"]):
        _raises(_kind_case("bjt", nodes))


@pytest.mark.parametrize("kind", ["va", "subckt"])
def test_va_and_subckt_node_counts(kind):
    _validate(_kind_case(kind, ["out"]))
    _validate(_kind_case(kind, ["out", "x", "0", "in", "out"]))
    _raises(_kind_case(kind, []))


# ------------------------------------------------------------------------------------------------ value / sources / model
@pytest.mark.parametrize("kind", ["resistor", "capacitor", "inductor"])
def test_passive_value_required_positive(kind):
    d = _kind_case(kind, ["out", "0"])
    d["elements"][-1]["value"] = 2.0
    _validate(d)
    for v in (None, 0.0, -2.0):
        d2 = copy.deepcopy(d)
        if v is None:
            del d2["elements"][-1]["value"]
        else:
            d2["elements"][-1]["value"] = v
        _raises(d2)


@pytest.mark.parametrize("kind", ["diode", "bjt", "va", "subckt"])
def test_value_forbidden_for_model_kinds(kind):
    d = _kind_case(kind, ["out", "x", "0"] if kind == "bjt" else ["out", "0"])
    _validate(d)
    d["elements"][-1]["value"] = 1.0
    _raises(d)


@pytest.mark.parametrize("kind", ["resistor", "capacitor", "inductor", "diode", "bjt", "va", "subckt"])
@pytest.mark.parametrize("field,value", [
    ("ac", 1.0),
    ("pulse", {"v1": 0, "v2": 1, "rise_s": 1e-9, "fall_s": 1e-9, "width_s": 1e-6, "period_s": 2e-6}),
    ("pwl", {"points": [[0.0, 0.0], [1e-6, 1.0]]})])
def test_source_fields_forbidden_for_other_kinds(kind, field, value):
    d = _kind_case(kind, ["out", "x", "0"] if kind == "bjt" else ["out", "0"])
    d["elements"][-1][field] = value
    _raises(d)


PULSE = {"v1": 0.0, "v2": 1.0, "rise_s": 1e-12, "fall_s": 1e-12, "width_s": 1.0, "period_s": 2.0}
PWL = {"points": [[0.0, 0.0], [1e-6, 1.0], [2e-6, 0.5]]}


@pytest.mark.parametrize("src", [0, 5])                         # vsource vin, isource i1
def test_sources_take_ac_pulse_pwl(src):
    d = _full()
    for f, v in (("ac", 1.0), ("pulse", PULSE), ("pwl", PWL)):
        dd = copy.deepcopy(d)
        dd["elements"][src][f] = v
        _validate(dd)
    dd = copy.deepcopy(d)
    dd["elements"][src].update(ac=1.0, pulse=PULSE)
    _validate(dd)
    dd = copy.deepcopy(d)
    dd["elements"][src].update(pulse=PULSE, pwl=PWL)                 # not together
    _raises(dd)


@pytest.mark.parametrize("kind", ["resistor", "capacitor", "inductor", "vsource", "isource"])
def test_model_forbidden(kind):
    d = _kind_case(kind, ["out", "0"])
    d["elements"][-1]["model"] = "dmod"
    _raises(d)


@pytest.mark.parametrize("kind", ["diode", "bjt", "va", "subckt"])
def test_model_required(kind):
    d = _kind_case(kind, ["out", "x", "0"] if kind == "bjt" else ["out", "0"])
    del d["elements"][-1]["model"]
    _raises(d)


@pytest.mark.parametrize("kind,model,ok", [
    ("diode", "dmod", True), ("diode", "qn", False), ("diode", "qp", False), ("diode", "vam", False),
    ("diode", "nomodel", False),
    ("bjt", "qn", True), ("bjt", "qp", True), ("bjt", "dmod", False), ("bjt", "vam", False),
    ("bjt", "nomodel", False),
    ("va", "vam", True), ("va", "dmod", False), ("va", "qn", False), ("va", "nomodel", False),
])
def test_model_kind_compatibility(kind, model, ok):
    d = _kind_case(kind, ["out", "x", "0"] if kind == "bjt" else ["out", "0"])
    d["elements"][-1]["model"] = model
    if ok:
        _validate(d)
    else:
        _raises(d)


def test_model_reference_case_insensitive():
    d = _full()
    _el(d, "d1")["model"] = "DMOD"
    _el(d, "q1")["model"] = "Qn"
    _el(d, "n1")["model"] = "VAM"
    _validate(d)


def test_subckt_needs_a_library():
    d = _full()
    _validate(d)
    d["libraries"] = []
    _raises(d)
    # the subcircuit name is not a Model: it is not looked up among the models
    d["libraries"] = [{"path": "a.lib"}]
    _el(d, "x1")["model"] = "anything_from_the_library"
    _validate(d)


def test_library_path_not_checked_and_not_expanded_at_validation():
    d = _full()
    d["libraries"] = [{"path": "~/does/not/exist.lib", "section": "hbt_typ"}, {"path": "/also/missing.inc"}]
    p = _validate(d)
    assert p.libraries[0].path == "~/does/not/exist.lib"                    # "~" is expanded when it is run
    assert p.libraries[0].section == "hbt_typ" and p.libraries[1].section is None


def test_names_unique_case_insensitive():
    d = _base()
    d["elements"].append({"name": "R1", "kind": "resistor", "nodes": ["out", "0"], "value": 5.0})
    _raises(d)
    d["elements"][-1]["name"] = "r2"
    _validate(d)
    d = _base()
    d["elements"].append({"name": "vin", "kind": "capacitor", "nodes": ["out", "0"], "value": 5.0})
    _raises(d)


def test_model_names_unique_case_insensitive():
    d = _full()
    d["models"].append({"name": "DMod", "kind": "diode"})
    _raises(d)
    d["models"][-1]["name"] = "dmod2"
    _validate(d)


def test_ground_node_required():
    d = _base()
    for e in d["elements"]:
        e["nodes"] = ["gnd" if n == "0" else n for n in e["nodes"]]
    _raises(d)
    d = _base()
    d["elements"] = [{"name": "r1", "kind": "resistor", "nodes": ["a", "b"], "value": 1.0},
                     {"name": "v1", "kind": "vsource", "nodes": ["a", "b"]}]
    d["probes"] = ["v(a)"]
    _raises(d)
    d["elements"][0]["nodes"] = ["a", "0"]
    _validate(d)


# ------------------------------------------------------------------------------------------------ Pulse / Pwl
def _with_pulse(**over):
    d = _base()
    pulse = dict(PULSE)
    pulse.update(over)
    d["elements"][0]["pulse"] = {k: v for k, v in pulse.items() if v is not None}
    return d


@pytest.mark.parametrize("field,value", [
    ("delay_s", -1e-9), ("rise_s", 0.0), ("rise_s", -1e-12), ("fall_s", 0.0), ("fall_s", -1.0),
    ("width_s", 0.0), ("width_s", -1.0), ("period_s", 0.0), ("period_s", -2.0)])
def test_pulse_rules(field, value):
    _raises(_with_pulse(**{field: value}))


@pytest.mark.parametrize("field", ["v1", "v2", "rise_s", "fall_s", "width_s", "period_s"])
def test_pulse_required_fields(field):
    d = _with_pulse()
    del d["elements"][0]["pulse"][field]
    _raises(d)


def test_pulse_valid():
    p = _validate(_with_pulse(delay_s=0.0, v1=-1.0, v2=3.3))
    assert p.elements[0].pulse.v1 == -1.0 and p.elements[0].pulse.delay_s == 0.0
    _validate(_with_pulse(delay_s=5e-9))


@pytest.mark.parametrize("points", [
    [], [[0.0, 1.0]], [[1e-6, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 1.0]], [[-1e-9, 0.0], [1e-6, 1.0]],
    [[0.0, 0.0], [2e-6, 1.0], [1e-6, 0.5]], [[0.0, 0.0], [1e-6, 1.0], [1e-6, 2.0]]])
def test_pwl_rules(points):
    d = _base()
    d["elements"][0]["pwl"] = {"points": points}
    _raises(d)


@pytest.mark.parametrize("points", [[[0.0, 0.0], [1e-9, 1.0]], [[1e-9, 2.0], [2e-9, -1.0], [5e-9, 0.0]]])
def test_pwl_valid(points):
    d = _base()
    d["elements"][0]["pwl"] = {"points": points}
    p = _validate(d)
    assert [tuple(x) for x in p.elements[0].pwl.points] == [tuple(x) for x in points]


# ------------------------------------------------------------------------------------------------ VerilogA / Model / Library
@pytest.mark.parametrize("module,source", [
    ("other", VA_SRC),                           # the source does not contain "module other"
    ("mydiode", ""),                             # empty source
    ("1diode", VA_SRC.replace("module mydiode", "module 1diode")),   # not a NAME
    ("my-diode", VA_SRC.replace("module mydiode", "module my-diode")),
    ("", VA_SRC),
])
def test_veriloga_rules(module, source):
    d = _full()
    d["models"][3]["va"] = {"module": module, "source": source}
    _raises(d)


def test_veriloga_required_for_va_and_forbidden_otherwise():
    d = _full()
    del d["models"][3]["va"]
    _raises(d)
    d = _full()
    d["models"][0]["va"] = {"module": "mydiode", "source": VA_SRC}
    _raises(d)
    d = _full()
    d["models"][1]["va"] = {"module": "mydiode", "source": VA_SRC}
    _raises(d)


@pytest.mark.parametrize("path,value", [
    ("models.0.name", "1dmod"), ("models.0.name", ""), ("models.0.name", "d-mod"), ("models.0.kind", "nmos"),
    ("models.0.kind", "Diode"), ("models.1.kind", "bjt"), ("models.0.params", {"is": "big"}),
])
def test_model_rules(path, value):
    _raises(_set(_full(), path, value))


def test_library_path_required():
    d = _full()
    d["libraries"] = [{"section": "typ"}]
    _raises(d)


# ------------------------------------------------------------------------------------------------ probes
@pytest.mark.parametrize("probe", ["v(out)", "v(in)", "v(0)", "v(in,out)", "v(out,0)", "i(vin)", "v(OUT)",
                                   "i(VIN)", "v(In,OUT)"])
def test_probe_valid(probe):
    _validate(_set(_base(), "probes", [probe]))


@pytest.mark.parametrize("probe", ["v(nowhere)", "v(out,nowhere)", "i(r1)", "i(c1)", "i(nosrc)", "v()", "x(out)",
                                   "v(out", "vout", "v(a,b,c)", "i()", "i(vin,0)", "out", "", "v(in,out,0)"])
def test_probe_invalid(probe):
    _raises(_set(_base(), "probes", [probe]))


def test_probe_isource_current_not_allowed():
    """i() names a vsource; an isource is not a vsource."""
    d = _full()
    d["probes"] = ["i(i1)"]
    _raises(d)
    d["probes"] = ["i(vin)", "v(x)", "v(out,x)"]
    _validate(d)


# ------------------------------------------------------------------------------------------------ analyses
def _dc(**k):
    a = {"kind": "dc", "source": "vin", "start": 0.0, "stop": 1.0, "step": 0.1}
    a.update(k)
    return _set(_base(), "analysis", a)


@pytest.mark.parametrize("over", [
    {"step": 0.0}, {"step": -0.1}, {"start": 1.0, "stop": 0.0, "step": 0.1}, {"source": "r1"}, {"source": "c1"},
    {"source": "nosrc"}, {"start": 0.0, "stop": 1.0, "step": 1e-6},             # 1000001 points
    {"start": 0.0, "stop": 1.0, "step": 0.99e-5},                                # 101011 points
    {"start": 0.5, "stop": 0.5, "step": 0.1}, {"start": 0.5, "stop": 0.5, "step": -0.1},
])
def test_dc_sweep_rules(over):
    _raises(_dc(**over))


@pytest.mark.parametrize("over", [
    {}, {"start": 1.0, "stop": 0.0, "step": -0.1}, {"start": 0.0, "stop": 1.0, "step": 1.0},       # 2 points
    {"start": 0.0, "stop": 1.0, "step": 1e-5},                                   # 100001 points
    {"start": -0.35, "stop": -1.02, "step": -0.005}, {"source": "VIN"},
])
def test_dc_sweep_valid(over):
    p = _validate(_dc(**over))
    assert p.analysis.kind == "dc"


def test_dc_sweep_of_isource():
    d = _full()
    d["analysis"] = {"kind": "dc", "source": "i1", "start": 0.0, "stop": 1e-3, "step": 1e-4}
    _validate(d)


def test_dc_sweep_fields_required():
    for f in ("source", "start", "stop", "step"):
        d = _dc()
        del d["analysis"][f]
        _raises(d)


def _ac(**k):
    a = {"kind": "ac", "points_per_decade": 10, "f_start_Hz": 10.0, "f_stop_Hz": 1e8}
    a.update(k)
    return _set(_base(), "analysis", a)


@pytest.mark.parametrize("over", [{"points_per_decade": 0}, {"points_per_decade": 1001}, {"points_per_decade": -1},
                                  {"f_start_Hz": 0.0}, {"f_start_Hz": -1.0}, {"f_start_Hz": 1e8},
                                  {"f_start_Hz": 1e9}])
def test_ac_rules(over):
    _raises(_ac(**over))


@pytest.mark.parametrize("over", [{}, {"points_per_decade": 1}, {"points_per_decade": 1000},
                                  {"f_start_Hz": 1e-3, "f_stop_Hz": 1.1e-3}])
def test_ac_valid(over):
    _validate(_ac(**over))


def _tran(**k):
    a = {"kind": "tran", "step_s": 1e-8, "stop_s": 1e-5}
    a.update(k)
    return _set(_base(), "analysis", a)


@pytest.mark.parametrize("over", [{"step_s": 0.0}, {"step_s": -1e-8}, {"step_s": 1e-5}, {"step_s": 2e-5},
                                  {"max_step_s": 0.0}, {"max_step_s": -1e-9}, {"stop_s": 0.0}])
def test_tran_rules(over):
    _raises(_tran(**over))


@pytest.mark.parametrize("over", [{}, {"max_step_s": 1e-8}, {"max_step_s": 1e-3}, {"step_s": 9.9e-6}])
def test_tran_valid(over):
    _validate(_tran(**over))


# ------------------------------------------------------------------------------------------------ unknown fields
@pytest.mark.parametrize("where", ["", "elements.0", "models.0", "models.3.va", "libraries.0", "analysis",
                                   "numerics", "elements.0.pulse", "elements.5.pwl"])
def test_unknown_fields_forbidden(where):
    d = _full()
    d["numerics"] = {}
    d["elements"][0]["pulse"] = dict(PULSE)
    d["elements"][5]["pwl"] = copy.deepcopy(PWL)
    _validate(copy.deepcopy(d))
    node = d
    for k in [w for w in where.split(".") if w]:
        node = node[int(k)] if isinstance(node, list) else node[k]
    node["colour"] = "red"
    _raises(d)


@pytest.mark.parametrize("analysis", [
    {"kind": "op", "source": "vin"},
    {"kind": "dc", "source": "vin", "start": 0.0, "stop": 1.0, "step": 0.1, "points": 3},
    {"kind": "ac", "points_per_decade": 10, "f_start_Hz": 10.0, "f_stop_Hz": 1e8, "phase": 0.0},
    {"kind": "tran", "step_s": 1e-8, "stop_s": 1e-5, "start_s": 0.0},
    {"kind": "noise"}, {"kind": "pz"}])
def test_analysis_unknown_fields_and_kinds(analysis):
    _raises(_set(_base(), "analysis", analysis))


# ------------------------------------------------------------------------------------------------ problem_id / with_changes
def test_problem_id_stable():
    from simws.spec import CircuitProblem
    a, b = _validate(_full()), _validate(_full())
    pid = a.problem_id()
    assert isinstance(pid, str) and pid and pid == b.problem_id()
    assert CircuitProblem.model_validate(a.model_dump(mode="json")).problem_id() == pid
    assert CircuitProblem.model_validate_json(a.model_dump_json()).problem_id() == pid
    assert a.with_changes({}).problem_id() == pid


@pytest.mark.parametrize("change", [
    {"title": "other"}, {"elements.1.value": 2e3}, {"elements.2.value": 2e-9}, {"elements.0.value": 2.0},
    {"elements.0.ac": 1.0}, {"elements.1.nodes": ["in", "x"]}, {"elements.6.params": {"area": 3.0}},
    {"models.0.params": {"is": 2e-14}}, {"models.1.params": {"bf": 200.0}}, {"models.3.va.source": VA_SRC + "\n"},
    {"libraries.0.path": "/nowhere/other.lib"}, {"libraries.0.section": "fast"},
    {"probes": ["v(x)"]}, {"probes": ["v(out)", "i(vin)"]}, {"temperature_C": 50.0}, {"tnom_C": 25.0},
    {"numerics.reltol": 1e-5}, {"numerics.abstol_A": 1e-12}, {"numerics.vntol_V": 1e-6}, {"numerics.gmin_S": 0.0},
    {"analysis": {"kind": "dc", "source": "vin", "start": 0.0, "stop": 1.0, "step": 0.1}},
    {"analysis": {"kind": "tran", "step_s": 1e-9, "stop_s": 1e-6}},
])
def test_problem_id_sensitive(change):
    p = _validate(_full())
    assert p.with_changes(change).problem_id() != p.problem_id()


def test_problem_id_sensitive_to_analysis_values():
    base = _validate(_dc())
    for k, v in (("start", 0.1), ("stop", 0.9), ("step", 0.05)):
        assert base.with_changes({f"analysis.{k}": v}).problem_id() != base.problem_id()
    a = _validate(_ac())
    for k, v in (("points_per_decade", 20), ("f_start_Hz", 1.0), ("f_stop_Hz", 1e7)):
        assert a.with_changes({f"analysis.{k}": v}).problem_id() != a.problem_id()
    t = _validate(_tran())
    for k, v in (("step_s", 2e-8), ("stop_s", 2e-5), ("max_step_s", 1e-8)):
        assert t.with_changes({f"analysis.{k}": v}).problem_id() != t.problem_id()


def test_with_changes_returns_validated_copy():
    from simws.spec import CircuitProblem
    p = _validate(_full())
    q = p.with_changes({"elements.1.value": 2.2e3, "temperature_C": 85.0, "models.0.params": {"is": 3e-14}})
    assert isinstance(q, CircuitProblem)
    assert q.elements[1].value == 2.2e3 and q.temperature_C == 85.0 and q.models[0].params == {"is": 3e-14}
    assert p.elements[1].value == 1e3 and p.temperature_C == 27.0
    r = p.with_changes({"analysis": {"kind": "ac", "points_per_decade": 5, "f_start_Hz": 1.0, "f_stop_Hz": 1e3}})
    assert r.analysis.kind == "ac" and r.analysis.points_per_decade == 5
    for bad in ({"elements.1.value": -1.0}, {"temperature_C": -300.0}, {"probes": ["v(nowhere)"]},
                {"elements.1.name": "vin"}, {"libraries": []}, {"numerics.reltol": 1.0},
                {"elements.6.model": "qn"}):
        with pytest.raises(ValueError):
            p.with_changes(bad)


# ------------------------------------------------------------------------------------------------ family
def test_family():
    from simws.spec import family
    from simws.spec.library import implant_si, npn_hbt_2d, pn_diode_1d, see_si_slab
    assert family(_validate(_base())) == "circuit"
    assert family(_validate(_full())) == "circuit"
    assert family(implant_si()) == "implant"
    assert family(see_si_slab()) == "see"
    assert family(pn_diode_1d()) == "device" and family(npn_hbt_2d()) == "device"


# ------------------------------------------------------------------------------------------------ library (A9.2)
def _lib():
    from simws.spec import library
    return library


def _nows(s):
    return re.sub(r"\s+", "", s)


def test_shockley_va_text():
    src = _lib().SHOCKLEY_VA
    assert isinstance(src, str) and src.strip()
    assert re.search(r"\bmodule\s+shockley_diode\b", src), src
    # the equation as stated; "1" may be written as a real literal ("1.0")
    assert re.search(r"I\(a,c\)<\+is\*\(limexp\(V\(a,c\)/\(n\*\$vt\)\)-1(\.0*)?\)", _nows(src)), src
    m = re.search(r"parameter\s+(?:real\s+)?is\s*=\s*([-+0-9.eE]+)", src)
    assert m and float(m.group(1)) == 1e-14, src
    m = re.search(r"parameter\s+(?:real\s+)?n\s*=\s*([-+0-9.eE]+)", src)
    assert m and float(m.group(1)) == 1.0, src
    assert re.search(r"\bendmodule\b", src)


def test_shockley_va_is_a_valid_veriloga_description():
    from simws.spec import VerilogA
    v = VerilogA(module="shockley_diode", source=_lib().SHOCKLEY_VA)
    assert v.module == "shockley_diode"


def _by_name(p):
    return {e.name: e for e in p.elements}


def test_rc_lowpass_ac_default():
    from simws.spec import AcSweep, CircuitProblem
    p = _lib().rc_lowpass()
    assert isinstance(p, CircuitProblem)
    els = _by_name(p)
    assert sorted(els) == ["c1", "r1", "vin"]
    v, r, c = els["vin"], els["r1"], els["c1"]
    assert v.kind == "vsource" and [n.lower() for n in v.nodes] == ["in", "0"]
    assert v.ac == 1.0 and (v.value or 0.0) == 0.0 and v.pulse is None and v.pwl is None
    assert r.kind == "resistor" and [n.lower() for n in r.nodes] == ["in", "out"] and r.value == 1e3
    assert c.kind == "capacitor" and [n.lower() for n in c.nodes] == ["out", "0"] and c.value == 1e-9
    assert p.probes == ["v(out)"]
    a = p.analysis
    assert isinstance(a, AcSweep)
    assert (a.points_per_decade, a.f_start_Hz, a.f_stop_Hz) == (10, 10.0, 1e8)


def test_rc_lowpass_tran_default():
    from simws.spec import Transient
    p = _lib().rc_lowpass(analysis="tran")
    els = _by_name(p)
    assert sorted(els) == ["c1", "r1", "vin"]
    v = els["vin"]
    assert v.kind == "vsource" and [n.lower() for n in v.nodes] == ["in", "0"] and v.pwl is None
    pu = v.pulse
    assert pu is not None
    assert (pu.v1, pu.v2, pu.delay_s, pu.rise_s, pu.fall_s, pu.width_s, pu.period_s) == \
        (0.0, 1.0, 0.0, 1e-12, 1e-12, 1.0, 2.0)
    assert els["r1"].value == 1e3 and els["c1"].value == 1e-9
    assert p.probes == ["v(out)"]
    a = p.analysis
    assert isinstance(a, Transient)
    assert (a.step_s, a.stop_s, a.max_step_s) == (1e-8, 1e-5, 1e-8)


def test_rc_lowpass_arguments():
    lib = _lib()
    p = lib.rc_lowpass(R_ohm=2.2e3, C_F=4.7e-10, analysis="ac", points_per_decade=20, f_start_Hz=1.0, f_stop_Hz=1e6)
    els = _by_name(p)
    assert els["r1"].value == 2.2e3 and els["c1"].value == 4.7e-10
    assert (p.analysis.points_per_decade, p.analysis.f_start_Hz, p.analysis.f_stop_Hz) == (20, 1.0, 1e6)
    t = lib.rc_lowpass(analysis="tran", step_s=2e-9, stop_s=3e-6)
    assert (t.analysis.step_s, t.analysis.stop_s, t.analysis.max_step_s) == (2e-9, 3e-6, 2e-9)


def test_divider_default_and_arguments():
    from simws.spec import Op
    lib = _lib()
    p = lib.divider()
    els = _by_name(p)
    assert sorted(els) == ["r1", "r2", "vin"]
    assert els["vin"].kind == "vsource" and [n.lower() for n in els["vin"].nodes] == ["in", "0"]
    assert els["vin"].value == 1.0
    assert els["r1"].kind == "resistor" and [n.lower() for n in els["r1"].nodes] == ["in", "mid"]
    assert els["r1"].value == 1e3
    assert els["r2"].kind == "resistor" and [n.lower() for n in els["r2"].nodes] == ["mid", "0"]
    assert els["r2"].value == 3e3
    assert isinstance(p.analysis, Op)
    assert sorted(p.probes) == ["i(vin)", "v(mid)"]
    q = lib.divider(R1_ohm=10.0, R2_ohm=20.0, V=-3.0)
    e = _by_name(q)
    assert (e["r1"].value, e["r2"].value, e["vin"].value) == (10.0, 20.0, -3.0)


def test_diode_iv_builtin_default():
    from simws.spec import DcSweep
    p = _lib().diode_iv()
    els = _by_name(p)
    assert sorted(els) == ["d1", "vin"]
    v, d = els["vin"], els["d1"]
    assert v.kind == "vsource" and [n.lower() for n in v.nodes] == ["a", "0"]
    assert d.kind == "diode" and [n.lower() for n in d.nodes] == ["a", "0"] and d.model.lower() == "dmod"
    assert len(p.models) == 1
    m = p.models[0]
    assert m.name.lower() == "dmod" and m.kind == "diode" and m.va is None
    assert m.params == {"is": 1e-14, "n": 1.0}
    a = p.analysis
    assert isinstance(a, DcSweep) and a.source.lower() == "vin"
    assert (a.start, a.stop, a.step) == (0.30, 0.65, 0.05)
    assert p.probes == ["i(vin)"]
    assert p.temperature_C == 27.0 and p.tnom_C == 27.0
    assert p.libraries == []


def test_diode_iv_va():
    p = _lib().diode_iv(model="va")
    els = _by_name(p)
    assert sorted(els) == ["d1", "vin"]
    d = els["d1"]
    assert d.kind == "va" and [n.lower() for n in d.nodes] == ["a", "0"]
    assert len(p.models) == 1
    m = p.models[0]
    assert m.kind == "va" and m.name.lower() == d.model.lower()
    assert m.va is not None and m.va.module == "shockley_diode" and m.va.source == _lib().SHOCKLEY_VA
    assert m.params == {"is": 1e-14, "n": 1.0}
    assert p.analysis.kind == "dc" and p.analysis.source.lower() == "vin"
    assert (p.analysis.start, p.analysis.stop, p.analysis.step) == (0.30, 0.65, 0.05)
    assert p.probes == ["i(vin)"]


@pytest.mark.parametrize("model", ["builtin", "va"])
def test_diode_iv_arguments(model):
    p = _lib().diode_iv(IS_A=2.5e-15, N=1.2, start=0.1, stop=0.5, step=0.1, model=model, temperature_C=50.0,
                        tnom_C=40.0)
    assert p.models[0].params == {"is": 2.5e-15, "n": 1.2}
    assert (p.analysis.start, p.analysis.stop, p.analysis.step) == (0.1, 0.5, 0.1)
    assert p.temperature_C == 50.0 and p.tnom_C == 40.0


def test_library_problems_distinct_ids():
    lib = _lib()
    ids = {lib.divider().problem_id(), lib.rc_lowpass().problem_id(), lib.rc_lowpass(analysis="tran").problem_id(),
           lib.diode_iv().problem_id(), lib.diode_iv(model="va").problem_id()}
    assert len(ids) == 5
    assert lib.divider().problem_id() == lib.divider().problem_id()
