"""Contract amendment A9.3 / A9.4 (with the correction A9.10): CircuitResult, CircuitCapability, the registry with
four families and five engines, the planner, and the ngspice translation (build_netlist, parse_raw, availability,
the OpenVAF cache, run's plumbing with stand-in executables, and a few runs of the real engine). Written from
docs/ARCHITECTURE_V1.md (A9, with the shared conventions of A7 / A8) only.

Raw files are written here in ngspice's ASCII raw format (as ngspice 47 writes them: Title, Date, Command, Plotname,
Flags, No. Variables, No. Points, Variables with one "\\t<index>\\t<name>\\t<type>" line each, Values with one
" <point>\\t<value 0>" line and n-1 "\\t<value>" lines per point, a blank line after each point; complex values
"re,im"). Variable names are the ones ngspice 47 gives (seen by running the bare ngspice 47 of SIMWS_SPICE_PREFIX
by hand: "v(mid)", "v(in,mid)", "i(v_vin)", scale vectors "v(v-sweep)", "i(i-sweep)", "frequency", "time"; for op
the first vector is a node voltage). Engine runs need ngspice (SIMWS_SPICE_PREFIX or <envs_root>/spice) and are
skipped without it."""
import hashlib
import os
import re
import shlex
from pathlib import Path

import numpy as np
import pydantic
import pytest

from simws.spec.library import implant_si, npn_hbt_2d, pn_diode_1d, see_si_slab

ALL_KINDS = ["resistor", "capacitor", "inductor", "vsource", "isource", "diode", "bjt", "va", "subckt"]
ALL_MODELS = ["diode", "npn", "pnp", "va"]
LETTER = {"resistor": "R", "capacitor": "C", "inductor": "L", "vsource": "V", "isource": "I", "diode": "D",
          "bjt": "Q", "va": "N", "subckt": "X"}
OTHER_ENGINES = ["channeling", "charon", "devsim", "devsim_eb", "g4implant", "g4see", "rustbca"]   # amended by the lead (A11.3, A13.3, A15.5)

VA_SRC = """`include "disciplines.vams"
module mydiode(a, c);
  inout a, c;
  electrical a, c;
  parameter real is = 1e-14;
  analog I(a, c) <+ is * (limexp(V(a, c) / $vt) - 1);
endmodule
"""
VA_SRC2 = VA_SRC.replace("module mydiode", "module otherva")


def _lib():
    from simws.spec import library
    return library


def _ng():
    from simws import engines
    return engines.module("ngspice")


def _need_engine():
    try:
        ok = _ng().available()
    except Exception as e:                                      # noqa: BLE001
        pytest.skip(f"ngspice engine module not importable: {e!r}")
    if not ok:
        pytest.skip("ngspice not available (SIMWS_SPICE_PREFIX / <envs_root>/spice)")


def _need_openvaf():
    _need_engine()
    if not _ng().openvaf_available():
        pytest.skip("OpenVAF not available")


def _cp(d):
    from simws.spec import CircuitProblem
    return CircuitProblem.model_validate(d)


def _base(**over):
    d = {"elements": [{"name": "vin", "kind": "vsource", "nodes": ["in", "0"], "value": 1.0},
                      {"name": "r1", "kind": "resistor", "nodes": ["in", "out"], "value": 1e3},
                      {"name": "c1", "kind": "capacitor", "nodes": ["out", "0"], "value": 1e-9}],
         "analysis": {"kind": "op"}, "probes": ["v(out)"]}
    d.update(over)
    return d


def _full():
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
    d["models"] = [{"name": "dmod", "kind": "diode", "params": {"is": 1e-14, "n": 1.05}},
                   {"name": "qn", "kind": "npn", "params": {"bf": 100.0}},
                   {"name": "qp", "kind": "pnp"},
                   {"name": "vam", "kind": "va", "params": {"is": 2e-14, "n": 1.1},
                    "va": {"module": "mydiode", "source": VA_SRC}}]
    d["libraries"] = [{"path": "/nowhere/models.lib", "section": "typ"}]
    return d


def _prefixes(strings):
    return sorted({s.split(":", 1)[0] for s in strings})


# ------------------------------------------------------------------------------------------------ CircuitResult
def test_circuit_result_model():
    import json

    from simws.engines.base import CircuitResult
    r = CircuitResult(engine="toy", problem_id="0" * 32, analysis="op", signals={"v(out)": [0.5]},
                      units={"v(out)": "V"})
    assert r.axis_name == "" and r.axis_unit == "" and r.axis == [] and r.signals_imag == {}
    assert r.flags == [] and r.diagnostics == {}
    json.loads(r.model_dump_json())
    for a in ("op", "dc", "ac", "tran"):
        CircuitResult(engine="toy", problem_id="x", analysis=a, signals={}, units={})
    with pytest.raises(pydantic.ValidationError):
        CircuitResult(engine="toy", problem_id="x", analysis="noise", signals={}, units={})
    with pytest.raises(pydantic.ValidationError):
        CircuitResult(engine="toy", problem_id="x", analysis="op", signals={}, units={}, colour="red")
    for missing in ("signals", "units", "engine", "problem_id", "analysis"):
        kw = dict(engine="toy", problem_id="x", analysis="op", signals={}, units={})
        del kw[missing]
        with pytest.raises(pydantic.ValidationError):
            CircuitResult(**kw)


# ------------------------------------------------------------------------------------------------ CircuitCapability
def _ccap(**over):
    from simws.engines.base import CircuitCapability
    d = dict(engine="toy", env="toyenv", description="test sheet", in_process=False, element_kinds=list(ALL_KINDS),
             model_kinds=list(ALL_MODELS), analyses=["op", "dc", "ac", "tran"], max_elements=10000,
             tested_element_kinds=["resistor", "capacitor", "vsource", "diode", "va", "subckt"],
             tested_analyses=["op", "dc", "ac", "tran"], tested_temperature_C=(26.85, 27.0))
    d.update(over)
    return CircuitCapability(**d)


def test_ccap_unknown_fields_and_defaults():
    with pytest.raises(pydantic.ValidationError):
        _ccap(gpus=4)
    assert _ccap().notes == ""


def test_ccap_supported_is_empty():
    c = _ccap()
    for p in (_cp(_base()), _cp(_full()), _lib().divider(), _lib().rc_lowpass(), _lib().diode_iv(model="va")):
        assert c.unsupported(p) == []


@pytest.mark.parametrize("field,over", [
    ("element_kinds", {"element_kinds": [k for k in ALL_KINDS if k != "inductor"]}),
    ("element_kinds", {"element_kinds": [k for k in ALL_KINDS if k not in ("bjt", "va")]}),
    ("model_kinds", {"model_kinds": ["diode", "va"]}),
    ("model_kinds", {"model_kinds": []}),
    ("analyses", {"analyses": ["dc", "ac", "tran"]}),
    ("max_elements", {"max_elements": 10}),
])
def test_ccap_reasons(field, over):
    r = _ccap(**over).unsupported(_cp(_full()))
    assert r and _prefixes(r) == [field], r


def test_ccap_analyses_each():
    for a, analysis in (("dc", {"kind": "dc", "source": "vin", "start": 0.0, "stop": 1.0, "step": 0.5}),
                        ("ac", {"kind": "ac", "points_per_decade": 1, "f_start_Hz": 1.0, "f_stop_Hz": 10.0}),
                        ("tran", {"kind": "tran", "step_s": 1e-9, "stop_s": 1e-6})):
        p = _cp(_base(analysis=analysis))
        others = [x for x in ("op", "dc", "ac", "tran") if x != a]
        r = _ccap(analyses=others).unsupported(p)
        assert _prefixes(r) == ["analyses"], r
        assert _ccap(analyses=[a]).unsupported(p) == []


def test_ccap_max_elements_bound():
    p = _cp(_base())                                              # 3 elements
    assert _ccap(max_elements=3).unsupported(p) == []
    r = _ccap(max_elements=2).unsupported(p)
    assert _prefixes(r) == ["max_elements"], r


def test_ccap_reasons_add_up():
    c = _ccap(element_kinds=["resistor", "vsource"], model_kinds=["diode"], analyses=["dc"], max_elements=5)
    assert _prefixes(c.unsupported(_cp(_full()))) == ["analyses", "element_kinds", "max_elements", "model_kinds"]


def test_ccap_outside_inside():
    c = _ccap()
    for p in (_cp(_base()), _lib().divider(), _lib().rc_lowpass(), _lib().rc_lowpass(analysis="tran"),
              _lib().diode_iv(), _lib().diode_iv(model="va"), _lib().diode_iv(temperature_C=26.85, tnom_C=26.85)):
        assert c.outside_tested_range(p) == [], p


@pytest.mark.parametrize("kind", ["inductor", "isource", "bjt"])
def test_ccap_untested_element_kind(kind):
    d = _base()
    el = {"name": "zz", "kind": kind, "nodes": ["out", "0"]}
    if kind == "inductor":
        el["value"] = 1e-6
    if kind == "bjt":
        el.update(nodes=["out", "in", "0"], model="qn")
        d["models"] = [{"name": "qn", "kind": "npn"}]
    d["elements"].append(el)
    r = _ccap().outside_tested_range(_cp(d))
    assert _prefixes(r) == ["element_kinds"], r


def test_ccap_untested_analysis():
    p = _cp(_base(analysis={"kind": "tran", "step_s": 1e-9, "stop_s": 1e-6}))
    r = _ccap(tested_analyses=["op", "dc"]).outside_tested_range(p)
    assert _prefixes(r) == ["analyses"], r
    assert _ccap(tested_analyses=["tran"]).outside_tested_range(p) == []


@pytest.mark.parametrize("T,inside", [(26.85, True), (27.0, True), (26.9, True), (26.84, False), (27.01, False),
                                      (-40.0, False), (125.0, False)])
def test_ccap_temperature_bounds_included(T, inside):
    p = _cp(_base(temperature_C=T))
    r = _ccap().outside_tested_range(p)
    if inside:
        assert r == [], r
    else:
        assert _prefixes(r) == ["temperature_C"], r


def test_ccap_tnom_is_not_temperature():
    assert _ccap().outside_tested_range(_cp(_base(tnom_C=60.0))) == []


def test_ccap_libraries_always_untested():
    d = _base()
    d["elements"].append({"name": "x1", "kind": "subckt", "nodes": ["out", "0"], "model": "sub"})
    d["libraries"] = [{"path": "/x/a.lib"}]
    r = _ccap().outside_tested_range(_cp(d))
    assert _prefixes(r) == ["libraries"], r
    d["libraries"].append({"path": "/x/b.lib", "section": "s"})
    r = _ccap().outside_tested_range(_cp(d))
    assert _prefixes(r) == ["libraries"], r


def test_ccap_outside_all_together():
    p = _cp(_full()).with_changes({"temperature_C": 85.0,
                                   "analysis": {"kind": "tran", "step_s": 1e-9, "stop_s": 1e-6}})
    r = _ccap(tested_analyses=["op"]).outside_tested_range(p)
    assert _prefixes(r) == ["analyses", "element_kinds", "libraries", "temperature_C"], r


# ------------------------------------------------------------------------------------------------ registry
def test_registry_has_ngspice():
    from simws import engines
    assert engines.ENGINES["ngspice"] == "simws.engines.ngspice_engine"
    for n, m in (("devsim", "simws.engines.devsim_engine"), ("charon", "simws.engines.charon_engine"),
                 ("rustbca", "simws.engines.rustbca_engine"), ("g4see", "simws.engines.g4see_engine")):
        assert engines.ENGINES[n] == m
    assert sorted(engines.ENGINES) == ["channeling", "charon", "devsim", "devsim_eb", "g4implant", "g4see", "ngspice", "rustbca"]   # amended by the lead (A11.3, A13.3, A15.5)
    assert engines.names() == sorted(engines.ENGINES)


def test_family_and_adapter_name_all_engines():
    from simws import engines
    assert engines.family("ngspice") == "circuit"
    assert engines.adapter_name("ngspice") == "circuit.ngspice"
    assert _ng().FAMILY == "circuit"
    exp = {"devsim": "device", "charon": "device", "rustbca": "implant", "g4see": "see", "ngspice": "circuit"}
    for n, f in exp.items():
        assert engines.family(n) == f
        assert engines.adapter_name(n) == f"{f}.{n}"


def test_ngspice_module_interface():
    from simws import engines
    from simws.engines.base import CircuitCapability
    mod = _ng()
    for attr in ("available", "openvaf_available", "unsupported", "build_netlist", "compile_va", "parse_raw", "run"):
        assert callable(getattr(mod, attr, None)), attr
    assert isinstance(engines.capability("ngspice"), CircuitCapability)
    assert isinstance(mod.available(), bool) and isinstance(mod.openvaf_available(), bool)
    lib = _lib()
    for p in (lib.divider(), lib.rc_lowpass(), lib.rc_lowpass(analysis="tran"), lib.diode_iv(),
              lib.diode_iv(model="va"), _cp(_full())):
        assert engines.unsupported("ngspice", p) == mod.unsupported(p) == []


def test_problem_kind_refusal_all_directions(monkeypatch):
    """A7.3 for four families: exactly one 'problem kind:' reason across families, without calling the engine
    module's own check (replaced by functions that fail when called)."""
    from simws import engines

    def boom(problem):
        raise AssertionError("the engine module's own check must not be called across families")

    for n in OTHER_ENGINES + ["ngspice"]:
        monkeypatch.setattr(engines.module(n), "unsupported", boom)
    circ = _lib().divider()
    for n in OTHER_ENGINES:
        r = engines.unsupported(n, circ)
        assert len(r) == 1 and r[0].startswith("problem kind:"), (n, r)
    for other in (pn_diode_1d(), npn_hbt_2d(), implant_si(), see_si_slab()):
        r = engines.unsupported("ngspice", other)
        assert len(r) == 1 and r[0].startswith("problem kind:"), r
    for n, other in (("rustbca", see_si_slab()), ("g4see", implant_si()), ("devsim", see_si_slab())):
        r = engines.unsupported(n, other)
        assert len(r) == 1 and r[0].startswith("problem kind:"), (n, r)


def test_same_family_uses_module_check():
    from simws import engines
    p = _big_problem(10001)
    r = engines.unsupported("ngspice", p)
    assert any(s.startswith("max_elements:") for s in r), r


# ------------------------------------------------------------------------------------------------ planner
def test_plan_circuit_problem():
    from simws.engines.planner import plan
    p = _lib().divider()
    pl = plan(p)
    assert pl.problem_id == p.problem_id()
    assert pl.primary == "ngspice" and pl.checkers == []
    assert sorted(pl.refused) == OTHER_ENGINES
    for n in OTHER_ENGINES:
        assert len(pl.refused[n]) == 1 and pl.refused[n][0].startswith("problem kind:")
    assert pl.warnings == {}
    assert "ngspice" in pl.summary()


def test_plan_circuit_outside_tested_warns():
    from simws.engines.planner import plan
    pl = plan(_cp(_full()))
    assert pl.primary == "ngspice"
    assert list(pl.warnings) == ["ngspice"]
    assert _prefixes(pl.warnings["ngspice"]) == ["element_kinds", "libraries"]


def test_plan_circuit_refused_by_capability():
    from simws.engines.planner import plan
    pl = plan(_big_problem(10001))
    assert pl.primary is None
    assert any(s.startswith("max_elements:") for s in pl.refused["ngspice"])


def test_plan_other_families_refuse_ngspice():
    from simws.engines.planner import plan
    for p, primary in ((pn_diode_1d(V=[0.1]), "devsim"), (npn_hbt_2d(), "devsim"), (implant_si("P"), "rustbca"),
                       (see_si_slab(), "g4see")):
        pl = plan(p)
        assert pl.primary == primary
        assert "ngspice" in pl.refused and "ngspice" not in pl.checkers
        assert len(pl.refused["ngspice"]) == 1 and pl.refused["ngspice"][0].startswith("problem kind:")


# ------------------------------------------------------------------------------------------------ capability sheet
def test_ngspice_capability_exact():
    from simws import engines
    c = engines.capability("ngspice")
    assert c.engine == "ngspice" and c.env == "spice" and c.in_process is False
    assert sorted(c.element_kinds) == sorted(ALL_KINDS) and len(c.element_kinds) == 9
    assert sorted(c.model_kinds) == sorted(ALL_MODELS) and len(c.model_kinds) == 4
    assert c.analyses == ["op", "dc", "ac", "tran"]
    assert c.max_elements == 10000
    assert c.tested_element_kinds == ["resistor", "capacitor", "vsource", "diode", "va", "subckt"]
    assert c.tested_analyses == ["op", "dc", "ac", "tran"]
    assert tuple(c.tested_temperature_C) == (26.85, 27.0)


def _big_problem(n):
    els = [{"name": "vin", "kind": "vsource", "nodes": ["n0", "0"], "value": 1.0}]
    els += [{"name": f"r{i}", "kind": "resistor", "nodes": [f"n{i - 1}", f"n{i}"], "value": 1.0}
            for i in range(1, n - 1)]
    els.append({"name": "rend", "kind": "resistor", "nodes": [f"n{n - 2}", "0"], "value": 1.0})
    assert len(els) == n
    return _cp({"elements": els, "analysis": {"kind": "op"}, "probes": ["v(n1)"]})


def test_ngspice_max_elements_bound():
    from simws import engines
    assert engines.unsupported("ngspice", _big_problem(10000)) == []
    r = engines.unsupported("ngspice", _big_problem(10001))
    assert _prefixes(r) == ["max_elements"], r


def test_acceptance_cases_inside_tested_range():
    """R30 to R35 are inside the tested range (A9.4: 'the cases of R30 to R36'); R36 uses a library."""
    from simws import engines
    c = engines.capability("ngspice")
    lib = _lib()
    for p in (lib.divider(), lib.rc_lowpass(), lib.rc_lowpass(analysis="tran"), lib.diode_iv(),
              lib.diode_iv(model="va"),
              lib.diode_iv(1e-20, 1.0, 0.30, 0.60, 0.05, model="builtin", temperature_C=300.0 - 273.15,
                           tnom_C=300.0 - 273.15)):
        assert c.outside_tested_range(p) == [], p
    assert _prefixes(c.outside_tested_range(r36_problem())) == ["libraries"]


IHP_PDK = "~/sim_workspace/ihp_sg13g2_pdk"
IHP_LIB = IHP_PDK + "/ihp-sg13g2/libs.tech/ngspice/models/cornerHBT.lib"


def r36_problem():
    """The legacy Gummel set-up of benchmarks/ihp_npn13g2_baseline (A9.8 R36), with the legacy numerics."""
    return _cp({"title": "IHP SG13G2 npn13G2 Gummel (hbt_typ, 27 C)",
                "libraries": [{"path": IHP_LIB, "section": "hbt_typ"}],
                "elements": [{"name": "vc", "kind": "vsource", "nodes": ["c", "0"], "value": 0.0},
                             {"name": "vb", "kind": "vsource", "nodes": ["b", "0"], "value": 0.0},
                             {"name": "ve", "kind": "vsource", "nodes": ["e", "0"], "value": -0.6},
                             {"name": "x1", "kind": "subckt", "nodes": ["c", "b", "e", "0"], "model": "npn13G2",
                              "params": {"Nx": 1}}],
                "analysis": {"kind": "dc", "source": "ve", "start": -0.35, "stop": -1.02, "step": -0.005},
                "probes": ["i(vc)", "i(vb)"], "temperature_C": 27.0, "tnom_C": 27.0,
                "numerics": {"reltol": 1e-5, "abstol_A": 1e-16, "gmin_S": 1e-15}})


# ------------------------------------------------------------------------------------------------ build_netlist
def _join(netlist):
    """Lines of a SPICE netlist with '+' continuation lines joined; comments and blank lines dropped (the title,
    line 0, is kept as it is)."""
    raw = netlist.splitlines()
    out = [raw[0]]
    for line in raw[1:]:
        s = line.strip()
        if not s or s.startswith("*"):
            continue
        if s.startswith("+") and len(out) > 1:
            out[-1] += " " + s[1:].strip()
        else:
            out.append(s)
    return out


class Net:
    def __init__(self, text):
        self.text = text
        self.lines = _join(text)
        low = [x.lower() for x in self.lines]
        assert ".control" in low and ".endc" in low, text
        self.i_ctl, self.i_endc = low.index(".control"), low.index(".endc")
        assert self.i_ctl < self.i_endc
        self.control = self.lines[self.i_ctl + 1:self.i_endc]
        self.outside = self.lines[1:self.i_ctl] + self.lines[self.i_endc + 1:]

    def elements(self):
        res = {}
        for line in self.outside:
            m = re.match(r"^([RCLVIDQNX])_(\S+)\s", line + " ", re.I)
            if m:
                res[line.split()[0]] = line
        return res

    def element(self, spice_name):
        els = self.elements()
        for k, v in els.items():
            if k.lower() == spice_name.lower():
                return v
        raise AssertionError(f"no element line {spice_name} in\n{self.text}")

    def models(self):
        return [x for x in self.outside if x.lower().startswith(".model")]

    def options(self):
        d = {}
        for x in self.outside:
            if x.lower().startswith(".options") or x.lower().startswith(".option "):
                for tok in x.split()[1:]:
                    k, _, v = tok.partition("=")
                    d[k.lower()] = v
        return d


def _net(p, osdi=None):
    return Net(_ng().build_netlist(p, osdi=osdi) if osdi is not None else _ng().build_netlist(p))


def _f(tok, expected, rel=1e-12):
    v = float(tok)
    assert v == pytest.approx(expected, rel=rel, abs=0 if expected else 1e-300), (tok, expected)


def _kv(tokens):
    return {t.split("=", 1)[0].lower(): float(t.split("=", 1)[1]) for t in tokens}


def test_build_netlist_is_pure(monkeypatch, tmp_path):
    import subprocess
    monkeypatch.setenv("SIMWS_SPICE_PREFIX", str(tmp_path))

    def no_process(*a, **k):
        raise AssertionError("build_netlist must not start a process")
    monkeypatch.setattr(subprocess, "run", no_process)
    monkeypatch.setattr(subprocess, "Popen", no_process)
    for p in (_lib().divider(), _lib().rc_lowpass(), _lib().diode_iv()):
        n = _ng().build_netlist(p)
        assert isinstance(n, str) and "write result.raw" in n.lower()
    n = _ng().build_netlist(_lib().diode_iv(model="va"), osdi={"shockley_diode": str(tmp_path / "x.osdi")})
    assert "pre_osdi" in n


def test_build_netlist_deterministic():
    p = _lib().rc_lowpass()
    assert _ng().build_netlist(p) == _ng().build_netlist(p.model_copy(deep=True))


def test_netlist_title_first_end_last():
    p0 = _lib().divider()
    n = _net(p0)
    assert p0.title in n.lines[0]                    # A9.2 does not fix the library problem's title
    assert n.lines[-1].lower() == ".end"
    assert n.i_endc == len(n.lines) - 2
    p = _lib().divider().with_changes({"title": "my own title"})
    assert "my own title" in _net(p).lines[0]


def test_netlist_divider_elements():
    n = _net(_lib().divider())
    els = n.elements()
    assert sorted(k.lower() for k in els) == ["r_r1", "r_r2", "v_vin"]
    t = n.element("R_r1").split()
    assert t[0] == "R_r1" and [x.lower() for x in t[1:3]] == ["in", "mid"] and len(t) == 4
    _f(t[3], 1e3)
    t = n.element("R_r2").split()
    assert t[0] == "R_r2" and [x.lower() for x in t[1:3]] == ["mid", "0"]
    _f(t[3], 3e3)
    t = n.element("V_vin").split()
    assert t[0] == "V_vin" and [x.lower() for x in t[1:3]] == ["in", "0"]
    assert t[3].upper() == "DC" and len(t) == 5
    _f(t[4], 1.0)


def test_netlist_element_names_and_kind_letters():
    n = _net(_cp(_full()), osdi={"mydiode": "/tmp/mydiode.osdi"})
    got = sorted(n.elements())
    exp = sorted(f"{LETTER[e['kind']]}_{e['name']}" for e in _full()["elements"])
    assert [g.lower() for g in got] == [e.lower() for e in exp]
    for e in _full()["elements"]:
        line = n.element(f"{LETTER[e['kind']]}_{e['name']}")
        assert line.split()[0] == f"{LETTER[e['kind']]}_{e['name']}", line


def test_netlist_passive_values():
    n = _net(_cp(_full()), osdi={"mydiode": "/tmp/mydiode.osdi"})
    for name, nodes, val in (("C_c1", ["out", "0"], 1e-9), ("L_l1", ["out", "x"], 1e-6), ("R_rx", ["x", "0"], 50.0)):
        t = n.element(name).split()
        assert [x.lower() for x in t[1:3]] == nodes and len(t) == 4, t
        _f(t[3], val)


def test_netlist_model_elements():
    n = _net(_cp(_full()), osdi={"mydiode": "/tmp/mydiode.osdi"})
    t = n.element("D_d1").split()
    assert [x.lower() for x in t[1:4]] == ["x", "0", "dmod"] and _kv(t[4:]) == {"area": 2.0}, t
    t = n.element("Q_q1").split()
    assert [x.lower() for x in t[1:5]] == ["out", "x", "0", "qn"] and len(t) == 5, t
    t = n.element("Q_q2").split()
    assert [x.lower() for x in t[1:6]] == ["out", "x", "0", "0", "qp"] and len(t) == 6, t
    t = n.element("N_n1").split()
    assert [x.lower() for x in t[1:4]] == ["x", "0", "vam"] and len(t) == 4, t
    t = n.element("X_x1").split()
    assert [x.lower() for x in t[1:5]] == ["out", "x", "0", "npn13g2"] and _kv(t[5:]) == {"nx": 1.0}, t


def _src(line):
    """(dc, ac, pulse list, pwl list) parsed from a source line."""
    m = re.search(r"\bDC\s+(\S+)", line, re.I)
    dc = float(m.group(1)) if m else None
    m = re.search(r"\bAC\s+([^\s(]+)", line, re.I)
    ac = float(m.group(1)) if m else None
    m = re.search(r"\bPULSE\s*\(([^)]*)\)", line, re.I)
    pulse = [float(x) for x in re.split(r"[\s,]+", m.group(1).strip())] if m else None
    m = re.search(r"\bPWL\s*\(([^)]*)\)", line, re.I)
    pwl = [float(x) for x in re.split(r"[\s,]+", m.group(1).strip())] if m else None
    return dc, ac, pulse, pwl


def test_netlist_rc_ac_source():
    n = _net(_lib().rc_lowpass())
    line = n.element("V_vin")
    t = line.split()
    assert [x.lower() for x in t[1:3]] == ["in", "0"]
    dc, ac, pulse, pwl = _src(line)
    assert dc == 0.0 and ac == 1.0 and pulse is None and pwl is None
    assert re.search(r"\bDC\s+\S+\s+AC\s+\S+", line, re.I), line


def test_netlist_rc_tran_source():
    n = _net(_lib().rc_lowpass(analysis="tran"))
    dc, ac, pulse, pwl = _src(n.element("V_vin"))
    assert dc == 0.0 and ac is None and pwl is None
    assert pulse == pytest.approx([0.0, 1.0, 0.0, 1e-12, 1e-12, 1.0, 2.0], rel=1e-12)


def test_netlist_source_dc_default_and_negative():
    d = _base()
    del d["elements"][0]["value"]
    dc, ac, _, _ = _src(_net(_cp(d)).element("V_vin"))
    assert dc == 0.0 and ac is None
    d["elements"][0]["value"] = -2.75
    dc, _, _, _ = _src(_net(_cp(d)).element("V_vin"))
    assert dc == -2.75


def test_netlist_source_pulse_pwl_isource():
    d = _full()
    d["elements"][0].update(ac=0.25, pulse={"v1": -1.0, "v2": 2.5, "delay_s": 3e-9, "rise_s": 1e-10, "fall_s": 2e-10,
                                            "width_s": 5e-8, "period_s": 1.1e-7})
    d["elements"][5].update(value=2e-6, pwl={"points": [[0.0, 0.0], [1e-6, 1e-3], [3e-6, -2e-3]]})
    n = _net(_cp(d), osdi={"mydiode": "/tmp/mydiode.osdi"})
    dc, ac, pulse, pwl = _src(n.element("V_vin"))
    assert dc == 1.0 and ac == 0.25 and pwl is None
    assert pulse == pytest.approx([-1.0, 2.5, 3e-9, 1e-10, 2e-10, 5e-8, 1.1e-7], rel=1e-12)
    line = n.element("I_i1")
    assert [x.lower() for x in line.split()[1:3]] == ["0", "x"]
    dc, ac, pulse, pwl = _src(line)
    assert dc == 2e-6 and ac is None and pulse is None
    assert pwl == pytest.approx([0.0, 0.0, 1e-6, 1e-3, 3e-6, -2e-3], rel=1e-12)


def _model_line(n, name):
    for x in n.models():
        if x.split()[1].lower() == name.lower():
            return x
    raise AssertionError(f"no .model {name} in\n{n.text}")


def test_netlist_builtin_models():
    n = _net(_cp(_full()), osdi={"mydiode": "/tmp/mydiode.osdi"})
    assert len(n.models()) == 4
    for name, typ, params in (("dmod", "D", {"is": 1e-14, "n": 1.05}), ("qn", "NPN", {"bf": 100.0}),
                              ("qp", "PNP", {})):
        line = _model_line(n, name)
        m = re.match(r"^\.model\s+(\S+)\s+([A-Za-z]+)\s*(?:\((.*)\))?\s*$", line, re.I)
        assert m, line
        assert m.group(2).upper() == typ, line
        if params:
            assert m.group(3) is not None, line                              # built-in: parentheses (A9.10)
        inner = (m.group(3) or "").replace(",", " ").split()
        assert _kv(inner) == pytest.approx(params, rel=1e-12), line


def test_netlist_va_model_without_parentheses():
    """A9.10: a Verilog-A model is written '.model <name> <module> key=value ...' without parentheses."""
    n = _net(_cp(_full()), osdi={"mydiode": "/tmp/mydiode.osdi"})
    line = _model_line(n, "vam")
    assert "(" not in line and ")" not in line, line
    t = line.split()
    assert t[0].lower() == ".model" and t[2] == "mydiode"
    assert _kv(t[3:]) == pytest.approx({"is": 2e-14, "n": 1.1}, rel=1e-12), line


def test_netlist_diode_iv_models():
    n = _net(_lib().diode_iv(IS_A=3e-15, N=1.3))
    line = _model_line(n, _lib().diode_iv().models[0].name)
    m = re.match(r"^\.model\s+\S+\s+D\s*\((.*)\)\s*$", line, re.I)
    assert m, line
    assert _kv(m.group(1).replace(",", " ").split()) == pytest.approx({"is": 3e-15, "n": 1.3}, rel=1e-12)
    pv = _lib().diode_iv(IS_A=3e-15, N=1.3, model="va")
    n = _net(pv, osdi={"shockley_diode": "/c/shockley.osdi"})
    line = _model_line(n, pv.models[0].name)
    t = line.split()
    assert "(" not in line and t[2] == "shockley_diode"
    assert _kv(t[3:]) == pytest.approx({"is": 3e-15, "n": 1.3}, rel=1e-12)


def test_netlist_options():
    p = _lib().divider().with_changes({"numerics": {"reltol": 1e-5, "abstol_A": 1e-14, "vntol_V": 1e-7,
                                                    "gmin_S": 1e-13}, "temperature_C": 85.0, "tnom_C": 25.0})
    o = _net(p).options()
    for k, v in (("reltol", 1e-5), ("abstol", 1e-14), ("vntol", 1e-7), ("gmin", 1e-13), ("temp", 85.0),
                 ("tnom", 25.0)):
        assert k in o, (k, o)
        _f(o[k], v)
    o = _net(_lib().divider()).options()
    for k, v in (("reltol", 1e-6), ("abstol", 1e-15), ("vntol", 1e-9), ("gmin", 1e-15), ("temp", 27.0),
                 ("tnom", 27.0)):
        _f(o[k], v)
    o = _net(_lib().divider().with_changes({"numerics.gmin_S": 0.0})).options()
    assert float(o["gmin"]) == 0.0


def test_netlist_libraries(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    d = _base()
    d["elements"].append({"name": "x1", "kind": "subckt", "nodes": ["out", "0"], "model": "sub"})
    d["libraries"] = [{"path": "~/pdk/corner.lib", "section": "hbt_typ"}, {"path": "/abs/models.inc"}]
    n = _net(_cp(d))
    libs = [x for x in n.outside if x.lower().startswith(".lib") or x.lower().startswith(".include")]
    exp_path = os.path.expanduser("~/pdk/corner.lib")
    assert exp_path.startswith(str(tmp_path))
    assert libs == [f'.lib "{exp_path}" hbt_typ', '.include "/abs/models.inc"'], libs
    assert "~" not in "\n".join(libs)
    assert n.element("X_x1").split()[1:4] == ["out", "0", "sub"]


def _ctl(n):
    return [shlex.split(x) if "pre_osdi" in x.lower() else x.split() for x in n.control]


def test_netlist_control_op():
    n = _net(_lib().divider())
    c = _ctl(n)
    assert [x[0].lower() for x in c] == ["set", "op", "write"], n.control
    assert [x.lower() for x in c[0]] == ["set", "filetype=ascii"]
    assert len(c[1]) == 1
    assert c[2][1] == "result.raw"
    assert sorted(x.lower() for x in c[2][2:]) == ["i(v_vin)", "v(mid)"]


def test_netlist_control_dc():
    n = _net(_lib().diode_iv())
    c = _ctl(n)
    assert [x[0].lower() for x in c] == ["set", "dc", "write"], n.control
    a = c[1]
    assert len(a) == 5 and a[1].lower() == "v_vin", a
    _f(a[2], 0.30)
    _f(a[3], 0.65)
    _f(a[4], 0.05)
    assert [x.lower() for x in c[2][2:]] == ["i(v_vin)"]


def test_netlist_control_dc_isource_negative_step():
    d = _full()
    d["analysis"] = {"kind": "dc", "source": "i1", "start": 1e-3, "stop": -1e-3, "step": -2.5e-4}
    d["probes"] = ["v(x)", "v(out,x)", "i(vin)"]
    n = _net(_cp(d), osdi={"mydiode": "/tmp/mydiode.osdi"})
    c = _ctl(n)
    a = next(x for x in c if x[0].lower() == "dc")
    assert a[1].lower() == "i_i1"
    _f(a[2], 1e-3)
    _f(a[3], -1e-3)
    _f(a[4], -2.5e-4)
    w = c[-1]
    assert w[0].lower() == "write" and w[1] == "result.raw"
    assert sorted(x.lower() for x in w[2:]) == sorted(["v(x)", "v(out,x)", "i(v_vin)"])


def test_netlist_control_ac():
    n = _net(_lib().rc_lowpass())
    c = _ctl(n)
    assert [x[0].lower() for x in c] == ["set", "ac", "write"], n.control
    a = c[1]
    assert len(a) == 5 and a[1].lower() == "dec" and int(a[2]) == 10, a
    _f(a[3], 10.0)
    _f(a[4], 1e8)
    assert [x.lower() for x in c[2][2:]] == ["v(out)"]


def test_netlist_control_tran():
    n = _net(_lib().rc_lowpass(analysis="tran"))
    c = _ctl(n)
    assert [x[0].lower() for x in c] == ["set", "tran", "write"], n.control
    a = c[1]
    assert len(a) == 5, a
    _f(a[1], 1e-8)
    _f(a[2], 1e-5)
    assert float(a[3]) == 0.0
    _f(a[4], 1e-8)


def test_netlist_control_tran_without_max_step():
    """A9.4 writes 'tran step stop 0 max_step'; without max_step_s the text says nothing (only the fixed part is
    tested)."""
    p = _cp(_base(analysis={"kind": "tran", "step_s": 2e-9, "stop_s": 4e-6}))
    a = next(x for x in _ctl(_net(p)) if x[0].lower() == "tran")
    _f(a[1], 2e-9)
    _f(a[2], 4e-6)
    if len(a) > 3:
        assert float(a[3]) == 0.0


def test_netlist_control_probe_translation_case():
    d = _base(probes=["v(OUT)", "v(In,out)", "i(VIN)"])
    w = _ctl(_net(_cp(d)))[-1]
    assert sorted(x.lower() for x in w[2:]) == sorted(["v(out)", "v(in,out)", "i(v_vin)"])


def test_netlist_control_pre_osdi(tmp_path):
    d = _full()
    d["models"].append({"name": "vam2", "kind": "va", "va": {"module": "otherva", "source": VA_SRC2}})
    d["elements"].append({"name": "n2", "kind": "va", "nodes": ["out", "0"], "model": "vam2"})
    f1, f2 = str(tmp_path / "a" / "mydiode.osdi"), str(tmp_path / "b" / "otherva.osdi")
    n = _net(_cp(d), osdi={"mydiode": f1, "otherva": f2})
    c = _ctl(n)
    heads = [x[0].lower() for x in c]
    assert heads == ["pre_osdi", "pre_osdi", "set", "op", "write"], n.control
    assert sorted(x[1] for x in c[:2]) == sorted([f1, f2])
    assert len(c[0]) == 2 and len(c[1]) == 2
    _model_line(n, "vam2")
    assert _model_line(n, "vam2").split()[2] == "otherva"


def test_netlist_no_pre_osdi_without_va():
    for p in (_lib().divider(), _lib().diode_iv(), _lib().rc_lowpass()):
        assert "pre_osdi" not in _ng().build_netlist(p).lower()


def test_netlist_definitions_outside_control():
    n = _net(_cp(_full()), osdi={"mydiode": "/tmp/mydiode.osdi"})
    ctl = " ".join(n.control).lower()
    assert ".model" not in ctl and ".options" not in ctl and ".lib" not in ctl
    assert len(n.elements()) == len(_full()["elements"])


# ------------------------------------------------------------------------------------------------ parse_raw
def write_raw(path, plotname, variables, points, complex_=False, title="simws circuit"):
    """An ASCII raw file as ngspice 47 writes it. variables: [(name, type)]; points: one row per point (complex
    numbers when complex_)."""
    def fmt(v):
        if complex_:
            return f"{v.real:.15e},{v.imag:.15e}"
        return f"{v:.15e}"
    s = [f"Title: {title}", "Date: Wed Nov 18 12:00:00  2026", "Command: ngspice-47, Build Wed Nov 18 2026",
         f"Plotname: {plotname}", f"Flags: {'complex' if complex_ else 'real'}", f"No. Variables: {len(variables)}",
         f"No. Points: {len(points)}", "Variables:"]
    for i, (name, typ) in enumerate(variables):
        s.append(f"\t{i}\t{name}\t{typ}")
    s.append("Values:")
    for k, row in enumerate(points):
        s.append(f" {k}\t{fmt(row[0])}")
        for v in row[1:]:
            s.append(f"\t{fmt(v)}")
        s.append("")
    Path(path).write_text("\n".join(s) + "\n")
    return path


def _vectors(d, names):
    """The vectors named `names` from parse_raw's dict. A9.4 fixes only that parse_raw returns a dict; this looks
    for a mapping (the dict itself or one dict inside it) whose keys include every name (case-insensitive)."""
    assert isinstance(d, dict), type(d)
    if "names" in d and "values" in d:              # names, and the values as one row per point
        nm = [str(x).lower() for x in d["names"]]
        rows = list(d["values"])
        assert all(len(row) == len(nm) for row in rows), rows
        return {n: [row[nm.index(n.lower())] for row in rows] for n in names}
    cands = [d] + [v for v in d.values() if isinstance(v, dict)]
    for c in cands:
        keys = {str(k).lower(): k for k in c}
        if all(n.lower() in keys for n in names):
            return {n: c[keys[n.lower()]] for n in names}
    raise AssertionError(f"parse_raw result does not hold {names}: keys {list(d)}")


def _as_complex(v):
    a = np.asarray(v)
    if np.iscomplexobj(a):
        return a.astype(complex)
    a = np.asarray(v, float)
    if a.ndim == 2 and a.shape[1] == 2:
        return a[:, 0] + 1j * a[:, 1]
    raise AssertionError(f"complex values in an unknown form: {v!r}")


DIV_OP = ("Operating Point", [("v(in)", "voltage"), ("v(mid)", "voltage"), ("i(v_vin)", "current")],
          [[1.0, 0.75, -2.5e-4]])


def test_parse_raw_op(tmp_path):
    path = write_raw(tmp_path / "result.raw", *DIV_OP)
    v = _vectors(_ng().parse_raw(path), ["v(in)", "v(mid)", "i(v_vin)"])
    assert np.asarray(v["v(mid)"], float).tolist() == [0.75]
    assert np.asarray(v["i(v_vin)"], float).tolist() == [-2.5e-4]
    assert np.asarray(v["v(in)"], float).tolist() == [1.0]


def test_parse_raw_accepts_str_path(tmp_path):
    path = write_raw(tmp_path / "result.raw", *DIV_OP)
    v = _vectors(_ng().parse_raw(str(path)), ["v(mid)"])
    assert np.asarray(v["v(mid)"], float).tolist() == [0.75]


def test_parse_raw_dc(tmp_path):
    V = np.linspace(0.30, 0.65, 8)
    I = -1e-14 * np.expm1(V / 0.025852)
    rows = [[v, i, v - 0.1] for v, i in zip(V, I)]
    path = write_raw(tmp_path / "result.raw", "DC transfer characteristic",
                     [("v(v-sweep)", "voltage"), ("i(v_vin)", "current"), ("v(a,b)", "notype")], rows)
    v = _vectors(_ng().parse_raw(path), ["v(v-sweep)", "i(v_vin)", "v(a,b)"])
    assert np.allclose(np.asarray(v["v(v-sweep)"], float), V, rtol=1e-14, atol=0)
    assert np.allclose(np.asarray(v["i(v_vin)"], float), I, rtol=1e-14, atol=0)
    assert np.allclose(np.asarray(v["v(a,b)"], float), V - 0.1, rtol=1e-14, atol=0)


def test_parse_raw_isource_dc(tmp_path):
    rows = [[0.0, 0.0], [5e-4, 0.5], [1e-3, 1.0]]
    path = write_raw(tmp_path / "result.raw", "DC transfer characteristic",
                     [("i(i-sweep)", "current"), ("v(a)", "voltage")], rows)
    v = _vectors(_ng().parse_raw(path), ["i(i-sweep)", "v(a)"])
    assert np.asarray(v["v(a)"], float).tolist() == [0.0, 0.5, 1.0]


def test_parse_raw_tran(tmp_path):
    t = np.array([0.0, 3e-11, 6e-11, 1.2e-10, 1e-8, 2e-8])
    y = 1 - np.exp(-t / 1e-6)
    path = write_raw(tmp_path / "result.raw", "Transient Analysis", [("time", "time"), ("v(out)", "voltage")],
                     [[a, b] for a, b in zip(t, y)])
    v = _vectors(_ng().parse_raw(path), ["time", "v(out)"])
    assert np.allclose(np.asarray(v["time"], float), t, rtol=1e-14, atol=0)
    assert np.allclose(np.asarray(v["v(out)"], float), y, rtol=1e-14, atol=1e-300)


def test_parse_raw_ac_complex(tmp_path):
    f = 10.0 ** (1 + np.arange(11) / 10)
    h = 1 / (1 + 2j * np.pi * f * 1e-6)
    cur = -(1 - h) / 1e3
    rows = [[complex(fi, 0.0), hi, ci] for fi, hi, ci in zip(f, h, cur)]
    path = write_raw(tmp_path / "result.raw", "AC Analysis",
                     [("frequency", "frequency grid=3"), ("v(out)", "voltage"), ("i(v_vin)", "current")], rows,
                     complex_=True)
    v = _vectors(_ng().parse_raw(path), ["frequency", "v(out)", "i(v_vin)"])
    assert np.allclose(_as_complex(v["v(out)"]), h, rtol=1e-14, atol=0)
    assert np.allclose(_as_complex(v["i(v_vin)"]), cur, rtol=1e-14, atol=0)
    fv = np.asarray(v["frequency"])
    fr = fv.real if np.iscomplexobj(fv) else (_as_complex(fv).real if np.asarray(fv).ndim == 2 else fv)
    assert np.allclose(np.asarray(fr, float), f, rtol=1e-14, atol=0)


# ------------------------------------------------------------------------------------------------ availability
def _prefix(root, ngspice=None, openvaf=None, mode=0o755):
    b = Path(root) / "bin"
    b.mkdir(parents=True, exist_ok=True)
    for name, body in (("ngspice", ngspice), ("openvaf", openvaf)):
        if body is not None:
            (b / name).write_text(body)
            (b / name).chmod(mode)
    return Path(root)


def test_not_available_with_empty_prefix(monkeypatch, tmp_path):
    from simws.engines.base import EngineNotAvailable
    monkeypatch.setenv("SIMWS_SPICE_PREFIX", str(tmp_path))
    mod = _ng()
    assert mod.available() is False
    assert mod.openvaf_available() is False
    with pytest.raises(EngineNotAvailable):
        mod.run(_lib().divider(), tmp_path / "work")
    with pytest.raises(EngineNotAvailable):
        mod.run(_lib().diode_iv(model="va"), tmp_path / "work2")


def test_available_with_executables(monkeypatch, tmp_path):
    root = _prefix(tmp_path / "pre", ngspice="#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("SIMWS_SPICE_PREFIX", str(root))
    assert _ng().available() is True and _ng().openvaf_available() is False
    _prefix(root, openvaf="#!/bin/sh\nexit 0\n")
    assert _ng().openvaf_available() is True


def test_va_needs_openvaf(monkeypatch, tmp_path):
    from simws.engines.base import EngineNotAvailable
    marker = tmp_path / "started"
    root = _prefix(tmp_path / "pre", ngspice=f"#!/bin/sh\ntouch {marker}\nexit 0\n")
    monkeypatch.setenv("SIMWS_SPICE_PREFIX", str(root))
    monkeypatch.setenv("SIMWS_CACHE", str(tmp_path / "cache"))
    with pytest.raises(EngineNotAvailable):
        _ng().run(_lib().diode_iv(model="va"), tmp_path / "work")
    assert not marker.exists()


def test_run_refuses_unsupported(monkeypatch, tmp_path):
    from simws.engines.base import Unsupported
    marker = tmp_path / "started"
    root = _prefix(tmp_path / "pre", ngspice=f"#!/bin/sh\ntouch {marker}\nexit 0\n",
                   openvaf=f"#!/bin/sh\ntouch {marker}\nexit 0\n")
    monkeypatch.setenv("SIMWS_SPICE_PREFIX", str(root))
    with pytest.raises(Unsupported) as ei:
        _ng().run(_big_problem(10001), tmp_path / "work")
    assert any(s.startswith("max_elements:") for s in ei.value.reasons), ei.value.reasons
    assert not marker.exists()


# ------------------------------------------------------------------------------------------------ stand-in executables
def _fake_ngspice(raw_src, rec, lines=(), exit_code=0, write_raw_file=True):
    """A stand-in ngspice: records its arguments and folder, copies every file argument into rec/, prints `lines`
    (also into the file given after -o, if any), copies a prepared raw file to result.raw in its folder."""
    body = ["#!/bin/sh", f'echo "$@" >> {rec}/args.txt', f"pwd >> {rec}/cwd.txt", f"echo x >> {rec}/calls.txt",
            "i=0", 'log=""', "prev=\"\"",
            'for a in "$@"; do i=$((i+1)); if [ -f "$a" ]; then cp "$a" ' + f'"{rec}/arg_$i"; fi; '
            'if [ "$prev" = "-o" ]; then log="$a"; fi; prev="$a"; done',
            'echo "fake ngspice running"']
    for x in lines:
        body.append(f'echo "{x}"')
        body.append(f'if [ -n "$log" ]; then echo "{x}" >> "$log"; fi')
    if write_raw_file and raw_src is not None:
        body.append(f"cp {raw_src} result.raw")
    body.append(f"exit {exit_code}")
    return "\n".join(body) + "\n"


def _fake_openvaf(rec, tag="v1"):
    return ("#!/bin/sh\n"
            f"# fake openvaf {tag}\n"
            f'echo "$@" >> {rec}/openvaf_calls.txt\n'
            'out=""; in=""\n'
            'while [ $# -gt 0 ]; do case "$1" in -o|--output) out="$2"; shift 2;; *.va) in="$1"; shift;; '
            '*) shift;; esac; done\n'
            '[ -z "$out" ] && out="${in%.va}.osdi"\n'
            f'echo "fake osdi {tag}" > "$out"\n'
            "exit 0\n")


def _seen_netlists(rec):
    out = []
    for f in sorted(Path(rec).glob("arg_*")):
        t = f.read_text(errors="replace")
        if ".control" in t.lower():
            out.append(t)
    return out


def _calls(rec, name):
    f = Path(rec) / name
    return len(f.read_text().splitlines()) if f.exists() else 0


@pytest.fixture
def stand_in(monkeypatch, tmp_path):
    """install(raw_rows...) -> (rec folder, prefix). SIMWS_CACHE points into tmp_path."""
    rec = tmp_path / "rec"
    rec.mkdir()
    monkeypatch.setenv("SIMWS_CACHE", str(tmp_path / "cache"))

    def install(raw_path, openvaf=True, tag="v1", **k):
        root = _prefix(tmp_path / "pre", ngspice=_fake_ngspice(raw_path, rec, **k),
                       openvaf=_fake_openvaf(rec, tag) if openvaf else None)
        monkeypatch.setenv("SIMWS_SPICE_PREFIX", str(root))
        return rec, root
    return install


def test_run_op_with_stand_in(stand_in, tmp_path):
    from simws.engines.base import CircuitResult
    raw = write_raw(tmp_path / "prepared.raw", *DIV_OP)
    rec, _ = stand_in(raw)
    p = _lib().divider()
    work = tmp_path / "work"
    work.mkdir()
    r = _ng().run(p, work)
    assert isinstance(r, CircuitResult)
    assert r.engine == "ngspice" and r.problem_id == p.problem_id() and r.analysis == "op"
    assert r.axis_name == "" and r.axis_unit == ""
    assert sorted(r.signals) == sorted(p.probes)
    assert r.signals["v(mid)"] == [0.75]
    assert r.signals["i(vin)"] == [-2.5e-4]                              # the SPICE sign is kept
    assert r.units == {"v(mid)": "V", "i(vin)": "A"}
    assert r.signals_imag == {}
    assert not any(f.startswith("circuit:external_library") for f in r.flags), r.flags
    seen = _seen_netlists(rec)
    assert seen and seen[-1] == _ng().build_netlist(p)                   # the netlist of A9.4 is what runs
    assert _calls(rec, "openvaf_calls.txt") == 0


def test_run_dc_with_stand_in(stand_in, tmp_path):
    p = _lib().diode_iv()
    V = np.linspace(0.30, 0.65, 8)
    I = -1e-14 * np.expm1(V / 0.025852)
    raw = write_raw(tmp_path / "prepared.raw", "DC transfer characteristic",
                    [("v(v-sweep)", "voltage"), ("i(v_vin)", "current")], [[v, i] for v, i in zip(V, I)])
    stand_in(raw)
    r = _ng().run(p, tmp_path)
    assert r.analysis == "dc" and r.axis_name.lower() == "vin" and r.axis_unit == "V"
    assert np.allclose(r.axis, V, rtol=1e-14, atol=0)
    assert np.allclose(r.signals["i(vin)"], I, rtol=1e-14, atol=0)
    assert r.units == {"i(vin)": "A"}


def test_run_dc_isource_axis_unit(stand_in, tmp_path):
    d = {"elements": [{"name": "i1", "kind": "isource", "nodes": ["0", "a"], "value": 0.0},
                      {"name": "r1", "kind": "resistor", "nodes": ["a", "0"], "value": 1e3}],
         "analysis": {"kind": "dc", "source": "i1", "start": 0.0, "stop": 1e-3, "step": 5e-4}, "probes": ["v(a)"]}
    raw = write_raw(tmp_path / "prepared.raw", "DC transfer characteristic",
                    [("i(i-sweep)", "current"), ("v(a)", "voltage")], [[0.0, 0.0], [5e-4, 0.5], [1e-3, 1.0]])
    stand_in(raw)
    r = _ng().run(_cp(d), tmp_path)
    assert r.axis_name.lower() == "i1" and r.axis_unit == "A"
    assert np.allclose(r.axis, [0.0, 5e-4, 1e-3]) and r.signals["v(a)"] == [0.0, 0.5, 1.0]
    assert r.units == {"v(a)": "V"}


def test_run_ac_with_stand_in(stand_in, tmp_path):
    p = _lib().rc_lowpass(points_per_decade=1, f_start_Hz=10.0, f_stop_Hz=1e3)
    f = np.array([10.0, 100.0, 1000.0])
    h = 1 / (1 + 2j * np.pi * f * 1e-6)
    raw = write_raw(tmp_path / "prepared.raw", "AC Analysis", [("frequency", "frequency grid=3"),
                                                                ("v(out)", "voltage")],
                    [[complex(a, 0), b] for a, b in zip(f, h)], complex_=True)
    stand_in(raw)
    r = _ng().run(p, tmp_path)
    assert r.analysis == "ac" and r.axis_name == "f" and r.axis_unit == "Hz"
    assert np.allclose(r.axis, f, rtol=1e-14)
    assert np.allclose(r.signals["v(out)"], h.real, rtol=1e-13, atol=0)
    assert np.allclose(r.signals_imag["v(out)"], h.imag, rtol=1e-13, atol=0)
    assert r.units == {"v(out)": "V"}


def test_run_tran_with_stand_in(stand_in, tmp_path):
    p = _lib().rc_lowpass(analysis="tran")
    t = np.array([0.0, 1e-8, 2e-8, 5e-6, 1e-5])
    y = 1 - np.exp(-t / 1e-6)
    raw = write_raw(tmp_path / "prepared.raw", "Transient Analysis", [("time", "time"), ("v(out)", "voltage")],
                    [[a, b] for a, b in zip(t, y)])
    stand_in(raw)
    r = _ng().run(p, tmp_path)
    assert r.analysis == "tran" and r.axis_name == "t" and r.axis_unit == "s"
    assert np.allclose(r.axis, t) and np.allclose(r.signals["v(out)"], y, rtol=1e-14, atol=1e-300)
    assert r.signals_imag == {}


def test_run_difference_probe_and_case(stand_in, tmp_path):
    p = _lib().divider().with_changes({"probes": ["v(IN,mid)", "i(VIN)"]})
    raw = write_raw(tmp_path / "prepared.raw", "Operating Point",
                    [("v(in)", "voltage"), ("v(in,mid)", "notype"), ("i(v_vin)", "current")], [[1.0, 0.25, -2.5e-4]])
    stand_in(raw)
    r = _ng().run(p, tmp_path)
    assert sorted(r.signals) == sorted(p.probes)                          # keyed by the probe as written
    assert r.signals["v(IN,mid)"] == [0.25] and r.signals["i(VIN)"] == [-2.5e-4]
    assert r.units == {"v(IN,mid)": "V", "i(VIN)": "A"}


def test_run_failure_raises_with_log_tail(stand_in, tmp_path):
    lines = [f"L{i:03d} line" for i in range(1, 31)]
    stand_in(None, lines=lines, exit_code=1, write_raw_file=False)
    with pytest.raises(RuntimeError) as ei:
        _ng().run(_lib().divider(), tmp_path / "w")
    msg = str(ei.value)
    for i in range(11, 31):
        assert f"L{i:03d}" in msg, (i, msg)
    for i in (1, 5, 9):
        assert f"L{i:03d}" not in msg, (i, msg)


def test_run_no_raw_file_raises_with_log_tail(stand_in, tmp_path):
    lines = [f"M{i:03d} line" for i in range(1, 26)] + ["Error during 'write': no writable vector found."]
    stand_in(None, lines=lines, exit_code=0, write_raw_file=False)
    with pytest.raises(RuntimeError) as ei:
        _ng().run(_lib().divider(), tmp_path / "w")
    msg = str(ei.value)
    assert "no writable vector found" in msg and "M007" in msg and "M005" not in msg, msg


def test_run_library_flag(stand_in, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    libf = tmp_path / "home" / "pdk" / "sub.lib"
    libf.parent.mkdir(parents=True)
    libf.write_text(".subckt myres a b\nR1 a b 1k\n.ends\n")
    incf = tmp_path / "other.inc"
    incf.write_text(".subckt myres2 a b\nR1 a b 3k\n.ends\n")
    d = {"elements": [{"name": "vin", "kind": "vsource", "nodes": ["in", "0"], "value": 1.0},
                      {"name": "x1", "kind": "subckt", "nodes": ["in", "mid"], "model": "myres"},
                      {"name": "x2", "kind": "subckt", "nodes": ["mid", "0"], "model": "myres2"}],
         "libraries": [{"path": "~/pdk/sub.lib", "section": "typ"}, {"path": str(incf)}],
         "analysis": {"kind": "op"}, "probes": ["v(mid)"]}
    raw = write_raw(tmp_path / "prepared.raw", "Operating Point", [("v(in)", "voltage"), ("v(mid)", "voltage")],
                    [[1.0, 0.75]])
    stand_in(raw)
    r = _ng().run(_cp(d), tmp_path / "w")
    for f in (libf, incf):
        h = hashlib.sha256(f.read_bytes()).hexdigest()[:12]
        exp = f"circuit:external_library: {os.path.expanduser(str(f))} sha256 {h}"
        assert exp in r.flags, (exp, r.flags)
    assert len([f for f in r.flags if f.startswith("circuit:external_library:")]) == 2


DC_RAW_VA = ("DC transfer characteristic", [("v(v-sweep)", "voltage"), ("i(v_vin)", "current")],
             [[v, -1e-14 * np.expm1(v / 0.025852)] for v in np.linspace(0.30, 0.65, 8)])


def _osdi_dirs(cache):
    d = Path(cache) / "osdi"
    return sorted(x for x in d.iterdir() if x.is_dir()) if d.exists() else []


def test_run_va_compiles_and_caches(stand_in, tmp_path):
    raw = write_raw(tmp_path / "prepared.raw", *DC_RAW_VA)
    rec, root = stand_in(raw)
    cache = tmp_path / "cache"
    p = _lib().diode_iv(model="va")
    r1 = _ng().run(p, tmp_path / "w1")
    assert _calls(rec, "openvaf_calls.txt") == 1
    dirs = _osdi_dirs(cache)
    assert len(dirs) == 1, dirs
    assert re.fullmatch(r"[0-9a-f]{64}", dirs[0].name), dirs[0].name      # a SHA-256
    files = [f for f in dirs[0].rglob("*") if f.is_file()]
    osdi_files = [f for f in files if "fake osdi" in f.read_text(errors="replace")]
    assert len(osdi_files) == 1, files
    assert not r1.diagnostics.get("osdi_cached"), r1.diagnostics
    net = _seen_netlists(rec)[-1]
    pre = [shlex.split(x.strip()) for x in net.splitlines() if x.strip().lower().startswith("pre_osdi")]
    assert len(pre) == 1 and os.path.realpath(pre[0][1]) == os.path.realpath(osdi_files[0]), (pre, osdi_files)
    assert net == _ng().build_netlist(p, osdi={"shockley_diode": pre[0][1]})
    r2 = _ng().run(p, tmp_path / "w2")
    assert _calls(rec, "openvaf_calls.txt") == 1                          # reused, not compiled again
    assert r2.diagnostics.get("osdi_cached") is True, r2.diagnostics
    assert r2.signals == r1.signals
    assert _osdi_dirs(cache) == dirs


def test_osdi_key_depends_on_source_and_openvaf(stand_in, tmp_path):
    raw = write_raw(tmp_path / "prepared.raw", *DC_RAW_VA)
    rec, root = stand_in(raw)
    cache = tmp_path / "cache"
    p = _lib().diode_iv(model="va")
    _ng().run(p, tmp_path / "w1")
    k1 = _osdi_dirs(cache)
    q = p.with_changes({"models.0.va.source": p.models[0].va.source + "\n// changed\n"})
    r = _ng().run(q, tmp_path / "w2")
    k2 = _osdi_dirs(cache)
    assert len(k2) == 2 and not r.diagnostics.get("osdi_cached")
    assert _calls(rec, "openvaf_calls.txt") == 2
    # another OpenVAF executable (different bytes): another key, compiled again
    (root / "bin" / "openvaf").write_text(_fake_openvaf(rec, tag="v2"))
    r = _ng().run(p, tmp_path / "w3")
    assert len(_osdi_dirs(cache)) == 3 and not r.diagnostics.get("osdi_cached")
    assert _calls(rec, "openvaf_calls.txt") == 3


def test_osdi_cache_default_location(stand_in, tmp_path, monkeypatch):
    raw = write_raw(tmp_path / "prepared.raw", *DC_RAW_VA)
    stand_in(raw)
    monkeypatch.delenv("SIMWS_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _ng().run(_lib().diode_iv(model="va"), tmp_path / "w")
    assert len(_osdi_dirs(tmp_path / "home" / ".cache" / "simws")) == 1


def test_compile_va_direct(stand_in, tmp_path):
    from simws.spec import VerilogA
    rec, _ = stand_in(None)
    va = VerilogA(module="shockley_diode", source=_lib().SHOCKLEY_VA)
    cdir = tmp_path / "cc"
    out = _ng().compile_va(va, cdir)
    assert isinstance(out, Path) and out.is_file()
    assert os.path.realpath(out).startswith(os.path.realpath(cdir))
    assert "fake osdi" in out.read_text()
    assert _ng().compile_va(va, cdir) == out
    assert _calls(rec, "openvaf_calls.txt") == 1


# ------------------------------------------------------------------------------------------------ real engine
def test_engine_divider(tmp_path):
    _need_engine()
    p = _lib().divider()
    r = _ng().run(p, tmp_path)
    assert r.engine == "ngspice" and r.analysis == "op" and r.problem_id == p.problem_id()
    assert r.units == {"v(mid)": "V", "i(vin)": "A"}
    assert r.signals["v(mid)"][0] == pytest.approx(0.75, abs=1e-9)
    assert r.signals["i(vin)"][0] == pytest.approx(-2.5e-4, abs=1e-12)


def test_engine_diode_dc(tmp_path):
    _need_engine()
    p = _lib().diode_iv()
    r = _ng().run(p, tmp_path)
    assert r.analysis == "dc" and r.axis_name.lower() == "vin" and r.axis_unit == "V"
    assert len(r.axis) == 8 and np.allclose(r.axis, np.linspace(0.30, 0.65, 8), atol=1e-12)
    assert all(i < 0 for i in r.signals["i(vin)"])                       # current enters the source at node "a"


def test_engine_ac_and_tran(tmp_path):
    _need_engine()
    r = _ng().run(_lib().rc_lowpass(), tmp_path / "ac")
    assert r.analysis == "ac" and r.axis_name == "f" and r.axis_unit == "Hz" and len(r.axis) == 71
    assert len(r.signals["v(out)"]) == 71 and len(r.signals_imag["v(out)"]) == 71
    t = _ng().run(_lib().rc_lowpass(analysis="tran"), tmp_path / "tran")
    assert t.analysis == "tran" and t.axis_name == "t" and t.axis_unit == "s"
    assert t.axis[0] == 0.0 and t.axis[-1] == pytest.approx(1e-5, rel=1e-9)
    assert max(np.diff(t.axis)) <= 1e-8 * (1 + 1e-6)


def test_engine_difference_probe(tmp_path):
    _need_engine()
    p = _lib().divider().with_changes({"probes": ["v(in,mid)", "v(mid)", "i(vin)"]})
    r = _ng().run(p, tmp_path)
    assert r.signals["v(in,mid)"][0] == pytest.approx(0.25, abs=1e-9)


def test_engine_library_and_flag(tmp_path, monkeypatch):
    _need_engine()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    libf = tmp_path / "home" / "pdk" / "res.lib"
    libf.parent.mkdir(parents=True)
    libf.write_text("* test library\n.lib typ\n.subckt myres a b\nR1 a b 1k\n.ends\n.endl typ\n")
    incf = tmp_path / "res3k.inc"
    incf.write_text(".subckt myres3k a b\nR1 a b 3k\n.ends\n")
    d = {"elements": [{"name": "vin", "kind": "vsource", "nodes": ["in", "0"], "value": 1.0},
                      {"name": "x1", "kind": "subckt", "nodes": ["in", "mid"], "model": "myres"},
                      {"name": "x2", "kind": "subckt", "nodes": ["mid", "0"], "model": "myres3k"}],
         "libraries": [{"path": "~/pdk/res.lib", "section": "typ"}, {"path": str(incf)}],
         "analysis": {"kind": "op"}, "probes": ["v(mid)", "i(vin)"]}
    r = _ng().run(_cp(d), tmp_path / "w")
    assert r.signals["v(mid)"][0] == pytest.approx(0.75, abs=1e-9)
    for f in (libf, incf):
        h = hashlib.sha256(f.read_bytes()).hexdigest()[:12]
        assert f"circuit:external_library: {f} sha256 {h}" in r.flags, r.flags


def test_engine_missing_library_raises(tmp_path):
    _need_engine()
    d = {"elements": [{"name": "vin", "kind": "vsource", "nodes": ["in", "0"], "value": 1.0},
                      {"name": "x1", "kind": "subckt", "nodes": ["in", "0"], "model": "nothing"}],
         "libraries": [{"path": str(tmp_path / "missing.lib")}], "analysis": {"kind": "op"}, "probes": ["v(in)"]}
    with pytest.raises(RuntimeError):
        _ng().run(_cp(d), tmp_path / "w")


def test_engine_va_runs(tmp_path, monkeypatch):
    _need_openvaf()
    monkeypatch.setenv("SIMWS_CACHE", str(tmp_path / "cache"))
    r = _ng().run(_lib().diode_iv(model="va"), tmp_path / "w")
    assert len(r.signals["i(vin)"]) == 8 and all(i < 0 for i in r.signals["i(vin)"])
    assert len(_osdi_dirs(tmp_path / "cache")) == 1
