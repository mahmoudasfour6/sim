"""Contract amendment A6 (docs/ARCHITECTURE_V1.md): small-signal (AC) analysis on the new path.

Written from the contract text alone (and the legacy reference simws.tcad.small_signal / devsim.hbt_ac), without
reading the implementation. Every expected number is derived in the test: by hand from a two-port written out here
(A6.2 / A6.5 through a monkeypatched engine run), from closed-form device physics (the DEVSIM diode sanity check),
or from the legacy adapter run in the same test (R9).

Notes on the contract text used here:
- `Bias` and `SmallSignal` are pydantic models; R9 writes `Bias("collector", 1.0, 0.1)` positionally, which a plain
  pydantic model does not accept, so these tests use keyword arguments everywhere.
- The A6.6 refusal is checked for status and message only (A6.6 does not say "before anything runs", unlike A4.2).
"""
import math

import numpy as np
import pydantic
import pytest

from simws.spec.library import npn_hbt_2d, pn_diode_1d
from simws.spec.schema import Problem

VE = pydantic.ValidationError
FLOOR_1D = 1e-4          # A/m^2 (contract section 5)
FLOOR_2D = 1e-8          # A/m
RTOL = 1e-12             # fake engine: pure arithmetic and unit factors only
Y_NAMES = [f"ac.y{ij}_{part}" for ij in ("11", "12", "21", "22") for part in ("re", "im")]


def _ss(**kw):
    from simws.spec.schema import SmallSignal
    return SmallSignal(**kw)


def _bias(contact, value_V, step_V=None):
    from simws.spec.schema import Bias
    return Bias(contact=contact, value_V=value_V) if step_V is None else \
        Bias(contact=contact, value_V=value_V, step_V=step_V)


def _with_analysis(p: Problem, analysis) -> Problem:
    """A validated copy of p with another analysis (Problem.model_validate on the dumped dict)."""
    d = p.model_dump(mode="json")
    d["analysis"] = analysis.model_dump(mode="json") if hasattr(analysis, "model_dump") else analysis
    return Problem.model_validate(d)


def diode_ss(freqs=(1e3,), nodes=201, bias_V=0.0):
    return _with_analysis(pn_diode_1d(nodes=nodes), _ss(bias=[_bias("anode", bias_V)], ports=("anode", "cathode"),
                                                        frequencies_Hz=list(freqs)))


def hbt_ss(freqs=(1e6, 1e8, 5e9)):
    return _with_analysis(npn_hbt_2d(), _ss(bias=[_bias("collector", 1.0, 0.1), _bias("base", 0.75, 0.05)],
                                            ports=("base", "collector"), frequencies_Hz=list(freqs)))


def _params(p, **extra):
    return {"problem": p.model_dump(mode="json"), **extra}


# ================================================================================================ A6.1 description
def test_spec_version_unchanged():
    from simws.spec.schema import SPEC_VERSION
    assert SPEC_VERSION == "1.0"


# Problem ids computed on 16 Nov 2026 with the PRE-A6 schema (git HEAD ae4978f, simws/spec unmodified, checked with
# `git diff --quiet HEAD -- simws/spec`) by running exactly these library calls. They are measured values of the
# unchanged description, not invented ones: A6.1 says every existing problem id stays as it is.
PRE_A6_IDS = {
    "pn_diode_1d()": "52b23f57732ec9aa197c56d61f3fb049",
    "pn_diode_1d(V=[0.3, 0.5])": "99e0ef3d98e4796e955a89e4217de128",
    "pn_diode_1d(nodes=401)": "e70277bcbb7090171645e64050bf79fe",
    "npn_hbt_2d()": "55cf78ed7e298d344d60b782acad29b8",
    "npn_hbt_2d(VBE=[0.6, 0.7])": "0440723e17e1b6cd17f9b51527d9145f",
    "npn_hbt_2d(eb_refine=True, cryo=True, T=77.0, VBE=[1.0])": "f4a4b31381e43abf4a43c9320917f29b",
}


@pytest.mark.parametrize("expr", sorted(PRE_A6_IDS))
def test_existing_problem_ids_unchanged(expr):
    p = eval(expr, {"pn_diode_1d": pn_diode_1d, "npn_hbt_2d": npn_hbt_2d})
    assert p.problem_id() == PRE_A6_IDS[expr]


def test_existing_problem_dump_has_no_new_keys():
    d = pn_diode_1d(V=[0.3]).model_dump(mode="json")
    assert set(d) == {"spec_version", "device", "models", "analysis", "numerics"}
    assert set(d["analysis"]) == {"kind", "contact", "values_V", "step_V", "tracking", "fixed_V"}
    assert set(pn_diode_1d().model_dump(mode="json")["analysis"]) == {"kind"}


def test_small_signal_fields_and_defaults():
    from simws.spec.schema import Bias, SmallSignal
    b = Bias(contact="anode", value_V=0.3)
    assert b.step_V == 0.05
    s = SmallSignal(bias=[b], ports=("anode", "cathode"), frequencies_Hz=[1e3, 1e6])
    assert s.kind == "small_signal"
    assert tuple(s.ports) == ("anode", "cathode")
    with pytest.raises(VE):
        Bias(contact="anode", value_V=0.3, colour="red")
    with pytest.raises(VE):
        SmallSignal(bias=[b], ports=("anode", "cathode"), frequencies_Hz=[1e3], noise=True)


@pytest.mark.parametrize("step", [0.0, -0.05])
def test_bias_step_positive(step):
    with pytest.raises(VE):
        _bias("anode", 0.3, step)


@pytest.mark.parametrize("freqs", [[], [0.0], [-1e3], [1e3, 0.0], [1e3, 1e3], [1e6, 1e3], [1e3, 1e6, 1e5]])
def test_frequencies_refused(freqs):
    with pytest.raises(VE):
        diode_ss(freqs=freqs)


@pytest.mark.parametrize("freqs", [[1e3], [1e-3, 1.0, 1e12]])
def test_frequencies_accepted(freqs):
    assert diode_ss(freqs=freqs).analysis.frequencies_Hz == freqs


def test_equal_ports_refused():
    with pytest.raises(VE):
        diode_ss_ports = _ss(bias=[_bias("anode", 0.0)], ports=("anode", "anode"), frequencies_Hz=[1e3])
        Problem(device=pn_diode_1d().device, analysis=diode_ss_ports)


@pytest.mark.parametrize("bias,ports", [
    ([("gate", 0.0)], ("anode", "cathode")),                     # unknown contact in bias
    ([("anode", 0.0)], ("anode", "drain")),                      # unknown contact in ports
    ([("anode", 0.0)], ("source", "cathode")),
    ([("anode", 0.0), ("anode", 0.1)], ("anode", "cathode")),    # a contact twice in bias
])
def test_contact_rules(bias, ports):
    dev = pn_diode_1d().device
    with pytest.raises(VE):
        Problem(device=dev, analysis=_ss(bias=[_bias(c, v) for c, v in bias], ports=ports, frequencies_Hz=[1e3]))


def test_contact_rules_from_dict():
    d = pn_diode_1d().model_dump(mode="json")
    d["analysis"] = {"kind": "small_signal", "bias": [{"contact": "gate", "value_V": 0.0}],
                     "ports": ["anode", "cathode"], "frequencies_Hz": [1e3]}
    with pytest.raises(VE):
        Problem.model_validate(d)


def test_valid_problems_and_discriminator():
    from simws.spec.schema import SmallSignal
    p = diode_ss(freqs=[1e3, 1e4])
    assert isinstance(p.analysis, SmallSignal) and p.analysis.kind == "small_signal"
    d = p.model_dump(mode="json")
    assert d["analysis"]["kind"] == "small_signal"
    q = Problem.model_validate(d)
    assert isinstance(q.analysis, SmallSignal)
    assert q.problem_id() == p.problem_id()
    assert Problem.model_validate_json(p.model_dump_json()).problem_id() == p.problem_id()
    # a bias on both contacts, other ports order, the HBT with an AC-grounded emitter
    Problem(device=pn_diode_1d().device, analysis=_ss(bias=[_bias("cathode", 0.0), _bias("anode", -1.0)],
                                                      ports=("cathode", "anode"), frequencies_Hz=[1.0]))
    h = hbt_ss()
    assert [b.contact for b in h.analysis.bias] == ["collector", "base"]          # order kept (ramp order)
    assert h.analysis.bias[0].step_V == 0.1
    # the analysis is part of the id
    assert diode_ss(freqs=[1e3]).problem_id() != diode_ss(freqs=[1e4]).problem_id()
    assert diode_ss().problem_id() != pn_diode_1d(nodes=201).problem_id()


def test_analysis_union_contains_small_signal():
    import typing

    from simws.spec import schema
    assert schema.SmallSignal in typing.get_args(schema.Analysis)
    assert schema.Equilibrium in typing.get_args(schema.Analysis)
    assert schema.DCSweep in typing.get_args(schema.Analysis)


def test_with_changes_on_small_signal():
    p = hbt_ss()
    q = p.with_changes({"analysis.frequencies_Hz": [1e7, 1e9]})
    assert q.analysis.frequencies_Hz == [1e7, 1e9]
    with pytest.raises(VE):
        p.with_changes({"analysis.ports": ["base", "base"]})


# ================================================================================================ A6.2 EngineResult
def test_engine_result_new_fields_default_empty():
    from simws.engines.base import EngineResult
    r = EngineResult(engine="toy", problem_id="0" * 32, analysis="dc_sweep", sweep_contact="anode", sweep_V=[0.1],
                     currents={"anode": [1.0], "cathode": [-1.0]}, current_unit="A/cm**2")
    assert r.frequencies_Hz == [] and r.y_real == {} and r.y_imag == {} and r.y_unit is None


def test_engine_result_small_signal():
    import json

    from simws.engines.base import EngineResult
    kw = dict(engine="toy", problem_id="0" * 32, analysis="small_signal", sweep_V=[0.0],
              currents={"a": [1.0], "c": [-1.0]}, current_unit="A/cm**2", frequencies_Hz=[1e3, 1e4],
              y_real={k: [1.0, 1.0] for k in ("11", "12", "21", "22")},
              y_imag={k: [0.1, 1.0] for k in ("11", "12", "21", "22")})
    for unit in ("S/cm**2", "S/cm"):
        r = EngineResult(**kw, y_unit=unit)
        assert r.sweep_contact is None and r.analysis == "small_signal"
        json.loads(r.model_dump_json())
    with pytest.raises(VE):
        EngineResult(**kw, y_unit="S")


# ================================================================================================ A6.5 adapter (fake)
# The DEVSIM engine module's run() is replaced (as in test_device_adapter.py / test_mesh_study.py), so the adapter's
# A6.5 arithmetic is checked against a two-port written out here. Test values, not physics:
# hybrid-pi transistor per cm of depth: y11 = g_pi + j w (C_pi + C_mu), y12 = -j w C_mu, y21 = g_m - j w C_mu,
# y22 = g_o + j w C_mu (port 1 = base, port 2 = collector).
HP = dict(g_m=4.0, g_pi=0.04, g_o=1e-3, C_pi=1.3e-11, C_mu=1e-12)          # S/cm, F/cm
# DC operating point (A/cm), binary-exact so that the three currents sum to exactly 0
DC_2D = {"base": 2.0 ** -10, "collector": 2.0 ** -3, "emitter": -(2.0 ** -10 + 2.0 ** -3)}


def hybrid_pi(f, g_m, g_pi, g_o, C_pi, C_mu):
    w = 2 * math.pi * f
    return {"11": complex(g_pi, w * (C_pi + C_mu)), "12": complex(0.0, -w * C_mu),
            "21": complex(g_m, -w * C_mu), "22": complex(g_o, w * C_mu)}


def make_fake(y_of, currents, calls, dim=2, flags=()):
    """Engine run() returning the small-signal EngineResult of the two-port y_of(f) -> {"11": complex, ...}."""
    def run(problem, workdir=None):
        from simws.engines.base import EngineResult
        p = problem if isinstance(problem, Problem) else Problem.model_validate(problem)
        calls.append(p)
        assert p.analysis.kind == "small_signal"
        fs = list(p.analysis.frequencies_Hz)
        ys = [y_of(f) for f in fs]
        return EngineResult(engine="devsim", problem_id=p.problem_id(), analysis="small_signal", sweep_contact=None,
                            sweep_V=[0.0], currents={k: [v] for k, v in currents.items()},
                            current_unit="A/cm" if dim == 2 else "A/cm**2", frequencies_Hz=fs,
                            y_real={k: [y[k].real for y in ys] for k in ("11", "12", "21", "22")},
                            y_imag={k: [y[k].imag for y in ys] for k in ("11", "12", "21", "22")},
                            y_unit="S/cm" if dim == 2 else "S/cm**2", flags=list(flags))
    return run


def _patch(monkeypatch, run, engine="devsim"):
    from simws import engines
    monkeypatch.setattr(engines.module(engine), "run", run)


def expected_figures(y_of, fs):
    """A6.5 written out: h21 per frequency; f_T single-pole with beta0 = |h21(f_lowest)| and |h21(f_highest)|;
    f_max = f sqrt(U) at the highest frequency; None where A6.3 says undefined."""
    h = [abs(y_of(f)["21"] / y_of(f)["11"]) for f in fs]
    fT = None
    if len(fs) >= 2 and h[0] > h[-1]:
        fT = h[0] * fs[-1] / math.sqrt((h[0] / h[-1]) ** 2 - 1)
    y = y_of(fs[-1])
    den = 4 * (y["11"].real * y["22"].real - y["12"].real * y["21"].real)
    U = abs(y["21"] - y["12"]) ** 2 / den if den > 0 else math.inf
    fmax = fs[-1] * math.sqrt(U) if (math.isfinite(U) and U > 0) else None
    return h, fT, fmax


def _names(r):
    return sorted(q.name for q in r.quantities)


def test_a6_5_hbt_quantities(hub, monkeypatch):
    fs = [1e6, 1e8, 5e9]
    y_of = lambda f: hybrid_pi(f, **HP)                                     # noqa: E731
    calls = []
    _patch(monkeypatch, make_fake(y_of, DC_2D, calls))
    p = hbt_ss(fs)
    r = hub.submit("device.devsim", params=_params(p))
    assert r.status == "ok", (r.status, r.message)
    assert len(calls) == 1 and calls[0].problem_id() == p.problem_id()
    expected_names = sorted(Y_NAMES + ["ac.h21_abs", "ac.fT", "ac.fmax", "dc.I.base", "dc.I.collector",
                                       "dc.I.emitter", "dc.kirchhoff_residual"])
    assert _names(r) == expected_names
    # Y parameters: S/m = S/cm x 100, axis f in Hz
    for ij in ("11", "12", "21", "22"):
        for part, get in (("re", lambda z: z.real), ("im", lambda z: z.imag)):
            q = r.q(f"ac.y{ij}_{part}")
            assert q.unit == "S/m", (q.name, q.unit)
            assert len(q.axes) == 1 and q.axes[0].name == "f" and q.axes[0].unit == "Hz"
            assert q.axes[0].values == fs
            exp = [get(y_of(f)[ij]) * 100.0 for f in fs]
            assert q.array() == pytest.approx(exp, rel=RTOL, abs=0.0), q.name
    h, fT, fmax = expected_figures(y_of, fs)
    qh = r.q("ac.h21_abs")
    assert qh.unit in ("", "1", "dimensionless"), qh.unit
    assert qh.axes[0].name == "f" and qh.axes[0].unit == "Hz" and qh.axes[0].values == fs
    assert qh.array() == pytest.approx(h, rel=RTOL)
    for name, val in (("ac.fT", fT), ("ac.fmax", fmax)):
        q = r.q(name)
        assert q.unit == "Hz" and q.axes == []
        assert isinstance(q.value, (int, float)), q.value
        assert q.value == pytest.approx(val, rel=RTOL), name
    # independent sanity on the hand numbers: the single-pole fit of an (almost) single-pole device
    assert fT == pytest.approx(HP["g_m"] / (2 * math.pi * (HP["C_pi"] + HP["C_mu"])), rel=0.05)
    assert fmax == pytest.approx(5e9 * HP["g_m"] / (2 * math.sqrt(HP["g_pi"] * HP["g_o"])), rel=1e-12)
    # operating point: scalars in A/m
    for c, v in DC_2D.items():
        q = r.q(f"dc.I.{c}")
        assert q.unit == "A/m" and q.axes == [] and isinstance(q.value, (int, float))
        assert q.value == pytest.approx(v * 100.0, rel=RTOL)
        assert q.conventions == {"terminal_current": "into_device+"}
    k = r.q("dc.kirchhoff_residual")
    assert k.axes == [] and k.value == pytest.approx(0.0, abs=1e-15)
    assert "ac:no_rolloff" not in r.flags and "ac:U_unbounded" not in r.flags


def test_a6_5_kirchhoff_residual_value(hub, monkeypatch):
    """dc.kirchhoff_residual "as for equilibrium" (A1.8): |sum of terminal currents| / (largest |I| + floor),
    floor 1e-8 A/m in 2D. Currents (A/cm): base 1e-3, collector 0.1, emitter -(0.101 - 1e-9): |sum| = 1e-9 A/cm
    = 1e-7 A/m, well inside the section-5 Kirchhoff rule 1e-4 x max|I| + floor."""
    cur = {"base": 1e-3, "collector": 0.1, "emitter": -(0.101 - 1e-9)}
    _patch(monkeypatch, make_fake(lambda f: hybrid_pi(f, **HP), cur, []))
    r = hub.submit("device.devsim", params=_params(hbt_ss([1e6, 5e9])))
    assert r.status == "ok", (r.status, r.message)
    I = [v * 100.0 for v in cur.values()]
    exp = abs(sum(I)) / (max(abs(x) for x in I) + FLOOR_2D)
    # 1e-6 relative: the sum is a cancellation of numbers ~10 A/m down to ~1e-7 A/m (rounding ~1e-15 / 1e-8)
    assert r.q("dc.kirchhoff_residual").value == pytest.approx(exp, rel=1e-6)


def test_a6_5_no_rolloff(hub, monkeypatch):
    """No capacitance: |h21| = g_m / g_pi at every frequency, so beta0 <= |h21| at the highest frequency."""
    hp = dict(HP, C_pi=0.0, C_mu=0.0)
    _patch(monkeypatch, make_fake(lambda f: hybrid_pi(f, **hp), DC_2D, []))
    fs = [1e6, 5e9]
    r = hub.submit("device.devsim", params=_params(hbt_ss(fs)))
    assert r.status == "ok", (r.status, r.message)
    assert "ac:no_rolloff" in r.flags, r.flags
    assert "ac.fT" not in _names(r)
    assert "ac:U_unbounded" not in r.flags
    _, _, fmax = expected_figures(lambda f: hybrid_pi(f, **hp), fs)
    assert r.q("ac.fmax").value == pytest.approx(fmax, rel=RTOL)


def test_a6_5_rising_h21_is_no_rolloff(hub, monkeypatch):
    """|h21| rising with frequency: y11 = g_pi / (1 + j f/f0) (Re y11 > 0) has |y11| falling, so with a constant
    y21 beta0 = |h21(f_lowest)| < |h21(f_highest)| -> no ac.fT, flag "ac:no_rolloff"."""
    def y_of(f):
        y11 = HP["g_pi"] / (1 + 1j * f / 1e8)
        return {"11": y11, "12": 0j, "21": complex(HP["g_m"], 0.0), "22": complex(HP["g_o"], 0.0)}
    _patch(monkeypatch, make_fake(y_of, DC_2D, []))
    r = hub.submit("device.devsim", params=_params(hbt_ss([1e6, 5e9])))
    assert r.status == "ok", (r.status, r.message)
    assert "ac:no_rolloff" in r.flags and "ac.fT" not in _names(r)


def test_a6_5_u_unbounded(hub, monkeypatch):
    """g_o < 0: Mason's denominator 4 (g_pi g_o - 0) < 0, U = inf at the highest frequency -> no ac.fmax."""
    hp = dict(HP, g_o=-1e-3)
    y_of = lambda f: hybrid_pi(f, **hp)                                     # noqa: E731
    _patch(monkeypatch, make_fake(y_of, DC_2D, []))
    fs = [1e6, 5e9]
    r = hub.submit("device.devsim", params=_params(hbt_ss(fs)))
    assert r.status == "ok", (r.status, r.message)
    assert "ac:U_unbounded" in r.flags, r.flags
    assert "ac.fmax" not in _names(r)
    _, fT, _ = expected_figures(y_of, fs)
    assert r.q("ac.fT").value == pytest.approx(fT, rel=RTOL)
    assert "ac:no_rolloff" not in r.flags


def test_a6_5_u_unbounded_only_at_highest_frequency_counts(hub, monkeypatch):
    """U is evaluated at the highest frequency only: a denominator <= 0 at a lower frequency does not matter."""
    def y_of(f):
        y = hybrid_pi(f, **HP)
        if f < 1e9:
            y["22"] = complex(-1e-3, y["22"].imag)
        return y
    _patch(monkeypatch, make_fake(y_of, DC_2D, []))
    fs = [1e6, 5e9]
    r = hub.submit("device.devsim", params=_params(hbt_ss(fs)))
    assert r.status == "ok", (r.status, r.message)
    assert "ac:U_unbounded" not in r.flags
    _, _, fmax = expected_figures(y_of, fs)
    assert r.q("ac.fmax").value == pytest.approx(fmax, rel=RTOL)


def test_a6_5_single_frequency_has_no_fT(hub, monkeypatch):
    """ac.fT needs at least two frequencies. ac.fmax is f_max at the highest (= only) frequency: still defined.
    (Whether "ac:no_rolloff" is also set here is not fixed by A6.5, so it is not checked.)"""
    y_of = lambda f: hybrid_pi(f, **HP)                                     # noqa: E731
    _patch(monkeypatch, make_fake(y_of, DC_2D, []))
    r = hub.submit("device.devsim", params=_params(hbt_ss([5e9])))
    assert r.status == "ok", (r.status, r.message)
    assert "ac.fT" not in _names(r)
    _, _, fmax = expected_figures(y_of, [5e9])
    assert r.q("ac.fmax").value == pytest.approx(fmax, rel=RTOL)
    assert r.q("ac.h21_abs").axes[0].values == [5e9]


def test_a6_5_1d_units_and_degenerate_two_terminal(hub, monkeypatch):
    """1D: Y in S/m**2 = S/cm^2 x 1e4, currents in A/m**2. A two-terminal device as a two-port
    (y11 = y22 = -y12 = -y21 = G + j w C): |h21| = 1 at every frequency -> no f_T ("ac:no_rolloff");
    Mason's denominator 4 (G^2 - G^2) = 0 -> U = inf -> no f_max ("ac:U_unbounded")."""
    G, C = 1e-3, 3e-8                                           # S/cm^2, F/cm^2 (test values)

    def y_of(f):
        y = complex(G, 2 * math.pi * f * C)
        return {"11": y, "12": -y, "21": -y, "22": y}
    cur = {"anode": 2.0 ** -20, "cathode": -(2.0 ** -20)}
    _patch(monkeypatch, make_fake(y_of, cur, [], dim=1))
    fs = [1e3, 1e6]
    r = hub.submit("device.devsim", params=_params(diode_ss(fs)))
    assert r.status == "ok", (r.status, r.message)
    for ij, sign in (("11", 1), ("12", -1), ("21", -1), ("22", 1)):
        qr, qi = r.q(f"ac.y{ij}_re"), r.q(f"ac.y{ij}_im")
        assert qr.unit == "S/m**2" and qi.unit == "S/m**2"
        assert qr.array() == pytest.approx([sign * G * 1e4] * 2, rel=RTOL)
        assert qi.array() == pytest.approx([sign * 2 * math.pi * f * C * 1e4 for f in fs], rel=RTOL)
    assert r.q("ac.h21_abs").array() == pytest.approx([1.0, 1.0], rel=RTOL)
    for c, v in cur.items():
        q = r.q(f"dc.I.{c}")
        assert q.unit == "A/m**2" and q.axes == [] and q.value == pytest.approx(v * 1e4, rel=RTOL)
    assert "ac.fT" not in _names(r) and "ac:no_rolloff" in r.flags
    assert "ac.fmax" not in _names(r) and "ac:U_unbounded" in r.flags


def test_a6_5_engine_flags_passed_through(hub, monkeypatch):
    _patch(monkeypatch, make_fake(lambda f: hybrid_pi(f, **HP), DC_2D, [],
                                  flags=["ac:frozen_mobility_linearisation"]))
    r = hub.submit("device.devsim", params=_params(hbt_ss([1e6, 5e9])))
    assert r.status == "ok", (r.status, r.message)
    assert "ac:frozen_mobility_linearisation" in r.flags


def test_a6_5_manifests():
    from simws.hub import adapter as A
    assert A.manifest("device.devsim").version == "1.2"
    assert A.manifest("device.charon").version == "1.2"


# ================================================================================================ A6.6 limits
@pytest.mark.parametrize("how", ["param", "paper"])
def test_a6_6_mesh_study_refused(hub, monkeypatch, how):
    calls = []
    _patch(monkeypatch, make_fake(lambda f: hybrid_pi(f, **HP), DC_2D, calls))
    p = hbt_ss([1e6, 5e9])
    if how == "param":
        r = hub.submit("device.devsim", params=_params(p, mesh_study=True))
    else:
        r = hub.submit("device.devsim", params=_params(p), fidelity="paper")
    assert r.status == "invalid_input", (r.status, r.message)
    assert "mesh_study: not available for small_signal" in r.message, r.message
    print(f"A6.6 ({how}): engine runs before the refusal: {len(calls)}")


def test_a6_6_charon_refuses_small_signal(hub, charon_problem):
    p = _with_analysis(charon_problem(), _ss(bias=[_bias("anode", 0.0)], ports=("anode", "cathode"),
                                             frequencies_Hz=[1e3]))
    from simws import engines
    reasons = engines.unsupported("charon", p)
    assert any(s.startswith("analyses: 'small_signal' not in") for s in reasons), reasons
    r = hub.submit("device.charon", params=_params(p))
    assert r.status == "invalid_input", (r.status, r.message)
    assert "unsupported by charon" in r.message and "analyses: 'small_signal' not in" in r.message, r.message


def test_a6_4_devsim_supports_small_signal_problems():
    from simws import engines
    assert engines.unsupported("devsim", hbt_ss()) == []
    assert engines.unsupported("devsim", diode_ss()) == []


# ================================================================================================ real DEVSIM (sanity)
def _diode_one_port(hub):
    """Sanity check, not an acceptance gate. The zero-biased pn diode as a one-port, through a two-contact
    small-signal problem (ports anode, cathode). At 1 kHz and 10 kHz (omega tau = 6e-4 .. 6e-3 for tau = 1e-7 s,
    so quasi-static) and 0 V the diffusion capacitance ~ q n_i^2 ... is negligible and Im(y11) / (2 pi f) is the
    depletion capacitance per cm^2.

    Reference (abrupt junction, depletion approximation): C = eps / W, W = sqrt(2 eps V_bi (NA + ND) / (q NA ND)),
    V_bi = Vt ln(NA ND / n_i^2) with n_i from simws.materials (the density the engine uses with band-gap narrowing
    off) and eps = eps_r eps0 of the same material.

    Tolerance 6 %, fixed BEFORE the run: the depletion approximation ignores the majority-carrier (Debye) tails at
    the depletion edges; the classical correction replaces V_bi by V_bi - 2 Vt, i.e. predicts the drift-diffusion C
    about sqrt(V_bi / (V_bi - 2 Vt)) - 1 = 3.5 % above the plain formula for V_bi ~ 0.77 V; 6 % covers that and the
    10 nm mesh. Current conservation of a two-terminal device: y11 = -y12 = -y21 = y22 to 1e-6 relative (linear
    algebra on the same solve; 1e-6 leaves room for the AC solve's iterative tolerance)."""
    pytest.importorskip("devsim")
    from simws.materials import load
    from simws.materials.semiconductor import intrinsic_density
    from simws.units import EPS0_F_PER_CM, K_B_EV, Q_E
    NA, ND, T = 1e17, 1e16, 300.0
    fs = [1e3, 1e4]
    p = _with_analysis(pn_diode_1d(nodes=401), _ss(bias=[_bias("anode", 0.0)], ports=("anode", "cathode"),
                                                   frequencies_Hz=fs))
    r = hub.submit("device.devsim", params=_params(p))
    assert r.status == "ok", (r.status, r.message)
    mat = load("silicon")
    n_i = intrinsic_density(mat, T)
    eps = mat.get("eps_r", "dimensionless") * EPS0_F_PER_CM
    Vt = K_B_EV * T
    Vbi = Vt * math.log(NA * ND / n_i ** 2)
    W = math.sqrt(2 * eps * Vbi * (NA + ND) / (Q_E * NA * ND))
    C_ref = eps / W                                                         # F/cm^2
    y = {ij: r.q(f"ac.y{ij}_re").array() / 1e4 + 1j * r.q(f"ac.y{ij}_im").array() / 1e4
         for ij in ("11", "12", "21", "22")}                                # S/cm^2
    C_sim = y["11"].imag / (2 * np.pi * np.asarray(fs))
    rel = C_sim / C_ref - 1
    print(f"diode C: n_i={n_i:.4g} Vbi={Vbi:.5f} V W={W * 1e4:.5f} um C_ref={C_ref:.6e} F/cm^2 "
          f"C_sim={C_sim} rel={rel}; y11={y['11']} y12={y['12']} y21={y['21']} y22={y['22']}")
    assert np.all(np.abs(rel) <= 0.06), rel
    assert all(abs(r.q(f"dc.I.{c}").value) < FLOOR_1D for c in ("anode", "cathode"))
    return y


def test_devsim_diode_one_port_depletion_capacitance(hub):
    """The capacitance check and the zero operating-point currents (see _diode_one_port)."""
    _diode_one_port(hub)


def _diode_y(hub):
    """The Y parameters (S/cm^2) of the zero-biased diode one-port at 1 kHz and 10 kHz."""
    return _diode_one_port(hub)


@pytest.mark.xfail(strict=True, reason="RECORDED FAIL (contract A6.9): the checker's tolerance of 1e-6, fixed before the "
                                       "run, is not met by the REAL parts (7.8e-5 of |y| at 1 kHz). The legacy AC code "
                                       "shows the same error, so it is DEVSIM's AC solve, not the translation. The "
                                       "tolerance is not changed.")
def test_devsim_diode_one_port_conservation_recorded_fail(hub):
    """Current conservation of a two-terminal device: y11 = -y12 = -y21 = y22 to 1e-6 relative (the checker's
    original check, split out of the capacitance test by the lead; tolerance unchanged)."""
    y = _diode_y(hub)
    for ij, sign in (("12", -1), ("21", -1), ("22", 1)):
        d = np.abs(sign * y[ij] - y["11"]) / np.abs(y["11"])
        print(f"y{ij} vs y11: rel diff {d}")
        assert np.all(d <= 1e-6), (ij, d)


def test_devsim_diode_one_port_conservation_imaginary_post_hoc(hub):
    """POST HOC (lead, contract A6.9, written after the check above failed): the capacitive (imaginary) parts do
    conserve current. Tolerance 1e-9 relative: three orders tighter than the failed check and about four orders
    above what the checker measured (1e-13), so it is not fitted to the observation."""
    y = _diode_y(hub)
    for ij, sign in (("12", -1), ("21", -1), ("22", 1)):
        d = np.abs(sign * y[ij].imag - y["11"].imag) / np.abs(y["11"].imag)
        assert np.all(d <= 1e-9), (ij, d)


# ================================================================================================ R9 (slow)
@pytest.mark.slow
def test_r9_hbt_ac_matches_legacy(hub):
    """R9 exactly as written in A6.7 (tolerances fixed there): legacy devsim.hbt_ac at VBE 0.75 V, VCE 1.0 V;
    new path device.devsim on the library HBT with the legacy frequencies f_lo = 1e-4 fT_c, f_hi = 0.1 fT_c,
    fT_c = 1 / (2 pi tau_charge)."""
    pytest.importorskip("devsim")
    import time

    from simws.spec.schema import Numerics
    t0 = time.time()
    old = hub.submit("devsim.hbt_ac", params={"VBE": [0.75], "VCE": 1.0})
    assert old.status == "ok", (old.status, old.message)
    t1 = time.time()
    tau = float(np.atleast_1d(old.q("hbt.tau_charge").to("s"))[0])
    fT_c = 1.0 / (2 * math.pi * tau)
    f_lo, f_hi = 1e-4 * fT_c, 0.1 * fT_c
    d = npn_hbt_2d().model_dump(mode="json")
    d["analysis"] = _ss(bias=[_bias("collector", 1.0, 0.1), _bias("base", 0.75, 0.05)], ports=("base", "collector"),
                        frequencies_Hz=[f_lo, f_hi]).model_dump(mode="json")
    d["numerics"] = Numerics(relative_error=1e-10, absolute_error=1e10, max_iterations=60).model_dump(mode="json")
    p = Problem.model_validate(d)
    new = hub.submit("device.devsim", params=_params(p))
    assert new.status == "ok", (new.status, new.message)
    t2 = time.time()
    pairs = [("ac.fT", "hbt.fT", "Hz"), ("ac.fmax", "hbt.fmax", "Hz"), ("dc.I.collector", "hbt.IC", "A/m"),
             ("dc.I.base", "hbt.IB", "A/m")]
    fails = []
    print(f"R9 tau={tau:.6e} s fT_c={fT_c:.6e} Hz f_lo={f_lo:.6e} f_hi={f_hi:.6e}; legacy {t1 - t0:.0f} s, "
          f"new {t2 - t1:.0f} s; new flags {new.flags}")
    for nn, on, unit in pairs:
        a = float(np.atleast_1d(new.q(nn).to(unit))[0])
        b = float(np.atleast_1d(old.q(on).to(unit))[0])
        rel = abs(a / b - 1)
        print(f"R9 {nn}={a:.10e} vs {on}={b:.10e} {unit}: rel diff {rel:.3e}")
        if rel > 1e-6:
            fails.append(f"{nn}: rel diff {rel:.3e} > 1e-6")
    assert not fails, fails
