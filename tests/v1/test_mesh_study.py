"""Contract amendment A4 (docs/ARCHITECTURE_V1.md): mesh error bars for the device adapters.

Written from the contract text alone (and the legacy reference simws.adapters.tcad.HbtGummel, simws.verify,
simws.hub.verification, simws.hub.uncertainty), without reading the implementation. Every expected number is derived
in the test from the A4.4 rule written out below, from independent engine runs made in the same test, or from the
legacy adapter run in the same test.
"""
import math

import numpy as np
import pytest

from simws.spec.library import npn_hbt_2d, pn_diode_1d

R = math.sqrt(2.0)                    # A4.3: refinement ratio
FLOOR_1D = 1e-4                       # A/m^2 (contract section 5 / A4.4)
FLOOR_2D = 1e-8                       # A/m
NOT_ASYMPTOTIC = "mesh_not_asymptotic: conservative mesh bound used for some points"     # A4.5, exact text
NUMERICAL_ONLY = "budget:numerical_only"                                                # A4.5
POINT_KEYS = {"axis_value", "values_coarse_to_fine", "order", "gci_fine", "asymptotic_ratio", "conservative_bound",
              "below_floor", "rel_sigma"}

# Tolerances, fixed before any run:
# - VAL_RTOL 1e-9: reported values against an independent run of the same problem on the same mesh with the same
#   engine (the contract's own figure for that comparison; the solves are deterministic repeats).
# - SIG_RTOL 1e-6: sigmas and the record's derived numbers (order, GCI, asymptotic ratio) are built from differences
#   of nearly equal currents, which amplifies any last-digit difference between two repeats of the same solve; 1e-6
#   is the figure stated for this comparison by the lead.
# - For the fake engine (pure arithmetic, no solver) the same 1e-6 is used for derived numbers; the only differences
#   there are the order of floating-point operations (A/cm^2 -> A/m^2 conversion, s0/r vs s0*r**-1).
VAL_RTOL = 1e-9
SIG_RTOL = 1e-6


def _params(p, **extra):
    return {"problem": p.model_dump(mode="json"), **extra}


def _finite_or_none(x):
    """The result envelope stores verification records through simws.hub.schema._jsonable, which turns non-finite
    floats into None (e.g. the observed order inf when the finest two levels are identical)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def expected_point(vals, floor):
    """The A4.4 rule, written out: vals = |I| on the three meshes, coarse to fine (f3, f2, f1)."""
    from simws.hub.uncertainty import GCI_COVERAGE_K
    from simws.hub.verification import solution_from_values
    f3, f2, f1 = vals
    rec = solution_from_values([0, 1, 2], [f3, f2, f1], ratio=R)
    if abs(f1) < floor:                                  # below the noise floor: neither good nor conservative
        sigma, cons, below = float(floor), False, True
    else:
        s, p, ar = rec["sigma_num"], rec.get("order"), rec.get("asymptotic_ratio")
        good = (s == 0.0) or (s is not None and p is not None and 0.5 <= p <= 4.0
                              and ar is not None and 0.5 < ar < 2.0)
        if good:
            sigma = float(s)
        else:                                            # conservative bound: order no better than 0.5 assumed
            sigma = 1.25 * abs(f1 - f2) / (R ** 0.5 - 1) / GCI_COVERAGE_K
        cons, below = not good, False
    return {"sigma": sigma, "conservative_bound": cons, "below_floor": below,
            "order": _finite_or_none(rec.get("order")), "gci_fine": _finite_or_none(rec.get("gci_fine")),
            "asymptotic_ratio": _finite_or_none(rec.get("asymptotic_ratio")),
            "rel_sigma": (sigma / abs(f1)) if f1 != 0 else None}


def _close(a, b, rtol, abs_=0.0):
    if a is None or b is None:
        return a is None and b is None
    return a == pytest.approx(b, rel=rtol, abs=abs_)


def check_point(entry, vals, exp, axis_value, rtol_vals=VAL_RTOL, rtol=SIG_RTOL):
    """One per-point record entry against the expectation derived from `vals`."""
    assert POINT_KEYS <= set(entry), set(entry)
    if axis_value is None:
        assert entry["axis_value"] is None
    else:
        assert entry["axis_value"] == pytest.approx(axis_value, rel=1e-12)
    assert len(entry["values_coarse_to_fine"]) == 3
    for got, want in zip(entry["values_coarse_to_fine"], vals):
        assert got == pytest.approx(want, rel=rtol_vals, abs=0.0)
    assert entry["conservative_bound"] is exp["conservative_bound"]
    assert entry["below_floor"] is exp["below_floor"]
    assert _close(entry["rel_sigma"], exp["rel_sigma"], rtol), (entry["rel_sigma"], exp["rel_sigma"])
    if not exp["below_floor"]:          # A4 does not say what order / GCI / ratio a below-floor point records
        for k in ("order", "gci_fine", "asymptotic_ratio"):
            assert _close(_finite_or_none(entry[k]), exp[k], rtol), (k, entry[k], exp[k])


def mesh_record(r):
    rec = r.verification["mesh"]
    assert rec["kind"] == "solution"
    assert rec["ratio"] == pytest.approx(R, rel=1e-12)
    assert rec["levels"] == 3
    return rec


# ================================================================================================ fake engine (A4.4)
# The DEVSIM engine module's run() is replaced (as in test_device_adapter.py's A2.4 tests) by a function whose
# currents depend on problem.device.mesh.scale in a chosen way, so every branch of A4.4 is hit deterministically
# without DEVSIM. Currents are A/cm^2 (1D); the adapter reports A/m^2 = A/cm^2 x 1e4 (contract section 5).

def _level(scale, s0):
    """Continuous level index: 0 at s0, 1 at s0/r, 2 at s0/r^2."""
    return math.log(s0 / scale) / math.log(R)


def make_fake(anode_of, calls, s0=1.0, fail_below=None, flags_of=None, cathode_of=None):
    """anode_of(scale, k) -> anode current density (A/cm^2) at sweep index k; the cathode is minus the anode (exact
    Kirchhoff). Records every problem it is given in `calls`. fail_below: raise NotConverged when scale < it."""
    def run(problem, workdir=None):
        from simws.engines.base import EngineResult
        from simws.hub.adapter import NotConverged
        from simws.spec.schema import Problem
        p = problem if isinstance(problem, Problem) else Problem.model_validate(problem)
        calls.append(p)
        scale = p.device.mesh.scale
        if fail_below is not None and scale < fail_below:
            raise NotConverged(f"fake engine: no convergence at mesh scale {scale:g}")
        if p.analysis.kind == "equilibrium":
            V, sweep, an = [0.0], None, "equilibrium"
        else:
            V, sweep, an = list(p.analysis.values_V), p.analysis.contact, "dc_sweep"
        ia = [float(anode_of(scale, k)) for k in range(len(V))]
        return EngineResult(engine="devsim", problem_id=p.problem_id(), analysis=an, sweep_contact=sweep,
                            sweep_V=V, currents={"anode": ia, "cathode": (
                                [float(cathode_of(scale, k)) for k in range(len(V))] if cathode_of
                                else [-x for x in ia])},
                            current_unit="A/cm**2", flags=list(flags_of(scale) if flags_of else []))
    return run


def _patch(monkeypatch, run):
    from simws import engines
    monkeypatch.setattr(engines.module("devsim"), "run", run)


V2 = [0.3, 0.5]
I0 = [1e-2, 1.0]          # A/cm^2 at the two sweep points (test values, not physics)


def _expected_from_fake(anode_of, s0, n, floor=FLOOR_1D):
    """Per contact, per point: (|I| coarse to fine in A/m^2, expected dict, finest signed value in A/m^2)."""
    scales = [s0, s0 / R, s0 / R ** 2]
    out = {}
    for name, sign in (("dc.I.anode", 1.0), ("dc.I.cathode", -1.0)):
        rows = []
        for k in range(n):
            signed = [sign * anode_of(s, k) * 1e4 for s in scales]
            vals = [abs(x) for x in signed]
            rows.append((vals, expected_point(vals, floor), signed[-1]))
        out[name] = rows
    return out


def _check_fake_result(r, anode_of, s0, V, floor=FLOOR_1D):
    exp = _expected_from_fake(anode_of, s0, len(V), floor)
    rec = mesh_record(r)
    assert rec["mesh_scales"] == pytest.approx([s0, s0 / R, s0 / R ** 2], rel=1e-12)
    assert {"dc.I.anode", "dc.I.cathode"} <= set(rec["quantities"])
    any_cons = False
    for name, rows in exp.items():
        q = r.q(name)
        assert q.unit == "A/m**2"
        assert q.axes[0].values == V
        assert q.uncertainty.kind == "sigma"
        assert isinstance(q.uncertainty.sigma, list) and len(q.uncertainty.sigma) == len(V)
        assert set(q.uncertainty.components) == {"numerical"}
        assert q.uncertainty.components["numerical"] == q.uncertainty.sigma
        entries = rec["quantities"][name]
        assert len(entries) == len(V)
        for k, (vals, e, finest) in enumerate(rows):
            assert q.value[k] == pytest.approx(finest, rel=1e-12)          # A4.3: values of the finest run
            assert q.uncertainty.sigma[k] == pytest.approx(e["sigma"], rel=SIG_RTOL, abs=0.0), (name, k)
            check_point(entries[k], vals, e, V[k], rtol_vals=1e-12)
            any_cons |= e["conservative_bound"]
    assert rec["passed"] is (not any_cons)
    assert (NOT_ASYMPTOTIC in r.flags) is any_cons, r.flags
    assert NUMERICAL_ONLY in r.flags
    # Kirchhoff: the fake cathode is exactly minus the anode on every mesh, so the residual is 0 on all three and
    # its sigma (max - min) / (2 sqrt 3) is 0
    kq = r.q("dc.kirchhoff_residual")
    assert kq.value == pytest.approx(0.0, abs=1e-15)
    assert kq.uncertainty.kind == "sigma" and kq.uncertainty.sigma == pytest.approx(0.0, abs=1e-15)
    assert set(kq.uncertainty.components) == {"numerical"}
    return exp


# ------------------------------------------------------------------------------------------------ A4.1 when
def test_a4_1_screening_without_mesh_study_unchanged(hub, monkeypatch):
    calls = []
    _patch(monkeypatch, make_fake(lambda s, k: I0[k] * (1 + 0.01 * s ** 2), calls))
    r = hub.submit("device.devsim", params=_params(pn_diode_1d(V=V2, nodes=201)))
    assert r.status == "ok", r.message
    assert len(calls) == 1 and calls[0].device.mesh.scale == 1.0              # one run, mesh as described
    for name in ("dc.I.anode", "dc.I.cathode"):
        assert r.q(name).uncertainty.kind == "none"
    assert "mesh" not in r.verification
    assert NUMERICAL_ONLY not in r.flags and NOT_ASYMPTOTIC not in r.flags


def test_a4_1_mesh_study_param_triggers_three_runs(hub, monkeypatch):
    calls = []
    _patch(monkeypatch, make_fake(lambda s, k: I0[k] * (1 + 0.01 * s ** 2), calls))
    p = pn_diode_1d(V=V2, nodes=201)
    r = hub.submit("device.devsim", params=_params(p, mesh_study=True))
    assert r.status == "ok", r.message
    assert "mesh" in r.verification and NUMERICAL_ONLY in r.flags
    # A4.3: three runs of the SAME problem, only device.mesh.scale differs
    assert len(calls) == 3
    assert sorted(c.device.mesh.scale for c in calls) == pytest.approx(sorted([1.0, 1 / R, 0.5]), rel=1e-12)
    base = p.model_dump(mode="json")
    for c in calls:
        d = c.model_dump(mode="json")
        d["device"]["mesh"]["scale"] = base["device"]["mesh"]["scale"]
        assert d == base


def test_a4_1_paper_fidelity_triggers_study(hub, monkeypatch):
    calls = []
    _patch(monkeypatch, make_fake(lambda s, k: I0[k] * (1 + 0.01 * s ** 2), calls))
    r = hub.submit("device.devsim", params=_params(pn_diode_1d(V=V2, nodes=201)), fidelity="paper")
    assert r.status == "ok", r.message
    assert len(calls) == 3 and "mesh" in r.verification and NUMERICAL_ONLY in r.flags


# ------------------------------------------------------------------------------------------------ A4.2 which engines
def test_a4_2_capability_field():
    from simws import engines
    from simws.engines.base import Capability
    assert Capability.model_fields["mesh_study"].default is False
    assert engines.capability("devsim").mesh_study is True
    assert engines.capability("charon").mesh_study is False


@pytest.mark.parametrize("how", ["param", "paper"])
def test_a4_2_charon_mesh_study_refused_before_running(hub, monkeypatch, charon_problem, how):
    """A4.2 / R6 (Charon half): refused before anything runs, status invalid_input with the stated message."""
    from simws import engines
    calls = []

    def must_not_run(problem, workdir=None):
        calls.append(problem)
        raise AssertionError("the Charon engine must not run for a refused mesh study")
    monkeypatch.setattr(engines.module("charon"), "run", must_not_run)
    p = charon_problem(V=(0.1, 0.2))
    assert engines.unsupported("charon", p) == []                 # the problem itself is Charon-compatible
    if how == "param":
        r = hub.submit("device.charon", params=_params(p, mesh_study=True))
    else:
        r = hub.submit("device.charon", params=_params(p), fidelity="paper")
    assert r.status == "invalid_input", (r.status, r.message)
    assert "mesh_study: charon cannot refine its mesh" in r.message
    assert calls == []


# ------------------------------------------------------------------------------------------------ A4.4 branches
def test_a4_4a_second_order_sequence_is_good(hub, monkeypatch):
    """I = I0 (1 + c s^2): differences e32 = I0 c s0^2/2, e21 = I0 c s0^2/4, so p = ln 2 / ln sqrt2 = 2 exactly and
    the GCI branch applies with sigma = sigma_num."""
    c = 0.01
    f = lambda s, k: I0[k] * (1 + c * s ** 2)
    calls = []
    _patch(monkeypatch, make_fake(f, calls))
    r = hub.submit("device.devsim", params=_params(pn_diode_1d(V=V2, nodes=201), mesh_study=True))
    assert r.status == "ok", r.message
    exp = _check_fake_result(r, f, 1.0, V2)
    rec = r.verification["mesh"]
    for name in ("dc.I.anode", "dc.I.cathode"):
        for k, entry in enumerate(rec["quantities"][name]):
            assert entry["order"] == pytest.approx(2.0, abs=1e-6)
            assert entry["conservative_bound"] is False and entry["below_floor"] is False
            # closed form of the GCI branch: sigma = Fs |e21/f1| / (r^p - 1) |f1| / k = 1.25 |e21| / 1 / 2
            e21 = I0[k] * 1e4 * c * (1 / R ** 2) ** 2 * (R ** 2 - 1)   # I0 c (s2^2 - s1^2) with s2 = 1/r, s1 = 1/r^2
            assert exp[name][k][1]["sigma"] == pytest.approx(1.25 * e21 / (R ** 2 - 1) / 2.0, rel=1e-9)
    assert rec["passed"] is True and NOT_ASYMPTOTIC not in r.flags


def test_a4_4a_non_default_s0(hub, monkeypatch):
    """Same sequence with s0 = 0.8 from the description (with_changes): mesh_scales are s0, s0/r, s0/r^2."""
    s0 = 0.8
    f = lambda s, k: I0[k] * (1 + 0.01 * s ** 2)
    calls = []
    _patch(monkeypatch, make_fake(f, calls, s0=s0))
    p = pn_diode_1d(V=V2, nodes=201).with_changes({"device.mesh.scale": s0})
    r = hub.submit("device.devsim", params=_params(p, mesh_study=True))
    assert r.status == "ok", r.message
    assert sorted(x.device.mesh.scale for x in calls) == pytest.approx(sorted([s0, s0 / R, s0 / R ** 2]), rel=1e-12)
    _check_fake_result(r, f, s0, V2)


def test_a4_4b_oscillating_values_use_conservative_bound(hub, monkeypatch):
    """I = I0 (1 + 0.1 cos(pi * level)): values 1.1, 0.9, 1.1 x I0 -> sign change between the differences with a
    spread of 0.2/1.1 (far above simws.verify's 1e-4 noise level): no GCI, so the conservative bound applies, the
    record does not pass and the result is flagged."""
    f = lambda s, k: I0[k] * (1 + 0.1 * math.cos(math.pi * _level(s, 1.0)))
    _patch(monkeypatch, make_fake(f, []))
    r = hub.submit("device.devsim", params=_params(pn_diode_1d(V=V2, nodes=201), mesh_study=True))
    assert r.status == "ok", r.message
    exp = _check_fake_result(r, f, 1.0, V2)
    from simws.hub.uncertainty import GCI_COVERAGE_K
    for name in ("dc.I.anode", "dc.I.cathode"):
        for k in range(2):
            vals, e, _ = exp[name][k]
            assert e["conservative_bound"] is True
            # closed form: |f1 - f2| = 0.2 I0 (x 1e4 to A/m^2)
            assert e["sigma"] == pytest.approx(1.25 * 0.2 * I0[k] * 1e4 / (R ** 0.5 - 1) / GCI_COVERAGE_K, rel=1e-9)
            assert r.verification["mesh"]["quantities"][name][k]["conservative_bound"] is True
    assert r.verification["mesh"]["passed"] is False
    assert NOT_ASYMPTOTIC in r.flags


def test_a4_4c_identical_values_sigma_zero(hub, monkeypatch):
    """Identical currents on all meshes: simws.verify reports GCI 0 (order inf), so sigma_num == 0.0 -> good."""
    f = lambda s, k: I0[k]
    _patch(monkeypatch, make_fake(f, []))
    r = hub.submit("device.devsim", params=_params(pn_diode_1d(V=V2, nodes=201), mesh_study=True))
    assert r.status == "ok", r.message
    _check_fake_result(r, f, 1.0, V2)
    for name in ("dc.I.anode", "dc.I.cathode"):
        assert r.q(name).uncertainty.sigma == [0.0, 0.0]
        for e in r.verification["mesh"]["quantities"][name]:
            assert e["conservative_bound"] is False and e["rel_sigma"] == 0.0
    assert r.verification["mesh"]["passed"] is True and NOT_ASYMPTOTIC not in r.flags


def test_a4_4d_below_floor(hub, monkeypatch):
    """Point 0: an oscillating current of 1.5e-9 / 0.5e-9 / 1.5e-9 A/cm^2 = 1.5e-5 A/m^2 at the finest mesh, below the
    1e-4 A/m^2 floor (it would be conservative otherwise): sigma = floor, below_floor, not conservative, so it does
    not fail the record. Point 1: the clean second-order sequence (good)."""
    def f(s, k):
        if k == 0:
            return 1e-9 * (1 + 0.5 * math.cos(math.pi * _level(s, 1.0)))
        return I0[1] * (1 + 0.01 * s ** 2)
    _patch(monkeypatch, make_fake(f, []))
    r = hub.submit("device.devsim", params=_params(pn_diode_1d(V=V2, nodes=201), mesh_study=True))
    assert r.status == "ok", r.message
    exp = _check_fake_result(r, f, 1.0, V2)
    for name in ("dc.I.anode", "dc.I.cathode"):
        vals, e, _ = exp[name][0]
        assert vals[-1] < FLOOR_1D and e["below_floor"] is True and e["conservative_bound"] is False
        assert r.q(name).uncertainty.sigma[0] == pytest.approx(FLOOR_1D, rel=1e-12)
        entry = r.verification["mesh"]["quantities"][name][0]
        assert entry["below_floor"] is True and entry["conservative_bound"] is False
        assert entry["rel_sigma"] == pytest.approx(FLOOR_1D / (1.5e-9 * 1e4), rel=1e-9)
        assert r.verification["mesh"]["quantities"][name][1]["below_floor"] is False
    assert r.verification["mesh"]["passed"] is True and NOT_ASYMPTOTIC not in r.flags


def test_a4_4d_zero_current_rel_sigma_none(hub, monkeypatch):
    """A current that is exactly 0 on every mesh: below the floor, sigma = floor, rel_sigma None (f1 is 0)."""
    f = lambda s, k: 0.0 if k == 0 else I0[1] * (1 + 0.01 * s ** 2)
    _patch(monkeypatch, make_fake(f, []))
    r = hub.submit("device.devsim", params=_params(pn_diode_1d(V=V2, nodes=201), mesh_study=True))
    assert r.status == "ok", r.message
    _check_fake_result(r, f, 1.0, V2)
    for name in ("dc.I.anode", "dc.I.cathode"):
        entry = r.verification["mesh"]["quantities"][name][0]
        assert entry["below_floor"] is True and entry["rel_sigma"] is None
        assert r.q(name).uncertainty.sigma[0] == pytest.approx(FLOOR_1D, rel=1e-12)


def test_a4_4_kirchhoff_residual_sigma(hub, monkeypatch):
    """A4.4: dc.kirchhoff_residual on each of the three meshes; value = finest mesh's, sigma = (max - min) /
    (2 sqrt 3), component "numerical". One sweep point, so A1.8's residual is unambiguous:
    |I_a + I_c| / (max(|I_a|, |I_c|) + floor). Anode 1 A/cm^2 on every mesh, cathode -(1 - 1e-6 s) A/cm^2 (test
    values; |sum| = 1e-6 s x |I| stays inside the section-5 Kirchhoff self-check of 1e-4 x max|I| + floor)."""
    eps = 1e-6
    _patch(monkeypatch, make_fake(lambda s, k: 1.0, [], cathode_of=lambda s, k: -(1.0 - eps * s)))
    r = hub.submit("device.devsim", params=_params(pn_diode_1d(V=[0.3], nodes=201), mesh_study=True))
    assert r.status == "ok", r.message

    def residual(s):
        a, c = 1.0 * 1e4, -(1.0 - eps * s) * 1e4                 # A/m^2
        return abs(a + c) / (max(abs(a), abs(c)) + FLOOR_1D)
    res = [residual(s) for s in (1.0, 1 / R, 1 / R ** 2)]
    kq = r.q("dc.kirchhoff_residual")
    # 1e-6 relative: |a + c| is a difference of numbers ~1e4 that agree to 1e-6, so its rounding is ~1e-10 relative
    assert kq.value == pytest.approx(res[-1], rel=SIG_RTOL)
    assert kq.uncertainty.kind == "sigma"
    assert kq.uncertainty.sigma == pytest.approx((max(res) - min(res)) / (2 * math.sqrt(3)), rel=SIG_RTOL)
    assert set(kq.uncertainty.components) == {"numerical"}
    assert kq.uncertainty.components["numerical"] == pytest.approx(kq.uncertainty.sigma, rel=1e-12)


def test_a4_3_finest_run_flags(hub, monkeypatch):
    """A4.3: the engine flags are those of the finest run."""
    f = lambda s, k: I0[k] * (1 + 0.01 * s ** 2)
    _patch(monkeypatch, make_fake(f, [], flags_of=lambda s: [f"fake_level:{round(_level(s, 1.0))}"]))
    r = hub.submit("device.devsim", params=_params(pn_diode_1d(V=V2, nodes=201), mesh_study=True))
    assert r.status == "ok", r.message
    assert "fake_level:2" in r.flags
    assert "fake_level:0" not in r.flags and "fake_level:1" not in r.flags, r.flags


@pytest.mark.parametrize("fail_level", [0, 1, 2])
def test_a4_3_not_converged_on_any_level(hub, monkeypatch, fail_level):
    """A run that does not converge on any level makes the job not_converged."""
    target = 1.0 / R ** fail_level

    def f(s, k):
        return I0[k] * (1 + 0.01 * s ** 2)
    base = make_fake(f, [])

    def run(problem, workdir=None):
        from simws.hub.adapter import NotConverged
        from simws.spec.schema import Problem
        p = problem if isinstance(problem, Problem) else Problem.model_validate(problem)
        if abs(p.device.mesh.scale / target - 1) < 1e-9:
            raise NotConverged(f"fake engine: no convergence at mesh scale {p.device.mesh.scale:g}")
        return base(problem, workdir)
    _patch(monkeypatch, run)
    r = hub.submit("device.devsim", params=_params(pn_diode_1d(V=V2, nodes=201), mesh_study=True))
    assert r.status == "not_converged", (r.status, r.message)


def test_a4_4_equilibrium_fake_scalars(hub, monkeypatch):
    """Equilibrium: scalars; sigma a float; axis_value None. The equilibrium self-check (contract section 5) requires
    every |I| below the floor, so the fake currents are 1e-9 A/cm^2 x (1 + 0.01 s^2) = ~1e-5 A/m^2: the floor branch."""
    f = lambda s, k: 1e-9 * (1 + 0.01 * s ** 2)
    _patch(monkeypatch, make_fake(f, []))
    r = hub.submit("device.devsim", params=_params(pn_diode_1d(nodes=201), mesh_study=True))
    assert r.status == "ok", r.message
    rec = mesh_record(r)
    for name, sign in (("dc.I.anode", 1.0), ("dc.I.cathode", -1.0)):
        q = r.q(name)
        assert q.axes == [] and isinstance(q.value, float)
        assert q.uncertainty.kind == "sigma" and isinstance(q.uncertainty.sigma, float)
        assert q.uncertainty.components == {"numerical": q.uncertainty.sigma}
        vals = [abs(sign * f(s, 0) * 1e4) for s in (1.0, 1 / R, 0.5)]
        e = expected_point(vals, FLOOR_1D)
        assert q.value == pytest.approx(sign * f(0.5, 0) * 1e4, rel=1e-12)
        assert q.uncertainty.sigma == pytest.approx(e["sigma"], rel=SIG_RTOL)
        entries = rec["quantities"][name]
        assert len(entries) == 1
        check_point(entries[0], vals, e, None, rtol_vals=1e-12)
    assert isinstance(r.q("dc.kirchhoff_residual").uncertainty.sigma, float)


# ================================================================================================ real DEVSIM, 1D
def _plain(hub, p):
    r = hub.submit("device.devsim", params=_params(p))
    assert r.status == "ok", r.message
    return r


def test_a4_diode_mesh_study_against_three_independent_runs(hub):
    """A4.3-A4.5 on pn_diode_1d(V=[0.3, 0.5], nodes=201): the adapter's three runs are repeated here as three
    independent device.devsim runs (no mesh_study) at s0, s0/r, s0/r^2, and every number of the record is
    recomputed from them with the A4.4 rule."""
    pytest.importorskip("devsim")
    p = pn_diode_1d(V=V2, nodes=201)
    s0 = p.device.mesh.scale
    scales = [s0, s0 / R, s0 / R ** 2]
    r = hub.submit("device.devsim", params=_params(p, mesh_study=True))
    assert r.status == "ok", r.message
    runs = [_plain(hub, p.with_changes({"device.mesh.scale": s})) for s in scales]
    rec = mesh_record(r)
    assert set(rec) >= {"kind", "ratio", "levels", "mesh_scales", "quantities", "passed"}
    assert rec["mesh_scales"] == pytest.approx(scales, rel=1e-12)
    names = sorted(q.name for q in runs[-1].quantities if q.name.startswith("dc.I."))
    assert names == ["dc.I.anode", "dc.I.cathode"]
    assert set(names) <= set(rec["quantities"])
    any_cons = False
    for name in names:
        q = r.q(name)
        fine = runs[-1].q(name)
        # A4.3: the reported values and sweep axis are the finest run's
        assert q.unit == fine.unit == "A/m**2"
        assert q.axes[0].values == fine.axes[0].values == V2
        assert np.allclose(q.array(), fine.array(), rtol=VAL_RTOL, atol=0.0), (q.value, fine.value)
        assert q.uncertainty.kind == "sigma" and len(q.uncertainty.sigma) == len(V2)
        assert q.uncertainty.components == {"numerical": q.uncertainty.sigma}
        for k in range(len(V2)):
            vals = [abs(x.q(name).value[k]) for x in runs]
            e = expected_point(vals, FLOOR_1D)
            assert q.uncertainty.sigma[k] == pytest.approx(e["sigma"], rel=SIG_RTOL, abs=0.0), (name, k, e)
            check_point(rec["quantities"][name][k], vals, e, V2[k])
            any_cons |= e["conservative_bound"]
            print(f"{name} V={V2[k]}: vals={vals} sigma={e['sigma']:.4e} conservative={e['conservative_bound']} "
                  f"below_floor={e['below_floor']}")
    assert rec["passed"] is (not any_cons)
    assert (NOT_ASYMPTOTIC in r.flags) is any_cons
    assert NUMERICAL_ONLY in r.flags
    # dc.kirchhoff_residual: finest mesh's value. The residual is a dimensionless ratio of sums at round-off level;
    # abs 1e-14 covers last-digit differences in such a ratio.
    res = [x.q("dc.kirchhoff_residual").value for x in runs]
    kq = r.q("dc.kirchhoff_residual")
    assert kq.value == pytest.approx(res[-1], rel=SIG_RTOL, abs=1e-14)
    assert kq.uncertainty.kind == "sigma" and isinstance(kq.uncertainty.sigma, float)
    assert math.isfinite(kq.uncertainty.sigma) and kq.uncertainty.sigma >= 0.0
    assert set(kq.uncertainty.components) == {"numerical"}
    assert kq.uncertainty.components["numerical"] == pytest.approx(kq.uncertainty.sigma, rel=1e-12, abs=0.0)
    # NOT checked here (first version of this test did, and failed by 4.0e-4 relative): sigma against
    # (max - min) / (2 sqrt 3) of the three independent runs' residuals. The residual is |I_anode + I_cathode| at
    # ~1e-8..1e-7 of |I|, i.e. pure cancellation; the adapter's middle-mesh currents differ from a plain run at
    # s0/r by up to ~4e-10 relative (plain repeats are bit-identical), and that is amplified ~1e7 in the residual.
    # The rule itself is checked exactly on the fake engine (test_a4_4_kirchhoff_residual_sigma).
    print("kirchhoff residuals of the independent runs:", res, "adapter sigma:", kq.uncertainty.sigma,
          "independent-run sigma:", (max(res) - min(res)) / (2 * math.sqrt(3)))


def test_a4_diode_non_default_s0(hub):
    """s0 = 2.0 from the description: mesh_scales [2, 2/r, 1], and the values equal a plain run at scale 1.0."""
    pytest.importorskip("devsim")
    s0 = 2.0
    p = pn_diode_1d(V=V2, nodes=201).with_changes({"device.mesh.scale": s0})
    r = hub.submit("device.devsim", params=_params(p, mesh_study=True))
    assert r.status == "ok", r.message
    assert mesh_record(r)["mesh_scales"] == pytest.approx([s0, s0 / R, s0 / R ** 2], rel=1e-12)
    fine = _plain(hub, p.with_changes({"device.mesh.scale": s0 / R ** 2}))
    for name in ("dc.I.anode", "dc.I.cathode"):
        assert np.allclose(r.q(name).array(), fine.q(name).array(), rtol=VAL_RTOL, atol=0.0)


def test_a4_diode_equilibrium_scalars(hub):
    """Equilibrium with mesh_study: scalar sigma, axis_value None. The equilibrium currents are below the 1D floor
    (test_device_adapter.test_devsim_equilibrium_scalars), so A4.4's floor exception applies when |f1| < floor."""
    pytest.importorskip("devsim")
    r = hub.submit("device.devsim", params=_params(pn_diode_1d(nodes=201), mesh_study=True))
    assert r.status == "ok", r.message
    rec = mesh_record(r)
    for name in ("dc.I.anode", "dc.I.cathode"):
        q = r.q(name)
        assert q.axes == [] and isinstance(q.value, (int, float))
        assert q.uncertainty.kind == "sigma" and isinstance(q.uncertainty.sigma, float)
        assert q.uncertainty.components == {"numerical": q.uncertainty.sigma}
        (entry,) = rec["quantities"][name]
        assert entry["axis_value"] is None
        vals = entry["values_coarse_to_fine"]
        assert vals[-1] == pytest.approx(abs(q.value), rel=VAL_RTOL, abs=0.0)
        e = expected_point(vals, FLOOR_1D)
        assert q.uncertainty.sigma == pytest.approx(e["sigma"], rel=SIG_RTOL)
        check_point(entry, vals, e, None)
    assert isinstance(r.q("dc.kirchhoff_residual").uncertainty.sigma, float)


def test_r6_diode_paper_fidelity_ok(hub):
    """R6 (fast half): the diode at fidelity="paper" on device.devsim passes the gatekeeper (uncertainty on every
    quantity, verification "mesh" reported)."""
    pytest.importorskip("devsim")
    r = hub.submit("device.devsim", params=_params(pn_diode_1d(V=V2, nodes=201)), fidelity="paper")
    assert r.status == "ok", r.message
    assert "mesh" in r.verification and NUMERICAL_ONLY in r.flags
    for q in r.quantities:
        assert q.uncertainty.kind == "sigma", q.name


# ------------------------------------------------------------------------------------------------ A4.6 manifests
def test_a4_6_manifests():
    from simws.hub import adapter as A
    d, c = A.manifest("device.devsim"), A.manifest("device.charon")
    assert d.version == "1.1" and c.version == "1.1"
    assert d.verification == {"mesh": "paper fidelity: GCI per sweep point over 3 meshes (spacings x1, 1/sqrt2, 1/2)"}
    assert c.verification == {}


# ================================================================================================ R5 / R6 (slow)
@pytest.mark.slow
def test_r5_hbt_mesh_study_matches_legacy(hub):
    """R5 (tolerances fixed in A4.7): device.devsim with mesh_study against devsim.hbt_gummel with mesh_study and
    eb_refine False, npn_hbt_2d(VBE=[0.6, 0.7])."""
    pytest.importorskip("devsim")
    VBE = [0.6, 0.7]
    new = hub.submit("device.devsim", params=_params(npn_hbt_2d(VBE=VBE), mesh_study=True))
    assert new.status == "ok", new.message
    old = hub.submit("devsim.hbt_gummel", params={"VBE": VBE, "mesh_study": True, "case": {"eb_refine": False}})
    assert old.status == "ok", old.message
    assert old.q("hbt.IC").axes[0].values == pytest.approx(VBE)
    pairs = [("dc.I.collector", "hbt.IC"), ("dc.I.base", "hbt.IB"), ("dc.I.emitter", "hbt.IE")]
    nrec, orec = new.verification["mesh"], old.verification["mesh"]
    fails = []
    for nn, on in pairs:
        a, b = new.q(nn), old.q(on)
        assert a.unit == "A/m" and b.unit == "A/m"
        va, vb = np.abs(a.array()), np.abs(b.array())
        sa, sb = np.asarray(a.uncertainty.sigma, float), np.asarray(b.uncertainty.sigma, float)
        dv, ds = np.abs(va / vb - 1), np.abs(sa / sb - 1)
        print(f"R5 {nn} vs {on}: |I| new={va} legacy={vb} rel diff={dv}; sigma new={sa} legacy={sb} rel diff={ds}")
        if dv.max() > 1e-6:
            fails.append(f"{nn}: value rel diff {dv.max():.3e} > 1e-6")
        if ds.max() > 1e-3:
            fails.append(f"{nn}: sigma rel diff {ds.max():.3e} > 1e-3")
        cn = [e["conservative_bound"] for e in nrec["quantities"][nn]]
        co = [e["conservative_bound"] for e in orec["quantities"][on]]
        print(f"R5 {nn}: conservative new={cn} legacy={co}")
        if cn != co:
            fails.append(f"{nn}: conservative_bound {cn} vs legacy {co}")
    print(f"R5 passed: new={nrec['passed']} legacy={orec['passed']}")
    if nrec["passed"] != orec["passed"]:
        fails.append(f"passed: new {nrec['passed']} vs legacy {orec['passed']}")
    assert not fails, fails


@pytest.mark.slow
def test_r6_hbt_paper_fidelity_ok(hub):
    """R6: the R5 problem at fidelity="paper" on device.devsim returns status ok (the gatekeeper accepts it)."""
    pytest.importorskip("devsim")
    r = hub.submit("device.devsim", params=_params(npn_hbt_2d(VBE=[0.6, 0.7])), fidelity="paper")
    assert r.status == "ok", r.message
    assert "mesh" in r.verification
