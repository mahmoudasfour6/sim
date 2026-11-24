"""simws.engines.devsim_engine: acceptance R1-R4 of contract section 4.4 (tolerances fixed there), engine-specific
refusals, and physics checks whose expected values are derived here from formulas, never typed in."""
import json
import math

import numpy as np
import pytest

from simws.spec.library import npn_hbt_2d, pn_diode_1d

pytest.importorskip("devsim")

K_B = 1.380649e-23          # J/K (exact, SI 2019)
Q_E = 1.602176634e-19       # C (exact, SI 2019)
# Kirchhoff floors of contract section 5, converted to engine units: 1e-4 A/m^2 = 1e-8 A/cm^2; 1e-8 A/m = 1e-10 A/cm
FLOOR_1D_A_CM2 = 1e-8
FLOOR_2D_A_CM = 1e-10


def _engine():
    from simws.engines import devsim_engine
    return devsim_engine


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a / b - 1.0)


# ------------------------------------------------------------------------------------------------ module surface
def test_module_surface():
    E = _engine()
    assert E.available() is True                       # devsim imports here (importorskip above)
    for f in ("CAPABILITY", "unsupported", "available", "run", "build"):
        assert hasattr(E, f), f


# ------------------------------------------------------------------------------------------------ R1 (1D diode)
V_R1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


@pytest.fixture(scope="module")
def diode_run():
    return _engine().run(pn_diode_1d(V=V_R1))


def test_R1_diode_matches_legacy(diode_run):
    from simws.validation.engine_pairs import _devsim_diode
    J_legacy, _ = _devsim_diode(V_R1)
    r = diode_run
    assert r.engine == "devsim" and r.analysis == "dc_sweep" and r.sweep_contact == "anode"
    assert r.problem_id == pn_diode_1d(V=V_R1).problem_id()
    assert r.sweep_V == V_R1
    assert r.current_unit == "A/cm**2"
    assert set(r.currents) == {"anode", "cathode"}
    assert all(len(v) == len(V_R1) for v in r.currents.values())
    rel = _rel(r.currents["anode"], J_legacy)          # same sign: both positive into the anode
    assert rel.max() <= 1e-6, (r.currents["anode"], J_legacy.tolist())


def test_diode_result_is_json_and_keeps_used(diode_run):
    json.loads(diode_run.model_dump_json())
    used = diode_run.diagnostics["used"]
    assert isinstance(used, dict) and "n_i_cm3" in used and "T_K" in used


def test_diode_kirchhoff_and_sign(diode_run):
    """Current into the anode = - current into the cathode (contract section 5 Kirchhoff bound); forward bias
    drives conventional current into the anode, so the anode current is positive (DEVSIM: positive into device)."""
    c = diode_run.currents
    for k in range(len(V_R1)):
        ia, ic = c["anode"][k], c["cathode"][k]
        assert abs(ia + ic) <= 1e-4 * max(abs(ia), abs(ic)) + FLOOR_1D_A_CM2, (k, ia, ic)
        assert ia > 0 and ic < 0
    assert all(b > a for a, b in zip(c["anode"], c["anode"][1:]))     # monotonic forward I-V


def test_diode_forward_slope_is_diffusion_like(diode_run):
    """Between 0.5 and 0.6 V the diode current is diffusion-dominated: J2/J1 ~ exp(dV/Vt), ideality n ~ 1.
    Tolerance (own choice, fixed before running): 0.995 <= n <= 1.05. Justification: the only other current is
    space-charge (SRH) recombination with n = 2; its share at ~0.55 V is bounded by q n_i W/(2 tau) exp(V/2Vt)
    with W ~ 0.17 um, n_i ~ 1e10 cm^-3, tau = 1e-7 s, i.e. ~1e-2 A/cm^2 against ~0.1-1 A/cm^2 total, a few %;
    a share f raises n by about f/2, so n <= 1.05 leaves a > 2x margin. n < 1 is unphysical (0.5 % slack for
    the discretisation). High injection is absent (p_n(0.6 V) ~ 1e14 << N_D = 1e16) and the series drop in the
    n side (rho ~ 0.6 Ohm cm over 3 um) is < 1 mV at 1 A/cm^2."""
    Vt = K_B * 300.0 / Q_E
    i5, i6 = V_R1.index(0.5), V_R1.index(0.6)
    J1, J2 = diode_run.currents["anode"][i5], diode_run.currents["anode"][i6]
    n = (0.1 / Vt) / math.log(J2 / J1)
    assert 0.995 <= n <= 1.05, n


def test_diode_reverse_current_negative_and_bounded():
    """Reverse bias: current into the anode is negative and below the full-device SRH generation limit.
    Bound (derived): |U| <= n_i / (tau_n + tau_p) for any n, p >= 0 when the trap is at midgap, so the volume
    generation current is at most q n_i L / (2 tau) with L the device length; the minority diffusion current
    from the contacts adds q n_i^2 D/(N W) ~ 1e-10 A/cm^2 (estimated), covered by a factor 2 on the bound."""
    from simws.materials import load
    from simws.materials.semiconductor import intrinsic_density
    p = pn_diode_1d(nodes=201, V=[-0.5])
    r = _engine().run(p)
    J = r.currents["anode"][0]
    n_i = intrinsic_density(load("silicon"), 300.0)
    bound = 2 * Q_E * n_i * (4.0e-4) / (2 * 1e-7)
    assert J < 0, J
    assert abs(J) < bound, (J, bound)


def test_diode_mesh_scale_changes_current_little():
    """Halving every spacing (scale 0.5) changes the 0.6 V current by less than 1 % (own bound, fixed before
    running): Scharfetter-Gummel fluxes are exact for the diffusion current between nodes; the unresolved part is
    the SRH recombination peak (width ~kT/qE ~ 3 nm vs 5 nm spacing), whose share at 0.6 V is a few % (see the
    ideality test) and whose node-lumping error charon_xcheck measured at <= 14 % on a 10 nm mesh, so < 0.5 %."""
    E = _engine()
    a = E.run(pn_diode_1d(V=[0.6])).currents["anode"][0]
    b = E.run(pn_diode_1d(V=[0.6]).with_changes({"device.mesh.scale": 0.5})).currents["anode"][0]
    assert a != b                                       # the scale is actually applied
    assert abs(b / a - 1) < 1e-2, (a, b)


def test_equilibrium_result_shape_and_R4_diode(diode_run):
    """R4 (1D): equilibrium currents below 1e-8 x the forward current at 0.6 V."""
    r = _engine().run(pn_diode_1d())
    assert r.analysis == "equilibrium" and r.sweep_contact is None and r.sweep_V == [0.0]
    assert set(r.currents) == {"anode", "cathode"} and r.current_unit == "A/cm**2"
    j06 = diode_run.currents["anode"][V_R1.index(0.6)]
    for name, v in r.currents.items():
        assert len(v) == 1 and abs(v[0]) < 1e-8 * abs(j06), (name, v, j06)


# ------------------------------------------------------------------------------------------------ R3 (node values)
def _snap(c_cm, breakpoints_um):
    """Node coordinate (cm, as DEVSIM stores it) -> micrometres, preserving its order relative to every description
    breakpoint b as the engine sees it (the contract: coordinates in cm = um x 1e-4, so the engine compares c_cm
    with b x 1e-4). c x 1e4 alone can round across a breakpoint; then the result is moved to b itself (c_cm equal
    to b x 1e-4) or to the adjacent float on the correct side. This is the coordinate the engine uses, expressed
    in the description's unit; the interval rule is then applied by doping_at / mole_fraction_at unchanged."""
    c_cm = np.asarray(c_cm, float)
    out = c_cm * 1e4
    for b in sorted(set(breakpoints_um)):
        bc = b * 1e-4
        eq = c_cm == bc
        out[eq] = b
        lo = (c_cm < bc) & (out >= b)
        out[lo] = np.nextafter(b, -np.inf)
        hi = (c_cm > bc) & (out <= b)
        out[hi] = np.nextafter(b, np.inf)
    return out


def _breakpoints(dev, axis):
    bp = set()
    for t in dev.doping:
        iv = t.box.x if axis == "x" else t.box.y
        if iv is not None:
            bp |= {b for b in iv if b is not None}
    for g in dev.grading:
        if g.axis == axis:
            bp |= {g.start_um, g.stop_um}
    lines = dev.mesh.x_lines if axis == "x" else dev.mesh.y_lines
    bp |= {p for p, _ in lines}
    for r in dev.regions:
        iv = r.box.x if axis == "x" else r.box.y
        bp |= set(iv)
    return bp


def _node_values(region, name):
    from devsim import get_node_model_values
    return np.array(get_node_model_values(device="d", region=region, name=name))


def test_R3_hbt_node_doping_and_ge_match_description():
    p = npn_hbt_2d()
    used = _engine().build(p)
    assert isinstance(used, dict)
    dev = p.device
    x = _snap(_node_values("si", "x"), _breakpoints(dev, "x"))
    y = _snap(_node_values("si", "y"), _breakpoints(dev, "y"))
    nd, na = dev.doping_at(x, y)
    net = _node_values("si", "NetDoping")
    assert np.allclose(net, nd - na, rtol=1e-12, atol=0)
    ge = _node_values("si", "GeX")
    ref = dev.mole_fraction_at(x, y)
    # 1e-12 relative (R3); where the fraction is zero, relative has no meaning: 1e-12 x the peak (0.15) absolute
    assert np.allclose(ge, ref, rtol=1e-12, atol=1e-12 * 0.15)
    assert ge.max() == pytest.approx(0.15, rel=1e-12)          # the peak node (y = t_e + w_b) is on the mesh


def test_R3_diode_node_doping_matches_description():
    p = pn_diode_1d(nodes=201)
    _engine().build(p)
    x = _snap(_node_values("si", "x"), _breakpoints(p.device, "x"))
    nd, na = p.device.doping_at(x)
    assert np.allclose(_node_values("si", "NetDoping"), nd - na, rtol=1e-12, atol=0)


# ------------------------------------------------------------------------------------------------ R2 / R4 (HBT)
VBE = [0.5, 0.6, 0.7]


@pytest.fixture(scope="module")
def hbt_run():
    return _engine().run(npn_hbt_2d(VBE=VBE))


@pytest.mark.slow
def test_R2_hbt_matches_legacy_gummel(hbt_run):
    from simws.tcad.hbt import UM, HBTCase, gummel
    g = gummel(HBTCase(), VBE=VBE)
    r = hbt_run
    assert r.current_unit == "A/cm" and r.sweep_contact == "base" and r.sweep_V == VBE
    assert set(r.currents) == {"emitter", "base", "collector"}
    IC = np.abs(r.currents["collector"]) * UM          # A/cm -> A/um (legacy units, absolute values)
    IB = np.abs(r.currents["base"]) * UM
    assert _rel(IC, g["IC_A"]).max() <= 1e-6, (IC, g["IC_A"])
    assert _rel(IB, g["IB_A"]).max() <= 1e-6, (IB, g["IB_A"])


@pytest.mark.slow
def test_hbt_signs_and_kirchhoff(hbt_run):
    """Forward-active npn: conventional current flows into collector and base and out of the emitter."""
    c = hbt_run.currents
    for k in range(len(VBE)):
        assert c["collector"][k] > 0 and c["base"][k] > 0 and c["emitter"][k] < 0
        vals = [c[n][k] for n in c]
        assert abs(sum(vals)) <= 1e-4 * max(map(abs, vals)) + FLOOR_2D_A_CM


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="R4 (HBT) is a RECORDED FAIL: 1.35e-8 measured against the 1e-8 fixed before the "
                                       "run. The tolerance is not changed; see contract amendment A1.2 and R4b below.")
def test_R4_hbt_equilibrium(hbt_run):
    """R4 (2D): equilibrium currents below 1e-8 x the forward current at 0.6 V. "The forward current" is taken
    as the largest terminal current at VBE = 0.6 V (the emitter current), the least strict reading.
    Note (checker, first run): the LEGACY build gives an emitter equilibrium current of -2.89e-12 A/cm, i.e.
    1.35e-8 x the 0.6 V emitter current, so R4 as written fails for the legacy physics too (see
    test_hbt_equilibrium_matches_legacy_build); reported to the lead, tolerance not changed."""
    r = _engine().run(npn_hbt_2d())
    assert r.analysis == "equilibrium" and r.sweep_V == [0.0] and r.current_unit == "A/cm"
    k = VBE.index(0.6)
    i06 = max(abs(v[k]) for v in hbt_run.currents.values())
    for name, v in r.currents.items():
        assert abs(v[0]) < 1e-8 * i06, (name, v, i06)


@pytest.mark.slow
def test_R4b_hbt_equilibrium_below_project_floor():
    """R4b (lead, amendment A1.2, written after R4 failed, so POST HOC): the zero-bias terminal currents are below
    the project's existing zero-bias floor, the one the legacy HBT adapter has used since 0.9a
    (HbtGummel.I_FLOOR_A_PER_M; the number is imported, not chosen here)."""
    from simws.adapters.tcad import HbtGummel
    floor_A_cm = HbtGummel.I_FLOOR_A_PER_M / 100.0            # A/m -> A/cm
    r = _engine().run(npn_hbt_2d())
    for name, v in r.currents.items():
        assert abs(v[0]) < floor_A_cm, (name, v, floor_A_cm)


def test_hbt_equilibrium_matches_legacy_build():
    """Equilibrium contact currents of the translated HBT equal those of the legacy build (same solve), within R2's
    1e-6 relative where the legacy current is above 1e-20 A/cm (below that both are numerical zeros: compared to
    1e-20 absolute, own choice). Separates a translator error from a property of the legacy equilibrium solve when
    R4 fails."""
    from simws.tcad.hbt import HBTCase, build
    from simws.tcad.physics import contact_current
    build(HBTCase())
    legacy = {c: contact_current("d", c) for c in ("emitter", "base", "collector")}
    r = _engine().run(npn_hbt_2d())
    for c, v in legacy.items():
        got = r.currents[c][0]
        if abs(v) > 1e-20:
            assert abs(got / v - 1) <= 1e-6, (c, got, v)
        else:
            assert abs(got) <= 1e-20, (c, got, v)


# ------------------------------------------------------------------------------------------------ refusals
def test_unsupported_2d_contact_on_side():
    from simws.engines.base import Unsupported
    E = _engine()
    p = npn_hbt_2d(VBE=[0.6]).with_changes({"device.contacts.2.side": "x_max"})
    reason = "contacts: 2D DEVSIM contacts must be on y_min or y_max ('collector' is on x_max)"
    assert reason in E.unsupported(p)
    q = npn_hbt_2d(VBE=[0.6]).with_changes({"device.contacts.0.side": "x_min", "device.contacts.0.span": [0.0, 0.03]})
    assert "contacts: 2D DEVSIM contacts must be on y_min or y_max ('emitter' is on x_min)" in E.unsupported(q)
    with pytest.raises(Unsupported) as ei:
        E.run(p)
    assert reason in ei.value.reasons


def test_unsupported_region_not_at_origin():
    from simws.engines.base import Unsupported
    E = _engine()
    p = npn_hbt_2d().with_changes({"device.regions.0.box.x": [0.1, 1.5]})
    assert "regions: the region box must start at 0" in E.unsupported(p)
    p1 = pn_diode_1d().with_changes({"device.regions.0.box.x": [0.5, 4.0]})
    assert "regions: the region box must start at 0" in E.unsupported(p1)
    with pytest.raises(Unsupported):
        E.run(p1)


def test_unsupported_two_grade_terms():
    # A23.7(b) allows several grading terms when ordered and non-overlapping (same element, axis and mode); this
    # test, written before A23, refused any second term. It now checks the refusal A23.7(b) keeps: overlapping terms.
    from simws.engines.base import Unsupported
    p = npn_hbt_2d()
    g = p.device.grading[0].model_dump(mode="json")
    g2 = {**g, "start_um": 0.05, "stop_um": 0.1}
    p2 = p.with_changes({"device.grading": [g, g2]})
    with pytest.raises(Unsupported):
        _engine().run(p2)


def test_unsupported_capability_reason_raises():
    from simws.engines.base import Unsupported
    p = pn_diode_1d(V=[0.1]).with_changes({"device.contacts.0.kind": "schottky", "device.contacts.0.barrier_eV": 0.7})
    with pytest.raises(Unsupported) as ei:
        _engine().run(p)
    assert "contact_kinds: 'schottky' not in ['ohmic']" in ei.value.reasons


def test_newton_failure_becomes_not_converged():
    """One Newton iteration per step cannot reach 1e-12 relative error on a 50 mV step: the ramp stalls."""
    from simws.hub.adapter import NotConverged
    p = pn_diode_1d(nodes=201, V=[0.3]).with_changes({"numerics.max_iterations": 1})
    with pytest.raises(NotConverged):
        _engine().run(p)


def test_run_accepts_workdir(tmp_path):
    r = _engine().run(pn_diode_1d(nodes=201, V=[0.2]), tmp_path)
    assert r.sweep_V == [0.2] and r.currents["anode"][0] > 0
