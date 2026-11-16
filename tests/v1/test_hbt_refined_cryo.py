"""Contract amendment A5 (docs/ARCHITECTURE_V1.md): refined HBT mesh (eb_refine) and cold-temperature HBT (cryo)
on the new path. Acceptance R7a/R7b/R7c (A5.2) and R8 (A5.3); tolerances are the ones fixed in A5 before any run.

Written from the contract text, simws/spec and the legacy reference (simws.tcad.hbt, simws.adapters.tcad.HbtGummel)
only. Expected mesh lines are rebuilt here from the expressions of simws.tcad.hbt.build for HBTCase(eb_refine=True);
expected currents come from the legacy code run in the same test. No reference value is typed in.
"""
import math
import os

import numpy as np
import pytest

from simws.spec.library import npn_hbt_2d

R = math.sqrt(2.0)                    # A4.3 refinement ratio (R7c repeats R5)
VAL_RTOL = 1e-6                       # A5.2 R7b, A5.3 R8, A4.7 R5 (values)
SIG_RTOL = 1e-3                       # A4.7 R5 (sigmas), repeated by R7c


def _params(p, **extra):
    return {"problem": p.model_dump(mode="json"), **extra}


def _legacy_refined_lines():
    """simws.tcad.hbt.build for HBTCase(eb_refine=True), mesh_scale 1, no extra x_lines: the (position, spacing)
    lines in um, rebuilt from the legacy expressions with the HBTCase field values."""
    from simws.tcad.hbt import HBTCase
    c = HBTCase(eb_refine=True)
    yb = c.t_e + c.w_b
    xl = sorted({(0, 0.05), (c.w_e - 0.05, 0.02), (c.w_e, 0.01), (c.w_e + 0.05, 0.02), (c.x_xb, 0.02),
                 (c.x_bc, 0.05), (c.width_um, 0.1)})
    we = c.w_e
    xl = sorted({(x, h) for x, h in xl if abs(x - we) > 0.009} |
                {(we - 0.008, 0.001), (we - 0.002, 0.0002), (we, 0.00005), (we + 0.002, 0.0002), (we + 0.008, 0.001)})
    yl = sorted([(0, 0.005), (c.t_e - 0.008, 0.001), (c.t_e - 0.002, 0.0002), (c.t_e, 0.00005),
                 (c.t_e + 0.002, 0.0002), (c.t_e + 0.008, 0.001), (yb - 0.003, 0.0005), (yb, 0.0002),
                 (yb + 0.003, 0.0005), (yb + 0.1, 0.01), (c.y_sc, 0.05), (c.depth_um, 0.1)])
    return [(float(x), float(h)) for x, h in xl], [(float(y), float(h)) for y, h in yl]


def _lines(lines):
    return [(float(p), float(h)) for p, h in lines]


def _without(d, path):
    """Deep copy of a model_dump dict without the dotted key `path`."""
    import copy
    d = copy.deepcopy(d)
    *head, last = path.split(".")
    cur = d
    for k in head:
        cur = cur[k]
    del cur[last]
    return d


CRYO_MODELS = {"statistics": "fermi_dirac", "mobility": "klaassen", "incomplete_ionization": "altermatt"}


# ================================================================================================ A5.1 library (fast)
def test_a5_1_defaults_leave_problem_id_unchanged():
    base = npn_hbt_2d(VBE=[0.6, 0.7])
    explicit = npn_hbt_2d(VBE=[0.6, 0.7], eb_refine=False, cryo=False)
    assert explicit.problem_id() == base.problem_id()
    assert explicit.model_dump(mode="json") == base.model_dump(mode="json")
    # also without a sweep (equilibrium)
    assert npn_hbt_2d(eb_refine=False, cryo=False).problem_id() == npn_hbt_2d().problem_id()


def test_a5_1_eb_refine_changes_only_the_mesh():
    base = npn_hbt_2d(VBE=[0.6, 0.7]).model_dump(mode="json")
    ref = npn_hbt_2d(VBE=[0.6, 0.7], eb_refine=True).model_dump(mode="json")
    assert _without(ref, "device.mesh") == _without(base, "device.mesh")
    assert ref["device"]["mesh"] != base["device"]["mesh"]
    assert npn_hbt_2d(VBE=[0.6, 0.7], eb_refine=True).problem_id() != npn_hbt_2d(VBE=[0.6, 0.7]).problem_id()


def test_a5_1_eb_refine_lines_are_the_legacy_lines():
    xl, yl = _legacy_refined_lines()
    m = npn_hbt_2d(eb_refine=True).device.mesh
    assert _lines(m.x_lines) == xl            # exact: same positions and spacings, sorted
    assert _lines(m.y_lines) == yl
    assert m.scale == 1.0
    # the mesh-scale argument is kept with eb_refine
    assert npn_hbt_2d(eb_refine=True, mesh_scale=0.5).device.mesh.scale == 0.5
    assert _lines(npn_hbt_2d(eb_refine=True, mesh_scale=0.5).device.mesh.x_lines) == xl
    # the spacing at the junctions named in A5.1: 0.05 nm at the emitter-base junction, 0.2 nm at base-collector
    from simws.tcad.hbt import HBTCase
    c = HBTCase()
    assert dict(yl)[c.t_e] == 0.00005 and dict(xl)[c.w_e] == 0.00005
    assert dict(yl)[c.t_e + c.w_b] == 0.0002


def test_a5_1_cryo_changes_only_the_three_models():
    base = npn_hbt_2d(VBE=[0.6, 0.7]).model_dump(mode="json")
    cold = npn_hbt_2d(VBE=[0.6, 0.7], cryo=True).model_dump(mode="json")
    for k, v in CRYO_MODELS.items():
        assert cold["models"][k] == v, (k, cold["models"][k])
        assert base["models"][k] != v, (k, base["models"][k])
    b, c = base, cold
    for k in CRYO_MODELS:
        b, c = _without(b, f"models.{k}"), _without(c, f"models.{k}")
    assert c == b


def test_a5_1_eb_refine_and_cryo_combine():
    base = npn_hbt_2d(VBE=[0.6, 0.7]).model_dump(mode="json")
    ref = npn_hbt_2d(VBE=[0.6, 0.7], eb_refine=True).model_dump(mode="json")
    cold = npn_hbt_2d(VBE=[0.6, 0.7], cryo=True).model_dump(mode="json")
    both = npn_hbt_2d(VBE=[0.6, 0.7], eb_refine=True, cryo=True).model_dump(mode="json")
    assert both["device"]["mesh"] == ref["device"]["mesh"]
    assert both["models"] == cold["models"]
    assert _without(_without(both, "device.mesh"), "models") == _without(_without(base, "device.mesh"), "models")
    xl, yl = _legacy_refined_lines()
    m = npn_hbt_2d(eb_refine=True, cryo=True).device.mesh
    assert _lines(m.x_lines) == xl and _lines(m.y_lines) == yl


# ================================================================================================ R7a (DEVSIM)
def _region_xy(region="si"):
    from devsim import get_node_model_values
    return (np.array(get_node_model_values(device="d", region=region, name="x")),
            np.array(get_node_model_values(device="d", region=region, name="y")))


def test_R7a_refined_node_coordinates_equal_legacy():          # ~10 s on the build machine: fast tier
    """R7a: mesh scale 1, x and y node arrays of the region from npn_hbt_2d(eb_refine=True) exactly equal
    (numpy.array_equal) those of simws.tcad.hbt.build(HBTCase(eb_refine=True))."""
    pytest.importorskip("devsim")
    from simws import engines
    from simws.tcad.hbt import HBTCase, build
    p = npn_hbt_2d(eb_refine=True)
    assert p.device.mesh.scale == 1.0
    engines.module("devsim").build(p)
    xn, yn = _region_xy(p.device.regions[0].name)
    build(HBTCase(eb_refine=True))
    xo, yo = _region_xy("si")
    print(f"R7a nodes: new {xn.size}, legacy {xo.size}")
    assert xn.size == xo.size, (xn.size, xo.size)
    assert np.array_equal(xn, xo)
    assert np.array_equal(yn, yo)


# ================================================================================================ R7b (slow)
@pytest.mark.slow
def test_R7b_refined_hbt_matches_legacy_gummel():
    """R7b: engine.run(npn_hbt_2d(eb_refine=True, VBE=[0.6])) vs gummel(HBTCase(eb_refine=True), VBE=[0.6]):
    |IC| and |IB| within 1e-6 relative (legacy A/um = A/cm x 1e-4, absolute values, as in R2)."""
    pytest.importorskip("devsim")
    from simws import engines
    from simws.tcad.hbt import UM, HBTCase, gummel
    V = [0.6]
    r = engines.module("devsim").run(npn_hbt_2d(eb_refine=True, VBE=V))
    assert r.current_unit == "A/cm" and r.sweep_contact == "base" and r.sweep_V == V
    IC = np.abs(np.asarray(r.currents["collector"], float)) * UM
    IB = np.abs(np.asarray(r.currents["base"], float)) * UM
    g = gummel(HBTCase(eb_refine=True), VBE=V)
    dIC = np.abs(IC / g["IC_A"] - 1.0)
    dIB = np.abs(IB / g["IB_A"] - 1.0)
    print(f"R7b IC new={IC} legacy={g['IC_A']} rel={dIC}; IB new={IB} legacy={g['IB_A']} rel={dIB}")
    assert dIC.max() <= VAL_RTOL, (IC, g["IC_A"])
    assert dIB.max() <= VAL_RTOL, (IB, g["IB_A"])


# ================================================================================================ R7c (overnight)
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("SIMWS_OVERNIGHT") != "1",
                    reason="R7c is for the overnight queue (hours): set SIMWS_OVERNIGHT=1")
def test_R7c_refined_hbt_mesh_study_matches_legacy(hub):
    """R7c: R5 (A4.7) repeated with eb_refine=True on both paths, same criteria: values within 1e-6, sigmas within
    1e-3 relative, verification["mesh"]["passed"] equal, every point's conservative_bound equal."""
    pytest.importorskip("devsim")
    VBE = [0.6, 0.7]
    new = hub.submit("device.devsim", params=_params(npn_hbt_2d(VBE=VBE, eb_refine=True), mesh_study=True))
    assert new.status == "ok", new.message
    old = hub.submit("devsim.hbt_gummel", params={"VBE": VBE, "mesh_study": True, "case": {"eb_refine": True}})
    assert old.status == "ok", old.message
    assert old.q("hbt.IC").axes[0].values == pytest.approx(VBE)
    nrec, orec = new.verification["mesh"], old.verification["mesh"]
    fails = []
    for nn, on in [("dc.I.collector", "hbt.IC"), ("dc.I.base", "hbt.IB"), ("dc.I.emitter", "hbt.IE")]:
        a, b = new.q(nn), old.q(on)
        assert a.unit == "A/m" and b.unit == "A/m"
        va, vb = np.abs(a.array()), np.abs(b.array())
        sa, sb = np.asarray(a.uncertainty.sigma, float), np.asarray(b.uncertainty.sigma, float)
        dv, ds = np.abs(va / vb - 1), np.abs(sa / sb - 1)
        print(f"R7c {nn} vs {on}: |I| new={va} legacy={vb} rel={dv}; sigma new={sa} legacy={sb} rel={ds}")
        if dv.max() > VAL_RTOL:
            fails.append(f"{nn}: value rel diff {dv.max():.3e} > {VAL_RTOL}")
        if ds.max() > SIG_RTOL:
            fails.append(f"{nn}: sigma rel diff {ds.max():.3e} > {SIG_RTOL}")
        cn = [e["conservative_bound"] for e in nrec["quantities"][nn]]
        co = [e["conservative_bound"] for e in orec["quantities"][on]]
        print(f"R7c {nn}: conservative new={cn} legacy={co}")
        if cn != co:
            fails.append(f"{nn}: conservative_bound {cn} vs legacy {co}")
    print(f"R7c passed: new={nrec['passed']} legacy={orec['passed']}")
    if nrec["passed"] != orec["passed"]:
        fails.append(f"passed: new {nrec['passed']} vs legacy {orec['passed']}")
    assert not fails, fails


# ================================================================================================ R8 (slow)
R8_CASES = {250.0: [0.65, 0.70], 150.0: [0.85, 0.90], 77.0: [1.00, 1.05]}     # A5.3


@pytest.mark.slow
@pytest.mark.parametrize("T", list(R8_CASES))
def test_R8_cold(hub, T):
    """R8: device.devsim on npn_hbt_2d(T=T, cryo=True, VBE=...) vs devsim.hbt_gummel with {"VBE", "T", "cryo": True},
    both through the hub at screening fidelity. At every bias the legacy result keeps (>= 1 per temperature),
    |dc.I.collector|, |dc.I.base|, |dc.I.emitter| equal hbt.IC, hbt.IB, |hbt.IE| within 1e-6 relative. Flags: below
    250 K the new result has a flag starting with "extrapolated:temperature_K"; a legacy "newton_rel_tol:" flag
    implies one on the new result."""
    pytest.importorskip("devsim")
    VBE = R8_CASES[T]
    new = hub.submit("device.devsim", params=_params(npn_hbt_2d(T=T, cryo=True, VBE=VBE)), fidelity="screening")
    assert new.status == "ok", (new.status, new.message, new.flags)
    old = hub.submit("devsim.hbt_gummel", params={"VBE": VBE, "T": T, "cryo": True}, fidelity="screening")
    assert old.status == "ok", (old.status, old.message, old.flags)
    print(f"R8 T={T:g} K new flags: {new.flags}")
    print(f"R8 T={T:g} K legacy flags: {old.flags}")

    kept = [float(v) for v in old.q("hbt.IC").axes[0].values]
    assert len(kept) >= 1, "legacy result kept no bias point"
    new_V = [float(v) for v in new.q("dc.I.collector").axes[0].values]
    assert new_V == pytest.approx(VBE, rel=0, abs=1e-12), new_V
    fails, worst = [], {}
    for nn, on in [("dc.I.collector", "hbt.IC"), ("dc.I.base", "hbt.IB"), ("dc.I.emitter", "hbt.IE")]:
        a, b = new.q(nn), old.q(on)
        assert a.unit == "A/m" and b.unit == "A/m", (a.unit, b.unit)
        va, vb = np.abs(a.array()), np.abs(b.array())
        assert [float(v) for v in b.axes[0].values] == kept
        rels = []
        for j, v in enumerate(kept):
            i = min(range(len(new_V)), key=lambda k: abs(new_V[k] - v))
            assert abs(new_V[i] - v) <= 1e-12, (v, new_V)
            rels.append(abs(va[i] / vb[j] - 1.0))
            print(f"R8 T={T:g} K VBE={v:g} {nn}: new={va[i]:.10e} legacy={vb[j]:.10e} rel={rels[-1]:.3e}")
        worst[nn] = max(rels)
        if worst[nn] > VAL_RTOL:
            fails.append(f"{nn}: max rel diff {worst[nn]:.3e} > {VAL_RTOL}")
    print(f"R8 T={T:g} K kept VBE={kept} max rel diff: " + ", ".join(f"{k}={v:.3e}" for k, v in worst.items()))

    if T < 250.0 and not any(f.startswith("extrapolated:temperature_K") for f in new.flags):
        fails.append(f"no 'extrapolated:temperature_K' flag below 250 K: {new.flags}")
    if any(f.startswith("newton_rel_tol:") for f in old.flags) and \
            not any(f.startswith("newton_rel_tol:") for f in new.flags):
        fails.append(f"legacy has a newton_rel_tol flag, new has none: {new.flags}")
    assert not fails, fails


def test_A5_4_cryo_unvalidated_flag():
    """Lead, contract A5.4: below 250 K the DEVSIM engine result carries the legacy 'cryo_unvalidated' warning;
    at 250 K and above it does not (fast 1D diode, one bias)."""
    pytest.importorskip("devsim")
    from simws import engines
    from simws.spec.library import pn_diode_1d
    dv = engines.module("devsim")
    cold = dv.run(pn_diode_1d(V=[0.3], nodes=101, T=200.0))
    warm = dv.run(pn_diode_1d(V=[0.3], nodes=101, T=300.0))
    assert any(f.startswith("cryo_unvalidated:") for f in cold.flags)
    assert not any(f.startswith("cryo_unvalidated") for f in warm.flags)
