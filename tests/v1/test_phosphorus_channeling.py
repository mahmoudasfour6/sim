"""Contract amendment A21 (24 Nov 2026): phosphorus channeling from the SIMS of Ghante et al. 1995 (simws's own fit).
Acceptance checks R140 to R145 of A21.4, the data rules of item 1, the model and its two conditions (items 2 and 8,
the one-component form of item 8 as corrected before any acceptance number), the engine and its flags (item 3) and
trust (item 5).

Written by the checker from docs/ARCHITECTURE_V1.md (A21 items 0 to 8; background A15, A16, A11.4, A13.7) and the
public interfaces: simws.radiation.channeling (model("P").select, the condition objects' parameters(E, dose) and
profile(x, E, dose, perturb), ghante_curves(use), OWN_FIT_FLAG, SIMS_FLAG), the data file
simws/data/channeling/ghante1995_phosphorus.json, the engine registry (simws.engines; channeling's unsupported, run,
share_beyond), the hub (adapter "implant.channeling"), simws.hub.trust.assess_stored, the planner and the chain
functions of simws.chain.process.

The checker's own model (`own_nc` below) is Suzuki's n_c as A15.1 / A21.8 describe it, written here: a joined half
Gaussian of peak 1 at Rpm (front dRp1, back dRp2) up to J = Rpm + dRp2, and beyond J kappa (JHG + exp(-((x - Rpm) /
L)^alpha)) with kappa = JHG(J) / (JHG(J) + tail(J)) (continuous at J, Suzuki eq. 7, the form the A15 checker
verified against the paper); normalised on [0, inf) in closed form (erf and the incomplete gamma function). It is
used (a) to check the form of the condition objects' profile, (b) for the leave-one-out parameters of R141 / R142,
which A21.4 asks to be built from the neighbours' stored fits (A21.2: linear in energy; linear in log10 dose),
not through the engine's interpolation, and (c) for the exact share beyond a target (normalisation, R144).

Readings made here (stated so the lead can object):
- "Above the floor" (R140) is taken as the data file states its fit rule: at or above floor_cm3 and deeper than
  5 nm.
- "The depth where it falls to X" is the deepest depth at which the curve is still at or above X; between that
  point and the next one the crossing is interpolated linearly in depth and in log10 C. The model is evaluated
  on a 0.01 nm grid from 0 to 1.5 um. A curve whose traced points never reach X has no such depth (Figure 2,
  1e13: peak 9.9e17): the depth part of R140 is not applicable there and is reported.
- "Peak concentration" (R141 to R143): the trace's largest point deeper than 5 nm (the fit's domain) against the
  model's maximum on the grid.
- R145: the crystal implant into a 1 um target with 2 nm bins, the anneal thickness 2 um (as A20's R131); the
  amorphous profile RustBCA, P 50 keV 1e14 tilt 10 deg, 2 nm bins, default stopping, 20 000 ions.
Tests that read the PDF (SHA-256, calibration ticks, traced points on the ink) look for SIMWS_GHANTE_PDF (default:
the build machine's scratch copy) and skip when it is absent.
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf, gamma as Gamma, gammaincc

REPO = Path(__file__).resolve().parents[2]
DATA_FILE = REPO / "simws" / "data" / "channeling" / "ghante1995_phosphorus.json"
GH = json.loads(DATA_FILE.read_text())
PDF = Path(os.environ.get("SIMWS_GHANTE_PDF", "/tmp/claude-0/-home-claude/d0abb2e9-411f-5090-9cb2-dd3ba3452a20/"
                                              "scratchpad/papers/ghante1995.pdf"))
ADAPTER = "implant.channeling"
KEYS = ("Rpm_um", "dRp1_um", "dRp2_um", "L_um", "alpha")
LENGTHS = ("Rpm_um", "dRp1_um", "dRp2_um", "L_um")
# A21.2, verbatim
OWN_FIT = "implant:own_fit: simws's fit to the SIMS of Ghante et al. 1995 (not a published model)"
SIMS = "budget:sims_accuracy: the SIMS are good to 50 % in concentration and 10 % in depth"
COND = {"A": (4.0, 22.0), "B": (10.0, 0.0)}
XF = np.linspace(0.0, 1.5, 150001)                       # model grid, 0.01 nm
FIT_CURVES = [c for c in GH["curves"] if c["use"] in ("fit_A", "fit_B")]


def cid(c):
    return f"fig{c['figure']}-{c['energy_keV']:g}keV-{c['dose_cm2']:.0e}"


# ------------------------------------------------------------------------------------------------ own model
def _jhg_cum(x, Rpm, s1, s2):
    """Integral from 0 to x of the joined half Gaussian of peak 1."""
    c = math.sqrt(math.pi / 2)
    x = np.asarray(x, float)
    left0 = s1 * c * erf(-Rpm / (math.sqrt(2) * s1))
    xl = np.minimum(x, Rpm)
    G = s1 * c * erf((xl - Rpm) / (math.sqrt(2) * s1)) - left0
    xr = np.maximum(x, Rpm)
    return G + s2 * c * erf((xr - Rpm) / (math.sqrt(2) * s2))


def _tail_upper(x, Rpm, L, a):
    """Integral from x (> Rpm) to infinity of exp(-((t - Rpm) / L)^a)."""
    z = ((np.asarray(x, float) - Rpm) / L) ** a
    return L / a * Gamma(1 / a) * gammaincc(1 / a, z)


def own_nc(P, scale=1.0):
    """(density 1/um, CDF) of the one-component model (A21.8) for parameters P (KEYS), lengths x scale."""
    Rpm, s1, s2, L = (P[k] * scale for k in LENGTHS)
    a = P["alpha"]
    J = Rpm + s2
    uJ = math.exp(-0.5)
    TJ = math.exp(-((J - Rpm) / L) ** a)
    k = uJ / (uJ + TJ)
    GJ = float(_jhg_cum(J, Rpm, s1, s2))
    Ginf = float(_jhg_cum(np.inf, Rpm, s1, s2))
    TailJ = float(_tail_upper(J, Rpm, L, a))
    H = GJ + k * ((Ginf - GJ) + TailJ)

    def dens(x):
        x = np.asarray(x, float)
        u = np.exp(-(x - Rpm) ** 2 / (2 * np.where(x < Rpm, s1, s2) ** 2))
        t = np.exp(-np.power(np.clip((x - Rpm) / L, 0, None), a))
        return np.where(x <= J, u, k * (u + t)) / H

    def cdf(x):
        x = np.asarray(x, float)
        xx = np.maximum(x, J)
        beyond = GJ + k * ((_jhg_cum(xx, Rpm, s1, s2) - GJ) + (TailJ - _tail_upper(xx, Rpm, L, a)))
        return np.where(x <= J, _jhg_cum(x, Rpm, s1, s2), beyond) / H
    return dens, cdf


def lin(x0, p0, x1, p1, x):
    t = (x - x0) / (x1 - x0)
    return {k: p0[k] + t * (p1[k] - p0[k]) for k in KEYS}


# ------------------------------------------------------------------------------------------------ helpers
def cond(name):
    from simws.radiation import channeling as ch
    return ch.model("P").select(*COND[name])


def cond_of(c):
    return "A" if c["use"] == "fit_A" else "B"


def model_conc(cnd, E, dose, x):
    """cm^-3 from the public condition object: density (1/um per ion) x 1e4 x dose."""
    return np.asarray(cnd.profile(np.asarray(x, float), E, dose, 0)["density"], float) * 1e4 * dose


def depth_at(x, c, level):
    """Deepest depth at which c >= level; linear in depth and log10 c to the next point (None if never)."""
    x, c = np.asarray(x, float), np.asarray(c, float)
    idx = np.nonzero(c >= level)[0]
    if idx.size == 0:
        return None
    i = idx[-1]
    if i == c.size - 1:
        return float(x[i])
    if c[i + 1] <= 0:
        return float(x[i + 1])
    t = (math.log10(c[i]) - math.log10(level)) / (math.log10(c[i]) - math.log10(c[i + 1]))
    return float(x[i] + t * (x[i + 1] - x[i]))


def trace(c):
    return np.asarray(c["depth_um"], float), np.asarray(c["conc_cm3"], float)


def fit_mask(c):
    x, y = trace(c)
    return (y >= c["floor_cm3"]) & (x > 0.005)


def trace_peak(c):
    x, y = trace(c)
    return float(y[x > 0.005].max())


def curve(fig, E, dose):
    out = [c for c in GH["curves"] if c["figure"] == fig and c["energy_keV"] == E and abs(c["dose_cm2"] / dose - 1)
           < 1e-9]
    assert len(out) == 1, (fig, E, dose)
    return out[0]


def p_problem(tilt, rot, E, dose, thickness_um=1.0, bin_um=0.002, **kw):
    from simws.spec.library import implant_si
    p = implant_si("P", E, tilt_deg=tilt, rotation_deg=rot, dose_cm2=dose, crystal=True, thickness_um=thickness_um)
    return p.with_changes({"depth_bin_um": bin_um, **kw})


def engine():
    import simws.engines as E
    return E.module("channeling")


def reasons(p):
    return engine().unsupported(p)


def all_flags(r):
    return list(r.flags) + [f for q in r.quantities for f in q.flags]


@pytest.fixture(scope="module")
def mhub(tmp_path_factory):
    from simws.hub.jobs import Hub
    return Hub(tmp_path_factory.mktemp("phosphorus_store"), mode="inprocess")


# ================================================================================================ item 1: data
def test_item1_curves_present():
    cs = GH["curves"]
    f1 = [c for c in cs if c["figure"] == 1]
    assert sorted(c["energy_keV"] for c in f1) == [15, 30, 50, 80, 120, 180]
    for c in f1:
        assert (c["dose_cm2"], c["tilt_deg"], c["rotation_deg"], c["use"], c["floor_cm3"]) == \
            (1e14, 4.0, 22.0, "fit_A", 2e17), cid(c)
    f2 = [c for c in cs if c["figure"] == 2]
    assert sorted(c["dose_cm2"] for c in f2) == [1e13, 1e14, 5e14, 2e15, 8e15]
    for c in f2:
        assert (c["energy_keV"], c["tilt_deg"], c["rotation_deg"], c["use"]) == (50.0, 10.0, 0.0, "fit_B"), cid(c)
    for fig, E, dose, use in ((3, 30.0, 1e14, "fit_B"), (6, 120.0, 1e14, "fit_B"), (5, 50.0, 5e14, "check")):
        c = curve(fig, E, dose)
        assert (c["tilt_deg"], c["rotation_deg"], c["use"]) == (10.0, 0.0, use), cid(c)
    assert len(cs) == 14
    assert {c["use"] for c in cs} == {"fit_A", "fit_B", "check"}


def test_item1_ghante_curves_by_use():
    from simws.radiation import channeling as ch
    for use in ("fit_A", "fit_B", "check"):
        assert [cid(c) for c in ch.ghante_curves(use)] == [cid(c) for c in GH["curves"] if c["use"] == use]


@pytest.mark.parametrize("c", GH["curves"], ids=cid)
def test_item1_points_and_floor(c):
    x, y = trace(c)
    assert x.size == y.size >= 50
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    assert np.all(np.diff(x) > 0) and x[0] >= 0
    assert np.all(y > 0)
    assert c["floor_cm3"] > 0
    below = int(np.sum(y < c["floor_cm3"]))
    print(f"item 1 {cid(c)}: {x.size} points, {below} below the floor {c['floor_cm3']:.0e} (kept), "
          f"{int(fit_mask(c).sum())} at/above it and deeper than 5 nm")
    assert below > 0                                           # points below the floor are kept


def test_item1_source_calibration_sha():
    s = GH["source"]
    assert len(s["pdf_sha256"]) == 64 and int(s["pdf_sha256"], 16) >= 0
    cal = s["calibration"]
    assert set(cal) == {"fig1", "fig2", "fig3", "fig5", "fig6"}
    for k, v in cal.items():
        assert len(v["x_px"]) == len(v["um"]) >= 2 and len(v["y_px"]) == len(v["log10_cm3"]) >= 2, k
    # A21.7(a): the authors as corrected
    assert s["cite"].startswith("V. Ghante, L. M. Lam, S. Morris, S.-H. Yang, A. F. Tasch, D. Kamenitsa, J. Sheng, "
                                "C. Magee")
    assert "396, 715 (1995)" in s["cite"] and "10.1557/PROC-396-715" in s["cite"]


def test_item1_fit_points_rule():
    """The stored fit used the points at/above the floor and deeper than 5 nm (the data file's rule); the check
    curve (Figure 5) has no fit."""
    for c in FIT_CURVES:
        m = fit_mask(c)
        x, y = trace(c)
        strict = int(np.sum((y > c["floor_cm3"]) & (x > 0.005)))
        assert c["fit_points"] == int(m.sum()), (cid(c), c["fit_points"], int(m.sum()), strict)
    chk = [c for c in GH["curves"] if c["use"] == "check"]
    assert len(chk) == 1 and "fit" not in chk[0]


def test_item0_trace_doses_reported():
    """A21.0 (recorded, not gated): each trace integrates to its nominal dose within 6 % (Fig. 1) and 2 % (Fig. 2)."""
    lines = []
    worst = {}
    for c in GH["curves"]:
        x, y = trace(c)
        d = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)) * 1e-4)
        r = d / c["dose_cm2"]
        worst[c["figure"]] = max(worst.get(c["figure"], 0.0), abs(r - 1))
        lines.append(f"{cid(c)} {r:.4f}")
    print("item 0, trace dose / nominal (trapezoid over the traced points):\n  " + "\n  ".join(lines))
    print(f"  worst per figure: {worst}")


# ================================================================================================ items 2 and 8: model
def test_item8_stored_fits_have_five_parameters():
    for c in FIT_CURVES:
        assert set(c["fit"]) == set(KEYS), cid(c)
        f = c["fit"]
        assert all(f[k] > 0 for k in KEYS), cid(c)


@pytest.mark.parametrize("c", FIT_CURVES, ids=cid)
def test_item8_stored_residuals(c):
    """The stored residual (rms of log10(model / trace) over the fitted points) is that of the stored parameters
    with the item 8 form (the checker's own model); stored to 4 decimals."""
    dens, _ = own_nc(c["fit"])
    x, y = trace(c)
    m = fit_mask(c)
    r = np.log10(dens(x[m]) * 1e4 * c["dose_cm2"] / y[m])
    rms = float(np.sqrt(np.mean(r ** 2)))
    print(f"item 8 {cid(c)}: rms log10 {rms:.5f}, stored {c['fit_rms_log10']}")
    assert abs(rms - c["fit_rms_log10"]) <= 6e-5


@pytest.mark.parametrize("c", FIT_CURVES, ids=cid)
def test_item8_condition_profile_is_one_component(c):
    """At each fitted curve's own condition the public profile has the item 8 form: the checker's n_c with the
    stored parameters, up to a constant factor (the normalisation is R144's); the join at Rpm + dRp2 is continuous
    and the maximum is at Rpm."""
    cn = cond(cond_of(c))
    E, d = c["energy_keV"], c["dose_cm2"]
    P = c["fit"]
    assert cn.parameters(E, d) == pytest.approx(P, rel=1e-12)
    x = np.linspace(0.0, 2.0, 200001)
    eng = np.asarray(cn.profile(x, E, d, 0)["density"], float)
    own = own_nc(P)[0](x)
    keep = own > 1e-10 * own.max()
    ratio = eng[keep] / own[keep]
    spread = float(ratio.max() / ratio.min() - 1)
    J = P["Rpm_um"] + P["dRp2_um"]
    lo, hi = (float(np.asarray(cn.profile(np.array([J * (1 + s)]), E, d, 0)["density"])[0]) for s in (-1e-10, 1e-10))
    xm = float(x[np.argmax(eng)])
    print(f"item 8 {cid(c)}: profile / own n_c constant within {spread:.2e} (scale {ratio.mean():.6f}); join "
          f"{abs(hi / lo - 1):.2e}; maximum at {xm * 1e3:.3f} nm, Rpm {P['Rpm_um'] * 1e3:.3f} nm")
    assert spread <= 1e-6
    assert abs(hi / lo - 1) <= 1e-6
    assert abs(xm - P["Rpm_um"]) <= 1e-5


def _stored(use, **sel):
    return [c for c in GH["curves"] if c["use"] == use and all(c[k] == v for k, v in sel.items())]


def test_item2_condition_A_linear_in_energy():
    A = {c["energy_keV"]: c["fit"] for c in _stored("fit_A")}
    cn = cond("A")
    for E, P in A.items():
        assert cn.parameters(E, 1e14) == pytest.approx(P, rel=1e-12)
    for E, (lo, hi) in {20.0: (15, 30), 65.0: (50, 80), 100.0: (80, 120), 150.0: (120, 180)}.items():
        assert cn.parameters(E, 1e14) == pytest.approx(lin(lo, A[lo], hi, A[hi], E), rel=1e-12), E


def test_item2_condition_B_energy_set_at_1e14():
    """(B) at 1e14: 30, 50, 120 keV from Figures 3, 2 (A21.8: Figure 2's 1e14 curve), 6; linear in energy."""
    f3, f2, f6 = curve(3, 30.0, 1e14)["fit"], curve(2, 50.0, 1e14)["fit"], curve(6, 120.0, 1e14)["fit"]
    cn = cond("B")
    assert cn.parameters(30.0, 1e14) == pytest.approx(f3, rel=1e-12)
    assert cn.parameters(50.0, 1e14) == pytest.approx(f2, rel=1e-12)
    assert cn.parameters(120.0, 1e14) == pytest.approx(f6, rel=1e-12)
    assert cn.parameters(40.0, 1e14) == pytest.approx(lin(30, f3, 50, f2, 40.0), rel=1e-12)
    assert cn.parameters(75.0, 1e14) == pytest.approx(lin(50, f2, 120, f6, 75.0), rel=1e-12)


def test_item2_condition_B_dose_set_at_50keV():
    D = {c["dose_cm2"]: c["fit"] for c in _stored("fit_B", figure=2)}
    cn = cond("B")
    for d, P in D.items():
        assert cn.parameters(50.0, d) == pytest.approx(P, rel=1e-12)
    for d, (lo, hi) in {3e13: (1e13, 1e14), 3e14: (1e14, 5e14), 1e15: (5e14, 2e15), 4e15: (2e15, 8e15)}.items():
        want = lin(math.log10(lo), D[lo], math.log10(hi), D[hi], math.log10(d))
        assert cn.parameters(50.0, d) == pytest.approx(want, rel=1e-12), d


def test_item2_select_two_conditions():
    from simws.radiation import channeling as ch
    m = ch.model("P")
    a, b = m.select(4.0, 22.0), m.select(10.0, 0.0)
    assert a is not None and b is not None and a is not b
    assert (a.tilt_deg, a.rotation_deg) == (4.0, 22.0) and (b.tilt_deg, b.rotation_deg) == (10.0, 0.0)
    assert m.select(4.5, 21.5) is not None and m.select(9.5, 0.5) is not None
    for t, r in ((4.6, 22.0), (4.0, 22.6), (10.6, 0.0), (10.0, 0.6), (7.0, 22.0), (7.0, 11.0), (4.0, 0.0),
                 (10.0, 22.0), (4.0, None), (10.0, None)):
        assert m.select(t, r) is None, (t, r)


# ================================================================================================ item 3: engine
def test_item3_flags_engine_and_hub(mhub):
    from simws.radiation import channeling as ch
    assert ch.OWN_FIT_FLAG == OWN_FIT and ch.SIMS_FLAG == SIMS
    for t, r, E, d in ((4.0, 22.0, 50.0, 1e14), (10.0, 0.0, 50.0, 2e15)):
        p = p_problem(t, r, E, d)
        res = engine().run(p)
        assert OWN_FIT in res.flags and SIMS in res.flags, res.flags
        h = mhub.submit(ADAPTER, params={"problem": p.model_dump(mode="json")})
        assert h.status == "ok", h.message
        fl = all_flags(h)
        print(f"item 3 flags ({t}, {r}, {E} keV, {d:.0e}): {fl}")
        assert OWN_FIT in fl and SIMS in fl


CASES3 = [(10.0, 0.0, 50.0, 5e14, 1.0), (4.0, 22.0, 50.0, 1e14, 1.0), (4.0, 22.0, 120.0, 1e14, 8.0),
          (10.0, 0.0, 50.0, 1e13, 3.0)]


@pytest.mark.parametrize("t,r,E,d,T", CASES3)
def test_item3_sigma_is_half_spread_of_lengths_10pct(t, r, E, d, T):
    """A21.3: every length (Rpm, dRp1, dRp2, L) moved by -/+ 10 %, the reported sigma half the spread; the profile
    is the bin average (A15.5), here exact from the checker's CDF. Tolerances are numerical (the checker's): 1e-3
    of the largest sigma per bin; 1e-4 relative for the sigma of Rp and dRp."""
    p = p_problem(t, r, E, d, thickness_um=T)
    res = engine().run(p)
    P = cond("A" if t == 4.0 else "B").parameters(E, d)
    b = p.depth_bin_um
    n = len(res.profile_per_um)
    edges = np.arange(n + 1) * b
    xc = 0.5 * (edges[1:] + edges[:-1])
    prof = {s: np.diff(own_nc(P, 1.0 + 0.1 * s)[1](edges)) / b for s in (-1, 1)}
    sig_own = 0.5 * np.abs(prof[1] - prof[-1])
    sig = np.asarray(res.profile_sigma_per_um, float)
    e = float(np.abs(sig - sig_own).max() / sig_own.max())

    def mom(y):
        w = y / y.sum()
        m = float((w * xc).sum())
        return m, math.sqrt(float((w * (xc - m) ** 2).sum()))
    (m_lo, s_lo), (m_hi, s_hi) = mom(prof[-1]), mom(prof[1])
    srp, sdrp = res.diagnostics["Rp_sigma_um"], res.diagnostics["dRp_sigma_um"]
    print(f"item 3 ({t}, {r}, {E} keV, {d:.0e}): profile sigma differs by {e:.2e} of its largest value; sigma(Rp) "
          f"{srp * 1e3:.4f} nm (own {0.5 * abs(m_hi - m_lo) * 1e3:.4f}); sigma(dRp) {sdrp * 1e3:.4f} nm "
          f"(own {0.5 * abs(s_hi - s_lo) * 1e3:.4f}); channeled_fraction sigma {res.channeled_fraction_sigma:.2e}")
    assert e <= 1e-3
    assert srp == pytest.approx(0.5 * abs(m_hi - m_lo), rel=1e-4)
    assert sdrp == pytest.approx(0.5 * abs(s_hi - s_lo), rel=1e-4)


@pytest.mark.parametrize("c", FIT_CURVES, ids=cid)
def test_item8_channeled_fraction_is_share_beyond_Rpm_2dRp2(c, mhub):
    """A21.8: implant.channeled_fraction for phosphorus is the share of the profile deeper than Rpm + 2 dRp2
    (information only; the checker's numerical tolerance 1e-3 absolute)."""
    P = c["fit"]
    t, r = COND[cond_of(c)]
    want = 1.0 - float(own_nc(P)[1](P["Rpm_um"] + 2 * P["dRp2_um"]))
    got = engine().run(p_problem(t, r, c["energy_keV"], c["dose_cm2"], thickness_um=10.0,
                                 bin_um=0.002)).channeled_fraction
    print(f"item 8 {cid(c)}: channeled_fraction {got:.6f}, own share beyond Rpm + 2 dRp2 {want:.6f} "
          f"({got - want:+.2e})")
    assert abs(got - want) <= 1e-3


# ================================================================================================ R140: the fit
@pytest.mark.parametrize("c", FIT_CURVES, ids=cid)
def test_R140_fit(c):
    cn = cond(cond_of(c))
    E, d = c["energy_keV"], c["dose_cm2"]
    x, y = trace(c)
    m = fit_mask(c)
    r = model_conc(cn, E, d, x[m]) / y[m]
    within = float(np.mean((r <= 1.5) & (r >= 1 / 1.5)))
    dt = depth_at(x, y, 1e18)
    dm = depth_at(XF, model_conc(cn, E, d, XF), 1e18)
    rel = None if dt is None else dm / dt - 1
    print(f"R140 {cid(c)}: {100 * within:.2f} % of {int(m.sum())} points within a factor 1.5 (limit 95 %); depth at "
          f"1e18: model {dm if dm is None else round(dm * 1e3, 2)} nm, trace "
          f"{dt if dt is None else round(dt * 1e3, 2)} nm" + ("" if rel is None else f" ({100 * rel:+.2f} %, limit 10 %)"))
    assert within >= 0.95
    if dt is None:
        print(f"R140 {cid(c)}: the trace never reaches 1e18 (peak {y.max():.3g}); depth part not applicable")
    else:
        assert dm is not None and abs(rel) <= 0.10


# ================================================================================================ R141 to R143
def _loo_check(P, dose, c, label, limit, d3e17=False):
    dens, _ = own_nc(P)
    cm = dens(XF) * 1e4 * dose
    x, y = trace(c)
    pk = float(cm.max()) / trace_peak(c)
    dm, dt = depth_at(XF, cm, 1e18), depth_at(x, y, 1e18)
    rel = dm / dt - 1
    line = (f"{label}: peak model / trace {pk:.4f} (limit factor 1.5); depth at 1e18 model {dm * 1e3:.2f} nm, trace "
            f"{dt * 1e3:.2f} nm ({100 * rel:+.2f} %, limit {100 * limit:.0f} %)")
    ok = (1 / 1.5 <= pk <= 1.5) and abs(rel) <= limit
    if d3e17:
        dm3, dt3 = depth_at(XF, cm, 3e17), depth_at(x, y, 3e17)
        rel3 = dm3 / dt3 - 1
        line += f"; depth at 3e17 model {dm3 * 1e3:.2f} nm, trace {dt3 * 1e3:.2f} nm ({100 * rel3:+.2f} %)"
        ok = ok and abs(rel3) <= limit
    print(line)
    return ok


@pytest.mark.parametrize("E,lo,hi", [(50.0, 30.0, 80.0), (80.0, 50.0, 120.0), (120.0, 80.0, 180.0)])
def test_R141_energy_leave_one_out(E, lo, hi):
    A = {c["energy_keV"]: c["fit"] for c in _stored("fit_A")}
    P = lin(lo, A[lo], hi, A[hi], E)
    assert _loo_check(P, 1e14, curve(1, E, 1e14), f"R141 A {E:g} keV from {lo:g} / {hi:g}", 0.15)


@pytest.mark.parametrize("d,lo,hi", [(5e14, 1e14, 2e15), (2e15, 5e14, 8e15)])
def test_R142_dose_leave_one_out(d, lo, hi):
    D = {c["dose_cm2"]: c["fit"] for c in _stored("fit_B", figure=2)}
    P = lin(math.log10(lo), D[lo], math.log10(hi), D[hi], math.log10(d))
    assert _loo_check(P, d, curve(2, 50.0, d), f"R142 B 50 keV {d:.0e} from {lo:.0e} / {hi:.0e}", 0.15)


def test_R143_replicate_figure5():
    """The model at (B) 50 keV 5e14 (through the public condition object) against Figure 5's 0 deg curve."""
    from simws.radiation import channeling as ch
    f5 = ch.ghante_curves("check")
    assert len(f5) == 1 and f5[0]["figure"] == 5
    P = cond("B").parameters(50.0, 5e14)
    cm = model_conc(cond("B"), 50.0, 5e14, XF)
    own = own_nc(P)[0](XF) * 1e4 * 5e14
    assert np.allclose(cm, own, rtol=1e-6, atol=1e-6 * own.max())
    assert _loo_check(P, 5e14, f5[0], "R143 B 50 keV 5e14 against Figure 5", 0.10, d3e17=True)


# ================================================================================================ R144: engine
NORM_CASES = [(4.0, 22.0, 15.0, 1e14), (4.0, 22.0, 30.0, 1e14), (4.0, 22.0, 50.0, 1e14), (4.0, 22.0, 80.0, 1e14),
              (4.0, 22.0, 120.0, 1e14), (4.0, 22.0, 180.0, 1e14), (4.0, 22.0, 65.0, 1e14),
              (10.0, 0.0, 30.0, 1e14), (10.0, 0.0, 75.0, 1e14), (10.0, 0.0, 120.0, 1e14),
              (10.0, 0.0, 50.0, 1e13), (10.0, 0.0, 50.0, 3e14), (10.0, 0.0, 50.0, 8e15)]


@pytest.mark.parametrize("t,r,E,d", NORM_CASES)
def test_R144_normalisation(t, r, E, d):
    """A15 R81 for phosphorus: the integral of N over the depth equals the dose within 1e-4 relative, with a target
    deep enough to hold the profile. The engine reports the target's bins; what lies beyond is the checker's exact
    share (below 2e-5 for the target chosen)."""
    P = cond("A" if t == 4.0 else "B").parameters(E, d)
    _, cdf = own_nc(P)
    T = next(T for T in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0) if 1 - float(cdf(T)) < 2e-5)
    p = p_problem(t, r, E, d, thickness_um=T, bin_um=max(0.002, T / 5000))
    assert reasons(p) == [], reasons(p)
    res = engine().run(p)
    inside = float(np.sum(res.profile_per_um) * p.depth_bin_um)
    beyond = 1 - float(cdf(T))
    err = inside + beyond - 1
    print(f"R144 normalisation ({t}, {r}, {E} keV, {d:.0e}), target {T} um: inside {inside:.7f} + beyond "
          f"{beyond:.2e} - 1 = {err:+.2e} (limit 1e-4); engine's share_beyond_target "
          f"{res.diagnostics.get('share_beyond_target'):.2e}")
    assert abs(err) <= 1e-4


@pytest.mark.parametrize("t,r,E,d,T", [(10.0, 0.0, 50.0, 1e13, 0.3), (4.0, 22.0, 180.0, 1e14, 1.0),
                                       (4.0, 22.0, 15.0, 1e14, 0.08)])
def test_R144_thin_target_refused(t, r, E, d, T):
    from simws.engines.base import Unsupported
    P = cond("A" if t == 4.0 else "B").parameters(E, d)
    lost = 1 - float(own_nc(P)[1](T))
    p = p_problem(t, r, E, d, thickness_um=T)
    rs = reasons(p)
    print(f"R144 thin target ({E} keV, {d:.0e}, {T} um; own share beyond {lost:.2e}): {rs}")
    assert lost > 1e-4
    assert any(s.startswith("structures:") for s in rs), rs
    with pytest.raises(Unsupported):
        engine().run(p)


def _refusals():
    A = lambda **kw: dict(t=4.0, r=22.0, E=50.0, d=1e14, **kw)          # noqa: E731
    B = lambda **kw: dict(t=10.0, r=0.0, E=50.0, d=1e14, **kw)          # noqa: E731
    out = [
        ("A 14.9 keV", {**A(), "E": 14.9}, "energy:"), ("A 180.5 keV", {**A(), "E": 180.5}, "energy:"),
        ("A 2e14", {**A(), "d": 2e14}, "dose:"), ("A 5e13", {**A(), "d": 5e13}, "dose:"),
        ("A no dose", {**A(), "d": None}, "dose:"),
        ("B 29 keV", {**B(), "E": 29.0}, "energy:"), ("B 121 keV", {**B(), "E": 121.0}, "energy:"),
        ("B 50 keV 9e12", {**B(), "d": 9e12}, "dose:"), ("B 50 keV 9e15", {**B(), "d": 9e15}, "dose:"),
        ("B 30 keV 5e14", {**B(), "E": 30.0, "d": 5e14}, "dose:"),
        ("B 120 keV 1e13", {**B(), "E": 120.0, "d": 1e13}, "dose:"),
        ("B 75 keV 2e15", {**B(), "E": 75.0, "d": 2e15}, "dose:"),
        ("B no dose", {**B(), "d": None}, "dose:"),
    ]
    for t, r in ((7.0, 22.0), (7.0, 30.0), (0.0, 0.0), (4.0, 0.0), (10.0, 22.0), (4.6, 22.0), (4.0, 22.6),
                 (10.6, 0.0), (10.0, 0.6), (4.0, None), (10.0, None), (10.0, 45.0)):
        out.append((f"angles {t}/{r}", {**B(), "t": t, "r": r}, "angles:"))
    return out


REFUSALS = _refusals()


@pytest.mark.parametrize("name,kw,prefix", REFUSALS, ids=[x[0] for x in REFUSALS])
def test_R144_refusals(name, kw, prefix):
    p = p_problem(kw["t"], kw["r"], kw["E"], kw["d"], thickness_um=2.0)
    rs = reasons(p)
    print(f"R144 {name}: {rs}")
    assert any(s.startswith(prefix) for s in rs), rs


def test_R144_structures_refused():
    two = p_problem(10.0, 0.0, 50.0, 1e14, thickness_um=2.0)
    si = two.model_dump(mode="json")["target"]["layers"][0]
    two = two.with_changes({"target.layers": [{**si, "name": "a", "thickness_um": 1.0},
                                              {**si, "name": "b", "thickness_um": 1.0}]})
    amorph = p_problem(10.0, 0.0, 50.0, 1e14).with_changes({"target.structure": "amorphous"})
    for p in (two, amorph):
        rs = reasons(p)
        assert any(s.startswith("structures:") for s in rs), rs


@pytest.mark.parametrize("t,r,E,d,T", [(4.5, 21.5, 15.0, 1e14, 1.0), (3.5, 22.5, 180.0, 1e14, 10.0),
                                       (9.5, 0.5, 50.0, 8e15, 1.0), (10.5, 0.0, 30.0, 1e14, 1.0),
                                       (10.0, 0.0, 120.0, 1e14, 2.0), (10.0, 0.0, 50.0, 1e13, 3.0)])
def test_R144_accepts_the_two_conditions(t, r, E, d, T):
    p = p_problem(t, r, E, d, thickness_um=T, bin_um=max(0.002, T / 5000))
    assert reasons(p) == [], reasons(p)


def test_R144_refusal_through_hub(mhub):
    p = p_problem(7.0, 22.0, 50.0, 1e14)
    h = mhub.submit(ADAPTER, params={"problem": p.model_dump(mode="json")})
    print("R144 hub:", h.status, h.message)
    assert h.status == "invalid_input" and "angles:" in h.message


def test_R144_share_beyond_is_public_and_exact():
    """share_beyond(model, problem): the share of the [0, inf)-normalised profile deeper than the target; against
    the checker's exact CDF (numerical tolerance 1e-6 absolute, the checker's)."""
    from simws.radiation import channeling as ch
    for t, r, E, d, T in ((10.0, 0.0, 50.0, 1e13, 0.3), (4.0, 22.0, 120.0, 1e14, 1.0)):
        p = p_problem(t, r, E, d, thickness_um=T)
        P = cond("A" if t == 4.0 else "B").parameters(E, d)
        got = engine().share_beyond(ch.model("P").select(t, r), p)
        want = 1 - float(own_nc(P)[1](T))
        print(f"R144 share_beyond ({E} keV, {d:.0e}, {T} um): {got:.6e}, own {want:.6e}")
        assert abs(got - want) <= 1e-6


# ================================================================================================ item 5: trust
def test_item5_trust_unverified(mhub):
    from simws.engines.planner import plan
    from simws.hub.trust import assess_stored
    p = p_problem(10.0, 0.0, 50.0, 1e14)
    h = mhub.submit(ADAPTER, params={"problem": p.model_dump(mode="json")})
    assert h.status == "ok", h.message
    assert assess_stored(mhub, h.id).level == "unverified"
    pl = plan(p)
    print("item 5 plan:", pl.primary, pl.checkers)
    assert pl.primary == "channeling" and pl.checkers == []


# ================================================================================================ R145 (reported)
@pytest.mark.slow
def test_R145_chain_reported(mhub):
    import simws.engines as E
    from simws.chain import process as P
    from simws.spec.library import implant_si
    for n in ("devsim_anneal", "fipy", "rustbca"):
        if not E.module(n).available():
            pytest.skip(f"{n} not available")

    def run(adapter, prob):
        h = mhub.submit(adapter, params={"problem": prob.model_dump(mode="json") if hasattr(prob, "model_dump")
                                         else prob})
        assert h.status == "ok", f"{adapter}: {h.message[-400:]}"
        return mhub.result(h.id)

    def junctions(ini):
        ap = P.anneal_problem("P", ini, background_cm3=1e16, background_type="acceptor", steps=[(1000.0, 600.0)],
                              model="fermi", thickness_um=2.0)
        return [float(run(a, ap).q("anneal.junction_depth").value) * 1e9 for a in
                ("anneal.devsim_anneal", "anneal.fipy")]
    cr = run(ADAPTER, p_problem(10.0, 0.0, 50.0, 1e14, thickness_um=1.0))
    am = run("implant.rustbca", implant_si("P", 50.0, tilt_deg=10.0, dose_cm2=1e14, thickness_um=1.0)
             .with_changes({"depth_bin_um": 0.002}))
    jc = junctions(P.initial_from_implant(cr))
    ja = junctions(P.initial_from_implant(am))
    dc, da = abs(jc[0] / jc[1] - 1), abs(ja[0] / ja[1] - 1)
    print(f"R145: P 50 keV 1e14 (B) -> 1000 C 600 s fermi on p 1e16: crystal junction devsim_anneal {jc[0]:.2f} nm, "
          f"fipy {jc[1]:.2f} nm (differ {100 * dc:.4f} %, limit 1 %); amorphous RustBCA {ja[0]:.2f} / {ja[1]:.2f} nm "
          f"(differ {100 * da:.4f} %); crystal / amorphous {jc[0] / ja[0]:.4f}")
    assert dc <= 0.01


# ================================================================================================ the PDF (skips without)
def _pdf_pages():
    if not PDF.is_file():
        pytest.skip(f"{PDF} not present")
    if hashlib.sha256(PDF.read_bytes()).hexdigest() != GH["source"]["pdf_sha256"]:
        pytest.fail("the PDF's SHA-256 differs from the data file's")
    if shutil.which("pdfimages") is None:
        pytest.skip("pdfimages not available")
    if not _PAGES:
        import tempfile

        from PIL import Image
        with tempfile.TemporaryDirectory() as out:
            subprocess.run(["pdfimages", "-png", str(PDF), str(Path(out) / "pg")], check=True)
            _PAGES.update({i: np.asarray(Image.open(Path(out) / f"pg-{i:03d}.png").convert("L")) < 128
                           for i in range(6)})
    return _PAGES


_PAGES = {}


PAGE = {1: 2, 2: 2, 3: 3, 5: 4, 6: 4}                    # figure -> page index (0-based) of the 400 dpi scan


def test_pdf_sha256():
    if not PDF.is_file():
        pytest.skip(f"{PDF} not present")
    assert hashlib.sha256(PDF.read_bytes()).hexdigest() == GH["source"]["pdf_sha256"]


def test_pdf_calibration_on_tick_marks():
    """Every calibration value sits on a tick mark: a dark run of 11 px inward from the left frame at each y tick,
    and upward from the bottom frame at each x tick (within +-2 px)."""
    pages = _pdf_pages()
    for fig, pg in PAGE.items():
        a = pages[pg]
        c = GH["source"]["calibration"][f"fig{fig}"]
        x0, x1 = (int(round(v)) for v in c["x_px"])
        for y in c["y_px"]:
            y = int(round(y))
            assert a[y - 2:y + 3, x0 + 4:x0 + 15].sum(axis=1).max() >= 10, (fig, "y", y)
        ylast = int(round(max(c["y_px"])))
        rows = [r for r in range(ylast - 5, ylast + 500) if a[r, x0:x1].mean() > 0.9]
        assert rows, (fig, "no bottom frame")
        yb = rows[0]
        for x in c["x_px"]:
            x = int(round(x))
            assert a[yb - 15:yb - 4, x - 2:x + 3].sum(axis=0).max() >= 10, (fig, "x", x)


@pytest.mark.parametrize("c", GH["curves"], ids=cid)
def test_pdf_traced_points_on_the_ink(c):
    """Every traced point at or above the floor, mapped through the file's calibration (linear in depth, piecewise
    linear in log10 C between the ticks), lies within 3 px (0.19 mm at 400 dpi) of ink."""
    from scipy.ndimage import binary_dilation
    pages = _pdf_pages()
    a = binary_dilation(pages[PAGE[c["figure"]]], iterations=3)
    cal = GH["source"]["calibration"][f"fig{c['figure']}"]
    x, y = trace(c)
    m = y >= c["floor_cm3"]
    px = cal["x_px"][0] + (x[m] - cal["um"][0]) * (cal["x_px"][1] - cal["x_px"][0]) / (cal["um"][1] - cal["um"][0])
    o = np.argsort(cal["log10_cm3"])
    py = np.interp(np.log10(y[m]), np.asarray(cal["log10_cm3"])[o], np.asarray(cal["y_px"])[o])
    on = a[np.round(py).astype(int), np.round(px).astype(int)]
    print(f"PDF {cid(c)}: {int(on.sum())} of {on.size} points within 3 px of ink")
    assert on.all()
