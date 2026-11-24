"""Contract amendment A22 (24 Nov 2026): transient enhanced diffusion (TED) from implant damage, the effective "+1"
model. Acceptance checks R150 to R153 of A22.4, and items 1, 2, 3, 5 and 8 (with the clarifications 7 and 8).

Written by the checker from docs/ARCHITECTURE_V1.md (A22 items 0 to 8; background A16 items 1-4, 8 and 10, A20,
A13.7 / A1.10 and section 6.1) and the three sources the contract names. Only public names are called:
simws.process.ted (tau_s, supersaturation, enhancement, damage, schedule, stolk_fig1, depth_at, the constants),
simws.spec.anneal.AnnealProblem, simws.chain.process (anneal_problem, gaussian_profile, initial_from_implant),
simws.spec.library.implant_si, the engine registry, the hub (adapters anneal.devsim_anneal, anneal.fipy,
implant.channeling, implant.rustbca), simws.measured.ingest, simws.compare (compare_results, record) and
simws.hub.trust.assess_stored. Every reference number below is computed here from the contract's formulas.

Readings of the contract made here (stated so the lead can object):
- R150 is gated on both engines (the item names no engine). The depth "where the result falls to" a level is read
  with item 8's rule (the deepest depth at or above the level, linear between points), for the model and for the
  traced SIMS alike. The shift ratio is (model depth - as-implanted depth) / (SIMS depth - as-implanted depth),
  the as-implanted depth taken from the traced as-implanted profile by the same rule (reported).
- R151's crystal case and R152 / R153 use A20's anneal setting for the crystal B profile: "fermi", n-type 1e16,
  thickness 2 um (A20 R131); R151's crystal case and R152 give no background, so 1e16 n-type (R153's) is used.
  R151 "junction or 1e17 depths within 1 %": both are gated, each within 1 %. Doses: A16.10's R94 rule, the
  relative difference of the final doses (anneal.dose) at most the relative difference of the engines' starting
  doses plus 1e-6; the starting doses computed here from A16.3 (DEVSIM: trapezoid on the nodes k dx; FiPy: the
  cell sum at (k + 1/2) dx).
- R152's "tau -> 0" case: the crystal B profile above at 1100 C for 1 h, both engines.
- Item 2, "damage = the dose and mean depth of the initial profile": both by A16.1's rule for the profile (the
  first value held from 0 to the first point, linear between points, zero beyond the last).
- Item 2's model as the engines solve it is also checked end to end against an exact solution: for "intrinsic"
  diffusion the TED factor multiplies D, so a Gaussian (with its image) stays a Gaussian with
  sigma^2 = sigma0^2 + 2 sum(D E t_TED + D t_rest). Gate: A16's R90 tolerance (1 % of the exact peak).
- Item 5's failing-measurement check: a second hub, the R150 TED result of devsim_anneal compared with the traced
  SIMS depths (should pass) and with a deliberately wrong "measurement" (the depths x 1.5, should fail).
"""
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

K_B = 8.617333262e-5                         # eV/K (A16.2: 8.617333e-5)
ANNEALS = ("anneal.devsim_anneal", "anneal.fipy")
ENGINE_OF = {"anneal.devsim_anneal": "devsim_anneal", "anneal.fipy": "fipy"}

# ------------------------------------------------------------------------------------- A22.2 / A22.7 constants
TAU_REF = 40.0
T_REF_C = 815.0
Q_REF = 5e13
RP_REF_NM = 59.02                            # A22.7(b): 59.02 +- 0.18 nm
RP_REF_SIGMA_NM = 0.18
E_TAU = 3.8
T_S_C = 790.0
S_PLOT = ((600.0, 1.3e5), (1100.0, 3.5e2))   # A22.0 (ii) / A22.2: the plot's two end points
MARKERS = ((200.0, 600.0), (60.0, 1800.0))   # Stolk Sec. IV: (enhancement, anneal time s) at 790 C
F_I = {"B": 1.0, "P": 1.0, "As": 0.4, "Sb": 0.02}
TED_FLAG = ("anneal:ted_effective: the +1 model with a uniform supersaturation; tau and S from Stolk et al. 1997 and "
            "the EE212 handout (simws's calibration); no clustering (boron above about 1e18 cm^-3 is immobile in "
            "reality), no depth dependence")

# R150 (A22.4)
R150_BG, R150_THICK, R150_STEP = 1e15, 1.5, (800.0, 2100.0)
R150_LEVELS = (3e17, 1e17)
R150_TOL = 0.15
# R151
R151_TOL = 0.01
# R152
R152_TOL = 0.02
# crystal profile (A20)
CRYSTAL_BG, CRYSTAL_THICK = 1e16, 2.0

SP = Path("/tmp/claude-0/-home-claude/d0abb2e9-411f-5090-9cb2-dd3ba3452a20/scratchpad")
STOLK_PDF = SP / "papers/ted/stolk1997.pdf"
H14_PDF = SP / "src/dunham8.pdf"
H12_PDF = SP / "src/dunham7.pdf"


def kelvin(T_C):
    return T_C + 273.15


# ------------------------------------------------------------------------------------- independent formulas
def my_tau(T_K, Q, rp_um):
    return TAU_REF * (Q / Q_REF) * (rp_um / (RP_REF_NM * 1e-3)) * math.exp(E_TAU / K_B * (1 / T_K - 1 / kelvin(T_REF_C)))


def my_E_S():
    (T1, S1), (T2, S2) = S_PLOT
    return math.log(S1 / S2) * K_B / (1 / kelvin(T1) - 1 / kelvin(T2))


def my_S_values():
    tau = my_tau(kelvin(T_S_C), Q_REF, RP_REF_NM * 1e-3)
    return [(E * t - t + tau) / tau for E, t in MARKERS], tau


def my_S_ref():
    vals, _ = my_S_values()
    return float(np.mean(vals))


def my_S(T_K):
    return my_S_ref() * math.exp(my_E_S() / K_B * (1 / T_K - 1 / kelvin(T_S_C)))


def my_enh(dopant, T_K):
    S = my_S(T_K)
    return F_I[dopant] * S + (1 - F_I[dopant]) / S


def a16_dense(profile, n_extra=0):
    """The profile by A16.1's rule as (x, c) points including the hold from 0 to the first point."""
    x = np.asarray(profile.depth_um, float)
    c = np.asarray(profile.concentration_cm3, float)
    if x[0] > 0:
        x = np.concatenate(([0.0], x))
        c = np.concatenate(([c[0]], c))
    return x, c


def my_dose_cm2(profile):
    x, c = a16_dense(profile)
    return float(np.sum(0.5 * (c[1:] + c[:-1]) * np.diff(x)) * 1e-4)


def my_mean_depth_um(profile):
    """Exact mean depth of the piecewise-linear profile of A16.1 (each segment integrated exactly)."""
    x, c = a16_dense(profile)
    x0, x1, c0, c1 = x[:-1], x[1:], c[:-1], c[1:]
    h = x1 - x0
    m0 = 0.5 * (c0 + c1) * h
    m1 = h * (c0 * (2 * x0 + x1) + c1 * (x0 + 2 * x1)) / 6.0       # int of x * linear c over [x0, x1]
    return float(np.sum(m1) / np.sum(m0))


def my_depth_at(x, c, level):
    """Item 8: the deepest depth at or above the level, linear between points; None when never at or above."""
    x = np.asarray(x, float)
    c = np.asarray(c, float)
    idx = [i for i in range(len(c)) if c[i] >= level]
    if not idx:
        return None
    i = idx[-1]
    if i == len(c) - 1:
        return float(x[i])
    return float(x[i] + (x[i + 1] - x[i]) * (c[i] - level) / (c[i] - c[i + 1]))


def junction(x, c, Nb):
    """A16.4: the first depth where C falls to the background, linear between points."""
    if c[0] <= Nb:
        return None
    below = np.nonzero(np.asarray(c) <= Nb)[0]
    if below.size == 0:
        return None
    i = below[0]
    return x[i - 1] + (c[i - 1] - Nb) * (x[i] - x[i - 1]) / (c[i - 1] - c[i])


# ------------------------------------------------------------------------------------- hub helpers
def need(*engines):
    import simws.engines as E
    for n in engines:
        if not E.module(n).available():
            pytest.skip(f"engine {n} not available")


@pytest.fixture(scope="module")
def mhub(tmp_path_factory):
    from simws.hub.jobs import Hub
    return Hub(tmp_path_factory.mktemp("ted_checker_store"), mode="inprocess")


_RUNS = {}


def submit(hub, adapter, problem):
    key = (id(hub), adapter, problem.problem_id())
    if key not in _RUNS:
        r = hub.submit(adapter, params={"problem": problem.model_dump(mode="json")})
        assert r.status == "ok", f"{adapter}: {r.status}: {r.message[-500:]}"
        _RUNS[key] = (r.id, hub.result(r.id))
    return _RUNS[key]


def profile_of(res):
    q = res.q("anneal.concentration")
    return np.asarray(q.axes[0].values, float) * 1e6, np.asarray(q.array(), float) * 1e-6


def scalar(res, name):
    q = next((q for q in res.quantities if q.name == name), None)
    if q is None or q.value is None:
        return None
    return float(q.value)


def stolk():
    from simws.process import ted
    return ted.stolk_fig1()


def r150_problem(mode):
    from simws.chain import process as P
    ini, _, _ = stolk()
    return P.anneal_problem("B", ini, background_cm3=R150_BG, background_type="donor", steps=[R150_STEP],
                            model="fermi", thickness_um=R150_THICK).with_changes({"models.ted": mode})


_CRYSTAL = {}


def crystal_initial(hub):
    from simws.chain import process as P
    from simws.spec.library import implant_si
    need("channeling")
    if id(hub) not in _CRYSTAL:
        p = implant_si("B", 35.0, tilt_deg=7.0, rotation_deg=30.0, dose_cm2=1e14, crystal=True,
                       thickness_um=1.5).with_changes({"depth_bin_um": 0.002})
        _, res = submit(hub, "implant.channeling", p)
        _CRYSTAL[id(hub)] = P.initial_from_implant(res)
    return _CRYSTAL[id(hub)]


def crystal_problem(hub, T_C, t_s, mode):
    from simws.chain import process as P
    return P.anneal_problem("B", crystal_initial(hub), background_cm3=CRYSTAL_BG, background_type="donor",
                            steps=[(T_C, t_s)], model="fermi",
                            thickness_um=CRYSTAL_THICK).with_changes({"models.ted": mode})


def starting_doses(problem):
    """A16.3: each engine's own interpolated starting profile, cm^-2."""
    dx = problem.numerics.dx_um
    M = problem.n_intervals
    xn = np.arange(M + 1) * dx
    cn = problem.initial.at(xn)
    dev = float(np.sum(0.5 * (cn[1:] + cn[:-1])) * dx * 1e-4)
    xc = (np.arange(M) + 0.5) * dx
    fip = float(np.sum(problem.initial.at(xc)) * dx * 1e-4)
    return {"anneal.devsim_anneal": dev, "anneal.fipy": fip}


# ===================================================================================== sources (skip when absent)
def _pdftotext(pdf, first=None, last=None, layout=False):
    if not pdf.exists() or shutil.which("pdftotext") is None:
        pytest.skip(f"{pdf} or pdftotext not available")
    cmd = ["pdftotext"] + (["-layout"] if layout else [])
    if first:
        cmd += ["-f", str(first), "-l", str(last)]
    out = subprocess.run(cmd + [str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    return out


def test_source_stolk_constants():
    """Stolk et al. 1997 Sec. III / IV: 40 s at 815 C, 4 h at 670 C, 3.8 +- 0.2 eV, 200 (10 min) and 60 (30 min) at
    790 C; Fig. 1 B 30 keV 1.5e14, 800 C 35 min. (pdftotext renders '~' as ';' and '+-' as '6'.)"""
    t = " ".join(_pdftotext(STOLK_PDF).split())
    assert re.search(r"decay times ranging from .?40 s at 815 °C to .?4 h at 670 °C", t)
    assert re.search(r"activation energy of 3\.8.0\.2 eV for the dissolution", t)
    m = re.search(r"drops from about (\d+)3@I \*# to about (\d+)3@I \*# upon increasing the annealing time from "
                  r"(\d+) to (\d+) min", t)
    assert m and m.groups() == ("200", "60", "10", "30"), m and m.groups()
    assert "for 790 °C, 30 min" in t and "annealing at 790 °C for 10 min" in t
    assert re.search(r"5310\s?13/cm2, 40 keV Si", t)            # the marker implant: 5e13, 40 keV Si
    assert re.search(r"1\.5310\s?14/cm2, 30 keV 10B\! before and after transient enhanced diffusion at 800 °C for 35 "
                     r"min", t)


def test_source_handout14_supersaturation_plot(tmp_path):
    """Handout #14 page 23: the supersaturation curve's end points read from the rendered page (400 dpi)."""
    if not H14_PDF.exists() or shutil.which("pdftoppm") is None:
        pytest.skip("handout #14 or pdftoppm not available")
    PIL = pytest.importorskip("PIL.Image")
    subprocess.run(["pdftoppm", "-f", "23", "-l", "23", "-r", "400", "-png", str(H14_PDF), str(tmp_path / "p")],
                   check=True)
    png = next(tmp_path.glob("p*.png"))
    im = np.asarray(PIL.open(png).convert("RGB")).astype(int)
    top = im[: int(im.shape[0] * 0.42)]                       # the upper plot only
    red = (top[..., 0] > 180) & (top[..., 1] < 80) & (top[..., 2] < 80)
    blue = (top[..., 2] > 150) & (top[..., 0] < 80) & (top[..., 1] < 80)
    # frame: the left axis is the blue column with the most pixels; x ticks are the short columns between
    colsum = blue.sum(0)
    left = int(np.argmax(colsum[: top.shape[1] // 2]))
    right = top.shape[1] // 2 + int(np.argmax(colsum[top.shape[1] // 2:]))
    # x calibration: major ticks (tall blue columns between the frame edges), labels 600..1100 C
    tick_cols = [x for x in range(left + 30, right - 30) if colsum[x] > 100]
    groups = np.split(np.array(tick_cols), np.nonzero(np.diff(tick_cols) > 3)[0] + 1)
    xt = [float(g.mean()) for g in groups]
    assert len(xt) == 6, xt                                    # 600, 700, ..., 1100 C
    x_of_T = np.polyfit([600, 700, 800, 900, 1000, 1100], xt, 1)
    # y calibration: decade ticks on the left axis (rows with a long blue run right of the axis)
    seg = blue[:, left + 5: left + 55].sum(1)
    rows = [y for y in range(len(seg)) if seg[y] >= 25]
    rg = np.split(np.array(rows), np.nonzero(np.diff(rows) > 3)[0] + 1)
    yt = [float(g.mean()) for g in rg]
    assert len(yt) == 5, yt                                     # 1e6 (top) .. 1e2 (bottom)
    lg_of_y = np.polyfit(yt, [6, 5, 4, 3, 2], 1)
    ys, xs = np.nonzero(red)
    T = lambda px: (px - x_of_T[1]) / x_of_T[0]                 # noqa: E731
    xa, xb = xs.min(), xs.max()
    Sa = 10 ** np.polyval(lg_of_y, ys[xs <= xa + 2].mean())
    Sb = 10 ** np.polyval(lg_of_y, ys[xs >= xb - 2].mean())
    Ta, Tb = T(xa), T(xb)
    E_read = math.log(Sa / Sb) * K_B / (1 / kelvin(Ta) - 1 / kelvin(Tb))
    print(f"\n[source] handout #14 p.23 curve: start {Ta:.0f} C, S {Sa:.3g}; end {Tb:.0f} C, S {Sb:.3g}; "
          f"E_S from these readings {E_read:.4f} eV (contract 1.2227 eV from 1.3e5 / 3.5e2 at 600 / 1100 C)")
    assert abs(Ta - 600) < 10 and abs(Tb - 1100) < 10
    assert abs(Sa / 1.3e5 - 1) < 0.05 and abs(Sb / 3.5e2 - 1) < 0.05


def test_source_handout12_fI_table():
    t = _pdftotext(H12_PDF, 24, 24, layout=True)
    got = {}
    for name, sym in (("Silicon", "Si"), ("Boron", "B"), ("Phosphorus", "P"), ("Arsenic", "As"), ("Antimony", "Sb")):
        m = re.search(name + r"\s+([0-9.]+)\s+([0-9.]+)", t)
        assert m, name
        got[sym] = (float(m.group(1)), float(m.group(2)))
    for d, fi in F_I.items():
        assert got[d] == (fi, round(1 - fi, 2)), (d, got[d])


# ===================================================================================== item 1: the description
def _a16_problem():
    from simws.chain import process as P
    ini = P.gaussian_profile(0.1, 0.03, 1e14)
    return P.anneal_problem("B", ini, steps=[(1000.0, 1800.0)], thickness_um=1.0)


def test_item1_none_left_out_and_ids_unchanged():
    from simws.spec.anneal import AnnealProblem
    p = _a16_problem()
    d = p.model_dump(mode="json")
    assert "ted" not in d["models"]
    d_none = dict(d, models=dict(d["models"], ted="none"))
    p_none = AnnealProblem.model_validate(d_none)
    assert p_none.models.ted == "none"
    assert "ted" not in p_none.model_dump(mode="json")["models"]
    assert p_none.problem_id() == p.problem_id()
    p2 = AnnealProblem.model_validate(d)
    assert p2.models.ted == "none" and p2.problem_id() == p.problem_id()


def test_item1_plus_one_changes_id():
    p = _a16_problem()
    q = p.with_changes({"models.ted": "plus_one"})
    assert q.models.ted == "plus_one"
    assert q.model_dump(mode="json")["models"]["ted"] == "plus_one"
    assert q.problem_id() != p.problem_id()


def test_item1_invalid_ted_value():
    from simws.spec.anneal import AnnealProblem
    d = _a16_problem().model_dump(mode="json")
    d["models"]["ted"] = "plus_two"
    with pytest.raises(Exception):
        AnnealProblem.model_validate(d)


# ===================================================================================== item 2: constants and formulas
def test_item2_constants():
    from simws.process import ted
    assert ted.TAU_REF_S == TAU_REF
    assert ted.T_REF_K == pytest.approx(kelvin(T_REF_C), abs=1e-9)
    assert ted.Q_REF_CM2 == Q_REF
    assert abs(ted.RP_REF_UM * 1e3 - RP_REF_NM) < 5e-3
    assert ted.E_TAU_EV == E_TAU
    assert ted.T_S_K == pytest.approx(kelvin(T_S_C), abs=1e-9)
    assert dict(ted.F_I) == F_I


def test_item2_rp_ref_from_rustbca(mhub):
    """A22.2 / 7(b): Rp_ref = mean depth of 40 keV Si in amorphous Si, RustBCA, Kr-C, LS, 20 000 ions, seed 1."""
    need("rustbca")
    from simws.process import ted
    from simws.spec.library import implant_si
    p = implant_si("B", 40.0, thickness_um=1.0).with_changes(
        {"beam.ion.symbol": "Si", "beam.ion.Z": 14, "beam.ion.mass_amu": 27.9769265, "models.potential": "kr_c",
         "models.electronic_stopping": "lindhard_scharff"})
    assert p.statistics.n_ions == 20000 and p.statistics.seed == 1
    _, res = submit(mhub, "implant.rustbca", p)
    q = res.q("implant.Rp")
    assert q.unit == "m"
    rp_nm = float(q.value) * 1e9
    sig = float(q.uncertainty.sigma) * 1e9
    print(f"\n[item 2] RustBCA 40 keV 28Si Kr-C LS 20000 ions seed 1: Rp {rp_nm:.3f} +- {sig:.3f} nm; stored "
          f"{ted.RP_REF_UM * 1e3:.3f} nm (contract 59.02 +- 0.18)")
    assert abs(rp_nm - ted.RP_REF_UM * 1e3) <= RP_REF_SIGMA_NM


def test_item2_tau_formula():
    from simws.process import ted
    for T_C in (650.0, 700.0, 790.0, 800.0, 815.0, 900.0, 1000.0, 1100.0):
        for Q in (1e13, 5e13, 1.14e14, 1e15):
            for rp in (0.02, 0.059, 0.126):
                assert ted.tau_s(kelvin(T_C), Q, rp) == pytest.approx(my_tau(kelvin(T_C), Q, rp), rel=1e-12)


def test_item2_E_S_from_plot_end_points():
    from simws.process import ted
    E = my_E_S()
    assert round(E, 2) == 1.22 and round(E, 4) == 1.2227
    assert ted.E_S_EV == pytest.approx(E, rel=1e-12)


def test_item2_S_ref_from_markers():
    from simws.process import ted
    (s10, s30), tau790 = my_S_values()
    S_ref = my_S_ref()
    print(f"\n[item 2] tau(790 C, Stolk's 5e13 40 keV Si) {tau790:.2f} s; S from (200, 10 min) {s10:.1f}, "
          f"from (60, 30 min) {s30:.1f}; S_ref {S_ref:.1f} (contract 1152, 1025, 1088); implementation "
          f"{ted.S_REF:.3f}")
    assert tau790 < 600                               # the calibration formula needs t > tau
    assert round(s10) == 1152 and round(s30) == 1025 and round(S_ref) == 1088
    assert ted.S_REF == pytest.approx(S_ref, rel=1e-12)


def test_item2_S_formula():
    from simws.process import ted
    for T_C in (600.0, 700.0, 790.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0):
        assert ted.supersaturation(kelvin(T_C)) == pytest.approx(my_S(kelvin(T_C)), rel=1e-12)


def test_item2_enhancement_fI_table():
    from simws.process import ted
    for d in F_I:
        for T_C in (700.0, 800.0, 900.0, 1000.0, 1100.0):
            T = kelvin(T_C)
            assert ted.enhancement(d, T) == pytest.approx(my_enh(d, T), rel=1e-12)
            S = ted.supersaturation(T)
            assert ted.enhancement(d, T) == pytest.approx(F_I[d] * S + (1 - F_I[d]) / S, rel=1e-12)


def test_item2_damage_profile_from_zero():
    """A finely sampled profile starting at 0 (the A16 Gaussian, 1 nm points): Q and the mean depth."""
    from simws.process import ted
    p = _a16_problem()
    Q, rp = ted.damage(p)
    assert Q == pytest.approx(my_dose_cm2(p.initial), rel=1e-9)
    assert rp == pytest.approx(my_mean_depth_um(p.initial), rel=1e-6)
    assert Q == pytest.approx(1e14, rel=1e-3)


def test_item2_damage_mean_depth_of_the_linear_profile():
    """A16.1: linear between points; the mean depth of that profile (each segment integrated exactly), on a coarse
    profile starting at 0 (no hold involved)."""
    from simws.chain import process as P
    from simws.process import ted
    from simws.spec.anneal import DepthProfile
    ini = DepthProfile(depth_um=[0.0, 0.05, 0.1, 0.3], concentration_cm3=[1e18, 5e18, 1e18, 0.0])
    q = P.anneal_problem("B", ini, thickness_um=1.0)
    Q2, rp2 = ted.damage(q)
    print(f"\n[item 2] coarse linear profile from 0: damage Rp {rp2 * 1e3:.3f} nm, exact mean of the linear profile "
          f"{my_mean_depth_um(ini) * 1e3:.3f} nm")
    assert Q2 == pytest.approx(my_dose_cm2(ini), rel=1e-12)
    assert rp2 == pytest.approx(my_mean_depth_um(ini), rel=1e-9)


def test_item2_damage_profile_with_held_first_value():
    """A16.1: the first value holds from 0 to the first point; Q and Rp are both of that profile. A flat profile
    (1e18 from 0 to 0.1 um: hold to 0.05, then one flat segment) has the exact mean 0.05 um and dose 1e13."""
    from simws.chain import process as P
    from simws.process import ted
    from simws.spec.anneal import DepthProfile
    ini = DepthProfile(depth_um=[0.05, 0.1], concentration_cm3=[1e18, 1e18])
    q = P.anneal_problem("B", ini, thickness_um=1.0)
    Q, rp = ted.damage(q)
    print(f"\n[item 2] 1e18 held from 0 to 0.05 um, flat to 0.1 um: damage Q {Q:.6e} (A16.1 1e13), Rp "
          f"{rp * 1e3:.3f} nm (A16.1 50 nm)")
    assert my_dose_cm2(ini) == pytest.approx(1e13, rel=1e-12) and my_mean_depth_um(ini) == pytest.approx(0.05)
    assert Q == pytest.approx(1e13, rel=1e-12)
    assert rp == pytest.approx(0.05, rel=1e-9)


def test_item2_damage_stolk_fig1():
    """A22.7(a): the traced as-implanted dose is 1.14e14 and tau uses it."""
    from simws.process import ted
    ini, _, src = stolk()
    p = r150_problem("plus_one")
    Q, rp = ted.damage(p)
    x = np.asarray(ini.depth_um)
    print(f"\n[item 2] Stolk Fig. 1 initial: {len(x)} points to {x[-1]:.3f} um, Q {Q:.4e} cm^-2, mean depth "
          f"{rp * 1e3:.2f} nm (A16.1 rule {my_mean_depth_um(ini) * 1e3:.2f} nm); tau(800 C) "
          f"{ted.tau_s(kelvin(800.0), Q, rp):.1f} s, S(800 C) {ted.supersaturation(kelvin(800.0)):.1f}, "
          f"E_B {ted.enhancement('B', kelvin(800.0)):.1f}")
    assert f"{Q:.3g}" == "1.14e+14"
    assert Q == pytest.approx(my_dose_cm2(ini), rel=1e-12)
    # the mean depth is gated by the two tests above; its effect here is reported (tau is proportional to it)
    print(f"[item 2] Stolk Fig. 1: damage Rp / A16.1 mean depth - 1 = {rp / my_mean_depth_um(ini) - 1:+.2e} "
          f"(the profile starts at {x[0] * 1e3:.2f} nm with {ini.concentration_cm3[0]:.3g} cm^-3)")
    # the tail extension: to 0.7 um with the fitted slope; check the slope is the fit of the last part of the trace
    import json
    d = json.loads((Path(ted.__file__).resolve().parent.parent / "data/ted/stolk1997_fig1.json").read_text())
    xa = np.asarray(d["as_implanted"]["depth_um"], float)
    ca = np.asarray(d["as_implanted"]["conc_cm3"], float)
    sel = xa >= 0.33
    slope = np.polyfit(xa[sel], np.log10(ca[sel]), 1)[0]
    rec = src["tail_extension"]["as_implanted_log10_slope_per_um"]
    print(f"[item 2] tail fit 0.33 um to end ({sel.sum()} points): {slope:.3f} decades/um; recorded {rec:.4f}")
    assert round(rec, 2) == -3.14 and round(slope, 2) == round(rec, 2)
    assert abs(x[-1] - 0.7) < 0.003
    tail = x > xa.max() + 1e-9
    c = np.asarray(ini.concentration_cm3)
    lg = np.log10(c[tail])
    assert np.allclose(np.diff(lg) / np.diff(x[tail]), rec, rtol=1e-6)


def _sched_problem(steps, mode="plus_one", N=200):
    from simws.chain import process as P
    ini = P.gaussian_profile(0.1, 0.03, 1e14)
    return P.anneal_problem("B", ini, steps=steps, thickness_um=1.0, steps_per_anneal=N).with_changes(
        {"models.ted": mode})


def test_item2_schedule_single_step_extra_point():
    from simws.process import ted
    p = _sched_problem([(1000.0, 60.0)])
    Q, rp = my_dose_cm2(p.initial), my_mean_depth_um(p.initial)
    tau = my_tau(kelvin(1000.0), Q, rp)
    assert tau < 60.0
    (pts, t_end), = ted.schedule(p)
    a16 = p.time_points_s(p.steps[0])
    assert t_end == pytest.approx(tau, rel=1e-6)
    extra = [t for t in pts if not np.any(np.isclose(a16, t, rtol=1e-12, atol=0))]
    assert len(extra) == 1 and extra[0] == pytest.approx(tau, rel=1e-6)
    assert len(pts) == len(a16) + 1 and np.all(np.diff(pts) > 0)
    assert np.all(np.isin(a16, pts))


def test_item2_schedule_multistep_consumption():
    """Each step consumes its duration over its tau; TED ends when the sum reaches 1; no TED afterwards."""
    from simws.process import ted
    steps = [(800.0, 120.0), (1000.0, 60.0), (900.0, 30.0)]
    p = _sched_problem(steps)
    Q, rp = my_dose_cm2(p.initial), my_mean_depth_um(p.initial)
    t1 = my_tau(kelvin(800.0), Q, rp)
    t2 = my_tau(kelvin(1000.0), Q, rp)
    assert 120.0 / t1 < 1
    end2 = (1 - 120.0 / t1) * t2
    assert end2 < 60.0
    sch = ted.schedule(p)
    # step 1: TED all through, A16.1 points only
    assert np.array_equal(sch[0][0], p.time_points_s(p.steps[0])) and sch[0][1] >= 120.0
    # step 2: ends at end2, which is an extra time point
    pts2 = sch[1][0]
    a16 = p.time_points_s(p.steps[1])
    assert sch[1][1] == pytest.approx(end2, rel=1e-6)
    assert len(pts2) == len(a16) + 1 and np.any(np.isclose(pts2, end2, rtol=1e-5, atol=0))
    # step 3: no TED, A16.1 points only
    assert np.array_equal(sch[2][0], p.time_points_s(p.steps[2])) and sch[2][1] <= 0.0
    print(f"\n[item 2] schedule: tau(800 C) {t1:.1f} s (step 1 consumes {120 / t1:.3f}), tau(1000 C) {t2:.4f} s, "
          f"TED ends {end2:.4f} s into step 2")


def test_item2_schedule_none_and_never_ending():
    from simws.process import ted
    p = _sched_problem([(1000.0, 60.0), (800.0, 30.0)], mode="none")
    for (pts, _), st in zip(ted.schedule(p), p.steps):
        assert np.array_equal(pts, p.time_points_s(st))
    q = _sched_problem([(700.0, 60.0), (700.0, 30.0)])           # tau(700 C) >> 90 s
    Q, rp = my_dose_cm2(q.initial), my_mean_depth_um(q.initial)
    assert my_tau(kelvin(700.0), Q, rp) > 90
    for (pts, t_end), st in zip(ted.schedule(q), q.steps):
        assert np.array_equal(pts, q.time_points_s(st)) and t_end >= st.time_s


# ------------------------------------------------------------------- item 2 / 3: both engines against an exact answer
EXACT_STEPS = [(800.0, 120.0), (1000.0, 60.0)]


def _exact_problem(mode):
    from simws.chain import process as P
    ini = P.gaussian_profile(0.1, 0.03, 1e14)                # with its image (A16.7 R90(b))
    return P.anneal_problem("B", ini, steps=EXACT_STEPS, model="intrinsic", thickness_um=1.0).with_changes(
        {"models.ted": mode})


def _exact_Dt_cm2(p, ted_on):
    Dt = 0.0
    Q, rp = my_dose_cm2(p.initial), my_mean_depth_um(p.initial)
    rem = 1.0
    for st in p.steps:
        T = kelvin(st.temperature_C)
        D = 1.0 * math.exp(-3.5 / (K_B * T))                 # A16.2 eq. (13), B
        if ted_on and rem > 0:
            tau = my_tau(T, Q, rp)
            t_ted = min(st.time_s, rem * tau)
            rem -= t_ted / tau
            Dt += D * (my_enh("B", T) * t_ted + (st.time_s - t_ted))
        else:
            Dt += D * st.time_s
    return Dt


@pytest.mark.parametrize("adapter", ANNEALS)
def test_item2_engines_against_exact_gaussian(mhub, adapter):
    need(ENGINE_OF[adapter])
    out = {}
    for mode in ("none", "plus_one"):
        p = _exact_problem(mode)
        _, res = submit(mhub, adapter, p)
        x, c = profile_of(res)
        Dt = _exact_Dt_cm2(p, mode == "plus_one")
        s2 = (0.03e-4) ** 2 + 2 * Dt
        xc = x * 1e-4
        ex = 1e14 / math.sqrt(2 * math.pi * s2) * (np.exp(-(xc - 0.1e-4) ** 2 / (2 * s2))
                                                    + np.exp(-(xc + 0.1e-4) ** 2 / (2 * s2)))
        out[mode] = (float(np.max(np.abs(c - ex)) / ex.max()), Dt, c)
    dev_none, Dt0, c0 = out["none"]
    dev_ted, Dt1, c1 = out["plus_one"]
    sens = float(np.max(np.abs(c1 - c0)) / c0.max())
    print(f"\n[item 2] {adapter} exact Gaussian, 800 C 120 s + 1000 C 60 s: Dt {Dt0:.3e} -> {Dt1:.3e} cm^2 with TED; "
          f"largest deviation {dev_none * 100:.3f} % (no TED), {dev_ted * 100:.3f} % (TED) of the peak, limit 1 %; "
          f"TED moves the profile by {sens * 100:.1f} % of the peak")
    assert sens > 0.05                                        # the check is sensitive to TED
    assert dev_none <= 0.01
    assert dev_ted <= 0.01


def _exact_profile(p, Dt):
    xs = np.asarray(p.output_depths_um(), float) * 1e-4
    s2 = (0.03e-4) ** 2 + 2 * Dt
    return 1e14 / math.sqrt(2 * math.pi * s2) * (np.exp(-(xs - 0.1e-4) ** 2 / (2 * s2))
                                                  + np.exp(-(xs + 0.1e-4) ** 2 / (2 * s2)))


@pytest.mark.parametrize("adapter", ANNEALS)
def test_item2_engines_ted_end_is_a_time_point(mhub, adapter):
    """A22.2: the moment TED ends is a time point. 1000 C 60 s with steps_per_anneal 10: A16.1's first point is
    0.6 s and tau(1000 C) is about 0.38 s, so without the extra point TED would last 0.6 s (about 3 % on the peak)."""
    need(ENGINE_OF[adapter])
    from simws.chain import process as P
    ini = P.gaussian_profile(0.1, 0.03, 1e14)
    p = P.anneal_problem("B", ini, steps=[(1000.0, 60.0)], model="intrinsic", thickness_um=1.0,
                         steps_per_anneal=10).with_changes({"models.ted": "plus_one"})
    T = kelvin(1000.0)
    tau = my_tau(T, my_dose_cm2(p.initial), my_mean_depth_um(p.initial))
    first = p.time_points_s(p.steps[0])[0]
    assert tau < first
    D = math.exp(-3.5 / (K_B * T))
    ex = _exact_profile(p, D * (my_enh("B", T) * tau + 60.0 - tau))
    wrong = _exact_profile(p, D * (my_enh("B", T) * first + 60.0 - first))
    _, res = submit(mhub, adapter, p)
    _, c = profile_of(res)
    dev = float(np.max(np.abs(c - ex)) / ex.max())
    sep = float(np.max(np.abs(wrong - ex)) / ex.max())
    print(f"\n[item 2] {adapter} TED end inside the first A16.1 interval (tau {tau:.4f} s < {first:.2f} s): largest "
          f"deviation from the exact answer {dev * 100:.3f} % of the peak (limit 1 %); TED to the next point would "
          f"be {sep * 100:.2f} % off")
    assert sep > 0.02
    assert dev <= 0.01


# ------------------------------------------------------------------- item 2: the flag; item 3: both engines
def test_item3_both_engines_accept_plus_one():
    import simws.engines as E
    p = r150_problem("plus_one")
    for n in ("devsim_anneal", "fipy"):
        assert E.unsupported(n, p) == [], n
        assert E.family(n) == "anneal"


@pytest.mark.parametrize("adapter", ANNEALS)
def test_item2_ted_flag_only_on_ted_results(mhub, adapter):
    need(ENGINE_OF[adapter])
    _, r_ted = submit(mhub, adapter, r150_problem("plus_one"))
    _, r_none = submit(mhub, adapter, r150_problem("none"))
    assert TED_FLAG in r_ted.flags
    assert not any(f.startswith("anneal:ted_effective") for f in r_none.flags)
    _, r_ex = submit(mhub, adapter, _exact_problem("plus_one"))
    assert TED_FLAG in r_ex.flags
    _, r_ex0 = submit(mhub, adapter, _exact_problem("none"))
    assert not any(f.startswith("anneal:ted_effective") for f in r_ex0.flags)


# ===================================================================================== R150
def _r150_measured():
    ini, (dx, dc), _ = stolk()
    meas = {lv: my_depth_at(dx, dc, lv) for lv in R150_LEVELS}
    before = {lv: my_depth_at(ini.depth_um, ini.concentration_cm3, lv) for lv in R150_LEVELS}
    return meas, before


@pytest.mark.parametrize("adapter", ANNEALS)
def test_R150_stolk_fig1(mhub, adapter):
    need(ENGINE_OF[adapter])
    meas, before = _r150_measured()
    _, (dx, dc), _ = stolk()
    rows = {}
    for mode in ("none", "plus_one"):
        _, res = submit(mhub, adapter, r150_problem(mode))
        x, c = profile_of(res)
        rows[mode] = ({lv: my_depth_at(x, c, lv) for lv in R150_LEVELS}, float(c.max()))
    print(f"\n[R150] {adapter}: traced SIMS after 800 C 35 min: x(3e17) {meas[3e17] * 1e3:.1f} nm, x(1e17) "
          f"{meas[1e17] * 1e3:.1f} nm, peak {dc.max():.3g}; as implanted x(3e17) {before[3e17] * 1e3:.1f}, x(1e17) "
          f"{before[1e17] * 1e3:.1f} nm")
    worst = 0.0
    for mode in ("none", "plus_one"):
        got, peak = rows[mode]
        dev = {lv: got[lv] / meas[lv] - 1 for lv in R150_LEVELS}
        shift = {lv: (got[lv] - before[lv]) / (meas[lv] - before[lv]) for lv in R150_LEVELS}
        print(f"[R150] {adapter} {mode:8s}: x(3e17) {got[3e17] * 1e3:.1f} nm ({dev[3e17] * 100:+.1f} %), x(1e17) "
              f"{got[1e17] * 1e3:.1f} nm ({dev[1e17] * 100:+.1f} %), shift ratio (model / SIMS) 3e17 "
              f"{shift[3e17]:.3f}, 1e17 {shift[1e17]:.3f}; peak {peak:.3g} cm^-3")
        if mode == "plus_one":
            worst = max(abs(v) for v in dev.values())
    print(f"[R150] {adapter}: worst |dev| with TED {worst * 100:.1f} % (limit 15 %)")
    got, _ = rows["plus_one"]
    for lv in R150_LEVELS:
        assert abs(got[lv] / meas[lv] - 1) <= R150_TOL, (lv, got[lv], meas[lv])


# ===================================================================================== R151
def _r151_check(hub, mk, label):
    ids, res = {}, {}
    for a in ANNEALS:
        ids[a], res[a] = submit(hub, a, mk())
    p = mk()
    xs = {a: profile_of(res[a]) for a in ANNEALS}
    xj = {a: scalar(res[a], "anneal.junction_depth") for a in ANNEALS}
    d17 = {a: my_depth_at(*xs[a], 1e17) for a in ANNEALS}
    fin = {a: scalar(res[a], "anneal.dose") for a in ANNEALS}
    st = starting_doses(p)
    rj = abs(xj["anneal.devsim_anneal"] / xj["anneal.fipy"] - 1)
    r17 = abs(d17["anneal.devsim_anneal"] / d17["anneal.fipy"] - 1)
    rfin = abs(fin["anneal.devsim_anneal"] / fin["anneal.fipy"] - 1)
    rst = abs(st["anneal.devsim_anneal"] / st["anneal.fipy"] - 1)
    diag = {a: res[a].diagnostics.get("initial_dose_cm2") for a in ANNEALS}
    print(f"\n[R151] {label}: junction {xj['anneal.devsim_anneal'] * 1e9:.2f} / {xj['anneal.fipy'] * 1e9:.2f} nm "
          f"({rj * 100:.4f} %), x(1e17) {d17['anneal.devsim_anneal'] * 1e3:.2f} / {d17['anneal.fipy'] * 1e3:.2f} nm "
          f"({r17 * 100:.4f} %), limit 1 %; final doses differ {rfin:.3e}, starting doses {rst:.3e} "
          f"(limit {rst:.3e} + 1e-6); engines' own starting doses {diag}, computed {st}")
    assert TED_FLAG in res["anneal.devsim_anneal"].flags and TED_FLAG in res["anneal.fipy"].flags
    assert xj["anneal.devsim_anneal"] is not None and xj["anneal.fipy"] is not None
    assert rj <= R151_TOL
    assert r17 <= R151_TOL
    assert rfin <= rst + 1e-6
    return ids


def test_R151_stolk_case(mhub):
    need("devsim_anneal", "fipy")
    _r151_check(mhub, lambda: r150_problem("plus_one"), "R150's case (800 C 35 min)")


def test_R151_crystal_case(mhub):
    need("devsim_anneal", "fipy")
    _r151_check(mhub, lambda: crystal_problem(mhub, 1000.0, 10.0, "plus_one"), "B 35 keV 1e14 crystal, 1000 C 10 s")


# ===================================================================================== R152
@pytest.mark.parametrize("adapter", ANNEALS)
def test_R152_tau_to_zero_is_transient(mhub, adapter):
    need(ENGINE_OF[adapter])
    from simws.process import ted
    xj = {}
    for mode in ("none", "plus_one"):
        _, res = submit(mhub, adapter, crystal_problem(mhub, 1100.0, 3600.0, mode))
        xj[mode] = scalar(res, "anneal.junction_depth")
    p = crystal_problem(mhub, 1100.0, 3600.0, "plus_one")
    Q, rp = my_dose_cm2(p.initial), my_mean_depth_um(p.initial)
    tau = my_tau(kelvin(1100.0), Q, rp)
    rel = abs(xj["plus_one"] / xj["none"] - 1)
    print(f"\n[R152] {adapter} 1100 C 1 h: tau {tau:.3g} s, enhancement {ted.enhancement('B', kelvin(1100.0)):.1f}; "
          f"junction {xj['none'] * 1e9:.2f} nm without, {xj['plus_one'] * 1e9:.2f} nm with TED: {rel * 100:.3f} % "
          f"(limit 2 %)")
    assert tau < 3600.0 / 100
    assert rel < R152_TOL


def test_R152_enhancement_at_step_temperature():
    from simws.process import ted
    for d in F_I:
        for T_C in (800.0, 1000.0, 1100.0):
            T = kelvin(T_C)
            S = ted.supersaturation(T)
            assert ted.enhancement(d, T) == pytest.approx(F_I[d] * S + (1 - F_I[d]) / S, rel=1e-12)


def test_R152_reference_values():
    from simws.process import ted
    assert ted.tau_s(kelvin(T_REF_C), Q_REF, ted.RP_REF_UM) == pytest.approx(TAU_REF, rel=1e-12)
    assert ted.tau_s(kelvin(T_REF_C), Q_REF, RP_REF_NM * 1e-3) == pytest.approx(TAU_REF, rel=1e-3)
    assert ted.supersaturation(kelvin(T_S_C)) == pytest.approx(my_S_ref(), rel=1e-12)
    t670 = my_tau(kelvin(670.0), Q_REF, RP_REF_NM * 1e-3)
    print(f"\n[R152] tau(815 C) {ted.tau_s(kelvin(T_REF_C), Q_REF, ted.RP_REF_UM):.6f} s, S(790 C) "
          f"{ted.supersaturation(kelvin(T_S_C)):.2f}; reported: tau(670 C) for Stolk's implant {t670:.0f} s = "
          f"{t670 / 3600:.2f} h (Stolk: about 4 h)")


# ===================================================================================== R153 (reported)
def test_R153_reported(mhub):
    need("devsim_anneal", "fipy")
    from simws.process import ted
    lines = []
    for T_C, t_s in ((900.0, 10.0), (1000.0, 10.0), (1000.0, 1800.0)):
        row = {}
        for a in ANNEALS:
            for mode in ("none", "plus_one"):
                _, res = submit(mhub, a, crystal_problem(mhub, T_C, t_s, mode))
                row[(a, mode)] = scalar(res, "anneal.junction_depth")
        p = crystal_problem(mhub, T_C, t_s, "plus_one")
        Q, rp = my_dose_cm2(p.initial), my_mean_depth_um(p.initial)
        lines.append(f"[R153] {T_C:.0f} C {t_s:.0f} s (tau {my_tau(kelvin(T_C), Q, rp):.3g} s, E "
                     f"{ted.enhancement('B', kelvin(T_C)):.0f}): junction devsim {row[(ANNEALS[0], 'none')] * 1e9:.1f} "
                     f"-> {row[(ANNEALS[0], 'plus_one')] * 1e9:.1f} nm, fipy {row[(ANNEALS[1], 'none')] * 1e9:.1f} -> "
                     f"{row[(ANNEALS[1], 'plus_one')] * 1e9:.1f} nm (without -> with TED)")
        assert all(v is not None for v in row.values())
    p = crystal_problem(mhub, 1000.0, 10.0, "plus_one")
    print(f"\n[R153] crystal B 35 keV 1e14: Q {my_dose_cm2(p.initial):.4e} cm^-2, mean depth "
          f"{my_mean_depth_um(p.initial) * 1e3:.1f} nm; n-type 1e16, fermi, thickness 2 um")
    print("\n".join(lines))


# ===================================================================================== item 8: depth_at
def test_item8_depth_at_function():
    from simws.process import ted
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    c = [5e17, 2e17, 4e17, 1e17, 5e16, 0.0]                 # crosses 3e17 twice: the deepest crossing counts
    assert ted.depth_at(x, c, 3e17) == pytest.approx(my_depth_at(x, c, 3e17), rel=1e-12)
    assert ted.depth_at(x, c, 3e17) == pytest.approx(0.2 + 0.1 * (4e17 - 3e17) / (4e17 - 1e17), rel=1e-12)
    assert ted.depth_at(x, c, 1e17) == pytest.approx(0.3, rel=1e-12)          # exactly at the level
    assert ted.depth_at(x, [5e17] * 6, 1e17) == pytest.approx(0.5)           # last point still above
    v = ted.depth_at(x, c, 1e18)
    assert v is None or not np.isfinite(v)


@pytest.mark.parametrize("adapter", ANNEALS)
def test_item8_adapters_report_depths(mhub, adapter):
    need(ENGINE_OF[adapter])
    for mode in ("none", "plus_one"):
        _, res = submit(mhub, adapter, r150_problem(mode))
        x, c = profile_of(res)
        assert c[0] >= 3e17
        for lv, name in ((1e17, "anneal.depth_at_1e17"), (3e17, "anneal.depth_at_3e17")):
            q = res.q(name)
            assert q.unit == "m"
            assert float(q.value) == pytest.approx(my_depth_at(x, c, lv) * 1e-6, rel=1e-9)


def _buried(rp, sigma, peak):
    from simws.chain import process as P
    dose = peak * math.sqrt(2 * math.pi) * sigma * 1e-4
    ini = P.gaussian_profile(rp, sigma, dose, image=False)
    return P.anneal_problem("B", ini, steps=[(900.0, 10.0)], model="intrinsic", thickness_um=1.0)


@pytest.mark.parametrize("adapter", ANNEALS)
def test_item8_absent_when_profile_starts_below(mhub, adapter):
    need(ENGINE_OF[adapter])
    # surface ~3e14 < 1e17 < 3e17 < peak 1e18: neither depth
    _, r1 = submit(mhub, adapter, _buried(0.2, 0.05, 1e18))
    x, c = profile_of(r1)
    assert c[0] < 1e17 and c.max() > 3e17
    names = {q.name for q in r1.quantities}
    assert "anneal.depth_at_1e17" not in names and "anneal.depth_at_3e17" not in names
    # surface ~1.35e17: between the levels -> only the 1e17 depth
    _, r2 = submit(mhub, adapter, _buried(0.1, 0.05, 1e18))
    x, c = profile_of(r2)
    assert 1e17 <= c[0] < 3e17
    names = {q.name for q in r2.quantities}
    assert "anneal.depth_at_3e17" not in names
    assert float(r2.q("anneal.depth_at_1e17").value) == pytest.approx(my_depth_at(x, c, 1e17) * 1e-6, rel=1e-9)


# ===================================================================================== item 5: trust
def _ingest_sims(hub, name, factor=1.0):
    from simws.measured import ingest
    meas, _ = _r150_measured()
    return ingest(hub, name, None,
                  {"anneal.depth_at_3e17": ([meas[3e17] * 1e-6 * factor], "m"),
                   "anneal.depth_at_1e17": ([meas[1e17] * 1e-6 * factor], "m")},
                  {"source": "Stolk et al., J. Appl. Phys. 81, 6031 (1997), Fig. 1 (traced; A22 checker)"
                             + ("" if factor == 1.0 else f" x {factor} (deliberately wrong, checker test)"),
                   "digitized": True}, label="validation")


def test_item5_engine_pair_verified(mhub):
    """A22.5 / A16.8: the two engines' TED results compared on the junction depth (1 %) -> "verified"."""
    need("devsim_anneal", "fipy")
    from simws.compare import compare_results, record
    from simws.hub.trust import assess_stored
    p = crystal_problem(mhub, 1000.0, 10.0, "plus_one")
    a, _ = submit(mhub, "anneal.devsim_anneal", p)
    b, _ = submit(mhub, "anneal.fipy", p)
    cm = compare_results(mhub, a, "anneal.junction_depth", b, tolerance=0.01)
    record(mhub, cm)
    print(f"\n[item 5] crystal 1000 C 10 s TED: engine pair worst {cm.worst:.2e} (tol 0.01), passed {cm.passed}, "
          f"kind {cm.kind}; levels {assess_stored(mhub, a).level} / {assess_stored(mhub, b).level}")
    assert cm.kind == "engine_pair" and cm.passed
    assert assess_stored(mhub, a).level == "verified"
    assert assess_stored(mhub, b).level == "verified"


def test_item5_R150_validated(mhub):
    """A22.5: R150's TED result, verified by the engine pair, recorded against the traced SIMS depths (label
    "validation", 15 % relative) -> "validated" when R150 passes; nothing else is validated."""
    need("devsim_anneal", "fipy")
    from simws.compare import compare_results, record
    from simws.hub.trust import assess_stored
    p = r150_problem("plus_one")
    a, _ = submit(mhub, "anneal.devsim_anneal", p)
    b, _ = submit(mhub, "anneal.fipy", p)
    record(mhub, compare_results(mhub, a, "anneal.junction_depth", b, tolerance=0.01))
    assert assess_stored(mhub, a).level == "verified"
    m = _ingest_sims(mhub, "stolk1997_fig1_diffused_checker")
    assert "measured" in m.flags and assess_stored(mhub, m.id).level == "measured"
    passed = []
    for name in ("anneal.depth_at_3e17", "anneal.depth_at_1e17"):
        cm = compare_results(mhub, a, name, m.id, tolerance=R150_TOL)
        assert cm.kind == "measurement"
        record(mhub, cm)
        passed.append(cm.passed)
        print(f"\n[item 5] {name}: worst {cm.worst:.4f} (tol 0.15), passed {cm.passed}")
    lv = assess_stored(mhub, a).level
    lv_b = assess_stored(mhub, b).level
    print(f"[item 5] R150 TED devsim result: {lv}; fipy TED result (not compared with the SIMS): {lv_b}")
    assert lv == ("validated" if all(passed) else "verified")
    assert all(passed)
    assert lv_b == "verified"
    a0, _ = submit(mhub, "anneal.devsim_anneal", r150_problem("none"))
    assert assess_stored(mhub, a0).level != "validated"


def test_item5_failed_measurement_blocks_validated(tmp_path):
    """A1.10 / A13.7: one failed measurement comparison blocks "validated" (a separate hub)."""
    need("devsim_anneal")
    from simws.compare import compare_results, record
    from simws.hub.jobs import Hub
    from simws.hub.trust import assess_stored
    hub = Hub(tmp_path / "store_fail", mode="inprocess")
    a, _ = submit(hub, "anneal.devsim_anneal", r150_problem("plus_one"))
    good = _ingest_sims(hub, "stolk1997_fig1_good")
    ok = []
    for name in ("anneal.depth_at_3e17", "anneal.depth_at_1e17"):
        cm = compare_results(hub, a, name, good.id, tolerance=R150_TOL)
        record(hub, cm)
        ok.append(cm.passed)
    lv_good = assess_stored(hub, a).level
    bad = _ingest_sims(hub, "stolk1997_fig1_wrong", factor=1.5)
    cb = compare_results(hub, a, "anneal.depth_at_1e17", bad.id, tolerance=R150_TOL)
    record(hub, cb)
    t = assess_stored(hub, a)
    print(f"\n[item 5] separate hub: good comparisons passed {ok} -> {lv_good}; wrong measurement worst "
          f"{cb.worst:.3f}, passed {cb.passed} -> {t.level}; reasons {t.reasons}")
    assert not cb.passed and cb.kind == "measurement"
    if all(ok):
        assert lv_good == "validated"
    assert t.level != "validated"
    assert t.level == "unverified"                            # no engine pair recorded in this hub
