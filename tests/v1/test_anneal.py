"""Contract amendment A16 (23 Nov 2026): thermal anneal (dopant diffusion), profile doping, the process chain.
Acceptance checks R90 to R99 of A16.7, plus the rules of items 1 to 5 and 8.

Written by the checker from docs/ARCHITECTURE_V1.md (A16 items 0 to 9, section 6.1, A7.3, A11.4) and from the two
sources the contract names: the EE212 handout #12 (Plummer, Deal, Griffin; eq. 13 page 10, eq. 19 page 18, the
worked design pages 12-14) and Morin and Maita, Phys. Rev. 96, 28 (1954), eq. (3). The engine and adapter
implementations were not read. Only public names are called: AnnealProblem / DepthProfile (simws.spec.anneal),
ProfileTerm / DeviceSpec.doping_at (simws.spec.schema), the registry functions of simws.engines, the planner, the hub
(adapters "anneal.devsim_anneal" and "anneal.fipy"), compare_results / record, assess_stored, the public functions of
simws.process.diffusivity (evaluated and compared with the tables), simws.engines.devsim_engine.build (R97) and the
chain functions of simws.chain.process / simws.chain.diode (R99).

The exact solutions (Gaussian with sigma^2 = sigma0^2 + 2 D t and its image, erfc, eq. 14, the R96 closed form) are
built here, with D the intrinsic D of A16.2 from the contract's numbers (k = 8.617333e-5 eV/K). Tests that read the
PDFs look in SIMWS_ANNEAL_SOURCES (default: the build machine's scratch folder) and skip when the files are absent.
"""
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

ENGINES = ("anneal.devsim_anneal", "anneal.fipy")
K_EV = 8.617333e-5                                                     # A16.2
# k is given to 7 significant digits; a code using a k that rounds to it (e.g. CODATA 8.617333262e-5) is equal to
# the contract's number. D = D0 exp(-E/kT) then agrees to (E/kT) x K_REL; every D comparison below allows that.
K_REL = 0.5e-11 / K_EV


def ktol(E, T_C):
    return E / (K_EV * TK(T_C)) * K_REL + 1e-12
UM = 1e-4                                                               # cm per um
SOURCES = Path(os.environ.get("SIMWS_ANNEAL_SOURCES",
                              "/tmp/claude-0/-home-claude/d0abb2e9-411f-5090-9cb2-dd3ba3452a20/scratchpad/src"))
HANDOUT = SOURCES / "dunham7.pdf"
MORIN = SOURCES / "morin.pdf"
NOT_MODELLED = ("anneal:not_modelled: transient enhanced diffusion from implant damage, clustering, solid "
                "solubility, oxidation and segregation")
NI_FLAG = "model:ni_morin_maita_extrapolated"
NO_JUNCTION = "anneal:no_junction"

# A16.2, eq. (13): D0 (cm^2/s), E (eV)
INTRINSIC = {"B": (1.0, 3.5), "As": (9.17, 3.99), "Sb": (4.58, 3.88), "P": (4.70, 3.68)}
# A16.2, eq. (19): component -> (D0, E); "0" neutral, "+" (B, p/ni), "-" (n/ni), "=" ((n/ni)^2)
CHARGED = {"B": {"0": (0.05, 3.5), "+": (0.95, 3.5)},
           "As": {"0": (0.011, 3.44), "-": (31.0, 4.15)},
           "Sb": {"0": (0.214, 3.65), "-": (15.0, 4.08)},
           "P": {"0": (3.85, 3.66), "-": (4.44, 4.0), "=": (44.2, 4.37)}}

# problem ids at the base commit b374737 (contract only, before any A16 code), computed by the checker in a
# worktree of b374737 with every zero-argument builder of simws.spec.library
BASE_IDS = {"diode_iv": "c70674e64e582db8d71e6abb0ea81470", "divider": "650cd885614fdaa14c23b0b8e1129eab",
            "implant_si": "90285c5cb1b9a21088805bf9e60ba682", "n_bar_1d": "3595ce69f90e9cfaaaf89d025b745563",
            "npn_hbt_2d": "55cf78ed7e298d344d60b782acad29b8", "pn_diode_1d": "52b23f57732ec9aa197c56d61f3fb049",
            "rc_lowpass": "2e04b4f07e1b802a0786b19a1a763498", "see_si_slab": "a4e0942f20122f9bba3ca49b45871df9"}


# ------------------------------------------------------------------------------------------------ own physics
def TK(T_C):
    return T_C + 273.15


def arr(d0, e, T_C):
    return d0 * math.exp(-e / (K_EV * TK(T_C)))


def D_int(dopant, T_C):
    """A16.2 intrinsic D (cm^2/s) from the contract's (= the handout's) numbers."""
    return arr(*INTRINSIC[dopant], T_C)


def ni_own(T_C):
    """sqrt of Morin and Maita eq. (3): n p = 1.5e33 T^3 exp(-1.21 eV / kT)."""
    T = TK(T_C)
    return math.sqrt(1.5e33 * T ** 3 * math.exp(-1.21 / (K_EV * T)))


def fermi_own(dopant, C, Nb, bg_type, T_C):
    """A16.2 "fermi": (D_eff, h) at concentration C (cm^-3) on a uniform background Nb of type bg_type."""
    C = np.asarray(C, float)
    own = "acceptor" if dopant == "B" else "donor"
    N = C + Nb if bg_type == own else C - Nb
    ni = ni_own(T_C)
    r = (N / 2 + np.sqrt(N ** 2 / 4 + ni ** 2)) / ni
    comp = {k: arr(d0, e, T_C) for k, (d0, e) in CHARGED[dopant].items()}
    D = comp["0"] + comp.get("+", 0.0) * r + comp.get("-", 0.0) * r + comp.get("=", 0.0) * r ** 2
    h = 1 + C / np.sqrt(N ** 2 + 4 * ni ** 2)
    return D, h


def half_gauss(x_cm, Q, s_cm):
    return 2 * Q / (math.sqrt(2 * math.pi) * s_cm) * np.exp(-x_cm ** 2 / (2 * s_cm ** 2))


def gauss_image(x_cm, Rp_cm, Q, s_cm):
    a = Q / (math.sqrt(2 * math.pi) * s_cm)
    return a * (np.exp(-(x_cm - Rp_cm) ** 2 / (2 * s_cm ** 2)) + np.exp(-(x_cm + Rp_cm) ** 2 / (2 * s_cm ** 2)))


def gauss_plain(x_cm, Rp_cm, Q, s_cm):
    return Q / (math.sqrt(2 * math.pi) * s_cm) * np.exp(-(x_cm - Rp_cm) ** 2 / (2 * s_cm ** 2))


def junction_own(x, C, Nb):
    """A16.4: the first depth where C falls to the background, linear between points; None when C starts below it
    or stays above it."""
    if C[0] <= Nb:
        return None
    below = np.nonzero(C <= Nb)[0]
    if below.size == 0:
        return None
    i = below[0]
    return x[i - 1] + (C[i - 1] - Nb) * (x[i] - x[i - 1]) / (C[i - 1] - C[i])


def trap(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


# ------------------------------------------------------------------------------------------------ problems
def tab(fn, top_um, step_um=0.0005):
    x = np.round(np.arange(0.0, top_um + step_um / 2, step_um), 10)
    return {"depth_um": x.tolist(), "concentration_cm3": fn(x * UM).tolist()}


def problem(dopant, initial, steps, model="intrinsic", bg=0.0, bg_type="acceptor", thickness=1.0, **kw):
    from simws.spec.anneal import AnnealProblem
    d = dict(dopant=dopant, initial=initial, background_cm3=bg, background_type=bg_type, thickness_um=thickness,
             steps=[{"temperature_C": T, "time_s": t} for T, t in steps], models={"diffusivity": model})
    numerics = {k: kw.pop(k) for k in ("dx_um", "steps_per_anneal") if k in kw}
    if numerics:
        d["numerics"] = numerics
    d.update(kw)
    return AnnealProblem.model_validate(d)


R90A = dict(Q=1e14, s0=0.02)
R90B = dict(Q=1e14, s0=0.03, Rp=0.1)
R94 = {"As": dict(Rp=0.03, s=0.012, Q=1e15, bg_type="acceptor", T=1000.0, t=600.0),
       "P": dict(Rp=0.05, s=0.02, Q=5e15, bg_type="acceptor", T=1000.0, t=600.0),
       "B": dict(Rp=0.1, s=0.035, Q=1e15, bg_type="donor", T=1000.0, t=1800.0)}


def p_r90a(steps=((1000.0, 1800.0),)):
    return problem("B", tab(lambda x: half_gauss(x, R90A["Q"], R90A["s0"] * UM), 1.0), list(steps))


def p_r90b():
    return problem("B", tab(lambda x: gauss_image(x, R90B["Rp"] * UM, R90B["Q"], R90B["s0"] * UM), 1.0),
                   [(1000.0, 1800.0)])


def p_r91():
    return problem("B", {"depth_um": [0.0, 1.0], "concentration_cm3": [0.0, 0.0]}, [(1000.0, 1800.0)],
                   surface="fixed", surface_concentration_cm3=1e19)


def p_r94(dopant, model="fermi", form="plain"):
    """R94 starting profiles. "plain": the Gaussian (Rp, sigma, dose) as written in R94, cut at the surface;
    "image": the same with its image in the surface (zero slope at x = 0, as R90(b) and chain.gaussian_profile's
    default). R94 does not say which; both are run."""
    c = R94[dopant]
    g = gauss_plain if form == "plain" else (lambda x, Rp, Q, s: gauss_image(x, Rp, Q, s))
    return problem(dopant, tab(lambda x: g(x, c["Rp"] * UM, c["Q"], c["s"] * UM), 1.0),
                   [(c["T"], c["t"])], model=model, bg=1e16, bg_type=c["bg_type"])


R96 = dict(Q=4.3e13, s0=0.005, CB=1e15, T=1100.0, t=6.8 * 3600.0)


def p_r96():
    return problem("B", tab(lambda x: half_gauss(x, R96["Q"], R96["s0"] * UM), 0.1, 0.0005),
                   [(R96["T"], R96["t"])], bg=R96["CB"], bg_type="donor", thickness=10.0, dx_um=0.005)


# ------------------------------------------------------------------------------------------------ hub, cached runs
@pytest.fixture(scope="module")
def mhub(tmp_path_factory):
    from simws.hub.jobs import Hub
    return Hub(tmp_path_factory.mktemp("anneal_store"), mode="inprocess")


_RUNS = {}


def run(hub, adapter, p):
    key = (id(hub), adapter, p.problem_id())
    if key not in _RUNS:
        r = hub.submit(adapter, params={"problem": p.model_dump(mode="json")})
        assert r.status == "ok", f"{adapter}: {r.status}: {r.message[-400:]}"
        _RUNS[key] = hub.result(r.id)
    return _RUNS[key]


def profile(res):
    """(x um, C cm^-3) from anneal.concentration (1/m^3 on the depth axis in m)."""
    q = res.q("anneal.concentration")
    return np.asarray(q.axes[0].values, float) * 1e6, np.asarray(q.array(), float) * 1e-6


def scalar(res, name):
    q = next((q for q in res.quantities if q.name == name), None)
    if q is None or q.value is None:
        return None
    v = float(q.value)
    return None if math.isnan(v) else v


def own_initial_dose(p, adapter):
    """A16.3 / R93: the dose (cm^-2) of the initial profile interpolated onto the engine's own points: DEVSIM nodes
    at k dx (trapezoid), FiPy cells of width dx centred at (k + 1/2) dx (cell sum)."""
    dx, M = p.numerics.dx_um, p.n_intervals
    if adapter == "anneal.devsim_anneal":
        x = np.arange(M + 1) * dx
        return trap(p.initial.at(x), x) * UM
    xc = (np.arange(M) + 0.5) * dx
    return float(np.sum(p.initial.at(xc)) * dx * UM)


def all_flags(res):
    return list(res.flags) + [f for q in res.quantities for f in q.flags]


def unit_is(u, *wanted):
    norm = u.replace("**", "^").replace(" ", "")
    return norm in wanted


# ================================================================================================ sources (PDFs)
def _pdftext(path, page=None):
    if not path.exists() or shutil.which("pdftotext") is None:
        pytest.skip(f"{path.name} or pdftotext not available")
    args = ["pdftotext", "-layout"] + (["-f", str(page), "-l", str(page)] if page else []) + [str(path), "-"]
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def _despaced_lines(text):
    return [re.sub(r"\s+", "", ln) for ln in text.splitlines()]


def test_source_handout_eq13_equals_contract():
    """Page 10 (handout numbering = PDF page): eq. (13) table, columns Si B In As Sb P."""
    lines = _despaced_lines(_pdftext(HANDOUT, 10))
    d0 = next(s for s in lines if s.startswith("D0560") and s.endswith("cm2sec-1"))
    ea = next(s for s in lines if s.startswith("EA4.76") and s.endswith("eV"))
    # B, In, As, Sb, P
    assert d0 == "D0560" + "1.0" + "1.2" + "9.17" + "4.58" + "4.70" + "cm2sec-1"
    assert ea == "EA4.76" + "3.5" + "3.5" + "3.99" + "3.88" + "3.68" + "eV"
    pdf = {"B": (1.0, 3.5), "As": (9.17, 3.99), "Sb": (4.58, 3.88), "P": (4.70, 3.68)}
    assert pdf == INTRINSIC


def test_source_handout_eq19_equals_contract():
    """Page 18: eq. (19) table, columns Si B In As Sb P (only the filled cells)."""
    lines = _despaced_lines(_pdftext(HANDOUT, 18))
    want = {"D0.0": "560" + "0.05" + "0.6" + "0.011" + "0.214" + "3.85",
            "D0.E": "4.76" + "3.5" + "3.5" + "3.44" + "3.65" + "3.66",
            "D+.0": "0.95" + "0.6", "D+.E": "3.5" + "3.5",
            "D-.0": "31.0" + "15.0" + "4.44", "D-.E": "4.15" + "4.08" + "4.0",
            "D=.0": "44.2", "D=.E": "4.37"}
    for head, rest in want.items():
        assert head + rest in lines, head
    pdf = {"B": {"0": (0.05, 3.5), "+": (0.95, 3.5)}, "As": {"0": (0.011, 3.44), "-": (31.0, 4.15)},
           "Sb": {"0": (0.214, 3.65), "-": (15.0, 4.08)},
           "P": {"0": (3.85, 3.66), "-": (4.44, 4.0), "=": (44.2, 4.37)}}
    assert pdf == CHARGED


def test_source_handout_design_numbers():
    """Pages 13-14: x_j = 3 um, C_B = 1e15, D(1100 C) = 1.5e-13, t = 6.8 h, Q = 4.3e13, D(950 C) = 4.2e-15."""
    t13 = re.sub(r"\s+", "", _pdftext(HANDOUT, 13))
    t14 = re.sub(r"\s+", "", _pdftext(HANDOUT, 14))
    assert "xj=3µm" in t13 and "NBC=1x1015" in t13 and "1100˚C" in t13 and "D=1.5×10−13" in t13
    assert "=6.8hours" in t14 and "=4.3×1013" in t14 and "4.2×10−15" in t14 and "950˚C" in t14


def test_source_morin_eq3():
    """Morin and Maita eq. (3), page 7: n p = 1.5e33 T^3 exp(-1.21/kT); the OCR text keeps 1.5, 1.21 and (3)."""
    t = _pdftext(MORIN, 7)
    line = next(ln for ln in t.splitlines() if "1.21/" in ln)
    assert "1.5" in line and "exp" in line
    assert re.search(r"\(3\)", t[t.index(line):t.index(line) + 400])
    # the contract's 3.87e16 is sqrt(1.5e33) rounded
    assert abs(math.sqrt(1.5e33) / 3.87e16 - 1) < 1e-3


# ================================================================================================ item 2: code numbers
@pytest.mark.parametrize("dopant", ["B", "P", "As", "Sb"])
@pytest.mark.parametrize("T_C", [800.0, 950.0, 1000.0, 1100.0, 1200.0])
def test_item2_intrinsic_D_equals_table(dopant, T_C):
    from simws.process import diffusivity as Dm
    assert Dm.intrinsic_D(dopant, TK(T_C)) == pytest.approx(D_int(dopant, T_C), rel=ktol(INTRINSIC[dopant][1], T_C))


def test_item2_boltzmann_constant_as_printed():
    """The k implied by the code's D (B, eq. 13) rounds to the contract's 8.617333e-5 eV/K."""
    from simws.process import diffusivity as Dm
    for T_C in (800.0, 1200.0):
        k = 3.5 / (TK(T_C) * math.log(1.0 / Dm.intrinsic_D("B", TK(T_C))))
        assert abs(k - K_EV) <= 0.5e-11, k


@pytest.mark.parametrize("dopant", ["B", "P", "As", "Sb"])
@pytest.mark.parametrize("T_C", [800.0, 1000.0, 1200.0])
def test_item2_charged_components_equal_table(dopant, T_C):
    from simws.process import diffusivity as Dm
    comp = Dm.components(dopant, TK(T_C))
    own = sorted(arr(d0, e, T_C) for d0, e in CHARGED[dopant].values())
    Emax = max(e for _, e in CHARGED[dopant].values())
    assert sorted(float(v) for v in comp.values()) == pytest.approx(own, rel=ktol(Emax, T_C))


@pytest.mark.parametrize("T_C", [700.0, 1000.0, 1300.0])
def test_item2_ni_morin_maita(T_C):
    from simws.process import diffusivity as Dm
    ni = Dm.ni_cm3(TK(T_C))
    assert ni == pytest.approx(ni_own(T_C), rel=ktol(0.605, T_C))
    assert ni == pytest.approx(3.87e16 * TK(T_C) ** 1.5 * math.exp(-0.605 / (K_EV * TK(T_C))), rel=1e-3)


@pytest.mark.parametrize("dopant,bg_type", [("B", "donor"), ("B", "acceptor"), ("As", "acceptor"),
                                            ("P", "acceptor"), ("P", "donor"), ("Sb", "acceptor")])
def test_item2_fermi_K_equals_contract_formula(dopant, bg_type):
    """K = D_eff h of A16.2 at concentrations from far below to far above ni, background 1e16."""
    from simws.process import diffusivity as Dm
    T_C = 1000.0
    p = problem(dopant, {"depth_um": [0.0, 0.1], "concentration_cm3": [1e18, 0.0]}, [(T_C, 60.0)], model="fermi",
                bg=1e16, bg_type=bg_type)
    C = np.array([1e10, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 5e20])
    D, h = fermi_own(dopant, C, 1e16, bg_type, T_C)
    K = np.asarray(Dm.K(p, C, TK(T_C)), float)
    tol = ktol(max(e for _, e in CHARGED[dopant].values()) + 0.605, T_C)
    assert np.max(np.abs(K / (D * h) - 1)) < tol
    pi = problem(dopant, {"depth_um": [0.0, 0.1], "concentration_cm3": [1e18, 0.0]}, [(T_C, 60.0)], bg=1e16,
                 bg_type=bg_type)
    Ki = np.asarray(Dm.K(pi, C, TK(T_C)), float) * np.ones_like(C)
    assert np.max(np.abs(Ki / D_int(dopant, T_C) - 1)) < ktol(INTRINSIC[dopant][1], T_C)   # "intrinsic": h = 1


# ================================================================================================ item 1: description
def _base(**kw):
    d = dict(dopant="B", initial={"depth_um": [0.0, 0.5], "concentration_cm3": [1e18, 0.0]}, thickness_um=1.0,
             steps=[{"temperature_C": 1000, "time_s": 60}])
    d.update(kw)
    return d


def test_item1_valid_and_defaults():
    from simws.spec.anneal import AnnealProblem
    p = AnnealProblem.model_validate(_base())
    assert p.kind == "anneal" and p.models.diffusivity == "intrinsic" and p.surface == "reflecting"
    assert p.numerics.dx_um == 0.001 and p.numerics.steps_per_anneal == 200
    assert p.n_intervals == 1000
    x = p.output_depths_um()
    assert len(x) == 1001 and x[0] == 0 and x[-1] == pytest.approx(1.0)
    t = p.time_points_s(p.steps[0])
    assert len(t) == 200 and t[-1] == pytest.approx(60.0) and t[0] == pytest.approx(60.0 / 200 ** 2)
    k = np.arange(1, 201)
    assert np.allclose(t, 60.0 * (k / 200) ** 2, rtol=1e-14)


@pytest.mark.parametrize("bad", [
    dict(thickness_um=0.4),                                                    # thinner than the profile
    dict(surface="fixed"),                                                     # fixed without a concentration
    dict(surface_concentration_cm3=1e19),                                      # a concentration without "fixed"
    dict(thickness_um=30.0, initial={"depth_um": [0.0, 0.5], "concentration_cm3": [1.0, 0.0]}),  # 30000 intervals
    dict(numerics={"dx_um": 0.003}),                                           # 1/0.003 not whole
    dict(numerics={"dx_um": 0.0007}),
    dict(dopant="In"), dict(dopant="Ga"),
    dict(steps=[]),
    dict(steps=[{"temperature_C": 590, "time_s": 60}]), dict(steps=[{"temperature_C": 1410, "time_s": 60}]),
    dict(steps=[{"temperature_C": 1000, "time_s": 0}]),
    dict(models={"diffusivity": "pairs"}),
    dict(surface="absorbing"),
    dict(background_cm3=-1.0), dict(background_type="intrinsic"),
    dict(thickness_um=0.0),
    dict(initial={"depth_um": [0.0], "concentration_cm3": [1.0]}),
    dict(initial={"depth_um": [0.0, 0.0], "concentration_cm3": [1.0, 1.0]}),
    dict(initial={"depth_um": [0.2, 0.1], "concentration_cm3": [1.0, 1.0]}),
    dict(initial={"depth_um": [-0.1, 0.1], "concentration_cm3": [1.0, 1.0]}),
    dict(initial={"depth_um": [0.0, 0.1], "concentration_cm3": [1.0, -1.0]}),
    dict(initial={"depth_um": [0.0, 0.1, 0.2], "concentration_cm3": [1.0, 1.0]}),
    dict(initial={"depth_um": list(np.linspace(0, 0.5, 20001)), "concentration_cm3": [1.0] * 20001}),
])
def test_item1_refusals(bad):
    from simws.spec.anneal import AnnealProblem
    with pytest.raises(ValidationError):
        AnnealProblem.model_validate(_base(**bad))


def test_item1_limits_accepted():
    from simws.spec.anneal import AnnealProblem
    AnnealProblem.model_validate(_base(thickness_um=20.0))                     # exactly 20000 intervals
    AnnealProblem.model_validate(_base(thickness_um=0.5))                      # exactly the last depth
    AnnealProblem.model_validate(_base(steps=[{"temperature_C": 600, "time_s": 1},
                                              {"temperature_C": 1400, "time_s": 1}]))
    AnnealProblem.model_validate(_base(surface="fixed", surface_concentration_cm3=1e19))
    AnnealProblem.model_validate(_base(initial={"depth_um": list(np.linspace(0, 0.5, 20000)),
                                                "concentration_cm3": [1.0] * 20000}))


def test_item1_interpolation_rule():
    from simws.spec.anneal import DepthProfile
    p = DepthProfile(depth_um=[0.1, 0.2, 0.3], concentration_cm3=[4.0, 2.0, 1.0])
    got = p.at([0.0, 0.05, 0.1, 0.15, 0.25, 0.3, 0.3000001, 0.5])
    assert np.allclose(got, [4.0, 4.0, 4.0, 3.0, 1.5, 1.0, 0.0, 0.0])
    assert p.dose_cm2() == pytest.approx((4.0 * 0.1 + 0.3 + 0.15) * UM, rel=1e-12)


@pytest.mark.parametrize("adapter", ENGINES)
def test_item1_first_value_holds_from_zero_through_engines(mhub, adapter):
    """A profile starting at 0.1 um: the engines must hold its first value from 0 (A16.1); the reflecting dose of
    the engine's own interpolated profile (which includes that part) is kept (R93's rule, 1e-6)."""
    p = problem("B", {"depth_um": [0.1, 0.2], "concentration_cm3": [1e18, 0.0]}, [(1000.0, 60.0)])
    res = run(mhub, adapter, p)
    want = own_initial_dose(p, adapter)
    assert want == pytest.approx((1e18 * 0.1 + 0.5e18 * 0.1) * UM, rel=1e-3)   # the constant part is in it
    assert scalar(res, "anneal.dose") * 1e-4 == pytest.approx(want, rel=1e-6)


# ================================================================================================ R90
def _r90_check(res, exact_fn, sd_ref=None):
    x, C = profile(res)
    ex = exact_fn(x * UM)
    xf = np.linspace(0, x[-1], 200001)
    peak = float(np.max(exact_fn(xf * UM)))
    dev = float(np.max(np.abs(C - ex))) / peak
    sd = None
    if sd_ref is not None:
        sd = math.sqrt(trap(x ** 2 * C, x) / trap(C, x)) * UM
        sd = sd / sd_ref - 1
    return dev, sd


@pytest.mark.parametrize("adapter", ENGINES)
def test_R90a_half_gaussian(mhub, adapter):
    p = p_r90a()
    D = D_int("B", 1000.0)
    s = math.sqrt((R90A["s0"] * UM) ** 2 + 2 * D * 1800.0)
    dev, sd = _r90_check(run(mhub, adapter, p), lambda x: half_gauss(x, R90A["Q"], s), sd_ref=s)
    print(f"R90a {adapter}: max dev {100 * dev:.4f} % of peak (limit 1 %), std dev {100 * sd:+.4f} % (limit 0.5 %)")
    assert dev <= 0.01
    assert abs(sd) <= 0.005


@pytest.mark.parametrize("adapter", ENGINES)
def test_R90b_gaussian_with_image(mhub, adapter):
    p = p_r90b()
    D = D_int("B", 1000.0)
    s = math.sqrt((R90B["s0"] * UM) ** 2 + 2 * D * 1800.0)
    dev, _ = _r90_check(run(mhub, adapter, p), lambda x: gauss_image(x, R90B["Rp"] * UM, R90B["Q"], s))
    print(f"R90b {adapter}: max dev {100 * dev:.4f} % of peak (limit 1 %)")
    assert dev <= 0.01


# ================================================================================================ R91
@pytest.mark.parametrize("adapter", ENGINES)
def test_R91_erfc(mhub, adapter):
    from scipy.special import erfc
    p = p_r91()
    Cs, t = 1e19, 1800.0
    D = D_int("B", 1000.0)
    res = run(mhub, adapter, p)
    x, C = profile(res)
    ex = Cs * erfc(x * UM / (2 * math.sqrt(D * t)))
    dev = float(np.max(np.abs(C - ex))) / Cs
    Qex = 2 * Cs * math.sqrt(D * t / math.pi)
    dq = scalar(res, "anneal.dose") * 1e-4 / Qex - 1
    print(f"R91 {adapter}: max dev {100 * dev:.4f} % of Cs (limit 1 %), dose {100 * dq:+.4f} % (limit 1 %)")
    assert dev <= 0.01
    assert abs(dq) <= 0.01


# ================================================================================================ R92
@pytest.mark.parametrize("adapter", ENGINES)
def test_R92_successive_steps(mhub, adapter):
    p = p_r90a(steps=((1000.0, 900.0), (1100.0, 600.0)))
    Dt = D_int("B", 1000.0) * 900.0 + D_int("B", 1100.0) * 600.0                  # eq. (14)
    s = math.sqrt((R90A["s0"] * UM) ** 2 + 2 * Dt)
    res = run(mhub, adapter, p)
    dev, sd = _r90_check(res, lambda x: half_gauss(x, R90A["Q"], s), sd_ref=s)
    print(f"R92 {adapter}: max dev {100 * dev:.4f} % of peak (limit 1 %), std dev {100 * sd:+.4f} % (limit 0.5 %)")
    assert dev <= 0.01
    assert abs(sd) <= 0.005
    assert scalar(res, "anneal.Dt") * 1e4 == pytest.approx(Dt, rel=ktol(3.5, 1000.0))   # item 4: summed over steps


# ================================================================================================ R93
R93_CASES = {"R90a": p_r90a, "R90b": p_r90b, "R92": lambda: p_r90a(((1000.0, 900.0), (1100.0, 600.0))),
             "R94_As": lambda: p_r94("As"), "R94_P": lambda: p_r94("P"), "R94_B": lambda: p_r94("B"),
             "R94_As_image": lambda: p_r94("As", form="image"), "R94_P_image": lambda: p_r94("P", form="image"),
             "R94_B_image": lambda: p_r94("B", form="image"),
             "R94_As_intrinsic": lambda: p_r94("As", "intrinsic"), "R94_P_intrinsic": lambda: p_r94("P", "intrinsic"),
             "R96": p_r96}


@pytest.mark.parametrize("adapter", ENGINES)
@pytest.mark.parametrize("case", list(R93_CASES))
def test_R93_dose_kept(mhub, case, adapter):
    p = R93_CASES[case]()
    res = run(mhub, adapter, p)
    want = own_initial_dose(p, adapter)
    got = scalar(res, "anneal.dose") * 1e-4
    rel = abs(got / want - 1)
    print(f"R93 {case} {adapter}: dose rel diff {rel:.2e} (limit 1e-6)")
    assert rel <= 1e-6


# ================================================================================================ R94
@pytest.mark.parametrize("form", ["plain", "image"])
@pytest.mark.parametrize("dopant", ["As", "P", "B"])
def test_R94_engines_agree_fermi(mhub, dopant, form):
    p = p_r94(dopant, form=form)
    ra, rb = (run(mhub, a, p) for a in ENGINES)
    xa, Ca = profile(ra)
    xb, Cb = profile(rb)
    assert np.allclose(xa, xb, rtol=0, atol=1e-9)
    peak = max(Ca.max(), Cb.max())
    dprof = float(np.max(np.abs(Ca - Cb))) / peak
    ja, jb = scalar(ra, "anneal.junction_depth"), scalar(rb, "anneal.junction_depth")
    assert ja is not None and jb is not None
    dj = abs(ja / jb - 1)
    dq = abs(scalar(ra, "anneal.dose") / scalar(rb, "anneal.dose") - 1)
    di = abs(own_initial_dose(p, ENGINES[0]) / own_initial_dose(p, ENGINES[1]) - 1)
    print(f"R94 {dopant} {form}: initial doses on the engines' own points differ by {di:.2e}; profiles {100 * dprof:.4f} % of peak (limit 1 %), xj {ja * 1e9:.2f} / {jb * 1e9:.2f} nm "
          f"diff {100 * dj:.4f} % (limit 1 %), doses {dq:.2e} (limit 1e-6)")
    assert dprof <= 0.01
    assert dj <= 0.01
    # amended by the lead (A16.10): the doses are compared through each engine's own conservation; the difference
    # between the engines may not exceed the difference of their starting doses (sampling at k dx against
    # (k + 1/2) dx) by more than 1e-6
    assert dq <= di + 1e-6


# ================================================================================================ R95
@pytest.mark.parametrize("dopant", ["B", "P", "As", "Sb"])
def test_R95_low_concentration_limit(dopant):
    from simws.process import diffusivity as Dm
    T_C = 1000.0
    ni = ni_own(T_C)
    C = 1e-6 * ni                                                               # C << ni: n/ni - 1 = 5e-7
    sum19 = sum(arr(d0, e, T_C) for d0, e in CHARGED[dopant].values())          # n = ni
    tol = ktol(max(e for _, e in CHARGED[dopant].values()) + 0.605, T_C) + 1e-6
    assert float(Dm.fermi_D(dopant, C, 0.0, TK(T_C))) == pytest.approx(sum19, rel=tol)
    h = float(Dm.field_factor(C, 0.0, Dm.ni_cm3(TK(T_C))))
    assert abs(h - 1) <= 1e-6
    p = problem(dopant, {"depth_um": [0.0, 0.1], "concentration_cm3": [1e18, 0.0]}, [(T_C, 60.0)], model="fermi")
    assert float(Dm.K(p, C, TK(T_C))) == pytest.approx(sum19, rel=tol)


@pytest.mark.parametrize("dopant", ["B", "P", "As", "Sb"])
def test_R95_high_concentration_h_tends_to_2(dopant):
    from simws.process import diffusivity as Dm
    T_C = 1000.0
    ni = Dm.ni_cm3(TK(T_C))
    r = np.array([1.0, 10.0, 100.0, 1e3, 1e4])
    h = np.asarray(Dm.field_factor(r * ni, 0.0, ni), float)
    assert np.all(np.diff(h) > 0) and np.all(h < 2)
    assert abs(h[-1] - 2) <= 1e-6
    p = problem(dopant, {"depth_um": [0.0, 0.1], "concentration_cm3": [1e18, 0.0]}, [(T_C, 60.0)], model="fermi")
    Ceff = 1e4 * ni
    K = float(Dm.K(p, Ceff, TK(T_C)))
    D = float(Dm.fermi_D(dopant, Ceff, 0.0, TK(T_C)))
    assert K / D == pytest.approx(2.0, abs=1e-6)                               # K = D_eff h
    Down, _ = fermi_own(dopant, Ceff, 0.0, "acceptor", T_C)
    assert D == pytest.approx(float(Down), rel=ktol(max(e for _, e in CHARGED[dopant].values()) + 0.605, T_C))


@pytest.mark.parametrize("adapter", ENGINES)
def test_R95_low_concentration_engines_give_the_eq19_gaussian(mhub, adapter):
    """Checker's end-to-end form of the low-concentration limit: R90(a) at Q = 1e6 cm^-2 (peak ~1e10 << ni) with
    "fermi" is the Gaussian with D = the eq. (19) sum (n = ni); gated with the R90 tolerances."""
    T_C, t, Q = 1000.0, 1800.0, 1e6
    p = problem("B", tab(lambda x: half_gauss(x, Q, R90A["s0"] * UM), 1.0), [(T_C, t)], model="fermi")
    D = sum(arr(d0, e, T_C) for d0, e in CHARGED["B"].values())
    s = math.sqrt((R90A["s0"] * UM) ** 2 + 2 * D * t)
    dev, sd = _r90_check(run(mhub, adapter, p), lambda x: half_gauss(x, Q, s), sd_ref=s)
    print(f"R95 low-C {adapter}: max dev {100 * dev:.4f} % of peak, std dev {100 * sd:+.4f} %")
    assert dev <= 0.01 and abs(sd) <= 0.005


@pytest.mark.parametrize("adapter", ENGINES)
@pytest.mark.parametrize("dopant", ["As", "P"])
def test_R95_fermi_junction_deeper(mhub, dopant, adapter):
    jf = scalar(run(mhub, adapter, p_r94(dopant, "fermi")), "anneal.junction_depth")
    ji = scalar(run(mhub, adapter, p_r94(dopant, "intrinsic")), "anneal.junction_depth")
    print(f"R95 {dopant} {adapter}: xj fermi {jf * 1e9:.2f} nm, intrinsic {ji * 1e9:.2f} nm")
    assert jf > ji


# ================================================================================================ R96
@pytest.mark.parametrize("adapter", ENGINES)
def test_R96_handout_design(mhub, adapter):
    p = p_r96()
    D1100, D950 = D_int("B", 1100.0), D_int("B", 950.0)
    Dt = D1100 * R96["t"]
    xj_cf = math.sqrt(4 * Dt * math.log(R96["Q"] / (math.sqrt(math.pi * Dt) * R96["CB"]))) / UM
    res = run(mhub, adapter, p)
    xj = scalar(res, "anneal.junction_depth") * 1e6
    d = xj / xj_cf - 1
    print(f"R96 {adapter}: xj {xj:.4f} um, closed form {xj_cf:.4f} um ({100 * d:+.3f} %, limit 1 %); "
          f"handout 3 um ({100 * (xj / 3 - 1):+.2f} %); D(1100 C) {D1100:.3e} (handout 1.5e-13), "
          f"D(950 C) {D950:.3e} (handout 4.2e-15)")
    assert abs(d) <= 0.01


# ================================================================================================ R97
def _diode_with_profiles(terms, base=None):
    from simws.spec.library import pn_diode_1d
    from simws.spec.schema import Problem
    p = base if base is not None else pn_diode_1d()
    d = p.model_dump(mode="json")
    d["device"]["profiles"] = terms
    return Problem.model_validate(d)


def _gauss_term(species, axis, x0, x1, peak, mu, s, n=400):
    x = np.linspace(x0, x1, n)
    return {"species": species, "axis": axis, "position_um": x.tolist(),
            "concentration_cm3": (peak * np.exp(-(x - mu) ** 2 / (2 * s * s))).tolist()}


def _devsim_net_doping(p):
    import devsim
    from simws.engines import devsim_engine
    devsim_engine.build(p)
    out = []
    for r in p.device.regions:
        net = np.asarray(devsim.get_node_model_values(device="d", region=r.name, name="NetDoping"), float)
        x = np.asarray(devsim.get_node_model_values(device="d", region=r.name, name="x"), float) / UM
        y = None
        if p.device.dimension == 2:
            y = np.asarray(devsim.get_node_model_values(device="d", region=r.name, name="y"), float) / UM
        out.append((r.name, x, y, net))
    return out


def _r97_worst(p, nodes):
    """Largest per-node |NetDoping - (ND - NA)| / (ND + NA), doping_at evaluated at the node coordinates. DEVSIM
    stores cm, so a node on a box edge (e.g. y = 0.08 um) comes back as 0.08000000000000002 um after /1e-4 and would
    fall on the other side of the box rule (and a node at 0.9999999999999977 um is on DEVSIM's side of an edge at
    1.0). A node within 1e-9 um of a box edge of the description is therefore compared with doping_at at its raw
    coordinate and at the edge, and the closer of the two counts; every other node with its raw coordinate."""
    ex, ey = set(), set()
    for t in list(p.device.doping) + list(p.device.regions):
        for edges, iv in ((ex, t.box.x), (ey, t.box.y)):
            if iv is not None:
                edges.update(v for v in iv if v is not None)

    def snap(c, edges):
        c = np.array(c, float)
        for e in edges:
            c[np.abs(c - e) < 1e-9] = e
        return c

    def err(xx, yy, net):
        nd, na = (np.asarray(a) for a in (p.device.doping_at(xx) if yy is None else p.device.doping_at(xx, yy)))
        return np.abs(net - (nd - na)) / (nd + na)

    worst = 0.0
    for name, x, y, net in nodes:
        e = err(x, y, net)
        e = np.minimum(e, err(snap(x, ex), y, net))
        if y is not None:
            e = np.minimum(e, err(x, snap(y, ey), net))
            e = np.minimum(e, err(snap(x, ex), snap(y, ey), net))
        worst = max(worst, float(np.max(e)))
    return worst


def test_R97_devsim_netdoping_equals_doping_at_1d():
    p = _diode_with_profiles([_gauss_term("acceptor", "x", 0.0, 1.5, 5e18, 0.3, 0.2),
                              _gauss_term("donor", "x", 2.0, 4.0, 3e17, 3.0, 0.4)])
    worst = _r97_worst(p, _devsim_net_doping(p))
    print(f"R97 1D: worst |NetDoping - doping_at| / (ND + NA) = {worst:.2e} (limit 1e-12)")
    assert worst <= 1e-12


def test_R97_devsim_netdoping_equals_doping_at_2d_hbt():
    from simws.spec.library import npn_hbt_2d
    hbt = npn_hbt_2d()
    ext = hbt.device.extent()
    (x0, x1), (y0, y1) = ext.x, ext.y
    p = _diode_with_profiles([_gauss_term("donor", "y", y0, y1, 1e18, 0.5 * (y0 + y1), 0.2),
                              _gauss_term("acceptor", "x", x0, x1, 2e17, 0.5 * (x0 + x1), 0.3)], base=hbt)
    worst = _r97_worst(p, _devsim_net_doping(p))
    print(f"R97 2D HBT: worst |NetDoping - doping_at| / (ND + NA) = {worst:.2e} (limit 1e-12)")
    assert worst <= 1e-12


def test_R97_ids_unchanged():
    import inspect

    from simws.spec import library as L
    got = {}
    for n, f in inspect.getmembers(L, inspect.isfunction):
        if f.__module__ != L.__name__ or n.startswith("_"):
            continue
        sig = inspect.signature(f)
        if all(q.default is not inspect.Parameter.empty or q.kind in (q.VAR_POSITIONAL, q.VAR_KEYWORD)
               for q in sig.parameters.values()):
            got[n] = f().problem_id()
    assert got == BASE_IDS


# ================================================================================================ item 5
def test_item5_profile_term_rule():
    from simws.spec.schema import ProfileTerm
    t = ProfileTerm(species="donor", position_um=[0.1, 0.2, 0.3], concentration_cm3=[4.0, 2.0, 1.0])
    assert t.axis == "x"
    got = np.asarray(t.at([0.0, 0.0999, 0.1, 0.15, 0.25, 0.3, 0.3001, 1.0]))
    assert np.allclose(got, [0.0, 0.0, 4.0, 3.0, 1.5, 1.0, 0.0, 0.0])


@pytest.mark.parametrize("bad", [
    dict(position_um=[0.0], concentration_cm3=[1.0]),
    dict(position_um=[0.0, 0.0], concentration_cm3=[1.0, 1.0]),
    dict(position_um=[0.2, 0.1], concentration_cm3=[1.0, 1.0]),
    dict(position_um=[0.0, 0.1], concentration_cm3=[1.0, -1.0]),
    dict(position_um=[0.0, 0.1, 0.2], concentration_cm3=[1.0, 1.0]),
    dict(position_um=list(np.linspace(0, 1, 20001)), concentration_cm3=[1.0] * 20001),
    dict(species="neutral"), dict(axis="z"),
])
def test_item5_profile_term_refusals(bad):
    from simws.spec.schema import ProfileTerm
    d = dict(species="donor", position_um=[0.0, 0.1], concentration_cm3=[1.0, 1.0])
    d.update(bad)
    with pytest.raises(ValidationError):
        ProfileTerm.model_validate(d)


def test_item5_doping_at_adds_terms_to_boxes():
    from simws.spec.library import pn_diode_1d
    from simws.spec.schema import ProfileTerm
    base = pn_diode_1d()
    terms = [_gauss_term("acceptor", "x", 0.0, 1.5, 5e18, 0.3, 0.2), _gauss_term("donor", "x", 2.0, 4.0, 3e17, 3.0, 0.4),
             _gauss_term("donor", "x", 0.5, 2.5, 1e16, 1.0, 0.5)]
    p = _diode_with_profiles(terms)
    x = np.linspace(0, 4.0, 4001)
    nd0, na0 = (np.asarray(a) for a in base.device.doping_at(x))
    nd, na = (np.asarray(a) for a in p.device.doping_at(x))
    tnd = sum(ProfileTerm.model_validate(t).at(x) for t in terms if t["species"] == "donor")
    tna = sum(ProfileTerm.model_validate(t).at(x) for t in terms if t["species"] == "acceptor")
    assert np.allclose(nd, nd0 + tnd, rtol=1e-12, atol=0)
    assert np.allclose(na, na0 + tna, rtol=1e-12, atol=0)


def test_item5_doping_at_axis_y_on_2d():
    from simws.spec.library import npn_hbt_2d
    from simws.spec.schema import ProfileTerm
    base = npn_hbt_2d()
    term = {"species": "acceptor", "axis": "y", "position_um": [0.2, 0.6], "concentration_cm3": [1e18, 3e18]}
    p = _diode_with_profiles([term], base=base)
    x = np.array([0.1, 0.4, 0.7, 1.2, 0.3])
    y = np.array([0.1, 0.3, 0.5, 0.6, 1.0])
    nd0, na0 = (np.asarray(a) for a in base.device.doping_at(x, y))
    nd, na = (np.asarray(a) for a in p.device.doping_at(x, y))
    assert np.allclose(nd, nd0, rtol=1e-12, atol=0)
    assert np.allclose(na, na0 + ProfileTerm.model_validate(term).at(y), rtol=1e-12, atol=0)


def test_item5_empty_profiles_left_out_and_ids():
    from simws.spec.library import npn_hbt_2d, pn_diode_1d
    for p in (pn_diode_1d(), npn_hbt_2d()):
        assert "profiles" not in p.model_dump(mode="json")["device"]
    p = pn_diode_1d()
    assert p.problem_id() == BASE_IDS["pn_diode_1d"]
    q = _diode_with_profiles([_gauss_term("donor", "x", 2.0, 4.0, 3e17, 3.0, 0.4)])
    assert q.problem_id() != p.problem_id()
    assert _diode_with_profiles([]).problem_id() == p.problem_id()


def test_item5_1d_device_refuses_axis_y():
    with pytest.raises(ValidationError):
        _diode_with_profiles([{"species": "donor", "axis": "y", "position_um": [0.0, 1.0],
                               "concentration_cm3": [1e17, 1e17]}])


def test_item5_charon_refuses_profile_devsim_eb_does_not(charon_problem):
    import simws.engines as E
    p0 = charon_problem()
    assert E.unsupported("charon", p0) == [] or not any(r.startswith("doping:") for r in E.unsupported("charon", p0))
    p = _diode_with_profiles([_gauss_term("donor", "x", 0.5, 1.0, 1e17, 0.8, 0.1)], base=p0)
    reasons = E.unsupported("charon", p)
    assert any(r.startswith("doping:") for r in reasons), reasons
    assert not any(r.startswith("doping:") for r in E.unsupported("devsim", p))
    assert not any(r.startswith("doping:") for r in E.unsupported("devsim_eb", p))


# ================================================================================================ R98
def test_R98_device_and_implant_engines_refuse_anneal():
    import simws.engines as E
    p = p_r90a()
    others = [n for n in E.names() if E.family(n) != "anneal"]
    assert {E.family(n) for n in others} >= {"device", "implant"}
    for n in others:
        reasons = E.unsupported(n, p)
        assert len(reasons) == 1 and reasons[0].startswith("problem kind:"), (n, reasons)


def test_R98_anneal_engines_refuse_other_kinds():
    import simws.engines as E
    from simws.spec.library import implant_si, pn_diode_1d
    for n in ("devsim_anneal", "fipy"):
        for q in (pn_diode_1d(), implant_si()):
            reasons = E.unsupported(n, q)
            assert len(reasons) == 1 and reasons[0].startswith("problem kind:"), (n, reasons)


def test_R98_planner():
    from simws.engines.planner import plan
    import simws.engines as E
    pl = plan(p_r90a())
    assert {pl.primary, *pl.checkers} == {"devsim_anneal", "fipy"}
    for n in E.names():
        if E.family(n) != "anneal":
            assert n in pl.refused and pl.refused[n][0].startswith("problem kind:")
    pd = plan(p_r90a(), engines=["devsim", "rustbca"])
    assert pd.primary is None and set(pd.refused) == {"devsim", "rustbca"}


def test_R98_charon_refuses_profile_term(charon_problem):
    import simws.engines as E
    p = _diode_with_profiles([_gauss_term("acceptor", "x", 0.0, 0.5, 1e18, 0.1, 0.1)], base=charon_problem())
    assert any(r.startswith("doping:") for r in E.unsupported("charon", p))


# ================================================================================================ items 3 and 4
# a profile above zero through the whole thickness (the flag tests below are kept apart from the zero-tail case of
# test_item1_valid_sharp_profiles_run_ok)
POSITIVE = {"depth_um": [0.0, 0.1, 1.0], "concentration_cm3": [1e18, 1e15, 1e10]}


@pytest.mark.parametrize("adapter", ENGINES)
@pytest.mark.parametrize("T_C,t", [(750.0, 10.0), (800.0, 10.0), (1000.0, 60.0)])
def test_item1_valid_sharp_profiles_run_ok(mhub, adapter, T_C, t):
    """A valid description (item 1) with a profile that drops to zero (zero beyond the last point) must give an
    "ok" result on both engines: the concentration far ahead of the diffusion front is 0, never negative."""
    p = problem("B", {"depth_um": [0.0, 0.1], "concentration_cm3": [1e18, 0.0]}, [(T_C, t)])
    res = run(mhub, adapter, p)
    assert np.min(profile(res)[1]) >= 0
def test_item3_registry():
    import simws.engines as E
    assert "devsim_anneal" in E.names() and "fipy" in E.names()
    assert E.family("devsim_anneal") == "anneal" and E.family("fipy") == "anneal"
    assert E.code("devsim_anneal") == "devsim" and E.code("fipy") == "fipy"
    assert E.code("devsim_anneal") != E.code("fipy")
    assert E.adapter_name("devsim_anneal") == "anneal.devsim_anneal" and E.adapter_name("fipy") == "anneal.fipy"


@pytest.mark.parametrize("adapter", ENGINES)
@pytest.mark.parametrize("T_C,extrap", [(750.0, True), (800.0, False), (1200.0, False), (1250.0, True)])
def test_item3_extrapolated_outside_800_1200(mhub, adapter, T_C, extrap):
    p = problem("B", POSITIVE, [(T_C, 10.0)])
    flags = all_flags(run(mhub, adapter, p))
    assert any(f.startswith("extrapolated") for f in flags) == extrap, flags


@pytest.mark.parametrize("adapter", ENGINES)
def test_item3_extrapolated_when_one_step_outside(mhub, adapter):
    p = problem("B", POSITIVE, [(1000.0, 10.0), (1300.0, 1.0)])
    assert any(f.startswith("extrapolated") for f in all_flags(run(mhub, adapter, p)))


@pytest.mark.parametrize("adapter", ENGINES)
@pytest.mark.parametrize("model", ["intrinsic", "fermi"])
def test_item4_outputs_units_flags(mhub, adapter, model):
    p = p_r94("B", model)
    res = run(mhub, adapter, p)
    q = res.q("anneal.concentration")
    assert unit_is(q.unit, "1/m^3", "m^-3")
    assert q.axes and unit_is(q.axes[0].unit, "m")
    x = np.asarray(q.axes[0].values, float)
    assert np.allclose(x, p.output_depths_um() * 1e-6, rtol=0, atol=1e-15)    # x_k = k dx, k = 0..M
    assert unit_is(res.q("anneal.dose").unit, "1/m^2", "m^-2")
    assert unit_is(res.q("anneal.Dt").unit, "m^2")
    assert unit_is(res.q("anneal.junction_depth").unit, "m")
    C = np.asarray(q.array(), float)
    assert scalar(res, "anneal.dose") == pytest.approx(trap(C, x), rel=1e-9)  # trapezoid over the points
    assert scalar(res, "anneal.Dt") == pytest.approx(D_int("B", 1000.0) * 1800.0 * 1e-4, rel=ktol(3.5, 1000.0))
    xj = junction_own(x, C, 1e16 * 1e6)
    assert xj is not None and scalar(res, "anneal.junction_depth") == pytest.approx(xj, rel=1e-9)
    # item 2 flags, item 4 budget and citations
    assert NOT_MODELLED in res.flags
    assert (NI_FLAG in all_flags(res)) == (model == "fermi")
    assert NO_JUNCTION not in all_flags(res)
    assert "budget:none" in res.flags and "budget:model_not_included" in res.flags
    own = "devsim" if adapter == "anneal.devsim_anneal" else "fipy"
    assert "plummer2000" in res.citations and own in res.citations
    if model == "fermi":
        assert "morin1954" in res.citations


@pytest.mark.parametrize("adapter", ENGINES)
@pytest.mark.parametrize("case", ["stays_above", "starts_below", "same_type", "no_background"])
def test_item4_no_junction(mhub, adapter, case):
    flat = {"depth_um": [0.0, 1.0], "concentration_cm3": [1e17, 1e17]}
    low = {"depth_um": [0.0, 0.05, 1.0], "concentration_cm3": [5e14, 5e14, 1e10]}
    gauss = POSITIVE
    if case == "stays_above":
        p = problem("B", flat, [(1000.0, 60.0)], bg=1e16, bg_type="donor")
    elif case == "starts_below":
        p = problem("B", low, [(1000.0, 60.0)], bg=1e16, bg_type="donor")
    elif case == "same_type":
        p = problem("B", gauss, [(1000.0, 60.0)], bg=1e16, bg_type="acceptor")
    else:
        p = problem("B", gauss, [(1000.0, 60.0)])
    res = run(mhub, adapter, p)
    assert scalar(res, "anneal.junction_depth") is None
    assert NOT_MODELLED in res.flags
    if case in ("stays_above", "starts_below"):
        assert NO_JUNCTION in all_flags(res), all_flags(res)


@pytest.mark.parametrize("adapter", ENGINES)
def test_item4_junction_rule_single_step(mhub, adapter):
    """Junction from the reported points equals the rule of A16.4 (R90(a) on an n-type 1e16 background)."""
    p = problem("B", tab(lambda x: half_gauss(x, R90A["Q"], R90A["s0"] * UM), 1.0), [(1000.0, 1800.0)], bg=1e16,
                bg_type="donor")
    res = run(mhub, adapter, p)
    x, C = profile(res)
    xj = junction_own(x, C, 1e16)
    assert scalar(res, "anneal.junction_depth") * 1e6 == pytest.approx(xj, rel=1e-9)


# ================================================================================================ item 8: trust
def test_item8_trust_verified(hub):
    """The trust rule with R94's B case in the "image" form (see test_R94 for the "plain" form's doses)."""
    from simws.compare import compare_results, record
    from simws.hub.trust import assess_stored
    p = p_r94("B", form="image")
    ids = []
    for a in ENGINES:
        r = hub.submit(a, params={"problem": p.model_dump(mode="json")})
        assert r.status == "ok", r.message
        ids.append(r.id)
    for rid in ids:
        assert assess_stored(hub, rid).level == "unverified"
    a, b = ids
    for name, tol in (("anneal.junction_depth", 0.01),):   # amended by the lead (A16.10): junction depth only
        c = compare_results(hub, a, name, b, tolerance=tol)
        assert c.kind == "engine_pair" and c.passed, (name, c)
        record(hub, c)
    for rid in ids:
        t = assess_stored(hub, rid)
        assert t.level == "verified", t
        assert t.in_range


def test_item8_same_code_not_independent():
    import simws.engines as E
    assert E.code("devsim_anneal") == E.code("devsim")                         # DEVSIM's two engines share a code
    assert E.code("fipy") not in {E.code(n) for n in E.names() if n != "fipy"}


# ================================================================================================ R99
@pytest.mark.slow
def test_R99_chain(hub):
    import simws.engines as E
    from simws.chain import diode as Dd
    from simws.chain import process as P
    from simws.spec.library import implant_si
    for n in ("rustbca", "ngspice"):
        if not E.module(n).available():
            pytest.skip(f"{n} not available")
    imp = implant_si("B", 35, tilt_deg=7.0, dose_cm2=1e14, thickness_um=1.0)
    imp = imp.with_changes({"models.electronic_stopping": "empirical"})
    r = hub.submit("implant.rustbca", params={"problem": imp})
    assert r.status == "ok", r.message[-400:]
    ini = P.initial_from_implant(hub.result(r.id))
    ap = P.anneal_problem("B", ini, background_cm3=1e16, background_type="donor", steps=[(1000.0, 1800.0)],
                          model="fermi")
    res = {}
    for a in ENGINES:
        ra = hub.submit(a, params={"problem": ap})
        assert ra.status == "ok", ra.message[-400:]
        res[a] = hub.result(ra.id)
    ja, jb = (scalar(res[a], "anneal.junction_depth") for a in ENGINES)
    dj = abs(ja / jb - 1)
    dq = abs(scalar(res[ENGINES[0]], "anneal.dose") / scalar(res[ENGINES[1]], "anneal.dose") - 1)
    di = abs(own_initial_dose(ap, ENGINES[0]) / own_initial_dose(ap, ENGINES[1]) - 1)
    V = [round(v, 3) for v in np.arange(0.30, 0.6501, 0.05)]
    dp = P.diode_from_anneal(res["anneal.devsim_anneal"], ap, V=V)
    rd = hub.submit("device.devsim", params={"problem": dp})
    assert rd.status == "ok", rd.message[-400:]
    J = P.forward_J(hub.result(rd.id), dp)
    fit = Dd.fit_shockley(P.forward_V(dp), J, dp.models.temperature_K)
    area = 1e-10
    cp = Dd.circuit_from_fit(fit, area_m2=area, start=0.30, stop=0.65, step=0.05)
    rc = hub.submit("circuit.ngspice", params={"problem": cp})
    assert rc.status == "ok", rc.message[-400:]
    rcr = hub.result(rc.id)
    iq = next(q for q in rcr.quantities if q.name.startswith("circuit.i."))
    vs = np.asarray(iq.axes[0].values, float)
    i06 = abs(float(np.interp(0.6, vs, np.asarray(iq.array(), float))))
    dev06 = J[V.index(0.6)] * area
    print(f"\nR99: implant dose in profile {ini.dose_cm2():.4e} cm^-2; xj devsim {ja * 1e9:.2f} nm, fipy "
          f"{jb * 1e9:.2f} nm, diff {100 * dj:.4f} % (gate 1 %); doses differ {dq:.2e} (A16.8 asks 1e-6; not gated "
          f"here; initial doses on the engines' own points differ {di:.2e}); Js {fit['Js_A_m2']:.4e} A/m^2, N {fit['N']:.4f}; "
          f"ngspice I(0.6 V) {i06:.4e} A vs device {dev06:.4e} A ({100 * (i06 / dev06 - 1):+.3f} %)")
    assert dj <= 0.01
