"""Contract amendment A20 (24 Nov 2026): the crystal (channeling) implant in the process chain.
Acceptance checks R130 to R133 of A20.2, the band rule of A20.1 and the trust rule of A20.3.

Written by the checker from docs/ARCHITECTURE_V1.md (A20 items 0 to 3; background A15, A16, A11.4, A13.7). Only
public names are called: simws.spec.library.implant_si, the hub (adapters "implant.channeling", "implant.rustbca",
"anneal.devsim_anneal", "anneal.fipy", "device.devsim"), simws.chain.process (initial_from_implant, anneal_problem,
diode_from_anneal, forward_V, forward_J), simws.chain.diode.fit_shockley, simws.compare (compare_results, record),
simws.hub.trust.assess_stored and the registry of simws.engines (available()).

Readings of the contract made here (stated so the lead can object):
- R130 gives "depth bins 2 nm" after the boron beam only; the arsenic beam is run with 2 nm bins as well.
- "The dose of initial_from_implant" is the dose of the returned DepthProfile by the rule of A16.1 (linear between
  points, the first value held from 0 to the first point, zero beyond the last), computed here.
- R132's amorphous profiles: RustBCA, same ion, energy, dose, tilt 7 deg (no rotation: the amorphous description
  has none), the same target thickness and 2 nm bins, 20 000 ions (implant_si's default); boron with electronic
  stopping "empirical", arsenic with the default stopping.
- R133's diode: forward biases 0.30 to 0.65 V in steps of 0.05 V (as A16's R99), the ModelSpec defaults.
"""
import math

import numpy as np
import pytest

ANNEALS = ("anneal.devsim_anneal", "anneal.fipy")
UM_CM = 1e-4

# A20.2, R130: (ion, keV, dose cm^-2, tilt, rotation, target um); bins 2 nm
BEAMS = {"B": dict(E=35.0, dose=1e14, tilt=7.0, rot=30.0, T=1.5),
         "As": dict(E=120.0, dose=1e13, tilt=7.0, rot=22.0, T=1.0)}
BIN_UM = 0.002
# A20.2, R131: the anneal of each profile ("fermi", thickness 2 um)
ANNEAL = {"B": dict(bg_type="donor", T_C=1000.0, t=1800.0),
          "As": dict(bg_type="acceptor", T_C=1000.0, t=600.0)}
BG = 1e16
THICK = 2.0
V_DIODE = [round(v, 3) for v in np.arange(0.30, 0.6501, 0.05)]


# ------------------------------------------------------------------------------------------------ helpers
def need(*engines):
    import simws.engines as E
    for n in engines:
        if not E.module(n).available():
            pytest.skip(f"engine {n} not available")


def crystal_problem(ion):
    from simws.spec.library import implant_si
    b = BEAMS[ion]
    return implant_si(ion, b["E"], tilt_deg=b["tilt"], rotation_deg=b["rot"], dose_cm2=b["dose"], crystal=True,
                      thickness_um=b["T"]).with_changes({"depth_bin_um": BIN_UM})


def amorphous_problem(ion):
    from simws.spec.library import implant_si
    b = BEAMS[ion]
    p = implant_si(ion, b["E"], tilt_deg=b["tilt"], dose_cm2=b["dose"], thickness_um=b["T"])
    ch = {"depth_bin_um": BIN_UM}
    if ion == "B":
        ch["models.electronic_stopping"] = "empirical"
    return p.with_changes(ch)


@pytest.fixture(scope="module")
def mhub(tmp_path_factory):
    from simws.hub.jobs import Hub
    return Hub(tmp_path_factory.mktemp("crystal_chain_store"), mode="inprocess")


_RUNS = {}


def submit(hub, adapter, problem):
    """(id, stored result), cached per hub, adapter and problem id."""
    key = (id(hub), adapter, problem.problem_id())
    if key not in _RUNS:
        r = hub.submit(adapter, params={"problem": problem.model_dump(mode="json")})
        assert r.status == "ok", f"{adapter}: {r.status}: {r.message[-500:]}"
        _RUNS[key] = (r.id, hub.result(r.id))
    return _RUNS[key]


def dose_of(profile):
    """A16.1's rule: trapezoid over the points plus the first value held from 0 to the first point (cm^-2)."""
    x = np.asarray(profile.depth_um, float)
    c = np.asarray(profile.concentration_cm3, float)
    return float((np.sum(0.5 * (c[1:] + c[:-1]) * np.diff(x)) + c[0] * x[0]) * UM_CM)


def anneal_of(ion, initial):
    from simws.chain import process as P
    a = ANNEAL[ion]
    return P.anneal_problem(ion, initial, background_cm3=BG, background_type=a["bg_type"],
                            steps=[(a["T_C"], a["t"])], model="fermi", thickness_um=THICK)


def scalar(res, name):
    q = next((q for q in res.quantities if q.name == name), None)
    if q is None or q.value is None:
        return None
    v = float(q.value)
    return None if math.isnan(v) else v


def junction_own(x, C, Nb):
    """A16.4: the first depth where C falls to the background, linear between points."""
    if C[0] <= Nb:
        return None
    below = np.nonzero(C <= Nb)[0]
    if below.size == 0:
        return None
    i = below[0]
    return x[i - 1] + (C[i - 1] - Nb) * (x[i] - x[i - 1]) / (C[i - 1] - C[i])


def crystal_initial(hub, ion, band=0):
    from simws.chain import process as P
    need("channeling")
    _, res = submit(hub, "implant.channeling", crystal_problem(ion))
    return P.initial_from_implant(res, band=band)


def amorphous_initial(hub, ion):
    from simws.chain import process as P
    need("rustbca")
    _, res = submit(hub, "implant.rustbca", amorphous_problem(ion))
    return P.initial_from_implant(res)


def junction(hub, ion, initial, adapter):
    need(adapter.split(".")[1])
    ap = anneal_of(ion, initial)
    _, res = submit(hub, adapter, ap)
    xj = scalar(res, "anneal.junction_depth")
    assert xj is not None, f"{adapter}: no junction for {ion}"
    return xj * 1e6, res, ap


# ================================================================================================ A20.1: band rule
@pytest.mark.parametrize("ion", ["B", "As"])
@pytest.mark.parametrize("band", [-1, 0, 1])
def test_item1_band_is_concentration_plus_minus_sigma(mhub, ion, band):
    """band +1 / -1: implant.concentration plus / minus its reported sigma at every point, clipped at 0; band 0 the
    concentration itself (1/m^3 -> cm^-3, depth m -> um), on the result's own depth points."""
    from simws.chain import process as P
    need("channeling")
    _, res = submit(mhub, "implant.channeling", crystal_problem(ion))
    q = res.q("implant.concentration")
    c = np.asarray(q.array(), float) * 1e-6
    s = np.broadcast_to(np.asarray(q.uncertainty.sigma, float), c.shape) * 1e-6
    x = np.asarray(q.axes[0].values, float) * 1e6
    ini = P.initial_from_implant(res, band=band)
    want = np.clip(c + band * s, 0.0, None)
    got = np.asarray(ini.concentration_cm3, float)
    assert np.allclose(np.asarray(ini.depth_um), x, rtol=1e-12, atol=1e-15)
    assert np.allclose(got, want, rtol=1e-12, atol=1e-30 * c.max())
    clipped = int(np.sum(c + band * s < 0))
    print(f"A20.1 {ion} band {band:+d}: {clipped} of {c.size} points clipped at 0; sigma/C at the peak "
          f"{s[np.argmax(c)] / c.max():.4f}")
    if band == -1:
        assert np.all(got[c - s < 0] == 0.0)


@pytest.mark.parametrize("band", [-1, 1])
def test_item1_band_clips_at_zero(mhub, band):
    """The R130 profiles never reach C - sigma < 0, so the clip is exercised on a channeling result whose sigma
    exceeds C in the tail: phosphorus (A21), 50 keV, 1e14, tilt 10, rotation 0, 1 um, 2 nm bins."""
    from simws.chain import process as P
    from simws.spec.library import implant_si
    need("channeling")
    p = implant_si("P", 50.0, tilt_deg=10.0, rotation_deg=0.0, dose_cm2=1e14, crystal=True,
                   thickness_um=1.0).with_changes({"depth_bin_um": BIN_UM})
    _, res = submit(mhub, "implant.channeling", p)
    q = res.q("implant.concentration")
    c = np.asarray(q.array(), float) * 1e-6
    s = np.broadcast_to(np.asarray(q.uncertainty.sigma, float), c.shape) * 1e-6
    got = np.asarray(P.initial_from_implant(res, band=band).concentration_cm3, float)
    neg = c + band * s < 0
    print(f"A20.1 P band {band:+d}: {int(neg.sum())} of {c.size} points clipped")
    if band == -1:
        assert neg.any()
    assert np.all(got >= 0) and np.all(got[neg] == 0.0)
    assert np.allclose(got[~neg], (c + band * s)[~neg], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("band", [-1, 0, 1])
def test_item1_band_with_dose_from_profile(mhub, band):
    """With dose_cm2 given, the profile route: implant.profile x dose, plus / minus its sigma x dose."""
    from simws.chain import process as P
    need("channeling")
    _, res = submit(mhub, "implant.channeling", crystal_problem("B"))
    q = res.q("implant.profile")
    d = BEAMS["B"]["dose"]
    p = np.asarray(q.array(), float) * 1e-2 * d                     # 1/m per ion -> 1/cm, x cm^-2
    s = np.broadcast_to(np.asarray(q.uncertainty.sigma, float), p.shape) * 1e-2 * d
    ini = P.initial_from_implant(res, dose_cm2=d, band=band)
    assert np.allclose(ini.concentration_cm3, np.clip(p + band * s, 0.0, None), rtol=1e-12, atol=1e-30 * p.max())


def test_item1_channeling_result_goes_through_A16_chain_unchanged(mhub):
    """'Nothing else changes': band 0 without a dose is implant.concentration, as for any implant result; with the
    beam's dose given it is the same profile (concentration = dose x profile)."""
    from simws.chain import process as P
    need("channeling")
    _, res = submit(mhub, "implant.channeling", crystal_problem("B"))
    a = P.initial_from_implant(res)
    b = P.initial_from_implant(res, dose_cm2=BEAMS["B"]["dose"])
    assert np.allclose(a.concentration_cm3, b.concentration_cm3, rtol=1e-12)
    ap = anneal_of("B", a)
    assert ap.dopant == "B" and ap.models.diffusivity == "fermi" and ap.thickness_um == THICK


# ================================================================================================ R130: dose
@pytest.mark.parametrize("ion", ["B", "As"])
def test_R130_dose(mhub, ion):
    from simws.chain import process as P
    need("channeling")
    _, res = submit(mhub, "implant.channeling", crystal_problem(ion))
    d0 = BEAMS[ion]["dose"]
    doses = {b: dose_of(P.initial_from_implant(res, band=b)) for b in (-1, 0, 1)}
    rel = doses[0] / d0 - 1
    print(f"R130 {ion}: dose of initial_from_implant {doses[0]:.6e} cm^-2 against the beam's {d0:.1e} "
          f"({100 * rel:+.4f} %, limit 0.5 %); band -1 {100 * (doses[-1] / d0 - 1):+.3f} %, band +1 "
          f"{100 * (doses[1] / d0 - 1):+.3f} % (not gated)")
    assert abs(rel) <= 0.005


# ================================================================================================ R131: two anneal engines
@pytest.mark.parametrize("ion", ["B", "As"])
def test_R131_anneal_engines_agree_on_crystal_profile(mhub, ion):
    ini = crystal_initial(mhub, ion)
    xa, ra, ap = junction(mhub, ion, ini, ANNEALS[0])
    xb, rb, _ = junction(mhub, ion, ini, ANNEALS[1])
    rel = abs(xa / xb - 1)
    # each engine's junction follows A16.4's rule on its own reported profile
    for res, xj in ((ra, xa), (rb, xb)):
        q = res.q("anneal.concentration")
        x = np.asarray(q.axes[0].values, float) * 1e6
        C = np.asarray(q.array(), float) * 1e-6
        assert junction_own(x, C, BG) == pytest.approx(xj, rel=1e-9)
    print(f"R131 {ion}: junction devsim_anneal {xa * 1e3:.3f} nm, fipy {xb * 1e3:.3f} nm, differ "
          f"{100 * rel:.4f} % (limit 1 %)")
    assert rel <= 0.01


# ================================================================================================ A20.3: trust
def test_item3_trust_each_keeps_its_own_level(mhub):
    from simws.compare import compare_results, record
    from simws.hub.trust import assess_stored
    need("channeling", "devsim_anneal", "fipy")
    rid_imp, _ = submit(mhub, "implant.channeling", crystal_problem("B"))
    ini = crystal_initial(mhub, "B")
    ap = anneal_of("B", ini)
    ida, _ = submit(mhub, ANNEALS[0], ap)
    idb, _ = submit(mhub, ANNEALS[1], ap)
    assert assess_stored(mhub, rid_imp).level == "unverified"
    c = compare_results(mhub, ida, "anneal.junction_depth", idb, tolerance=0.01)
    assert c.kind == "engine_pair" and c.independent and c.passed, c
    record(mhub, c)
    ta, tb, ti = assess_stored(mhub, ida), assess_stored(mhub, idb), assess_stored(mhub, rid_imp)
    print(f"A20.3: anneal {ta.level} / {tb.level}, starting profile {ti.level} ({ti.reasons})")
    assert ta.level == "verified" and tb.level == "verified"
    assert ti.level == "unverified"


# ================================================================================================ R132: deeper junction
@pytest.mark.slow
@pytest.mark.parametrize("adapter", ANNEALS)
@pytest.mark.parametrize("ion", ["B", "As"])
def test_R132_channeling_deepens_the_junction(mhub, ion, adapter):
    xc, _, _ = junction(mhub, ion, crystal_initial(mhub, ion), adapter)
    xa, _, _ = junction(mhub, ion, amorphous_initial(mhub, ion), adapter)
    print(f"R132 {ion} {adapter}: crystal junction {xc * 1e3:.2f} nm, amorphous {xa * 1e3:.2f} nm "
          f"({100 * (xc / xa - 1):+.2f} %)")
    assert xc > xa


# ================================================================================================ R133 (reported)
@pytest.mark.slow
@pytest.mark.parametrize("ion", ["B", "As"])
def test_R133_band_junctions_reported(mhub, ion):
    xs = {}
    for band in (-1, 0, 1):
        xs[band], _, _ = junction(mhub, ion, crystal_initial(mhub, ion, band), ANNEALS[0])
    print(f"R133 {ion} (devsim_anneal): junction band -1 {xs[-1] * 1e3:.2f} nm, band 0 {xs[0] * 1e3:.2f} nm, "
          f"band +1 {xs[1] * 1e3:.2f} nm (spread {100 * (xs[1] - xs[-1]) / 2 / xs[0]:.2f} % half-width)")
    assert xs[-1] <= xs[0] <= xs[1]


def _diode(hub, ion, initial):
    from simws.chain import diode as Dd
    from simws.chain import process as P
    need("devsim")
    _, res, ap = junction(hub, ion, initial, ANNEALS[0])
    dp = P.diode_from_anneal(res, ap, V=V_DIODE)
    _, rd = submit(hub, "device.devsim", dp)
    J = P.forward_J(rd, dp)
    V = P.forward_V(dp)
    fit = Dd.fit_shockley(V, J, dp.models.temperature_K)
    return float(J[list(V).index(0.6)]), fit


@pytest.mark.slow
def test_R133_diode_reported(mhub):
    jc, fc = _diode(mhub, "B", crystal_initial(mhub, "B"))
    ja, fa = _diode(mhub, "B", amorphous_initial(mhub, "B"))
    print(f"R133 diode, B 35 keV 1e14 -> 1000 C 1800 s fermi -> DEVSIM: crystal J(0.6 V) {jc:.4e} A/m^2, Js "
          f"{fc['Js_A_m2']:.4e} A/m^2, N {fc['N']:.4f} (max rel residual {fc['max_rel_residual']:.2e}); amorphous "
          f"J(0.6 V) {ja:.4e} A/m^2, Js {fa['Js_A_m2']:.4e} A/m^2, N {fa['N']:.4f} "
          f"(max rel residual {fa['max_rel_residual']:.2e}); J ratio crystal/amorphous {jc / ja:.4f}")
    assert jc > 0 and ja > 0
