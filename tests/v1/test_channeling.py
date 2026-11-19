"""Contract amendment A15 ("A15, continued", 19 Nov 2026): channeling from the two papers the owner supplied.
Acceptance checks R80 to R87 of A15.6, plus the result rules of A15.5.

Written by the checker from docs/ARCHITECTURE_V1.md (section 2, A7, A12.4, A15 and "A15, continued") and from the two
papers (Tasch et al., J. Electrochem. Soc. 136, 810 (1989); Suzuki et al., IEEE Trans. Electron Devices 47, 44
(2000)). The implementation files (simws/radiation/channeling.py, simws/engines/channeling_engine.py, the
ImplantChanneling adapter) were not read. Only public names are called: engine "channeling" through simws.engines,
adapter "implant.channeling" through the hub, simws.spec.library.implant_si(crystal=True, rotation_deg=...), and the
ImplantResult fields named in A15.5.

The checker's own model code below implements A15 items 1 to 3 from the contract text and the papers (it is used for
R80, R83 and as the "main component alone" of R84, and to compare with the engine's profile). The contract names no
public Pearson helper, so R80 uses the checker's own Pearson integration (item 3); the engine's own Pearson code is
checked through R80e, which recovers the two components from the engine's profiles at two doses.

R82 digitises the figures with the checker's own method: the page images are taken at their native 300 dpi from the
PDFs with `pdfimages` (no resampling); the axis frame and the major ticks are found from the pixels; markers are
found as blobs (3x3 closing, hole filling, 7x7 opening) in a column window at each abscissa; overlapping markers are
split with single-marker heights measured in the same figure. Tests that need the PDFs look in the folder named by
SIMWS_CHANNELING_PAPERS (default: the build machine's scratch folder) for PDFs whose SHA-256 equals the data files'
and skip cleanly when they are absent."""
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "simws" / "data" / "channeling"
TASCH = json.loads((DATA / "tasch1989_boron.json").read_text())
SUZUKI = json.loads((DATA / "suzuki2000_arsenic.json").read_text())
PAPERS = Path(os.environ.get("SIMWS_CHANNELING_PAPERS",
                             "/tmp/claude-0/-home-claude/d0abb2e9-411f-5090-9cb2-dd3ba3452a20/scratchpad/papers"))
ADAPTER = "implant.channeling"
ENGINE = "channeling"
UM = 1e-6

# A15.2: the beam the two papers used
B_TILT, B_ROT = 7.0, 30.0
AS_TILT, AS_ROT = 7.0, 22.0
# R81 cases (keV, cm^-2)
R81_B = [(15, 2e13), (35, 5e14), (50, 2e15)]
R81_AS = [(40, 1e11), (120, 1e13), (200, 1e16)]
AS_THICK_UM = 4.0      # Suzuki's exponential-type tail at 200 keV (L ~ 167 nm) needs ~1.6 um to fall below 1e-4


# ------------------------------------------------------------------------------------------------ problem helpers
def crystal(ion, E, dose, **kw):
    from simws.spec.library import implant_si
    tilt = kw.pop("tilt_deg", B_TILT if ion == "B" else AS_TILT)
    rot = kw.pop("rotation_deg", B_ROT if ion == "B" else AS_ROT)
    kw.setdefault("thickness_um", 2.0 if ion == "B" else AS_THICK_UM)
    return implant_si(ion, E, tilt_deg=tilt, rotation_deg=rot, dose_cm2=dose, crystal=True, **kw)


def run_engine(p):
    import simws.engines as E
    return E.module(ENGINE).run(p)


def params(p):
    return {"problem": p.model_dump(mode="json")}


# ------------------------------------------------------------------------------------------------ own model: Pearson
def pearson(Rp, dRp, gamma, beta, n=40001):
    """A15.3 by the checker: integrate d ln f/dx = (x - a)/(b0 + b1 x + b2 x^2), x measured from Rp, on the interval
    around the mode where the denominator keeps its sign. Each side of the mode is mapped to w in [0, W]: towards a
    finite root r, x = r - (r - a) exp(-w) (the integrand times dx/dw stays finite at the root); towards infinity,
    x = a +- dRp sinh(w) out to 1e5 dRp (heavy type VII tails). Returns x (from Rp, ascending), the density f on x
    (normalised over the whole region) and its CDF."""
    A = 10 * beta - 12 * gamma ** 2 - 18
    b1 = -dRp * gamma * (beta + 3) / A
    a = b1
    b0 = -dRp ** 2 * (4 * beta - 3 * gamma ** 2) / A
    b2 = -(2 * beta - 3 * gamma ** 2 - 6) / A
    if b2 != 0:
        rts = np.roots([b2, b1, b0])
    elif b1 != 0:
        rts = np.array([-b0 / b1])
    else:
        rts = np.array([])
    real = [float(r.real) for r in np.atleast_1d(rts) if abs(complex(r).imag) <= 1e-12 * max(1.0, abs(r))]
    lo = max([r for r in real if r < a], default=-math.inf)
    hi = min([r for r in real if r > a], default=math.inf)
    sides = []
    for s, end in ((+1, hi), (-1, lo)):
        if math.isfinite(end):
            w = np.linspace(0.0, 30.0, n)
            x = end - (end - a) * np.exp(-w)
            dxdw = (end - a) * np.exp(-w)
        else:
            w = np.linspace(0.0, math.asinh(1e5), n)
            x = a + s * dRp * np.sinh(w)
            dxdw = s * dRp * np.cosh(w)
        g = (x - a) / (b0 + b1 * x + b2 * x * x) * dxdw
        lnf = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(w))])
        sides.append((w, x, dxdw, lnf))
    shift = max(s[3].max() for s in sides)
    mass = []
    for w, x, dxdw, lnf in sides:
        f = np.exp(lnf - shift)
        dm = f * np.abs(dxdw)
        cm = np.concatenate([[0.0], np.cumsum(0.5 * (dm[1:] + dm[:-1]) * np.diff(w))])
        mass.append((x, f, cm))
    (xr, fr, cr), (xl, fl, cl) = mass
    tot = cr[-1] + cl[-1]
    x = np.concatenate([xl[::-1], xr[1:]])
    f = np.concatenate([fl[::-1], fr[1:]]) / tot
    cdf = np.concatenate([(cl[-1] - cl)[::-1], cl[-1] + cr[1:]]) / tot
    return dict(x=x, f=f, cdf=cdf, sides=sides, shift=shift, tot=tot, a=a, lo=lo, hi=hi)


def pearson_moments(P):
    """Untruncated mean (from Rp), standard deviation, skewness and kurtosis by quadrature in w on both sides."""
    out = np.zeros(5)
    for w, x, dxdw, lnf in P["sides"]:
        f = np.exp(lnf - P["shift"]) * np.abs(dxdw) / P["tot"]
        for k in range(5):
            y = f * x ** k
            out[k] += float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(w)))
    m = out[1] / out[0]
    c2 = out[2] / out[0] - m ** 2
    c3 = out[3] / out[0] - 3 * m * out[2] / out[0] + 2 * m ** 3
    c4 = out[4] / out[0] - 4 * m * out[3] / out[0] + 6 * m ** 2 * out[2] / out[0] - 3 * m ** 4
    sd = math.sqrt(c2)
    return dict(norm=out[0], mean=m, sd=sd, skew=c3 / sd ** 3, kurt=c4 / sd ** 4)


def pearson_bins_trunc(Rp, dRp, gamma, beta, edges):
    """Truncated to depth >= 0, normalised there, averaged over the bins with these edges (same unit as Rp): mass per
    bin / bin width. Also returns the share beyond the last edge."""
    P = pearson(Rp, dRp, gamma, beta)
    F = lambda d: np.interp(np.asarray(d, float) - Rp, P["x"], P["cdf"], left=0.0, right=1.0)
    z = 1.0 - F(0.0)
    m = np.diff(F(edges)) / z
    return m / np.diff(edges), (1.0 - F(edges[-1])) / z


# ------------------------------------------------------------------------------------------------ own model: Tasch
def tasch_columns():
    t = TASCH["table_I"]
    return {E: dict(R1=t["R1_A"][i] * 1e-4, dRp1=t["dRp1_A"][i] * 1e-4, S1=t["S1"][i], K1=t["K1"][i],
                    R2=t["R2_A"][i] * 1e-4, dRp2=t["dRp2_A"][i] * 1e-4, S2=t["S2"][i], K2=t["K2"][i])
            for i, E in enumerate(t["energy_keV"])}


def tasch_params(E):
    """A15.2: Table I linearly in energy (15 to 50 keV); lengths in um."""
    cols = tasch_columns()
    Es = [15, 25, 35, 50]
    return {k: float(np.interp(E, Es, [cols[e][k] for e in Es])) for k in cols[15]}


def tasch_d1(E, dose):
    """A15.2: D1/DT linearly in log10(dose) along each of the 15, 25, 35, 50 keV curves, then linearly in energy."""
    f5 = TASCH["fig5_D1_over_DT"]
    ld = np.log10(f5["dose_cm2"][:4])
    along = [float(np.interp(math.log10(dose), ld, f5["curves"][str(e)][:4])) for e in (15, 25, 35, 50)]
    return float(np.interp(E, [15, 25, 35, 50], along))


def tasch_model_bins(E, dose, edges_um):
    p = tasch_params(E)
    f1, s1 = pearson_bins_trunc(p["R1"], p["dRp1"], p["S1"], p["K1"], edges_um)
    f2, s2 = pearson_bins_trunc(p["R2"], p["dRp2"], p["S2"], p["K2"], edges_um)
    w = tasch_d1(E, dose)
    return w * f1 + (1 - w) * f2, f1, f2, 1 - w


# ------------------------------------------------------------------------------------------------ own model: Suzuki
def suzuki_params(E):
    """A15.2: Fig. 4(a) linearly in energy (40 to 200 keV); lengths in um."""
    a = SUZUKI["fig4a"]
    Es = a["energy_keV"]
    g = lambda k, s=1e-3: float(np.interp(E, Es, a[k])) * s
    return dict(Rpm=g("Rpm_nm"), s1=g("dRp1_nm"), s2=g("dRp2_nm"), L=g("L_nm"), alpha=g("alpha", 1.0))


def suzuki_frac(E, dose):
    """A15.2: Phi_chan/Phi linearly in log10(dose) along the 40, 120, 200 keV curves of Fig. 4(b), then in energy."""
    b = SUZUKI["fig4b_channel_dose"]
    ld = np.log10(b["dose_cm2"])
    along = [float(np.interp(math.log10(dose), ld, np.array(b["curves"][str(e)]) / np.array(b["dose_cm2"])))
             for e in (40, 120, 200)]
    return float(np.interp(E, [40, 120, 200], along))


def _jhg_cum(x, Rpm, s1, s2):
    """G(x) = integral from 0 to x of the unit-height joined half Gaussian (Gibbons and Mylroie form, A15.1)."""
    from scipy.special import erf
    x = np.asarray(x, float)
    c = math.sqrt(math.pi / 2)
    left0 = s1 * c * erf(-Rpm / (math.sqrt(2) * s1))
    xl = np.minimum(x, Rpm)
    G = s1 * c * erf((xl - Rpm) / (math.sqrt(2) * s1)) - left0
    xr = np.maximum(x, Rpm)
    return G + s2 * c * erf((xr - Rpm) / (math.sqrt(2) * s2))


def _tail_cum(x, Rpm, L, alpha, x0):
    """Integral from x0 to x (x >= x0 > Rpm) of exp(-((t - Rpm)/L)^alpha)."""
    from scipy.special import gamma as G, gammainc
    a = 1.0 / alpha
    z = lambda t: ((np.asarray(t, float) - Rpm) / L) ** alpha
    return L / alpha * G(a) * (gammainc(a, z(x)) - gammainc(a, z(x0)))


def suzuki_cum(P):
    """CDFs (on [0, inf)) of n_a and n_c, eq. (3)-(8) with the squares of A15.1. Returns (cdf_a, cdf_c, kappa)."""
    Rpm, s1, s2, L, al = P["Rpm"], P["s1"], P["s2"], P["L"], P["alpha"]
    J = Rpm + s2
    uJ = math.exp(-0.5)
    TJ = math.exp(-((J - Rpm) / L) ** al)
    kappa = uJ / (uJ + TJ)                                                        # eq. (7)
    Ginf = float(_jhg_cum(np.inf, Rpm, s1, s2))
    GJ = float(_jhg_cum(J, Rpm, s1, s2))

    def Hc(x):
        x = np.asarray(x, float)
        xx = np.maximum(x, J)
        beyond = GJ + kappa * ((_jhg_cum(xx, Rpm, s1, s2) - GJ) + _tail_cum(xx, Rpm, L, al, J))
        return np.where(x <= J, _jhg_cum(x, Rpm, s1, s2), beyond)
    Hinf = GJ + kappa * ((Ginf - GJ) + _tail_cum(np.inf, Rpm, L, al, J))
    return (lambda x: _jhg_cum(x, Rpm, s1, s2) / Ginf), (lambda x: Hc(x) / Hinf), kappa, Ginf, Hinf


def suzuki_point(P, x):
    """Point values of n_a and n_c (1/um) from the formulas, for the join check of R83."""
    Rpm, s1, s2, L, al = P["Rpm"], P["s1"], P["s2"], P["L"], P["alpha"]
    _, _, kappa, Ginf, Hinf = suzuki_cum(P)
    u = math.exp(-(x - Rpm) ** 2 / (2 * (s1 if x < Rpm else s2) ** 2))
    left = u / Hinf
    right = kappa * (u + math.exp(-((x - Rpm) / L) ** al)) / Hinf if x > Rpm else left
    return u / Ginf, left, right


def suzuki_model_bins(E, dose, edges_um):
    P = suzuki_params(E)
    ca, cc, *_ = suzuki_cum(P)
    b = np.diff(edges_um)
    na = np.diff(ca(edges_um)) / b
    nc = np.diff(cc(edges_um)) / b
    c = suzuki_frac(E, dose)
    return (1 - c) * na + c * nc, na, nc, c


# ------------------------------------------------------------------------------------------------ profile helpers
def bins_of(p):
    b = p.depth_bin_um
    n = int(math.ceil(p.target.thickness_um / b - 1e-9))
    return np.arange(n + 1) * b


def depth_at_fraction(x, y, frac=1e-3):
    """First depth beyond the peak at which y falls to frac x peak; log-linear interpolation between bin centres."""
    y = np.asarray(y, float)
    k = int(np.argmax(y))
    thr = frac * y[k]
    for i in range(k, len(y) - 1):
        if y[i + 1] < thr:
            if y[i + 1] <= 0:
                return float(x[i + 1])
            t = (math.log(y[i]) - math.log(thr)) / (math.log(y[i]) - math.log(y[i + 1]))
            return float(x[i] + t * (x[i + 1] - x[i]))
    return float("nan")


def split_components(pa, ca, pb, cb):
    """Two profiles of the same energy with channeled fractions ca != cb: per bin
    P = (1 - c) f_main + c f_chan -> (f_main, f_chan)."""
    pa, pb = np.asarray(pa), np.asarray(pb)
    fc = (pa * (1 - cb) - pb * (1 - ca)) / (ca - cb)
    fm = (pa - ca * fc) / (1 - ca)
    return fm, fc


def binned_moments(x, y):
    w = y / y.sum()
    m = float((w * x).sum())
    c2 = float((w * (x - m) ** 2).sum())
    sd = math.sqrt(c2)
    return dict(mean=m, sd=sd, skew=float((w * (x - m) ** 3).sum()) / sd ** 3,
                kurt=float((w * (x - m) ** 4).sum()) / sd ** 4)


# ================================================================================================ R86: no id changes
def test_R86_ids_unchanged():
    from simws.spec.library import implant_si
    assert implant_si("P", 50).problem_id() == "5cfbc9d5e496882805061234f5349fe1"
    assert implant_si("B", 100).problem_id() == "7141925e1068b3eda07be7f312f0690d"
    d = implant_si("P", 50).model_dump(mode="json")
    assert "rotation_deg" not in d["beam"]                       # A15.4: left out when None
    assert d["target"]["structure"] == "amorphous"


def test_A15_4_description():
    from simws.spec.library import implant_si
    p = crystal("B", 35, 1e14)
    assert p.target.structure == "crystalline_si_100" and p.beam.rotation_deg == 30.0
    assert p.model_dump(mode="json")["beam"]["rotation_deg"] == 30.0
    with pytest.raises(ValueError):
        implant_si("B", 35, tilt_deg=7, rotation_deg=360.0, dose_cm2=1e14, crystal=True)
    with pytest.raises(ValueError):
        implant_si("B", 35, tilt_deg=7, rotation_deg=-1.0, dose_cm2=1e14, crystal=True)
    with pytest.raises(ValueError):
        p.with_changes({"target.structure": "crystalline_ge"})


# ================================================================================================ R80: Pearson code
def _r80_cases():
    out = []
    for tab, key in ((TASCH["table_I"], "I"), (TASCH["table_II_BF2"], "II")):
        for i, E in enumerate(tab["energy_keV"]):
            for comp in ("1", "2"):
                out.append(pytest.param(tab[f"R{comp}_A"][i], tab[f"dRp{comp}_A"][i], tab[f"S{comp}"][i],
                                        tab[f"K{comp}"][i], id=f"table{key}-{E}keV-f{comp}"))
    return out


@pytest.mark.parametrize("Rp,dRp,gamma,beta", _r80_cases())
def test_R80_pearson_moments(Rp, dRp, gamma, beta):
    """R80 with the checker's own Pearson integration (A15.3; no public helper is named): the untruncated density's
    numerical moments reproduce the table: Rp 0.5 %, dRp 1 %, gamma 0.02, beta 0.05."""
    P = pearson(float(Rp), float(dRp), float(gamma), float(beta))
    m = pearson_moments(P)
    print(f"R80 {Rp}/{dRp}/{gamma}/{beta}: mean {Rp + m['mean']:.3f} sd {m['sd']:.3f} skew {m['skew']:+.4f} "
          f"kurt {m['kurt']:.4f} region [{P['lo']:.0f}, {P['hi']:.0f}] A from Rp")
    assert abs(m["norm"] - 1) < 1e-9
    assert abs(m["mean"]) <= 0.005 * Rp
    assert abs(m["sd"] / dRp - 1) <= 0.01
    assert abs(m["skew"] - gamma) <= 0.02
    assert abs(m["kurt"] - beta) <= 0.05


@pytest.mark.parametrize("E", [15, 25, 35, 50])
def test_R80e_engine_components_boron(E):
    """Checker's extension of R80 to the engine's own Pearson code: the two components are recovered from the
    engine's profiles at 2e13 and 2e15 (P = (1 - c) f1 + c f2 per bin, c = the result's channeled_fraction). Their
    moments (truncated at the surface, 1 nm bins) must match the checker's truncated, binned components with the
    R80 tolerances; the largest bin difference is reported."""
    pa_ = crystal("B", E, 2e13).with_changes({"depth_bin_um": 0.001})
    pb_ = crystal("B", E, 2e15).with_changes({"depth_bin_um": 0.001})
    ra, rb = run_engine(pa_), run_engine(pb_)
    x = np.asarray(ra.profile_um)
    # main = non-channeled: P = (1 - c) f1 + c f2
    fm, fc = split_components(ra.profile_per_um, ra.channeled_fraction, rb.profile_per_um, rb.channeled_fraction)
    p = tasch_params(E)
    edges = bins_of(pa_)
    my1, _ = pearson_bins_trunc(p["R1"], p["dRp1"], p["S1"], p["K1"], edges)
    my2, _ = pearson_bins_trunc(p["R2"], p["dRp2"], p["S2"], p["K2"], edges)
    msgs = []
    for name, eng, mine in (("f1", fm, my1), ("f2", fc, my2)):
        me, mm = binned_moments(x, eng), binned_moments(x, mine)
        dev = float(np.max(np.abs(eng - mine)) / np.max(mine))
        print(f"R80e B {E} keV {name}: engine mean {me['mean'] * 1e4:.1f} sd {me['sd'] * 1e4:.1f} A skew "
              f"{me['skew']:+.3f} kurt {me['kurt']:.3f} | checker {mm['mean'] * 1e4:.1f} {mm['sd'] * 1e4:.1f} "
              f"{mm['skew']:+.3f} {mm['kurt']:.3f} | max bin diff {dev:.2e} of peak")
        if not (abs(me["mean"] / mm["mean"] - 1) <= 0.005 and abs(me["sd"] / mm["sd"] - 1) <= 0.01
                and abs(me["skew"] - mm["skew"]) <= 0.02 and abs(me["kurt"] - mm["kurt"]) <= 0.05):
            msgs.append(f"{name}: engine {me} checker {mm}")
    assert not msgs, msgs


# ================================================================================================ R81: normalisation
@pytest.mark.parametrize("ion,E,dose", [("B", *c) for c in R81_B] + [("As", *c) for c in R81_AS])
def test_R81_normalisation_engine(ion, E, dose):
    p = crystal(ion, E, dose)
    r = run_engine(p)
    x = np.asarray(r.profile_um)
    y = np.asarray(r.profile_per_um)
    b = p.depth_bin_um
    edges = bins_of(p)
    assert len(x) == len(edges) - 1 and np.allclose(x, 0.5 * (edges[1:] + edges[:-1]), rtol=0, atol=1e-12)
    integral = float(y.sum() * b)                       # per ion; times the dose = the integral of N
    N_int = integral * dose
    print(f"R81 {ion} {E} keV {dose:.0e}: integral of N / dose - 1 = {N_int / dose - 1:+.2e} "
          f"(diagnostics share_beyond_target {r.diagnostics.get('share_beyond_target')})")
    assert abs(N_int / dose - 1) <= 1e-4
    assert r.depth_um == [] and r.lateral_um == []      # A15.5: no ions are followed


@pytest.mark.parametrize("ion,E,dose", [("B", 15, 2e13), ("As", 200, 1e16)])
def test_R81_normalisation_adapter(hub, ion, E, dose):
    p = crystal(ion, E, dose)
    r = hub.submit(ADAPTER, params=params(p))
    assert r.status == "ok", r.message
    q = r.q("implant.concentration")
    prof = r.q("implant.profile")
    depth = np.asarray(q.axes[0].values)
    assert q.axes[0].name == "depth" and q.axes[0].unit == "m"
    b = p.depth_bin_um * UM
    tot = float(np.sum(q.to("1/m**3")) * b)             # ions per m^2
    print(f"R81 adapter {ion} {E} keV {dose:.0e}: integral of concentration / (dose x 1e4) - 1 = "
          f"{tot / (dose * 1e4) - 1:+.2e}; profile integral - 1 = {float(np.sum(prof.to('1/m')) * b) - 1:+.2e}")
    assert abs(tot / (dose * 1e4) - 1) <= 1e-4
    assert abs(float(np.sum(prof.to("1/m")) * b) - 1) <= 1e-4
    assert len(depth) == len(bins_of(p)) - 1


@pytest.mark.parametrize("ion,E,dose,thick", [
    ("B", 50, 2e15, 0.3),
    # amended by the lead (A15.8): the normalisation defect was fixed, so the recorded-FAIL marks are removed
    ("As", 200, 1e11, 0.5), ("As", 200, 1e11, 1.0)])
def test_R81_refuses_target_that_cuts_the_profile(ion, E, dose, thick):
    """'the engine refuses one that cuts off more than 1e-4 of it' (no reason prefix is stated)."""
    import simws.engines as E_
    from simws.engines.base import Unsupported
    p = crystal(ion, E, dose, thickness_um=thick)
    reasons = E_.unsupported(ENGINE, p)
    raised = None
    try:
        run_engine(p)
    except Unsupported as e:
        raised = e
    print(f"R81 thin target {ion} {E} keV {thick} um: unsupported={reasons}, run raised={raised!r}")
    assert reasons or raised is not None


# ================================================================================================ A15.5 result rules
@pytest.mark.parametrize("ion,E,dose", [("B", 35, 5e14), ("As", 120, 1e13)])
def test_A15_5_result_rules(hub, ion, E, dose):
    from simws.hub.trust import assess
    p = crystal(ion, E, dose)
    r = hub.submit(ADAPTER, params=params(p))
    assert r.status == "ok", r.message
    flags = r.flags + [f for q in r.quantities for f in q.flags]
    pub = [f for f in flags if f.startswith("implant:published_fit:")]
    assert pub and all(f.endswith("the models and statistics of the description are not used (A15)") for f in pub)
    assert any(f.startswith("budget:digitisation_only") for f in flags)
    assert ("tasch1989" if ion == "B" else "suzuki2000") in r.citations
    names = {q.name for q in r.quantities}
    assert {"implant.profile", "implant.concentration", "implant.Rp", "implant.dRp",
            "implant.channeled_fraction"} <= names
    assert assess(r).level == "unverified"
    # Rp and dRp: mean and standard deviation of the whole profile (reported against the adapter's own profile)
    prof = r.q("implant.profile")
    x = np.asarray(prof.axes[0].values)
    y = np.asarray(prof.to("1/m"))
    mean = float((x * y).sum() / y.sum())
    sd = math.sqrt(float(((x - mean) ** 2 * y).sum() / y.sum()))
    Rp, dRp = float(r.q("implant.Rp").to("m")), float(r.q("implant.dRp").to("m"))
    cf = float(r.q("implant.channeled_fraction").value)
    eng = run_engine(p)
    print(f"A15.5 {ion} {E} keV {dose:.0e}: implant.Rp {Rp / 1e-9:.3f} nm vs binned profile mean {mean / 1e-9:.3f}; "
          f"implant.dRp {dRp / 1e-9:.3f} vs {sd / 1e-9:.3f}; channeled_fraction {cf} "
          f"(engine {eng.channeled_fraction}); sigma(Rp) {r.q('implant.Rp').uncertainty.sigma}, "
          f"components {list(r.q('implant.Rp').uncertainty.components)}")
    assert cf == pytest.approx(eng.channeled_fraction, rel=1e-12)
    expected = 1 - tasch_d1(E, dose) if ion == "B" else suzuki_frac(E, dose)
    assert cf == pytest.approx(expected, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("ion,E,dose", [("B", 35, 5e14), ("B", 15, 2e13), ("As", 120, 1e13), ("As", 40, 1e16)])
def test_A15_5_sigma_of_channeled_fraction(hub, ion, E, dose):
    """A15.5: the figure values moved by their reading uncertainty (D1/DT -/+ 0.002; channel dose -/+ 5 %), half the
    spread is sigma. For the channeled fraction only that one group acts: sigma = 0.002 (boron), 0.05 x the
    fraction (arsenic)."""
    r = hub.submit(ADAPTER, params=params(crystal(ion, E, dose)))
    assert r.status == "ok", r.message
    q = r.q("implant.channeled_fraction")
    sig = float(np.asarray(q.uncertainty.sigma))
    f = float(q.value)
    expected = 0.002 if ion == "B" else 0.05 * f
    print(f"A15.5 sigma {ion} {E} keV {dose:.0e}: fraction {f:.6g} sigma {sig:.6g} (rule {expected:.6g})")
    assert sig == pytest.approx(expected, rel=1e-6)


_INTERP_FAIL = pytest.mark.xfail(strict=True, reason=(
    "recorded FAIL A15.2 (item 2 text, not an R gate): between the digitised doses the engine interpolates "
    "log10(Phi_chan) linearly in log10(dose) (straight lines on the log-log Fig. 4(b)), not Phi_chan/Phi linearly in "
    "log10(dose) as written: As 120 keV 3e14 gives 0.01246 against 0.01943 (-36 %), 160 keV 2e15 0.00198 against "
    "0.00288 (-31 %), 100 keV 3e15 -41 %, 60 keV 3e12 -0.3 %"))


@pytest.mark.parametrize("ion,E,dose", [("B", 15, 3e13), ("B", 42, 7e14), ("B", 20, 1.5e15),
                                         # amended by the lead (A15.8): interpolation now as A15.2 says
                                         ("As", 60, 3e12), ("As", 120, 3e14), ("As", 160, 2e15), ("As", 100, 3e15)])
def test_A15_2_interpolation_between_figure_points(ion, E, dose):
    """A15.2 between the digitised doses: D1/DT (boron) and Phi_chan/Phi (arsenic) linearly in log10(dose) along
    each curve and linearly in energy between the curves. The expected value is the same arithmetic on the data
    file (exact up to rounding)."""
    r = run_engine(crystal(ion, E, dose))
    exp_ = 1 - tasch_d1(E, dose) if ion == "B" else suzuki_frac(E, dose)
    print(f"A15.2 {ion} {E} keV {dose:.1e}: engine channeled fraction {r.channeled_fraction:.6g}, "
          f"contract rule {exp_:.6g} ({r.channeled_fraction / exp_ - 1:+.1%})")
    assert r.channeled_fraction == pytest.approx(exp_, rel=1e-9)


# ================================================================================================ engine vs own model
@pytest.mark.parametrize("ion,E,dose", [("B", *c) for c in R81_B] + [("As", *c) for c in R81_AS]
                         + [("B", 25, 1e14), ("As", 80, 5e13), ("As", 160, 2e15)])
def test_engine_profile_vs_checker_model_reported(ion, E, dose):
    """Reported (no contract tolerance): the engine's bin-averaged profile against the checker's own evaluation of
    A15 items 1 to 3 on the same bins."""
    p = crystal(ion, E, dose)
    r = run_engine(p)
    edges = bins_of(p)
    mine = (tasch_model_bins if ion == "B" else suzuki_model_bins)(E, dose, edges)[0]
    y = np.asarray(r.profile_per_um)
    dev = np.abs(y - mine) / mine.max()
    k = int(np.argmax(dev))
    rel = np.abs(y - mine) / np.maximum(mine, 1e-300)
    mask = mine > 1e-3 * mine.max()
    print(f"MODEL {ion} {E} keV {dose:.0e}: max |engine - checker| = {dev.max():.2e} of peak (at {r.profile_um[k]:.4f}"
          f" um); max relative where > 1e-3 of peak {rel[mask].max():.2e}; channeled fraction engine "
          f"{r.channeled_fraction:.6g} checker {(1 - tasch_d1(E, dose)) if ion == 'B' else suzuki_frac(E, dose):.6g}")


# ================================================================================================ R83: Suzuki's joins
@pytest.mark.parametrize("E", [40, 80, 120, 160, 200])
def test_R83_joins_formulas(E):
    """R83 on the checker's own evaluation of eq. (3)-(8) (the contract names no public point evaluator): n_c is
    continuous at Rpm + dRp2 within 1e-9 relative, and n_a and n_c peak at Rpm."""
    P = suzuki_params(E)
    J = P["Rpm"] + P["s2"]
    _, left, _ = suzuki_point(P, J * (1 - 1e-15))
    _, _, right = suzuki_point(P, J * (1 + 1e-15))
    assert abs(right / left - 1) <= 1e-9
    xs = np.linspace(0.0, 1.0, 200001)
    na = [suzuki_point(P, x)[0] for x in xs[::50]]
    nc = [suzuki_point(P, x)[1] if x < J else suzuki_point(P, x)[2] for x in xs[::50]]
    assert abs(xs[::50][int(np.argmax(na))] - P["Rpm"]) <= 2.5e-4
    assert abs(xs[::50][int(np.argmax(nc))] - P["Rpm"]) <= 2.5e-4


@pytest.mark.parametrize("E", [40, 80, 120, 160, 200])
def test_R83_joins_engine(E):
    """R83 on the engine: n_a and n_c recovered from the engine's profiles at 1e11 and 1e16 (1 nm bins) have their
    maximum within one depth bin of Rpm. Reported: the recovered n_c around the join against the checker's continuous
    n_c (a step at the join would show as a local deviation)."""
    pa_ = crystal("As", E, 1e11).with_changes({"depth_bin_um": 0.001})
    pb_ = crystal("As", E, 1e16).with_changes({"depth_bin_um": 0.001})
    ra, rb = run_engine(pa_), run_engine(pb_)
    x = np.asarray(ra.profile_um)
    na, nc = split_components(ra.profile_per_um, ra.channeled_fraction, rb.profile_per_um, rb.channeled_fraction)
    P = suzuki_params(E)
    b = pa_.depth_bin_um
    xa, xc = x[int(np.argmax(na))], x[int(np.argmax(nc))]
    _, my_na, my_nc, _ = suzuki_model_bins(E, 1e11, bins_of(pa_))
    J = P["Rpm"] + P["s2"]
    near = np.abs(x - J) <= 5 * b
    step = float(np.max(np.abs(nc[near] / my_nc[near] - 1)))
    print(f"R83 As {E} keV: Rpm {P['Rpm'] * 1e3:.2f} nm; argmax n_a {xa * 1e3:.2f} nm, n_c {xc * 1e3:.2f} nm "
          f"(bin {b * 1e3:.0f} nm); n_c vs checker within 5 bins of the join {J * 1e3:.2f} nm: max rel dev {step:.2e}; "
          f"overall n_a {np.max(np.abs(na - my_na)) / my_na.max():.2e}, "
          f"n_c {np.max(np.abs(nc - my_nc)) / my_nc.max():.2e}")
    assert abs(xa - P["Rpm"]) <= b
    assert abs(xc - P["Rpm"]) <= b


# ================================================================================================ R84: the papers
@pytest.mark.parametrize("E", [15, 25, 35, 50])
def test_R84_boron_fraction_falls_with_dose(E):
    doses = [2e13, 3e13, 5e13, 1e14, 2e14, 5e14, 1e15, 2e15]
    fr = [run_engine(crystal("B", E, d)).channeled_fraction for d in doses]
    print(f"R84 B {E} keV channeled fraction:", " ".join(f"{d:.0e}:{f:.4f}" for d, f in zip(doses, fr)))
    assert all(b < a for a, b in zip(fr, fr[1:]))


@pytest.mark.parametrize("E", [40, 120, 200])
def test_R84_arsenic_fraction_and_saturation(E):
    doses = [1e13, 2e13, 5e13, 1e14, 3e14, 1e15, 3e15, 1e16]
    fr = {d: run_engine(crystal("As", E, d)).channeled_fraction for d in doses}
    sat = fr[1e16] * 1e16 / (fr[1e14] * 1e14)
    print(f"R84 As {E} keV Phi_chan/Phi:", " ".join(f"{d:.0e}:{fr[d]:.4g}" for d in doses),
          f"| Phi_chan(1e16)/Phi_chan(1e14) = {sat:.3f}")
    vals = [fr[d] for d in doses]
    assert all(b < a for a, b in zip(vals, vals[1:]))
    assert sat < 1.5


@pytest.mark.parametrize("ion,E,dose", [("B", *c) for c in R81_B] + [("As", *c) for c in R81_AS])
def test_R84_tail_deeper_than_main_component(ion, E, dose):
    """The depth at which the engine's profile falls to 1e-3 of its peak exceeds that of the main component alone
    (the checker's own f1 for boron, n_a for arsenic, on the same bins)."""
    p = crystal(ion, E, dose)
    r = run_engine(p)
    x = np.asarray(r.profile_um)
    edges = bins_of(p)
    _, main, *_ = (tasch_model_bins if ion == "B" else suzuki_model_bins)(E, dose, edges)
    d_full = depth_at_fraction(x, r.profile_per_um)
    d_main = depth_at_fraction(x, main)
    print(f"R84 depth at 1e-3 of peak, {ion} {E} keV {dose:.0e}: profile {d_full * 1e3:.2f} nm, main component "
          f"alone {d_main * 1e3:.2f} nm (difference {(d_full - d_main) * 1e3:+.3f} nm)")
    assert d_full > d_main


# ================================================================================================ R87: refusals
def _reasons(p):
    import simws.engines as E
    return E.unsupported(ENGINE, p)


def _has(reasons, prefix):
    return any(r.startswith(prefix) for r in reasons)


def _layer(name, t, dens):
    return {"name": name, "thickness_um": t, "density_cm3": dens}


SI = 4.994e22


def _refusal_cases():
    from simws.spec.library import implant_si
    b = lambda **kw: crystal("B", kw.pop("E", 35), kw.pop("dose", 1e14), **kw)
    a = lambda **kw: crystal("As", kw.pop("E", 120), kw.pop("dose", 1e13), **kw)
    two = b().with_changes({"target.layers": [_layer("si", 0.5, {"Si": SI}), _layer("si2", 1.5, {"Si": SI})]})
    o_atom = {"symbol": "O", "Z": 8, "mass_amu": 15.999, "cutoff_eV": 1.5, "surface_binding_eV": 0.0,
              "bulk_binding_eV": 0.0}
    oxy = b().with_changes({"target.atoms": [b().model_dump(mode="json")["target"]["atoms"][0], o_atom],
                            "target.layers": [_layer("si", 2.0, {"Si": SI, "O": 1e21})]})
    return [
        ("tilt 0", b(tilt_deg=0.0), "angles:"), ("tilt 7.6", b(tilt_deg=7.6), "angles:"),
        ("rotation None", b(rotation_deg=None), "angles:"), ("B rotation 22", b(rotation_deg=22.0), "angles:"),
        ("B rotation 30.6", b(rotation_deg=30.6), "angles:"), ("As rotation 30", a(rotation_deg=30.0), "angles:"),
        ("As tilt 6.4", a(tilt_deg=6.4), "angles:"),
        ("P", implant_si("P", 50, tilt_deg=7, rotation_deg=30, dose_cm2=1e14, crystal=True), "ion:"),
        ("B 14 keV", b(E=14), "energy:"), ("B 51 keV", b(E=51), "energy:"), ("B 80 keV", b(E=80), "energy:"),
        ("As 39 keV", a(E=39), "energy:"), ("As 201 keV", a(E=201), "energy:"),
        ("B 1.9e13", b(dose=1.9e13), "dose:"), ("B 2.1e15", b(dose=2.1e15), "dose:"), ("B 1e16", b(dose=1e16), "dose:"),
        ("As 9e10", a(dose=9e10), "dose:"), ("As 1.1e16", a(dose=1.1e16), "dose:"),
        ("B no dose", b(dose=None), "dose:"), ("As no dose", a(dose=None), "dose:"),
        ("two layers", two, "structures:"), ("Si+O layer", oxy, "structures:"),
        ("amorphous", b().with_changes({"target.structure": "amorphous"}), "structures:"),
    ]


def test_R87_accepts_the_covered_region():
    ok = [crystal("B", 15, 2e13), crystal("B", 50, 2e15), crystal("B", 35, 1e14, tilt_deg=7.4, rotation_deg=29.6),
          crystal("As", 40, 1e11), crystal("As", 200, 1e16), crystal("As", 120, 1e13, tilt_deg=6.6, rotation_deg=22.4)]
    for p in ok:
        assert _reasons(p) == [], (p.beam, _reasons(p))


@pytest.mark.parametrize("case", range(23))
def test_R87_refusals(case):
    cases = _refusal_cases()
    assert len(cases) == 23
    name, p, prefix = cases[case]
    rs = _reasons(p)
    print(f"R87 {name}: {rs}")
    assert _has(rs, prefix), (name, rs)


def test_R87_refusal_through_hub(hub):
    p = crystal("B", 35, 1e14, tilt_deg=0.0)
    r = hub.submit(ADAPTER, params=params(p))
    print("R87 hub:", r.status, r.message)
    assert r.status == "invalid_input" and "angles:" in r.message


@pytest.mark.parametrize("engine", ["rustbca", "g4implant"])
def test_R87_bca_engines_refuse_crystal(engine):
    import simws.engines as E
    rs = E.unsupported(engine, crystal("B", 35, 1e14))
    assert _has(rs, "structures:"), rs


def test_plan_of_a_crystal_problem():
    """A7.3 planner rules applied to a crystal problem: channeling is the only implant engine that supports it."""
    from simws.engines.planner import plan
    pl = plan(crystal("B", 35, 1e14))
    print("plan:", pl.primary, pl.checkers, {k: v[:1] for k, v in pl.refused.items()})
    assert pl.primary == ENGINE and pl.checkers == []
    assert {"rustbca", "g4implant"} <= set(pl.refused)


# ================================================================================================ R82: the papers
def _find_pdf(sha):
    if not PAPERS.is_dir():
        return None
    for f in sorted(PAPERS.glob("*.pdf")):
        if hashlib.sha256(f.read_bytes()).hexdigest() == sha:
            return f
    return None


TASCH_PDF = _find_pdf(TASCH["source"]["pdf_sha256"])
SUZUKI_PDF = _find_pdf(SUZUKI["source"]["pdf_sha256"])
needs_tools = pytest.mark.skipif(not (shutil.which("pdfimages") and shutil.which("pdftotext")),
                                 reason="poppler-utils (pdfimages, pdftotext) not installed")
needs_tasch = pytest.mark.skipif(TASCH_PDF is None, reason=f"Tasch PDF with the data file's SHA-256 not in {PAPERS}")
needs_suzuki = pytest.mark.skipif(SUZUKI_PDF is None, reason=f"Suzuki PDF with the data file's SHA-256 not in {PAPERS}")


def _page_images(pdf, page):
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdfimages", "-f", str(page), "-l", str(page), "-png", str(pdf), str(Path(d) / "img")],
                       check=True, capture_output=True)
        return [np.array(Image.open(p).convert("L")) < 128 for p in sorted(Path(d).glob("img-*.png"))]


def _runs(prof, off):
    """Contiguous runs of non-zero entries: (weighted centre + off, max, length)."""
    idx = np.where(prof > 0)[0]
    if idx.size == 0:
        return []
    gs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [(float((g * prof[g]).sum() / prof[g].sum()) + off, int(prof[g].max()), len(g)) for g in gs]


def _frame(c, frac):
    out = []
    for axis in (1, 0):
        s = c.sum(axis)
        idx = np.where(s > frac * c.shape[axis])[0]
        gs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        out.append([(float((g * s[g]).sum() / s[g].sum()), int(g[0]), int(g[-1])) for g in gs])
    h, v = out
    return dict(top=h[0], bot=h[-1], left=v[0], right=v[-1])


def _opened(win, close, se):
    from scipy import ndimage as ndi
    F = ndi.binary_fill_holes(ndi.binary_closing(win, structure=np.ones((close, close))))
    return ndi.binary_opening(F, structure=np.ones((se, se)))


def _blobs(c, x0, half, r_lo, r_hi, exclude=(), se=7):
    """Marker blobs in the column window x0 +- half: 3x3 closing, hole filling, 7x7 opening; a second pass with a
    5x5 closing adds hollow markers whose outline has a 2-pixel gap (new, marker sized, mostly white inside)."""
    from scipy import ndimage as ndi
    a = int(round(x0)) - half
    win = c[r_lo:r_hi, a:a + 2 * half + 1]
    O1 = _opened(win, 3, se)
    lab1, n1 = ndi.label(O1)
    masks = [lab1 == k for k in range(1, n1 + 1)]
    lab2, n2 = ndi.label(_opened(win, 5, se))
    for k in range(1, n2 + 1):
        m = lab2 == k
        rr, cc = np.where(m)
        h, w = rr.max() - rr.min() + 1, cc.max() - cc.min() + 1
        if not (m & O1).any() and 10 <= h <= 17 and 10 <= w <= 17 and win[m].mean() < 0.5:
            masks.append(m)
    out = []
    for m in masks:
        rr, cc = np.where(m)
        b = dict(r0=int(rr.min()) + r_lo, r1=int(rr.max()) + r_lo, rc=float(rr.mean()) + r_lo,
                 cx=float(cc.mean()) + a, fill=float(win[m].mean()))
        b["h"] = b["r1"] - b["r0"] + 1
        b["mid"] = 0.5 * (b["r0"] + b["r1"])
        if any(x1 <= b["cx"] <= x2 and y1 <= b["rc"] <= y2 for x1, y1, x2, y2 in exclude):
            continue
        out.append(b)
    return sorted(out, key=lambda b: b["rc"])


_CACHE = {}


def digitise_tasch_fig5():
    """Tasch Fig. 5 (page 812 = PDF page 4, a 300 dpi bilevel scan). Frame and ticks from the pixels: the D1/DT
    ticks (1.0, 0.95, 0.9, 0.85, 0.8) left of the frame, the decade ticks (1e13 .. 1e17) below it; piecewise linear
    between ticks. At each dose the markers are taken bottom to top as 15, 25, 35, 50 keV; a fifth marker on top is
    the 80 keV circle (at 1e14 and 2e15). Returns {energy: {dose: D1/DT}} and the merged 1e16 blob's extent."""
    if "f5" in _CACHE:
        return _CACHE["f5"]
    page = [i for i in _page_images(TASCH_PDF, 4) if i.shape == (3375, 2400)][0]
    c = page[2380:3120, 1300:2330]                                   # the figure's region on the page
    f = _frame(c, 0.6)
    top, bot = f["top"][0], f["bot"][0]
    L, B = f["left"][1], f["bot"][2]
    yt = [t[0] for t in _runs(c[f["top"][1] - 6:B + 7, L - 16:L - 3].sum(1), f["top"][1] - 6)
          if t[1] >= 3 and t[2] >= 2]
    xt = [t[0] for t in _runs(c[B + 3:B + 16, L - 6:f["right"][2] + 7].sum(0), L - 6) if t[1] >= 9]
    assert len(yt) == 5 and len(xt) == 5, (yt, xt)
    yv, xd = [1.0, 0.95, 0.9, 0.85, 0.8], [13, 14, 15, 16, 17]
    val = lambda r: float(np.interp(r, yt, yv))
    excl = [(100, 60, 380, 120), (590, 380, 860, 580)]              # the title "B+ Implants" and the legend
    out = {e: {} for e in (15, 25, 35, 50, 80)}
    merged = None
    for dose in (2e13, 1e14, 5e14, 2e15, 1e16):
        x0 = float(np.interp(math.log10(dose), xd, xt))
        bl = _blobs(c, x0, 16, int(top) + 6, int(bot) - 5, exclude=excl)[::-1]       # bottom first
        if dose == 1e16:
            assert len(bl) == 1
            merged = (val(bl[0]["r1"]), val(bl[0]["r0"]))
            continue
        assert len(bl) in (4, 5), (dose, bl)
        for e, b in zip((15, 25, 35, 50, 80), bl):
            out[e][dose] = val(b["mid"])
    _CACHE["f5"] = (out, merged, dict(yticks=yt, xticks=xt))
    return _CACHE["f5"]


def _suzuki_fig(shape):
    imgs = [i for i in _page_images(SUZUKI_PDF, 3) if i.shape == shape]
    assert len(imgs) == 1
    return imgs[0]


def digitise_suzuki_fig4a():
    """Suzuki Fig. 4(a) (page 46 = PDF page 3, an embedded 772 x 588 bilevel image at 300 dpi). Frame and the major
    ticks inside it (50, 100, 150 nm left; 50, 100, 150 keV bottom); the right axis shares the frame (alpha 0.5 at
    the bottom, 1.5 at the top). Per energy: hollow blobs bottom to top are dRp1, L; filled blobs bottom to top are
    dRp2, Rp, alpha. Circles and squares: centre of the opened blob's rows. Triangles: bounding-box centre (apex to
    base, the apex followed up the raw pixels). Merged blobs are split with single-marker heights from the energies
    where all markers are separate: dRp2 under Rp at 40 keV; at 200 keV Rp under alpha, where the triangle's base is
    the narrowest row (neck) above the circle."""
    if "4a" in _CACHE:
        return _CACHE["4a"]
    c = _suzuki_fig((588, 772))
    f = _frame(c, 0.4)
    t1, b0, l1, r0 = f["top"][2], f["bot"][1], f["left"][2], f["right"][1]
    lt = [t[0] for t in _runs(c[t1 + 2:b0 - 1, l1 + 2:l1 + 10].sum(1), t1 + 2)
          if t[1] >= 6 and t[2] <= 4 and t1 + 15 < t[0] < b0 - 15]
    bt = [t[0] for t in _runs(c[b0 - 10:b0 - 1, l1 + 2:r0 - 1].sum(0), l1 + 2)
          if t[1] >= 6 and t[2] <= 4 and l1 + 15 < t[0] < r0 - 15]
    ycal = [f["top"][0]] + lt + [f["bot"][0]]
    xcal = [f["left"][0]] + bt + [f["right"][0]]
    assert len(ycal) == 5 and len(xcal) == 5, (ycal, xcal)
    nm = lambda r: float(np.interp(r, ycal, [200, 150, 100, 50, 0]))
    excl = [(140, 22, 345, 145), (345, 22, 475, 195), (505, 22, 650, 70)]   # text block, two legend boxes
    E_ = (40, 80, 120, 160, 200)
    raw = {}
    for E in E_:
        x0 = float(np.interp(E, [0, 50, 100, 150, 200], xcal))
        bl = _blobs(c, x0, 14, t1 + 3, b0 - 1, exclude=excl)
        opn = sorted([b for b in bl if b["fill"] < 0.6], key=lambda b: -b["rc"])
        fil = sorted([b for b in bl if b["fill"] >= 0.6], key=lambda b: -b["rc"])
        assert len(opn) == 2 and len(fil) in (2, 3), (E, bl)
        raw[E] = dict(x0=x0, opn=opn, fil=fil)
    clean = [E for E in E_ if len(raw[E]["fil"]) == 3]
    h_sq = float(np.median([raw[E]["fil"][0]["h"] for E in clean]))
    h_ci = float(np.median([raw[E]["fil"][1]["h"] for E in clean]))

    def apex(b):
        col, r = int(round(b["cx"])), b["r1"]
        while c[r - 1, col - 1:col + 2].any():
            r -= 1
        return r
    H_t = float(np.median([raw[E]["fil"][2]["r1"] - apex(raw[E]["fil"][2]) + 1 for E in clean]))
    out = {}
    for E in E_:
        R, o = raw[E], {}
        o["dRp1"], o["L"] = nm(R["opn"][0]["mid"]), nm(R["opn"][1]["mid"])
        fl = R["fil"]
        if len(fl) == 3:
            o["dRp2"], o["Rp"] = nm(fl[0]["mid"]), nm(fl[1]["mid"])
            arow = 0.5 * (apex(fl[2]) + fl[2]["r1"])
        else:
            lo, hi = fl
            if lo["h"] > 1.4 * h_sq:
                o["dRp2"], o["Rp"] = nm(lo["r1"] - (h_sq - 1) / 2), nm(lo["r0"] + (h_ci - 1) / 2)
                arow = 0.5 * (apex(hi) + hi["r1"])
            else:
                assert hi["h"] > 1.4 * h_ci, (E, fl)
                o["dRp2"], o["Rp"] = nm(lo["mid"]), nm(hi["r1"] - (h_ci - 1) / 2)
                a0 = int(round(R["x0"])) - 14
                rows = np.arange(int(hi["r1"] - h_ci - 4), int(hi["r1"] - h_ci + 5))
                base = int(rows[int(np.argmin([c[r, a0:a0 + 29].sum() for r in rows]))]) - 1
                arow = base - (H_t - 1) / 2
        o["alpha"] = 0.5 + nm(arow) / 200.0
        out[E] = o
    _CACHE["4a"] = (out, dict(h_sq=h_sq, h_ci=h_ci, H_t=H_t, ycal=ycal, xcal=xcal))
    return _CACHE["4a"]


def digitise_suzuki_fig4b():
    """Suzuki Fig. 4(b) (embedded 756 x 587 bilevel image). Log-log frame: dose 1e11 (left frame) to 1e16 (right),
    channel dose 1e10 (bottom) to 1e14 (top); interior decade ticks from both sides; piecewise linear in log10. At
    each dose the blobs fall into two clusters (largest vertical gap): 10 keV over 40 keV, 120 keV over 200 keV. A
    cluster that is one blob is a merged pair, split with the single-marker height (circles: the separate 10/40 keV
    markers; squares: the legend's square)."""
    if "4b" in _CACHE:
        return _CACHE["4b"]
    c = _suzuki_fig((587, 756))
    f = _frame(c, 0.4)
    t1, b0, l1, r0 = f["top"][2], f["bot"][1], f["left"][2], f["right"][1]
    top, bot, left, right = f["top"][0], f["bot"][0], f["left"][0], f["right"][0]
    maj = lambda ts, lo, hi: [t for t in ts if 6 <= t[1] <= 8 and t[2] <= 4 and lo + 15 < t[0] < hi - 15]
    li = maj(_runs(c[t1 + 2:b0 - 1, l1 + 2:l1 + 10].sum(1), t1 + 2), t1, b0)
    ri = maj(_runs(c[t1 + 2:b0 - 1, r0 - 10:r0 - 1].sum(1), t1 + 2), t1, b0)
    bi = maj(_runs(c[b0 - 10:b0 - 1, l1 + 2:r0 - 1].sum(0), l1 + 2), l1, r0)
    ys = {14: [top], 10: [bot]}
    for t in li + ri:
        k = 14 - 4 * (t[0] - top) / (bot - top)
        assert abs(k - round(k)) < 0.1
        ys.setdefault(int(round(k)), []).append(t[0])
    xs = {11: [left], 16: [right]}
    for t in bi:
        k = 11 + 5 * (t[0] - left) / (right - left)
        assert abs(k - round(k)) < 0.1
        xs.setdefault(int(round(k)), []).append(t[0])
    ydec = sorted(ys, reverse=True)
    ycal = [float(np.mean(ys[k])) for k in ydec]
    xdec = sorted(xs)
    xcal = [float(np.mean(xs[k])) for k in xdec]
    assert ydec == [14, 13, 12, 11, 10] and xdec == [11, 12, 13, 14, 15, 16], (ydec, xdec)
    lg = lambda r: float(np.interp(r, ycal, ydec))
    excl = [(150, 30, 340, 140), (510, 310, 712, 460)]              # text block and legend
    leg = _blobs(c, 545, 10, 400, 455)                                # the legend's filled square
    h_sq = float(np.median([b["h"] for b in leg]))
    doses = SUZUKI["fig4b_channel_dose"]["dose_cm2"]
    per = {}
    for D in doses:
        x0 = float(np.interp(math.log10(D), xdec, xcal))
        bl = [b for b in _blobs(c, x0, 14, t1 + 3, b0 - 1, exclude=excl)
              if b["h"] >= 10 and left - 4 <= b["cx"] <= right + 4]
        gaps = [bl[i + 1]["r0"] - bl[i]["r1"] for i in range(len(bl) - 1)]
        k = int(np.argmax(gaps)) + 1
        per[D] = (bl[:k], bl[k:])
        assert 1 <= k <= 2 and 1 <= len(bl) - k <= 2, (D, bl)
    h_ci = float(np.median([b["h"] for D in doses for b in per[D][0] if len(per[D][0]) == 2]))
    res = {"10": [], "40": [], "120": [], "200": []}
    for D in doses:
        for grp, (hn, ln), h in ((per[D][0], ("10", "40"), h_ci), (per[D][1], ("120", "200"), h_sq)):
            if len(grp) == 2:
                u, l = grp[0]["mid"], grp[1]["mid"]
            else:
                g = grp[0]
                u, l = g["r0"] + (h - 1) / 2, g["r1"] - (h - 1) / 2
            res[hn].append(10 ** lg(u))
            res[ln].append(10 ** lg(l))
    _CACHE["4b"] = (res, dict(h_ci=h_ci, h_sq=h_sq, ycal=ycal, xcal=xcal))
    return _CACHE["4b"]


@needs_tasch
@needs_suzuki
def test_R82_pdf_hashes():
    for pdf, d in ((TASCH_PDF, TASCH), (SUZUKI_PDF, SUZUKI)):
        assert hashlib.sha256(pdf.read_bytes()).hexdigest() == d["source"]["pdf_sha256"]


def _table_rows(text, header_re, n_cols):
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if re.search(header_re, l))
    head = [float(t) for t in lines[i].split()[-n_cols:]]
    rows = []
    for l in lines[i + 1:]:
        if not l.strip():
            continue
        rows.append([float(t) for t in l.split()[-n_cols:]])
        if len(rows) == 8:
            break
    return head, rows


@needs_tools
@needs_tasch
@pytest.mark.parametrize("table,key,ncol,header", [
    ("I", "table_I", 5, r"Parameter\s+15\s+25\s+35\s+50\s+80\s*$"),
    ("II", "table_II_BF2", 7, r"Parameter\s+15\s+25\s+35\s+45\s+55\s+65\s+80\s*$")])
def test_R82_typed_tables_exact(table, key, ncol, header):
    """Tasch Tables I and II: the data file's values equal the PDF's text layer exactly (pdftotext -layout; rows R1,
    dRp1, S1, K1, R2, dRp2, S2, K2; the last ncol numbers of each row)."""
    text = subprocess.run(["pdftotext", "-layout", str(TASCH_PDF), "-"], check=True, capture_output=True,
                          text=True).stdout
    head, rows = _table_rows(text, header, ncol)
    d = TASCH[key]
    assert head == [float(e) for e in d["energy_keV"]]
    names = ["R1_A", "dRp1_A", "S1", "K1", "R2_A", "dRp2_A", "S2", "K2"]
    for name, row in zip(names, rows):
        assert row == [float(v) for v in d[name]], (table, name, row, d[name])


@needs_tools
@needs_tasch
def test_R82_tasch_fig5():
    mine, merged, cal = digitise_tasch_fig5()
    f5 = TASCH["fig5_D1_over_DT"]
    worst = (0.0, None)
    lines = []
    for e in (15, 25, 35, 50):
        for j, dose in enumerate(f5["dose_cm2"][:4]):           # A15.2: points used = 2e13 .. 2e15
            v, w = mine[e][dose], f5["curves"][str(e)][j]
            lines.append(f"{e} keV {dose:.0e}: checker {v:.4f} file {w:.4f} ({v - w:+.4f})")
            if abs(v - w) > abs(worst[0]):
                worst = (v - w, (e, dose))
    print("R82 Fig. 5\n  " + "\n  ".join(lines))
    print(f"  80 keV (not used): checker {mine[80]}; file {f5['curves']['80']}")
    print(f"  1e16 merged blob (not used): {merged[0]:.4f} to {merged[1]:.4f}; file 25/35/50 keV "
          f"{[f5['curves'][k][4] for k in ('25', '35', '50')]}")
    print(f"  worst deviation {worst[0]:+.4f} at {worst[1]}; ticks {cal}")
    assert abs(worst[0]) <= 0.005


@needs_tools
@needs_suzuki
def test_R82_suzuki_fig4a():
    mine, cal = digitise_suzuki_fig4a()
    a = SUZUKI["fig4a"]
    fails, lines = [], []
    key = {"Rp": "Rpm_nm", "dRp1": "dRp1_nm", "dRp2": "dRp2_nm", "L": "L_nm", "alpha": "alpha"}
    worst = {}
    for i, E in enumerate(a["energy_keV"]):
        for k, fk in key.items():
            v, w = mine[E][k], a[fk][i]
            tol = 0.02 if k == "alpha" else max(2.0, 0.03 * w)
            lines.append(f"{E} keV {k}: checker {v:.3f} file {w:.3f} ({v - w:+.3f}, tol {tol:.2f})")
            if abs(v - w) / tol > worst.get(k, (0, None))[0]:
                worst[k] = (abs(v - w) / tol, E, v - w)
            if abs(v - w) > tol:
                fails.append(lines[-1])
    print("R82 Fig. 4(a)\n  " + "\n  ".join(lines))
    print("  worst |dev|/tol per quantity:", worst, "| calibration", cal)
    assert not fails, fails


@needs_tools
@needs_suzuki
def test_R82_suzuki_fig4b():
    mine, cal = digitise_suzuki_fig4b()
    b = SUZUKI["fig4b_channel_dose"]
    fails, lines, worst = [], [], (0.0, None)
    for e in ("10", "40", "120", "200"):
        for j, D in enumerate(b["dose_cm2"]):
            v, w = mine[e][j], b["curves"][e][j]
            rel = v / w - 1
            used = e != "10"                                  # A15.2 uses the 40, 120 and 200 keV curves
            lines.append(f"{e} keV {D:.0e}: checker {v:.4g} file {w:.4g} ({rel:+.1%}){'' if used else ' (not used)'}")
            if used and abs(rel) > abs(worst[0]):
                worst = (rel, (e, D))
            if used and abs(rel) > 0.10:
                fails.append(lines[-1])
    print("R82 Fig. 4(b)\n  " + "\n  ".join(lines))
    print(f"  worst used deviation {worst[0]:+.1%} at {worst[1]}; calibration {cal}")
    assert not fails, fails


# ================================================================================================ R85 (reported)
@pytest.mark.slow
def test_R85_reported(hub):
    """R85, reported, no gate. RustBCA through the hub, 20 000 ions, seed 1, amorphous silicon, tilt 7 deg (the
    papers' beam; the description has no rotation for an amorphous target): boron with electronic_stopping
    "empirical", implant.Rp against Table I R1; arsenic with Lindhard-Scharff, the peak position (2 nm bins, 9-bin
    moving average, vertex of a parabola through its maximum and +-3 bins) and implant.Rp against Fig. 4(a) Rpm."""
    pytest.importorskip("libRustBCA")
    from simws.spec.library import implant_si
    cols = tasch_columns()
    lines = []
    for E in (15, 25, 35, 50):
        p = implant_si("B", E, tilt_deg=7.0).with_changes({"models.electronic_stopping": "empirical"})
        r = hub.submit("implant.rustbca", params=params(p))
        assert r.status == "ok", r.message
        rp = float(r.q("implant.Rp").to("m")) / 1e-9
        s = float(r.q("implant.Rp").uncertainty.sigma) / 1e-9
        R1 = cols[E]["R1"] * 1e3
        lines.append(f"B {E} keV: RustBCA empirical Rp {rp:.1f} +- {s:.1f} nm; Tasch R1 {R1:.1f} nm "
                     f"({rp / R1 - 1:+.1%})")
    a = SUZUKI["fig4a"]
    for i, E in enumerate(a["energy_keV"]):
        p = implant_si("As", E, tilt_deg=7.0).with_changes({"depth_bin_um": 0.002})
        r = hub.submit("implant.rustbca", params=params(p))
        assert r.status == "ok", r.message
        prof = r.q("implant.profile")
        x = np.asarray(prof.axes[0].values) / 1e-9
        y = np.asarray(prof.to("1/m"))
        k = int(np.argmax(y))
        ys = np.convolve(y, np.ones(9) / 9, mode="same")             # 18 nm moving average against the noise
        j = int(np.argmax(ys))
        cf = np.polyfit(x[j - 3:j + 4], ys[j - 3:j + 4], 2)
        peak = -cf[1] / (2 * cf[0])
        rp = float(r.q("implant.Rp").to("m")) / 1e-9
        lines.append(f"As {E} keV: RustBCA LS peak {peak:.1f} nm (raw histogram max bin {x[k]:.0f} nm), mean Rp "
                     f"{rp:.1f} nm; Suzuki Rpm {a['Rpm_nm'][i]:.1f} nm (peak {peak / a['Rpm_nm'][i] - 1:+.1%})")
    print("R85\n  " + "\n  ".join(lines))
