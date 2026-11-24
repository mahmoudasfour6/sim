"""Contract amendment A25 (24 Nov 2026): total dose on the IBM 5AM SiGe HBT, interface traps and oxide charge at the EB
spacer and the STI sidewall. Acceptance checks R189 to R199 (A25.5) as clarified by "A25, clarifications" (a) to (e).

Written by the checker from the contract text (A25 items 0 to 6, the clarifications (a) to (e), A23b closed), sections
2, 6 and 7 of docs/ARCHITECTURE_V1.md and simws/spec. The new code is called only through its public names
(simws.spec.schema.Interface / DeviceSpec.interfaces, simws.spec.library.ibm5am_hbt(spacer, sti, mode),
simws.engines.devsim_engine.run / interface_currents / _build, simws.engines.charon_engine.unsupported,
simws.validation.catalog, simws.validation.ibm5am_tid) and the data files' documented content. Every gated quantity
(dIB/IB0 at a VBE or at a JC, the device scaling, interpolation, crossings, ideality, the measured values read off the
traces) is computed here from the model's raw per-cm arrays and the traced points; the lead's own helpers (ratio, at,
vbe_at_jc, ideality) are only printed next to ours.

Tiers. Fast (not slow): R190 (data, catalog, roles, ingest, the trace checks), R191 (the interface feature on a small
2D diode, a few seconds per solve), R192 (refusals), the 5AM's interfaces and inverse mode as A25.3 describes them, the
A25.4 knob ranges, the perimeter factor, ids unchanged by A25. Slow: every 5AM total-dose solve (R189, R193 to R199);
they need simws/data/validation/ibm5am_tid.json (A25.4's fit) and are skipped without it. Each mode and temperature is
one module-scoped dose_curve call (all doses the slow tests need), computed once.

Readings of the contract made here (stated so the lead can object):
- "within a factor F" is max(model / measured, measured / model) <= F; "within x %" is |a / b - 1| <= x.
- Device scaling (A25.3): the 2D half cell does not depend on L (A23.2: L enters only through x 2L and x P_E), so one
  per-cm sweep serves every chip. One device: I(V) = I'_pre(V) x 2L + (I'_post(V) - I'_pre(V)) x P_E, P_E = 2 (L + W_E),
  W_E = 0.5 um, for IB and for the collector-side current alike; dIB/IB0 = (I'_B,post - I'_B,pre) / I'_B,pre x
  P_E / (2L). The per-cm base current is the sweep's IB_cm; the collector-side current (the collector in forward mode,
  the physical emitter in inverse mode, A25.3) is taken from the sweep's JC_A_um2 (JC = I' x 2L / AE, AE = W_E x L).
- dIB/IB0 at a VBE: the excess and IB0 interpolated linearly in ln between the sweep's points (0.6 V is a grid point).
- dIB/IB0 "at JC = J": the base current after the dose at the VBE where the post-dose device JC reaches J, over the
  pre-irradiation base current at the VBE where the pre-irradiation JC reaches J (Sutton's fixed-JC extraction); each
  crossing is the first from below, linear in ln JC. The lead's at(pre, post, vbe_at_jc(pre, J)) (which assumes dose
  does not move JC) is printed beside it.
- Doses: the model is run at the traced dose of each point (rad(SiO2) / 1e6 = Mrad(SiO2), as plotted; A25.5 R196 says
  the X-ray doses are taken as Sutton plots them, so no conversion). Doses that agree within 0.5 % (the same nominal
  dose read off two figures, e.g. 0.3148 and 0.3152 Mrad) share one model run (the response's log-log slope is <= 1.3,
  so <= 0.7 % in dIB/IB0, far inside a factor 2). Fig. 39 (3 Mrad) and Fig. 72's post-irradiation curves (1, 3,
  6 Mrad) carry no traced dose: their stated nominal doses are used.
- The calibration points (A25.1, "1 and 6 Mrad") are the traced points nearest 1 and 6 Mrad (1.05 and 6.297 Mrad on
  the trace; every traced dose sits about 5 % above Sutton's nominal 0.3 / 0.6 / 1 / 3 / 6). R194's "other doses,
  0.1 to 3 Mrad" is every other point of the VBE = 0.6 V series (0.3148, 0.6297 and 3.148 Mrad, the last being the
  nominal 3 Mrad point). R194's log-log slope is between the 0.3148 and 3.148 Mrad points (model and measured).
- R190 "combined trace error": clarification (e) gives each trace's error as about 2 %; the ratio of two traces carries
  2 % + 2 %, so the gate is |ln(dJB / (dIB/IB0) / JB0)| <= 0.04 at every dose, with JB0 the median of the series'
  ratios (a free constant per series; the median keeps one outlier from moving it). Clarification (e) records the JC
  series as a FAIL (strict xfail here); the VBE series must pass. Fig. 22's assignment (clarification (d)) is checked
  as stated: the curve whose factor to Figs. 25 / 26's JB0 at 0.6 V is nearer (in ln) the forward factor (Figs.
  23 / 24's JB0 over Fig. 72's pre-irradiation JB at 0.6 V) is IB.
- R190 "ingest labels them so": every dose series of Figs. 24, 26, 33 and 73 through catalog.ingest_series; each
  stored set carries exactly the points point_roles gives its role, with their values; Figs. 23 / 25 (trace checks
  only) must not be calibration or validation data and are refused by ingest. Figs. 22, 39 and 72's post-irradiation
  curves are checked for loading, source and role only.
- R191 (clarification (a)): the 2D diode as described (n+ 1e19 for y < 0.5 um over p 1e17, 1 x 3 um, cathode on y_min,
  anode on y_max, the interface on x_max over y 0.6 to 2.9 um), Boltzmann, constant SRH tau 1e-6 s, no Auger, no
  band-gap narrowing (so n_ie = n_i), constant mobility (1000 / 400 cm^2/Vs: any mobility serves a balance); forward
  bias 0.5 V on the anode (low injection: n at the surface is ~1e12 against p 1e17). (b) gates: the hole balance
  I_p(anode) + I_p(cathode) = q sum(URecomb x NodeVolume) + the reported q integral U_s dl within 1 % with and without
  an interface (N_it 1e10, 3e10, 1e11, and 1e11 with N_ot 1e11); the reported interface current against the contract's
  formula evaluated here from the solved n, p on the boundary nodes (sigma v_th N_it (n p - n_i^2) / (n + p + 2 n_i),
  v_th = 1e7 cm/s, each node's dual length intersected with the span) within 1 % ("what the engine reports is what it
  solves"); linearity: I_s / N_it at 3e10 and 1e11 against 1e10 within 2 %; zero bias: |I_s| <= q x
  Numerics.absolute_error (the solver's absolute resolution, carriers / s per cm). (c) at equilibrium, N_ot 1e11, no
  traps: sum((p - n + NetDoping) x NodeVolume) changes by -N_ot x 2.3 um within 1 %. (a) "no interfaces": the diode's
  id and its currents computed by this tree and by the tree of commit ee80a7c (v1 update 14, before any A25 code)
  in two subprocesses are identical (bit for bit); the 5AM's ids likewise.
- R192: a refusal is a ValueError / ValidationError when the description is built, or a non-empty DEVSIM capability
  answer. "An interface on a contact's nodes": an interface whose span overlaps a contact's span on the same side by
  more than a point; an interface that only touches a contact's end (as the 5AM spacer touches the emitter at
  x = 0.21 um, clarification (b)) is accepted. Charon's reason must be the contract's sentence verbatim.
- R189: Fig. 22's IB is the curve fig22.assignment names (clarification (d)); the model's JB = |IB'| x 2L / AE (L = 1)
  and JC from the inverse sweep's JC_A_um2, at every traced VBE of each curve in 0.5 to 0.8 V (ln interpolation of the
  model, never extrapolated); IB gated (factor 3), IC reported.
- R197: n = 1 / (V_t d ln dIB / dVBE) from a central difference over +-10 mV of the 3 Mrad sweep (the 300 K grid holds
  VBE +- 10 mV around 0.55, 0.6 and 0.7 V); the measured n at a VBE is the traced point within 2 mV of it, else the
  trace interpolated linearly between its neighbours. One model n per mode (no source dependence) is gated against
  each source.
- R198: the model's JB after the dose against Fig. 72's post-irradiation 300 K JB at every traced VBE in 0.45 to
  0.75 V; the pre-irradiation model JB against Fig. 72's pre-irradiation JB in the same window is printed first.
- R199: Fig. 73's 77 K and 300 K points (gated, factor 2), and "Figs. 72 and 73": Fig. 72's 77 K post-irradiation
  curves give the measured dIB/IB0 at JC = 10 nA/um^2 as JB_post(V*) / JB_pre(V*) - 1 with V* where Fig. 72's 77 K
  pre-irradiation JC reaches 10 nA/um^2 (gated too, separately). The 77 K grid is V* rounded to 10 mV +- 80 mV in
  20 mV steps (as R174). The contract expects 77 K to FAIL (no temperature-dependent build-up); those tests are not
  marked xfail because no result has been recorded yet.
- Not tested here (no API named for it): A25.6's hub record of the knobs' file hash.
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
Q_E = 1.602176634e-19           # C
K_B = 1.380649e-23              # J/K

# ------------------------------------------------------------------------------------------- the contract's numbers
TID_RANGES = {"k_F": (1e9, 1e13), "k_I": (1e9, 1e13), "k_Q": (0.0, 1e13)}      # A25.4, cm^-2 per Mrad(SiO2)
W_E_UM = 0.5                                                                   # A25.3, drawn
SPACER_X = (0.21, 0.36)                                                        # A25.3
STI_X, STI_BELOW_XB = 1.2, 0.2                                                 # A25.3
SIGMA_DEFAULT = 1e-15                                                          # A25.2
VTH_300 = 1e7                                                                  # A25.2, cm/s at 300 K
CHARON_REASON = "interfaces: interface traps and oxide charge (A25) are not translated to Charon"
A23B = {"s_um": 0.80, "k_E": 1.0, "k_B": 0.681, "k_P": 1.822, "S_p": 6.24e5}   # A23b closed (3 digits)

R190_TRACE_ERR = 0.02                    # clarification (e): each trace's error, about 2 %
R190_COMBINED = 2 * R190_TRACE_ERR       # the ratio of two traces (reading in the docstring)
R191_BAL_TOL, R191_LIN_TOL, R191_GAUSS_TOL = 0.01, 0.02, 0.01
R191_V = 0.5
R191_NIT = (1e10, 3e10, 1e11)
R191_NOT = 1e11
R191_SPAN = (0.6, 2.9)
R193_F, R194_F, R195_F, R196_F, R198_F, R199_F, R189_F = 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0
R197_TOL, R197_VBE, R197_DOSE = 0.3, (0.55, 0.6, 0.7), 3.0
R194_SLOPE_REF = {"forward": 1.0, "inverse": 0.3}
R198_WIN, R198_DOSES = (0.45, 0.75), (1.0, 3.0, 6.0)
R189_WIN = (0.5, 0.8)
VBE_FIX, JC_FIX, JC_77 = 0.6, 1e-6, 10e-9       # V, A/um^2 (1 uA/um^2), A/um^2 (10 nA/um^2)
L_SMALL, L_BIG = 1.0, 2.5                       # um: the 0.5 x 1.0 chip (Figs. 22-26, 72, 73), 0.5 x 2.5 (Figs. 33, 39)
DOSE_MERGE = 0.005

PRE_A25_COMMIT = "ee80a7c"         # v1 update 14 (on GitHub and on the laptop): the tree before any A25 code or data
PDF_DIRS = [Path("/tmp/claude-0/-home-claude/dcddde37-0a9f-534f-8d49-446206203045/scratchpad/dl"),
            Path("/mnt/user-data/attachments/dcddde37-0a9f-534f-8d49-446206203045")]

A25_FIGS = ["sutton2004.fig22", "sutton2004.fig23", "sutton2004.fig24", "sutton2004.fig25", "sutton2004.fig26",
            "sutton2007.fig33", "sutton2007.fig39", "prakash2006.fig73"]
DOSE_SERIES = [("sutton2004.fig24", "VBE_0.6V"), ("sutton2004.fig24", "JC_1uA_um2"),
               ("sutton2004.fig26", "VBE_0.6V"), ("sutton2004.fig26", "JC_1uA_um2")] + \
              [("sutton2007.fig33", f"{s}/{m}") for s in ("Co60", "Xray_10keV", "proton_63MeV")
               for m in ("forward", "inverse")] + \
              [("prakash2006.fig73", "300.0"), ("prakash2006.fig73", "77.0")]
ROLES = ("calibration", "validation", "trend", "input")


# ------------------------------------------------------------------------------------------------ small helpers
def C():
    from simws.validation import catalog
    return catalog


def T():
    from simws.validation import ibm5am_tid
    return ibm5am_tid


def M():
    from simws.validation import ibm5am
    return ibm5am


def lib():
    from simws.spec import library
    return library


def factor(m, ref):
    if not (m > 0 and ref > 0) or not (math.isfinite(m) and math.isfinite(ref)):
        return math.inf
    return max(m / ref, ref / m)


def vt(Tk):
    return K_B * Tk / Q_E


def sorted_xy(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.argsort(x, kind="stable")
    return x[o], y[o]


def interp_ln(xq, x, y):
    """y at xq, linear in ln|y| between points; nan outside [x0, xn] or where a neighbour is <= 0."""
    x, y = sorted_xy(x, y)
    xq = float(xq)
    if xq < x[0] - 1e-12 or xq > x[-1] + 1e-12:
        return float("nan")
    i = int(np.searchsorted(x, xq))
    for j in range(len(x)):
        if abs(x[j] - xq) <= 1e-9:
            return float(y[j]) if y[j] > 0 else float("nan")
    a, b = i - 1, i
    if y[a] <= 0 or y[b] <= 0:
        return float("nan")
    w = (xq - x[a]) / (x[b] - x[a])
    return float(math.exp(math.log(y[a]) + w * (math.log(y[b]) - math.log(y[a]))))


def first_crossing(V, I, target):
    """First V where I reaches target from below, linear in ln I; None if never reached."""
    V, I = np.asarray(V, float), np.abs(np.asarray(I, float))
    for i in range(1, len(V)):
        if I[i - 1] < target <= I[i]:
            a, b = math.log(I[i - 1]), math.log(I[i])
            return float(V[i - 1] + (V[i] - V[i - 1]) * (math.log(target) - a) / (b - a))
    return None


def pfac(L_um):
    """P_E / (2 L), A25.3."""
    return 2.0 * (L_um + W_E_UM) / (2.0 * L_um)


def nearest_index(x, v):
    x = np.asarray(x, float)
    return int(np.argmin(np.abs(np.log(x) - math.log(v))))


# ================================================================================================== FAST TIER
# ---------------------------------------------------------------------------------------------- R190 (data)
def test_R190_rows_and_roles():
    cat = C().catalog()
    ids = [d["id"] for d in cat["datasets"]]
    for i in A25_FIGS + ["sutton2004.figs21_26", "sutton2007.figs33_39", "sutton2009.fig72"]:
        assert i in ids, i
    # A25.1: roles as A23 set them
    assert C().role("sutton2004.figs21_26") == "calibration"
    assert C().role("sutton2007.figs33_39") == "validation"
    assert C().role("prakash2006.fig73") == "validation"
    # (a) Fig. 22: validation, under sutton2004.figs21_26
    assert C().role("sutton2004.fig22") == "validation"
    assert "sutton2004.fig22" in (C().entry("sutton2004.figs21_26").get("parts") or [])
    for i in ("sutton2004.fig24", "sutton2004.fig26", "sutton2007.fig33", "sutton2007.fig39"):
        assert C().role(i) == "validation", i
    # Figs. 23 / 25 are used only to check the traces: neither calibration nor validation data
    for i in ("sutton2004.fig23", "sutton2004.fig25"):
        assert C().role(i) not in ("calibration", "validation"), (i, C().role(i))
        assert C().role(i) in ROLES
    # the calibration points: the VBE = 0.6 V series of Figs. 24 and 26 at 1 and 6 Mrad, four numbers, nothing else
    n_cal = 0
    for ds, path in DOSE_SERIES:
        s = C().series(ds, path)
        roles = list(C().point_roles(ds, path))
        dose = np.asarray(s["dose_rad_SiO2"], float)
        assert len(roles) == len(dose), (ds, path)
        want = ["validation"] * len(dose)
        if ds in ("sutton2004.fig24", "sutton2004.fig26") and path == "VBE_0.6V":
            for nominal in (1e6, 6e6):
                want[nearest_index(dose, nominal)] = "calibration"
        assert roles == want, (ds, path, roles, want)
        n_cal += roles.count("calibration")
        print(f"[R190] {ds} {path}: roles {roles}")
    for path in ("VBE_0.6V", "JC_1uA_um2"):
        for ds in ("sutton2004.fig23", "sutton2004.fig25"):
            assert "calibration" not in list(C().point_roles(ds, path)) and \
                   "validation" not in list(C().point_roles(ds, path)), (ds, path)
    assert n_cal == 4, n_cal
    print(f"[R190] {n_cal} calibration points (Figs. 24 / 26, VBE = 0.6 V, 1 and 6 Mrad); every other point validation")


def _check_axes(ds, block):
    axes = block.get("axes") or {}
    assert axes, ds
    for name, ax in axes.items():
        assert len(ax["value"]) == 2 and len(ax["page_pt"]) == 2 and ax["scale"] in ("linear", "log10"), (ds, name)


def test_R190_every_trace_loads_with_units_source_and_method():
    for ds in A25_FIGS + ["sutton2009.fig72"]:
        e = C().entry(ds)
        for k in ("id", "paper", "read_from", "device", "conditions", "role"):
            assert str(e.get(k, "")).strip(), (ds, k)
        block, src = C().figure(ds)
        assert str(src.get("cite", "")).strip() and str(src.get("url", "")).startswith("http"), ds
        sha = str(src.get("pdf_sha256", ""))
        assert len(sha) == 64 and all(ch in "0123456789abcdef" for ch in sha), (ds, sha)
        assert str(block.get("method", "")).startswith("vector"), (ds, block.get("method"))
        assert "precision" in block["method"] or "width" in block["method"], ds
        _check_axes(ds, block)
    # dose series (Figs. 23 to 26, 33, 73): doses in rad(SiO2), positive finite values, error bars min <= value <= max
    for ds, path in DOSE_SERIES + [("sutton2004.fig23", p) for p in ("VBE_0.6V", "JC_1uA_um2")] + \
            [("sutton2004.fig25", p) for p in ("VBE_0.6V", "JC_1uA_um2")]:
        s = C().series(ds, path)
        d, v = np.asarray(s["dose_rad_SiO2"], float), np.asarray(s["value"], float)
        assert len(d) == len(v) >= 4 and np.all(d > 0) and np.all(v > 0) and np.all(np.isfinite(v)), (ds, path)
        assert np.all(np.diff(d) > 0), (ds, path)
        for k in ("min", "max"):
            if k in s:
                assert len(s[k]) == len(v), (ds, path, k)
        if "min" in s and "max" in s:
            lo, hi = np.asarray(s["min"], float), np.asarray(s["max"], float)
            m = np.isfinite(lo) & np.isfinite(hi)
            assert np.all(lo[m] <= v[m] * (1 + 1e-9)) and np.all(hi[m] >= v[m] * (1 - 1e-9)), (ds, path)
    # (b): Figs. 24 / 26 have both series; the error bars of Fig. 24 were drawn
    for ds in ("sutton2004.fig24", "sutton2004.fig26"):
        for path in ("VBE_0.6V", "JC_1uA_um2"):
            assert "min" in C().series(ds, path) and "max" in C().series(ds, path), (ds, path)
    # (c) Fig. 33: three sources, both modes; (f) Fig. 73: 300 K and 77 K
    for s in ("Co60", "Xray_10keV", "proton_63MeV"):
        for m in ("forward", "inverse"):
            assert len(C().series("sutton2007.fig33", f"{s}/{m}")["value"]) >= 4
    for t in ("300.0", "77.0"):
        assert len(C().series("prakash2006.fig73", t)["value"]) >= 4
    # (d) Fig. 39: n vs VBE for the three sources, both modes
    for s in ("Co60", "Xray_10keV", "proton_63MeV"):
        for m in ("forward", "inverse"):
            f = C().series("sutton2007.fig39", f"{s}/{m}")
            V, n = np.asarray(f["VBE_V"], float), np.asarray(f["n"], float)
            assert len(V) == len(n) >= 10 and np.all(np.isfinite(n)) and np.all(n > 0), (s, m)
            assert V.min() <= 0.551 and V.max() >= 0.73, (s, m, V.min(), V.max())
    # (a) Fig. 22: two curves and the assignment
    f22 = C().figure("sutton2004.fig22")[0]
    a = f22["assignment"]
    assert {a["IB"], a["IC"]} == {"upper", "lower"}, a
    for c in ("upper", "lower"):
        V, J = np.asarray(f22[c]["VBE_V"], float), np.asarray(f22[c]["J_A_um2"], float)
        assert len(V) == len(J) > 10 and np.all(J > 0) and V.min() < 0.5 and V.max() > 0.8, c
    # (e) Fig. 72: the post-irradiation curves 1, 3, 6 Mrad at 300 K and 77 K, beside the pre-irradiation ones
    f72 = C().figure("sutton2009.fig72")[0]
    assert f72.get("curves"), "the pre-irradiation curves stay"
    post = f72["post_irradiation"]
    have = {(round(c["dose_rad_SiO2"] / 1e6, 3), float(c["T_K"])) for c in post}
    want = {(D, Tk) for D in (1.0, 3.0, 6.0) for Tk in (300.0, 77.0)}
    assert want <= have, (want - have)
    for c in post:
        V, J = np.asarray(c["VBE_V"], float), np.asarray(c["J_A_um2"], float)
        assert len(V) == len(J) > 10 and np.all(J > 0) and np.all(np.isfinite(V)), (c["dose_rad_SiO2"], c["T_K"])
    print(f"[R190] {len(A25_FIGS) + 1} figures load with a vector method, axes and a source; Fig. 72 post-irradiation "
          f"curves {sorted(have)}")


def test_R190_sha256_matches_the_thesis():
    found = {}
    for d in PDF_DIRS:
        if d.is_dir():
            for p in d.glob("*.pdf"):
                found[hashlib.sha256(p.read_bytes()).hexdigest()] = p
    if not found:
        pytest.skip("the thesis PDF is not on this machine (the field is checked above)")
    sha = C().figure("sutton2004.fig24")[1]["pdf_sha256"]
    print(f"[R190] sutton2009.json pdf_sha256 {sha[:12]}...: {found[sha].name if sha in found else 'NO MATCH'}")
    assert sha in found


@pytest.mark.parametrize("ds,path", DOSE_SERIES)
def test_R190_ingest_labels_as_the_catalog_says(hub, ds, path):
    from simws.hub.trust import assess_stored
    s = C().series(ds, path)
    dose, val = np.asarray(s["dose_rad_SiO2"], float), np.asarray(s["value"], float)
    roles = np.asarray(list(C().point_roles(ds, path)))
    out = C().ingest_series(hub, ds, path)
    assert set(out) == set(roles.tolist()), (ds, path, set(out), set(roles.tolist()))
    sha = C().figure(ds)[1]["pdf_sha256"]
    for role, r in out.items():
        assert r.status == "ok" and "measured" in r.flags, (ds, path, role)
        assert assess_stored(hub, r.id).level == "measured"
        assert r.provenance["measured"].get("pdf_sha256") == sha
        m = roles == role
        for q in r.quantities:
            assert q.label == role, (ds, path, q.name, q.label, role)
            assert str(q.source or "").strip(), (ds, q.name)
            x = np.asarray(q.axes[0].values, float)
            np.testing.assert_allclose(np.sort(x), np.sort(dose[m]), rtol=1e-12, atol=0)
            xs, ys = sorted_xy(x, q.array())
            np.testing.assert_allclose(ys, val[m][np.argsort(dose[m])], rtol=1e-12, atol=0)
        print(f"[R190] ingest {ds} {path}: {role} set of {int(m.sum())} points, labelled {role}")


@pytest.mark.parametrize("ds", ["sutton2004.fig23", "sutton2004.fig25"])
def test_R190_trace_check_figures_are_not_ingested(hub, ds):
    with pytest.raises(ValueError):
        C().ingest_series(hub, ds, "VBE_0.6V")


def _trace_check(ratio_ds, djb_ds, path):
    """|ln(dJB / (dIB/IB0) / JB0)| per dose, JB0 the median ratio (reading in the docstring)."""
    a, b = C().series(ratio_ds, path), C().series(djb_ds, path)
    da, db = np.asarray(a["dose_rad_SiO2"], float), np.asarray(b["dose_rad_SiO2"], float)
    assert len(da) == len(db) and np.allclose(da, db, rtol=1e-3), (ratio_ds, djb_ds, path)
    r = np.asarray(b["value"], float) / np.asarray(a["value"], float)
    jb0 = float(np.median(r))
    dev = np.abs(np.log(r / jb0))
    return da, dev, jb0


@pytest.mark.parametrize("pair", [("sutton2004.fig24", "sutton2004.fig23"), ("sutton2004.fig26", "sutton2004.fig25")])
def test_R190_figs23_25_consistent_with_24_26_vbe_series(pair):
    d, dev, jb0 = _trace_check(pair[0], pair[1], "VBE_0.6V")
    print(f"[R190] {pair[1]} / {pair[0]} VBE = 0.6 V: JB0 {jb0:.4g} (as printed); |dln| per dose "
          f"{np.round(dev, 4).tolist()} at {(d / 1e6).round(4).tolist()} Mrad; worst {dev.max():.4f} "
          f"(limit {R190_COMBINED})")
    assert dev.max() <= R190_COMBINED


@pytest.mark.parametrize("pair", [("sutton2004.fig24", "sutton2004.fig23"), ("sutton2004.fig26", "sutton2004.fig25")])
@pytest.mark.xfail(strict=True, reason="clarification (e): the JC = 1 uA/um^2 series of Figs. 23 / 24 agree within 0.6 % "
                   "except 0.3 Mrad (14 %), of 25 / 26 within 16 %: a property of the published points, a recorded FAIL")
def test_R190_figs23_25_consistent_with_24_26_jc_series(pair):
    d, dev, jb0 = _trace_check(pair[0], pair[1], "JC_1uA_um2")
    print(f"[R190] {pair[1]} / {pair[0]} JC = 1 uA/um^2: JB0 {jb0:.4g}; |dln| per dose {np.round(dev, 4).tolist()}; "
          f"worst {dev.max():.4f} (limit {R190_COMBINED})")
    assert dev.max() <= R190_COMBINED


def test_R190_fig22_assignment_as_clarification_d():
    _, _, jb0_f = _trace_check("sutton2004.fig24", "sutton2004.fig23", "VBE_0.6V")
    _, _, jb0_i = _trace_check("sutton2004.fig26", "sutton2004.fig25", "VBE_0.6V")
    f72 = C().figure("sutton2009.fig72")[0]
    pre300 = next(c for c in f72["curves"] if float(c["T_K"]) == 300.0)
    jb72 = interp_ln(VBE_FIX, pre300["VBE_V_B"], pre300["JB_A_um2"])
    f_fwd = jb0_f / jb72
    f22 = C().figure("sutton2004.fig22")[0]
    f_inv = {c: jb0_i / interp_ln(VBE_FIX, f22[c]["VBE_V"], f22[c]["J_A_um2"]) for c in ("upper", "lower")}
    nearer = min(f_inv, key=lambda c: abs(math.log(f_inv[c] / f_fwd)))
    print(f"[R190] Fig. 22 assignment: forward factor JB0(23/24) / JB(Fig. 72, 0.6 V) {f_fwd:.3f}; inverse "
          f"JB0(25/26) / curve at 0.6 V: upper {f_inv['upper']:.3f}, lower {f_inv['lower']:.3f}; nearer: {nearer}; "
          f"assignment IB = {f22['assignment']['IB']}")
    assert f22["assignment"]["IB"] == nearer and f22["assignment"]["IC"] != nearer


# ---------------------------------------------------------------------------------------------- R191 (the feature)
def _diode(interfaces=(), analysis=None):
    """Clarification (a)'s 2D diode: n+ 1e19 to 0.5 um over p 1e17, 1 x 3 um, contacts on y_min / y_max."""
    from simws.spec.schema import Box, Contact, DCSweep, DeviceSpec, DopingTerm, MeshHints, ModelSpec, Problem, Region
    kw = {"interfaces": list(interfaces)} if interfaces else {}
    dev = DeviceSpec(
        name="a25_diode_2d", dimension=2,
        regions=[Region(name="si", material="silicon", box=Box(x=(0.0, 1.0), y=(0.0, 3.0)))],
        contacts=[Contact(name="cathode", region="si", side="y_min"), Contact(name="anode", region="si", side="y_max")],
        doping=[DopingTerm(species="donor", concentration_cm3=1e19, box=Box(x=(None, None), y=(None, 0.5))),
                DopingTerm(species="acceptor", concentration_cm3=1e17, box=Box(x=(None, None), y=(0.5, None)))],
        mesh=MeshHints(x_lines=[(0.0, 0.1), (0.9, 0.02), (1.0, 0.01)],
                       y_lines=[(0.0, 0.05), (0.45, 0.01), (0.5, 0.005), (0.6, 0.02), (2.9, 0.05), (3.0, 0.05)]),
        description="A25 R191 diode (clarification (a))", **kw)
    m = ModelSpec(statistics="boltzmann", mobility="constant", mu_n_cm2_Vs=1000.0, mu_p_cm2_Vs=400.0, srh="constant",
                  tau_n_s=1e-6, tau_p_s=1e-6, auger=False, band_gap_narrowing="off")
    an = analysis if analysis is not None else DCSweep(contact="anode", values_V=[0.0, R191_V])
    return Problem(device=dev, models=m, analysis=an)


def _ifc(N_it=0.0, N_ot=0.0, span=R191_SPAN, side="x_max", name="side", **kw):
    from simws.spec.schema import Interface
    return Interface(name=name, region="si", side=side, span=span, N_it_cm2=N_it, N_ot_cm2=N_ot, **kw)


def _ni():
    from simws.materials import load
    from simws.materials.semiconductor import intrinsic_density
    return float(intrinsic_density(load("silicon"), 300.0))


_NODES = {}


def _nodes(problem, V):
    """The solved node values after the equilibrium solve and a ramp of the anode to V (cached per id and V)."""
    key = (problem.problem_id(), V)
    if key not in _NODES:
        import devsim

        from simws.engines import devsim_engine as D
        st = D._build(problem)
        if V:
            st.ramp("anode", V, 0.05)
        reg = st.region

        def g(n):
            return np.asarray(devsim.get_node_model_values(device="d", region=reg, name=n), float)
        out = {n: g(n) for n in ("x", "y", "NodeVolume", "URecomb", "Electrons", "Holes", "NetDoping")}
        out["Ip"] = {c: float(devsim.get_contact_current(device="d", contact=c, equation="HoleContinuityEquation"))
                     for c in ("anode", "cathode")}
        out["Is"] = {k: float(v) for k, v in D.interface_currents(problem, reg).items()}
        out["used"] = dict(getattr(st, "used", {}) or {})
        _NODES[key] = out
    return _NODES[key]


def _us_closed_form(nd, N_it, span_um=R191_SPAN, x_side_cm=1e-4, sigma=SIGMA_DEFAULT, Tk=300.0):
    """q x sum U_s x l over the boundary nodes, U_s = sigma v_th N_it (n p - n_i^2) / (n + p + 2 n_i) (A25.2), l the
    node's dual length on the side intersected with the span (A/cm)."""
    ni = _ni()
    on = np.abs(nd["x"] - x_side_cm) < 1e-12
    y = nd["y"][on]
    o = np.argsort(y)
    y, n, p = y[o], nd["Electrons"][on][o], nd["Holes"][on][o]
    lo_c, hi_c = span_um[0] * 1e-4, span_um[1] * 1e-4
    mids = np.concatenate(([y[0]], 0.5 * (y[1:] + y[:-1]), [y[-1]]))
    lo = np.maximum(mids[:-1], lo_c)
    hi = np.minimum(mids[1:], hi_c)
    length = np.clip(hi - lo, 0.0, None)
    s = sigma * VTH_300 * math.sqrt(Tk / 300.0) * N_it
    us = s * (n * p - ni ** 2) / (n + p + 2 * ni)
    return Q_E * float(np.sum(us * length)), float(length.sum())


_RUNS = {}


def _run(problem):
    pid = problem.problem_id()
    if pid not in _RUNS:
        from simws.engines import devsim_engine
        _RUNS[pid] = devsim_engine.run(problem)
    return _RUNS[pid]


def _is_reported(r, name="side"):
    d = r.diagnostics.get("interface_current_A_per_cm")
    assert d is not None and name in d, ("diagnostics interface_current_A_per_cm", list(r.diagnostics))
    return [float(v) for v in d[name]]


def test_R191a_no_interfaces_same_id_and_currents():
    pytest.importorskip("devsim")
    from simws.spec.schema import Problem
    p0 = _diode()
    d = p0.model_dump(mode="json")
    assert "interfaces" not in d["device"], "the serializer drops an empty interface list (A25.2)"
    d2 = json.loads(json.dumps(d))
    d2["device"]["interfaces"] = []
    p1 = Problem.model_validate(d2)
    assert p1.problem_id() == p0.problem_id()
    assert Problem.model_validate_json(p0.model_dump_json()).problem_id() == p0.problem_id()
    from simws.engines import devsim_engine
    r0, r1 = devsim_engine.run(p0), devsim_engine.run(p1)
    assert r0.currents == r1.currents
    with_traps = _diode([_ifc(N_it=1e11)])
    assert with_traps.problem_id() != p0.problem_id()
    print(f"[R191a] no interfaces: id {p0.problem_id()[:12]} (empty list gives the same id); currents "
          f"{ {k: v[-1] for k, v in r0.currents.items()} } A/cm")


_RUN_SCRIPT = r"""
import json, sys
from simws.spec.schema import Problem
from simws.engines import devsim_engine
p = Problem.model_validate(json.loads(open(sys.argv[1]).read()))
r = devsim_engine.run(p)
print("OUT " + json.dumps({"id": p.problem_id(), "currents": {k: [repr(x) for x in v] for k, v in r.currents.items()}}))
"""

_ID_SCRIPT = r"""
import json, sys
from simws.spec import library as L
k = json.loads(sys.argv[1])
out = {"base": L.ibm5am_hbt(**k).problem_id(),
       "grid": L.ibm5am_hbt(**k, VBE=[0.5, 0.55, 0.6, 0.65, 0.7], L_um=1.0).problem_id(),
       "cold": L.ibm5am_hbt(**k, VBE=[0.95, 1.0], L_um=1.0, T=77.0).problem_id()}
print("IDS " + json.dumps(out))
"""


def _pre_a25_tree(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    ok = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", PRE_A25_COMMIT + "^{commit}"], capture_output=True)
    if ok.returncode != 0:
        pytest.skip(f"commit {PRE_A25_COMMIT} not in this repository")
    old = tmp_path / "old"
    old.mkdir()
    arch = subprocess.run(["git", "-C", str(REPO), "archive", PRE_A25_COMMIT, "simws"], capture_output=True)
    assert arch.returncode == 0, arch.stderr[-500:]
    subprocess.run(["tar", "-x", "-C", str(old)], input=arch.stdout, check=True)
    now = tmp_path / "now"
    shutil.copytree(REPO / "simws", now / "simws", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return old, now


def _in_tree(tree: Path, args, tag):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(tree)
    p = subprocess.run([sys.executable, "-c"] + args, cwd=str(tree), env=env, capture_output=True, text=True,
                       timeout=600)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith(tag + " ")]
    assert p.returncode == 0 and line, (p.stdout[-1000:], p.stderr[-2000:])
    return json.loads(line[-1][len(tag) + 1:])


def test_R191a_diode_bit_identical_to_the_tree_before_A25(tmp_path):
    pytest.importorskip("devsim")
    old, now = _pre_a25_tree(tmp_path)
    f = tmp_path / "diode.json"
    f.write_text(_diode().model_dump_json())
    a = _in_tree(old, [_RUN_SCRIPT, str(f)], "OUT")
    b = _in_tree(now, [_RUN_SCRIPT, str(f)], "OUT")
    print(f"[R191a] diode without interfaces: id before A25 ({PRE_A25_COMMIT}) {a['id'][:12]}, now {b['id'][:12]}; "
          f"currents {'bit-identical' if a['currents'] == b['currents'] else 'DIFFER'}: {b['currents']}")
    assert a["id"] == b["id"] == _diode().problem_id()
    assert a["currents"] == b["currents"]


def test_5am_ids_without_interfaces_unchanged_by_A25(tmp_path):
    k = M().knobs()
    f = lib().ibm5am_hbt
    assert f(**k).problem_id() == f(**k, spacer=None, sti=None, mode="forward").problem_id()
    grid = [0.5, 0.6]
    assert f(**k, VBE=grid).problem_id() == f(**k, VBE=grid, spacer=None, sti=None).problem_id()
    assert "interfaces" not in f(**k).model_dump(mode="json")["device"]
    assert f(**k, VBE=grid, mode="inverse").problem_id() != f(**k, VBE=grid).problem_id()
    assert f(**k, spacer=(1e11, 0.0)).problem_id() != f(**k).problem_id()
    old, now = _pre_a25_tree(tmp_path)
    a = _in_tree(old, [_ID_SCRIPT, json.dumps(k)], "IDS")
    b = _in_tree(now, [_ID_SCRIPT, json.dumps(k)], "IDS")
    diff = sorted(x for x in a if a[x] != b.get(x))
    print(f"[R191a] 5AM ids (A23b knobs) before A25 ({PRE_A25_COMMIT}) and now: "
          f"{'all equal' if not diff else 'DIFFER for ' + str(diff)}")
    assert not diff, diff


@pytest.fixture(scope="module")
def r191_runs():
    pytest.importorskip("devsim")
    runs = {"none": _diode()}
    for N in R191_NIT:
        runs[N] = _diode([_ifc(N_it=N)])
    runs["it+ot"] = _diode([_ifc(N_it=1e11, N_ot=R191_NOT)])
    return runs


def test_R191b_hole_balance_with_and_without_interfaces(r191_runs):
    worst = 0.0
    for key, p in r191_runs.items():
        nd = _nodes(p, R191_V)
        ip = nd["Ip"]["anode"] + nd["Ip"]["cathode"]
        bulk = Q_E * float(np.sum(nd["URecomb"] * nd["NodeVolume"]))
        surf = sum(nd["Is"].values())
        rhs = bulk + surf
        d = abs(ip / rhs - 1)
        worst = max(worst, d)
        # the bulk term against the SRH closed form from the solved n, p (reported: what URecomb is)
        ni = _ni()
        n, pp = nd["Electrons"], nd["Holes"]
        srh = (n * pp - ni ** 2) / (1e-6 * (n + ni) + 1e-6 * (pp + ni))
        b2 = Q_E * float(np.sum(srh * nd["NodeVolume"]))
        print(f"[R191b] {key}: I_p(anode) + I_p(cathode) {ip:.6e} A/cm; q int U_bulk {bulk:.6e} + q int U_s {surf:.6e} "
              f"= {rhs:.6e}: {d * 100:.4f} % (limit {R191_BAL_TOL * 100:.0f} %); bulk SRH closed form {b2:.6e} "
              f"(engine n_i {nd['used'].get('n_i_cm3')} vs {ni:.4g})")
        assert d <= R191_BAL_TOL, key
        if key != "none":
            assert surf > 0, (key, "the interface recombines under forward bias")
        else:
            assert surf == 0.0 or not nd["Is"], nd["Is"]
    print(f"[R191b] worst balance {worst * 100:.4f} %")


def test_R191b_reported_interface_current_is_the_solved_one(r191_runs):
    for N in R191_NIT:
        p = r191_runs[N]
        nd = _nodes(p, R191_V)
        cf, length = _us_closed_form(nd, N)
        rep = nd["Is"]["side"]
        run_rep = _is_reported(_run(p))[-1]
        d = abs(rep / cf - 1)
        print(f"[R191b] N_it {N:.0e}: reported q int U_s {rep:.6e} A/cm (run diagnostics {run_rep:.6e}), from the "
              f"solved n, p on {length * 1e4:.4f} um of boundary {cf:.6e}: {d * 100:.4f} % (limit "
              f"{R191_BAL_TOL * 100:.0f} %)")
        assert abs(length * 1e4 - (R191_SPAN[1] - R191_SPAN[0])) < 1e-9
        assert d <= R191_BAL_TOL, N
        assert abs(run_rep / rep - 1) <= R191_BAL_TOL, (run_rep, rep)


def test_R191b_linear_in_N_it(r191_runs):
    base = _is_reported(_run(r191_runs[R191_NIT[0]]))[-1] / R191_NIT[0]
    for N in R191_NIT[1:]:
        per = _is_reported(_run(r191_runs[N]))[-1] / N
        d = abs(per / base - 1)
        print(f"[R191b] q int U_s / N_it at {N:.0e}: {per:.6e} against {base:.6e} at {R191_NIT[0]:.0e} "
              f"(A cm per trap): {d * 100:.3f} % from linear (limit {R191_LIN_TOL * 100:.0f} %)")
        assert d <= R191_LIN_TOL, N


def test_R191b_zero_bias_interface_current(r191_runs):
    floor = Q_E * r191_runs[R191_NIT[-1]].numerics.absolute_error
    for N in R191_NIT:
        r = _run(r191_runs[N])
        assert r.sweep_V[0] == 0.0
        i0, i1 = _is_reported(r)
        print(f"[R191b] N_it {N:.0e}: interface current at 0 V {i0:.3e} A/cm (limit q x absolute_error {floor:.3e}); "
              f"at {R191_V} V {i1:.3e}")
        assert abs(i0) <= floor, (N, i0)


def test_R191c_oxide_charge_gauss():
    pytest.importorskip("devsim")
    from simws.spec.schema import Equilibrium
    p0 = _diode(analysis=Equilibrium())
    p1 = _diode([_ifc(N_ot=R191_NOT)], analysis=Equilibrium())
    q = []
    for p in (p0, p1):
        nd = _nodes(p, 0.0)
        q.append(float(np.sum((nd["Holes"] - nd["Electrons"] + nd["NetDoping"]) * nd["NodeVolume"])))
    dq = q[1] - q[0]
    want = -R191_NOT * (R191_SPAN[1] - R191_SPAN[0]) * 1e-4
    d = abs(dq / want - 1)
    print(f"[R191c] N_ot {R191_NOT:.0e} cm^-2 over {R191_SPAN[1] - R191_SPAN[0]:.1f} um: change of the silicon charge "
          f"{dq:.6e} per cm (/q) against -N_ot x span {want:.6e}: {d * 100:.4f} % (limit {R191_GAUSS_TOL * 100:.0f} %)")
    assert d <= R191_GAUSS_TOL


# ---------------------------------------------------------------------------------------------- R192 (refusals)
def _refusal(make):
    """None when accepted; else the reason (spec error or DEVSIM capability answer)."""
    from pydantic import ValidationError

    from simws import engines
    try:
        p = make()
    except (ValueError, ValidationError) as e:
        return "spec: " + str(e).splitlines()[0][:160]
    r = engines.unsupported("devsim", p)
    return ("devsim: " + "; ".join(r)) if r else None


def test_R192_negative_values_refused():
    for kw in ({"N_it": -1e10}, {"N_ot": -1e10}, {"sigma_cm2": -1e-15}):
        r = _refusal(lambda: _diode([_ifc(**kw)]))
        print(f"[R192] {kw}: {r}")
        assert r, kw
    for kw in ({"N_it": 0.0}, {"N_ot": 0.0}, {"N_it": 1e12, "N_ot": 1e12, "sigma_cm2": 1e-14}):
        assert _refusal(lambda: _diode([_ifc(**kw)])) is None, kw
    print(f"[R192] (reported) sigma = 0: {_refusal(lambda: _diode([_ifc(N_it=1e10, sigma_cm2=0.0)]))}")
    k = M().knobs()
    for kw in ({"spacer": (-1e10, 0.0)}, {"spacer": (0.0, -1e10)}, {"sti": (-1e10, 0.0)}, {"sti": (0.0, -1e10)}):
        with pytest.raises(ValueError):
            lib().ibm5am_hbt(**k, **kw)
    print("[R192] negative N_it / N_ot / sigma refused (Interface and ibm5am_hbt)")


def test_R192_span_off_its_side_refused():
    for side, span in (("x_max", (0.6, 3.5)), ("x_max", (-0.1, 2.0)), ("x_min", (2.0, 3.01)), ("x_min", (-0.5, 0.2))):
        r = _refusal(lambda: _diode([_ifc(N_it=1e10, side=side, span=span)]))
        print(f"[R192] {side} span {span} (region x 0 to 1, y 0 to 3 um): {r}")
        assert r, (side, span)
    assert _refusal(lambda: _diode([_ifc(N_it=1e10, side="x_min", span=(0.0, 3.0))])) is None


def _diode_partial_anode(interfaces):
    from simws.spec.schema import Contact, Problem
    p = _diode()
    d = p.model_dump(mode="json")
    d["device"]["contacts"][1] = json.loads(Contact(name="anode", region="si", side="y_max", span=(0.0, 0.5))
                                            .model_dump_json())
    d["device"]["interfaces"] = [json.loads(i.model_dump_json()) for i in interfaces]
    return Problem.model_validate(d)


def test_R192_interface_on_a_contacts_nodes_refused():
    cases = [("y_max", None, "the whole anode side"), ("y_min", (0.2, 0.8), "inside the cathode"),
             ("y_max", (0.2, 0.8), "inside the anode")]
    for side, span, what in cases:
        r = _refusal(lambda: _diode([_ifc(N_it=1e10, side=side, span=span)]))
        print(f"[R192] interface on {side} {span} ({what}): {r}")
        assert r, (side, span)
    r = _refusal(lambda: _diode_partial_anode([_ifc(N_it=1e10, side="y_max", span=(0.4, 1.0))]))
    print(f"[R192] interface y_max (0.4, 1.0) over a partial anode (0, 0.5): {r}")
    assert r
    # touching the contact's end only (clarification (b): the 5AM spacer at the emitter edge) is accepted
    ok = _refusal(lambda: _diode_partial_anode([_ifc(N_it=1e10, side="y_max", span=(0.5, 1.0))]))
    print(f"[R192] interface y_max (0.5, 1.0) touching the partial anode's end: {ok or 'accepted'}")
    assert ok is None
    k = M().knobs()
    p = lib().ibm5am_hbt(**k, spacer=(1e11, 1e11), sti=(1e11, 1e11))
    assert _refusal(lambda: p) is None


def test_R192_1d_device_refused():
    from simws.spec.schema import Box, Contact, DeviceSpec, DopingTerm, MeshHints, Region
    from simws.spec.schema import Interface

    def make():
        return DeviceSpec(name="d1", dimension=1, regions=[Region(name="si", box=Box(x=(0.0, 1.0)))],
                          contacts=[Contact(name="a", region="si", side="x_min"),
                                    Contact(name="c", region="si", side="x_max")],
                          doping=[DopingTerm(species="acceptor", concentration_cm3=1e17)],
                          mesh=MeshHints(x_lines=[(0.0, 0.01), (1.0, 0.01)]),
                          interfaces=[Interface(name="s", region="si", side="x_max", N_it_cm2=1e10)])
    from pydantic import ValidationError
    with pytest.raises((ValueError, ValidationError)) as e:
        make()
    print(f"[R192] 1D device with an interface: {str(e.value).splitlines()[0][:160]}")


def test_R192_charon_refuses_interfaces():
    from simws.engines import charon_engine
    base = charon_engine.unsupported(_diode())
    with_ifc = charon_engine.unsupported(_diode([_ifc(N_it=1e10, N_ot=1e10)]))
    new = [r for r in with_ifc if r not in base]
    print(f"[R192] Charon: without interfaces {base}; with, new: {new}")
    assert CHARON_REASON in with_ifc
    assert CHARON_REASON not in base
    from simws import engines
    assert engines.unsupported("devsim", _diode([_ifc(N_it=1e10, N_ot=1e10)])) == \
        engines.unsupported("devsim", _diode())


# ---------------------------------------------------------------------------------------------- A25.3 / A25.4 (no solve)
def test_A25_3_the_5AMs_two_interfaces():
    k = M().knobs()
    p = lib().ibm5am_hbt(**k, spacer=(1.1e11, 2.2e11), sti=(3.3e11, 4.4e11))
    ifs = {i.name: i for i in p.device.interfaces}
    assert len(ifs) == 2, list(ifs)
    sp = next(i for i in ifs.values() if i.side == "y_min")
    st = next(i for i in ifs.values() if i.side == "x_max")
    assert sp.span == pytest.approx(SPACER_X), sp.span
    assert (sp.N_it_cm2, sp.N_ot_cm2) == (1.1e11, 2.2e11) and sp.sigma_cm2 == SIGMA_DEFAULT
    reg = p.device.region(st.region)
    assert reg.box.x[1] == pytest.approx(STI_X)
    # y_xb: the extrinsic base (acceptors for x >= 0.36 um, A23.2) ends at its box's lower edge in y
    xb = [t for t in p.device.doping if t.species == "acceptor" and t.box.x[0] is not None
          and abs(t.box.x[0] - SPACER_X[1]) < 1e-9]
    assert len(xb) == 1, [t.label for t in p.device.doping]
    y_xb = xb[0].box.y[1]
    assert st.span == pytest.approx((0.0, y_xb + STI_BELOW_XB)), (st.span, y_xb)
    assert (st.N_it_cm2, st.N_ot_cm2) == (3.3e11, 4.4e11) and st.sigma_cm2 == SIGMA_DEFAULT
    print(f"[A25.3] spacer y_min x {sp.span}; STI x_max y {st.span} (y_xb {y_xb:.4f} um + 0.2)")
    one = lib().ibm5am_hbt(**k, spacer=(1e11, 0.0))
    assert [i.side for i in one.device.interfaces] == ["y_min"]


def test_A25_3_inverse_mode_bias():
    from simws.spec.schema import DCSweep
    k = M().knobs()
    p = lib().ibm5am_hbt(**k, VBE=[0.5, 0.6], mode="inverse")
    a = p.analysis
    assert isinstance(a, DCSweep) and a.contact == "base" and a.values_V == [0.5, 0.6]
    tr = {t.contact: t.offset_V for t in a.tracking}
    assert tr == {"emitter": 0.0}, tr
    assert a.fixed_V.get("collector", 0.0) == 0.0
    f = lib().ibm5am_hbt(**k, VBE=[0.5, 0.6]).analysis
    assert f.contact == "base" and {t.contact: t.offset_V for t in f.tracking} == {"collector": 0.0}
    with pytest.raises(ValueError):
        lib().ibm5am_hbt(**k, mode="sideways")
    print("[A25.3] inverse: base swept, the physical emitter tracking it (VCB_inverse 0), the collector grounded")


def test_A25_4_ranges_and_traps():
    assert {n: tuple(v) for n, v in T().TID_RANGES.items()} == TID_RANGES
    good = {"k_F": 1e11, "k_I": 2e11, "k_Q": 3e11}
    T().check_knobs(good)
    for n, (lo, hi) in TID_RANGES.items():
        for ok in (lo, hi):
            T().check_knobs(dict(good, **{n: ok}))
        bad = [hi * 1.01] + ([lo * 0.99] if lo > 0 else [-1.0])
        for b in bad:
            with pytest.raises(ValueError):
                T().check_knobs(dict(good, **{n: b}))
            with pytest.raises(ValueError):
                T().traps(dict(good, **{n: b}), 1.0)
        print(f"[A25.4] {n}: {bad} refused by check_knobs and traps; {lo:g} and {hi:g} accepted")
    for D in (0.1, 1.0, 6.3):
        tr = T().traps(good, D)
        assert tuple(tr["spacer"]) == pytest.approx((good["k_F"] * D, good["k_Q"] * D), rel=1e-12)
        assert tuple(tr["sti"]) == pytest.approx((good["k_I"] * D, good["k_Q"] * D), rel=1e-12)
    print("[A25.4] traps: spacer (k_F D, k_Q D), STI (k_I D, k_Q D)")


def test_A25_3_perimeter_factor():
    for L in (0.5, 1.0, 2.5, 10.0):
        want = 2 * (L + W_E_UM) / (2 * L)
        assert T().perimeter_factor(L) == pytest.approx(want, rel=1e-12), L
    print(f"[A25.3] P_E / 2L = (L + 0.5) / L: 1.5 at L = 1, 1.2 at L = 2.5")


def test_A25_4_stored_fit():
    path = Path(T().TID_FILE)
    if not path.exists():
        pytest.skip(f"{path.name} not written yet (A25.4's fit has not been run)")
    d = json.loads(path.read_text())
    t = T().tid_knobs()
    T().check_knobs(t)
    for n, (lo, hi) in TID_RANGES.items():
        assert lo <= t[n] <= hi, (n, t[n])
    assert "residuals_log10" in d, list(d)
    print(f"[A25.4] stored knobs {t}; residuals {d.get('residuals_log10')}")


# ================================================================================================== SLOW TIER
# Run-time choices (the lead, after the first slow run took over 4 hours for one mode on the build machine): the
# sweeps run on a VBE grid of 0.04 V plus +-10 mV around R197's three points, at a fixed set of doses spanning every
# traced dose, in parallel; a value at another dose is interpolated between the two nearest doses, the excess base
# and collector currents linearly in ln(excess) against ln(dose) at every VBE (the model's response is smooth and
# monotonic in dose; between neighbouring doses of the grid it changes by at most a factor 2.2).
GRID300 = sorted({round(0.44 + 0.04 * i, 2) for i in range(10)} | {0.54, 0.56, 0.59, 0.61, 0.69, 0.71})
DOSES300 = [0.1799, 0.3148, 0.6297, 1.05, 1.799, 3.148, 6.297]          # Mrad(SiO2): traced doses, 0.18 to 6.3
DOSES77 = [0.105, 0.3148, 1.05, 3.148, 6.297]                           # Fig. 73's 77 K doses (traced)
WORKERS = int(os.environ.get("SIMWS_WORKERS", max(1, min(4, (os.cpu_count() or 2)))))


def _series_mrad(ds, path):
    s = C().series(ds, path)
    return np.asarray(s["dose_rad_SiO2"], float) / 1e6, np.asarray(s["value"], float)


def _doses(mode, Tk):
    return list(DOSES300 if Tk == 300.0 else DOSES77)


class Curve:
    """One dose_curve: the pre-irradiation sweep and the post-dose sweeps on the dose grid; at(D) gives the post-dose
    per-cm IB and JC at any dose inside the grid (interpolated as stated above)."""

    def __init__(self, dc, Tk):
        self.dc, self.T = dc, Tk
        self.pre = dc["pre"]
        self.grid = [float(D) for D in dc["dose_Mrad"]]
        self.post = {float(D): s for D, s in zip(dc["dose_Mrad"], dc["post"])}
        self.V = np.asarray(self.pre["VBE_V"], float)
        for s in self.post.values():
            assert np.allclose(np.asarray(s["VBE_V"], float), self.V), "pre and post on the same VBE grid"

    def sweep(self, D):
        """The post-dose sweep at D: the grid's own sweep when D is within 0.5 % of a grid dose, else interpolated."""
        r = min(self.grid, key=lambda x: abs(math.log(D / x)))
        if abs(D / r - 1) <= DOSE_MERGE + 1e-12:
            return self.post[r]
        assert self.grid[0] <= D <= self.grid[-1], (D, self.grid)
        hi = next(x for x in self.grid if x >= D)
        lo = max(x for x in self.grid if x <= D)
        w = math.log(D / lo) / math.log(hi / lo)
        out = {"VBE_V": self.V}
        for key in ("IB_cm", "JC_A_um2"):
            p0 = np.abs(np.asarray(self.pre[key], float))
            a = np.abs(np.asarray(self.post[lo][key], float)) - p0
            b = np.abs(np.asarray(self.post[hi][key], float)) - p0
            ok = (a > 0) & (b > 0)
            ex = np.where(ok, np.exp((1 - w) * np.log(np.where(ok, a, 1)) + w * np.log(np.where(ok, b, 1))),
                          (1 - w) * a + w * b)
            out[key] = p0 + ex
        return out

    def ib(self, s):
        return np.abs(np.asarray(s["IB_cm"], float))

    def jc(self, s):
        return np.abs(np.asarray(s["JC_A_um2"], float))

    def ratio_at_vbe(self, D, L, v):
        ib0, ibp = self.ib(self.pre), self.ib(self.sweep(D))
        return interp_ln(v, self.V, ibp - ib0) / interp_ln(v, self.V, ib0) * pfac(L)

    def ratio_at_jc(self, D, L, J):
        """(dIB/IB0 at the fixed JC, V_pre, V_post) for one device of length L (reading in the docstring)."""
        f = pfac(L)
        ib0, ibp = self.ib(self.pre), self.ib(self.sweep(D))
        jc0, jcp = self.jc(self.pre), self.jc(self.sweep(D))
        v0 = first_crossing(self.V, jc0, J)
        v1 = first_crossing(self.V, jc0 + (jcp - jc0) * f, J)
        if v0 is None or v1 is None:
            return float("nan"), v0, v1
        ibd = ib0 + (ibp - ib0) * f
        return interp_ln(v1, self.V, ibd) / interp_ln(v0, self.V, ib0) - 1.0, v0, v1

    def jb_device(self, s, L):
        """JB (A/um^2) of one device after the dose of sweep s (s = pre: before), A25.3 scaling."""
        ib0, ibp = self.ib(self.pre), self.ib(s)
        return (ib0 * 2 * L + (ibp - ib0) * 2 * (L + W_E_UM)) * 1e-4 / (W_E_UM * L)

    def ideality(self, D, v, h=0.01):
        ib0, ibp = self.ib(self.pre), self.ib(self.sweep(D))
        ex = ibp - ib0
        i1 = int(np.argmin(np.abs(self.V - (v - h))))
        i2 = int(np.argmin(np.abs(self.V - (v + h))))
        assert abs(self.V[i1] - (v - h)) < 1e-9 and abs(self.V[i2] - (v + h)) < 1e-9, (v, self.V)
        if ex[i1] <= 0 or ex[i2] <= 0:
            return float("nan")
        return (2 * h) / (vt(self.T) * math.log(ex[i2] / ex[i1]))


# A25 build-machine results (docs/ARCHITECTURE_V1.md, "A25, build-machine results"): the recorded FAILs, strict
# xfails with the numbers seen; no tolerance changed. Added to each test in its setup (so the mark is strict).
A25_RECORDED = {
    "test_R193_calibration_points[inverse]": "inverse 1.05 Mrad: 11.3 against 24.3 (factor 2.14 > 1.5)",
    "test_R194_other_doses_vbe[inverse]": "inverse 0.31 / 0.63 Mrad: factors 2.81 / 2.54 (the model's slope 0.94, "
                                          "measured 0.64)",
    "test_R195_jc_series[forward]": "forward at JC = 1 uA/um^2: factors 2.18 to 3.06 at 0.63 to 3.1 Mrad",
    "test_R195_jc_series[inverse]": "inverse at JC = 1 uA/um^2: factors 2.2 to 5.6",
    "test_R196_fig33[forward-proton_63MeV]": "0.31 Mrad: 0.30 against 0.077 (3.95); 3.15 Mrad: 3.27 against 10.5",
    "test_R196_fig33[forward-Xray_10keV]": "X-ray forward: factors up to 3.19 (dose enhancement, expected)",
    "test_R196_fig33[inverse-proton_63MeV]": "inverse protons: factors 3.27 to 2.49 below 1 Mrad",
    "test_R196_fig33[inverse-Co60]": "inverse Co-60: factors 5.28 to 1.9",
    "test_R197_ideality[forward-proton_63MeV]": "n at 0.7 V: 1.33 against 1.81",
    "test_R197_ideality[forward-Co60]": "n at 0.7 V: 1.33 against 1.89",
    "test_R197_ideality[forward-Xray_10keV]": "n at 0.6 V: 1.60 against 1.20; at 0.7 V 1.33 against 1.66",
    "test_R197_ideality[inverse-proton_63MeV]": "n at 0.7 V: 1.47 against 1.78 (|diff| 0.31)",
    "test_R198_fig72_post_irradiation_IB": "JB after 1 / 3 / 6 Mrad: worst factors 2.16 / 3.01 / 3.03",
    "test_R199_fig73[77.0]": "77 K: factors 6.4 to 13 (no temperature-dependent build-up; expected)",
    "test_R199_fig73[300.0]": "300 K 3.15 Mrad: 5.29 against 10.6 (factor 2.010 > 2)",
    "test_R199_fig72_77K": "77 K from Fig. 72: factors 11 to 18 (expected)",
}


@pytest.fixture(autouse=True)
def _a25_recorded(request):
    reason = A25_RECORDED.get(request.node.name)
    if reason:
        request.node.add_marker(pytest.mark.xfail(strict=True, reason="A25 recorded FAIL: " + reason))
    yield


@pytest.fixture(scope="module")
def tid():
    path = Path(T().TID_FILE)
    if not path.exists():
        pytest.skip(f"{path.name} not written yet: run A25.4's fit (ibm5am_tid.fit) first")
    t = T().tid_knobs()
    T().check_knobs(t)
    return t


@pytest.fixture(scope="module")
def k23b():
    k = M().knobs()
    for n, v in A23B.items():
        assert k[n] == pytest.approx(v, rel=5e-3), (n, k[n], v)
    return k


_CURVES = {}


def curve(t, k, mode, Tk):
    key = (mode, Tk)
    if key not in _CURVES:
        if Tk == 300.0:
            grid = GRID300
        else:
            pre77 = next(c for c in C().figure("sutton2009.fig72")[0]["curves"] if float(c["T_K"]) == Tk)
            vstar = first_crossing(*sorted_xy(pre77["VBE_V_C"], pre77["JC_A_um2"]), JC_77)
            c0 = round(vstar, 2)
            grid = [round(c0 + 0.02 * i, 2) for i in range(-4, 5)]
        doses = _doses(mode, Tk)
        try:
            dc = T().dose_curve(t, doses, mode, L_um=L_SMALL, VBE=grid, T=Tk, k=k, workers=WORKERS)
            _CURVES[key] = Curve(dc, Tk)
        except Exception as e:  # noqa: BLE001
            _CURVES[key] = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"[A25] dose_curve {mode} {Tk:.0f} K: {len(doses)} doses {doses} on {len(grid)} VBE points")
    c = _CURVES[key]
    if isinstance(c, str):
        pytest.fail(f"dose_curve {mode} at {Tk:.0f} K: no result ({c})")
    return c


@pytest.fixture(scope="module")
def fwd300(tid, k23b):
    return curve(tid, k23b, "forward", 300.0)


@pytest.fixture(scope="module")
def inv300(tid, k23b):
    return curve(tid, k23b, "inverse", 300.0)


@pytest.fixture(scope="module")
def fwd77(tid, k23b):
    return curve(tid, k23b, "forward", 77.0)


def _mode_curve(request, mode):
    return request.getfixturevalue("fwd300" if mode == "forward" else "inv300")


# ---------------------------------------------------------------------------------------------- R189
@pytest.mark.slow
def test_R189_inverse_gummel_against_fig22(inv300):
    f22 = C().figure("sutton2004.fig22")[0]
    a = f22["assignment"]
    c = inv300
    jb_m = c.jb_device(c.pre, L_SMALL)
    jc_m = c.jc(c.pre)
    out = {}
    for what, key, model in (("IB", a["IB"], jb_m), ("IC", a["IC"], jc_m)):
        V, J = sorted_xy(f22[key]["VBE_V"], f22[key]["J_A_um2"])
        m = (V >= R189_WIN[0] - 1e-9) & (V <= R189_WIN[1] + 1e-9)
        fs = []
        for v, j in zip(V[m], J[m]):
            mod = interp_ln(v, c.V, model)
            fs.append(factor(mod, j))
            print(f"[R189] {what} VBE {v:.3f} V: model {mod:.4e} against Fig. 22 ({key}) {j:.4e} A/um^2: "
                  f"factor {fs[-1]:.2f}" + (f" (limit {R189_F})" if what == "IB" else " (reported)"))
        out[what] = (fs, int(m.sum()))
    fs, n = out["IB"]
    print(f"[R189] IB: worst factor {max(fs):.2f} over {n} traced VBE in {R189_WIN} (limit {R189_F}); IC worst "
          f"{max(out['IC'][0]):.2f} (reported)")
    assert n > 5 and max(fs) <= R189_F


# ---------------------------------------------------------------------------------------------- R193 / R194 / R195
@pytest.mark.slow
@pytest.mark.parametrize("mode", ["forward", "inverse"])
def test_R193_calibration_points(request, mode):
    c = _mode_curve(request, mode)
    fig = "sutton2004.fig24" if mode == "forward" else "sutton2004.fig26"
    D, val = _series_mrad(fig, "VBE_0.6V")
    roles = np.asarray(list(C().point_roles(fig, "VBE_0.6V")))
    fs = []
    for d, v in zip(D[roles == "calibration"], val[roles == "calibration"]):
        mod = c.ratio_at_vbe(d, L_SMALL, VBE_FIX)
        fs.append(factor(mod, v))
        lead = T().at(c.pre, c.sweep(d), VBE_FIX)
        print(f"[R193] {mode} {d:.4g} Mrad, VBE 0.6 V: model dIB/IB0 {mod:.4g} (lead's at() {lead:.4g}) against "
              f"{fig} {v:.4g}: factor {fs[-1]:.3f} (limit {R193_F})")
    assert len(fs) == 2 and max(fs) <= R193_F, fs


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["forward", "inverse"])
def test_R194_other_doses_vbe(request, mode):
    c = _mode_curve(request, mode)
    fig = "sutton2004.fig24" if mode == "forward" else "sutton2004.fig26"
    D, val = _series_mrad(fig, "VBE_0.6V")
    roles = np.asarray(list(C().point_roles(fig, "VBE_0.6V")))
    fs = []
    for d, v in zip(D[roles == "validation"], val[roles == "validation"]):
        mod = c.ratio_at_vbe(d, L_SMALL, VBE_FIX)
        fs.append(factor(mod, v))
        print(f"[R194] {mode} {d:.4g} Mrad, VBE 0.6 V: model {mod:.4g} against {fig} {v:.4g}: factor {fs[-1]:.3f} "
              f"(limit {R194_F})")
    i3, i30 = nearest_index(D, 0.3), nearest_index(D, 3.0)
    sl_m = math.log(c.ratio_at_vbe(D[i30], L_SMALL, VBE_FIX) / c.ratio_at_vbe(D[i3], L_SMALL, VBE_FIX)) / \
        math.log(D[i30] / D[i3])
    sl_d = math.log(val[i30] / val[i3]) / math.log(D[i30] / D[i3])
    print(f"[R194] {mode} log-log slope {D[i3]:.4g} to {D[i30]:.4g} Mrad: model {sl_m:.3f}, measured {sl_d:.3f}, "
          f"Sutton D^{R194_SLOPE_REF[mode]} (reported)")
    assert len(fs) == 3 and max(fs) <= R194_F, fs


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["forward", "inverse"])
def test_R195_jc_series(request, mode):
    c = _mode_curve(request, mode)
    fig = "sutton2004.fig24" if mode == "forward" else "sutton2004.fig26"
    D, val = _series_mrad(fig, "JC_1uA_um2")
    fs = []
    for d, v in zip(D, val):
        mod, v0, v1 = c.ratio_at_jc(d, L_SMALL, JC_FIX)
        lead_v = T().vbe_at_jc(c.pre, JC_FIX)
        lead = T().at(c.pre, c.sweep(d), lead_v)
        fs.append(factor(mod, v))
        print(f"[R195] {mode} {d:.4g} Mrad, JC 1 uA/um^2 (VBE pre {v0}, post {v1}; lead's {lead_v:.4f} V, at() "
              f"{lead:.4g}): model {mod:.4g} against {fig} {v:.4g}: factor {fs[-1]:.3f} (limit {R195_F})")
    assert len(fs) == len(D) and max(fs) <= R195_F, fs


# ---------------------------------------------------------------------------------------------- R196 (Fig. 33)
@pytest.mark.slow
@pytest.mark.parametrize("source", ["proton_63MeV", "Co60", "Xray_10keV"])
@pytest.mark.parametrize("mode", ["forward", "inverse"])
def test_R196_fig33(request, source, mode):
    c = _mode_curve(request, mode)
    D, val = _series_mrad("sutton2007.fig33", f"{source}/{mode}")
    fs = []
    for d, v in zip(D, val):
        mod = c.ratio_at_vbe(d, L_BIG, VBE_FIX)
        fs.append(factor(mod, v))
        print(f"[R196] {source} {mode} {d:.4g} Mrad(SiO2), L 2.5 um, VBE 0.6 V: model {mod:.4g} against Fig. 33 "
              f"{v:.4g}: factor {fs[-1]:.3f} (limit {R196_F})")
    print(f"[R196] {source} {mode}: worst factor {max(fs):.3f} (limit {R196_F})"
          + ("; expected trouble (X-ray dose enhancement)" if source == "Xray_10keV" and mode == "forward" else ""))
    assert max(fs) <= R196_F, fs


# ---------------------------------------------------------------------------------------------- R197 (ideality)
def _n_measured(V, n, v):
    V, n = sorted_xy(V, n)
    i = int(np.argmin(np.abs(V - v)))
    if abs(V[i] - v) <= 0.002:
        return float(n[i]), "traced"
    return float(np.interp(v, V, n)), "interpolated"


@pytest.mark.slow
@pytest.mark.parametrize("source", ["proton_63MeV", "Co60", "Xray_10keV"])
@pytest.mark.parametrize("mode", ["forward", "inverse"])
def test_R197_ideality(request, source, mode):
    c = _mode_curve(request, mode)
    f = C().series("sutton2007.fig39", f"{source}/{mode}")
    ds = []
    for v in R197_VBE:
        nm = c.ideality(R197_DOSE, v)
        nd, how = _n_measured(f["VBE_V"], f["n"], v)
        try:
            lead = T().ideality(c.sweep(R197_DOSE), c.pre, v)
        except Exception as e:  # noqa: BLE001
            lead = f"{type(e).__name__}"
        ds.append(abs(nm - nd) if math.isfinite(nm) else math.inf)
        print(f"[R197] {source} {mode} {R197_DOSE} Mrad, VBE {v} V: model n {nm:.3f} (lead's {lead}) against Fig. 39 "
              f"{nd:.3f} ({how}): |diff| {ds[-1]:.3f} (limit {R197_TOL})")
    assert max(ds) <= R197_TOL, ds


# ---------------------------------------------------------------------------------------------- R198 (Fig. 72, 300 K)
@pytest.mark.slow
def test_R198_fig72_post_irradiation_IB(fwd300):
    c = fwd300
    f72 = C().figure("sutton2009.fig72")[0]
    pre = next(x for x in f72["curves"] if float(x["T_K"]) == 300.0)
    Vb, Jb = sorted_xy(pre["VBE_V_B"], pre["JB_A_um2"])
    m = (Vb >= R198_WIN[0] - 1e-9) & (Vb <= R198_WIN[1] + 1e-9)
    jb0 = c.jb_device(c.pre, L_SMALL)
    r = [interp_ln(v, c.V, jb0) / j for v, j in zip(Vb[m], Jb[m])]
    print(f"[R198] pre-irradiation JB model / Fig. 72 at {int(m.sum())} traced VBE in {R198_WIN}: "
          f"{min(r):.3f} to {max(r):.3f} (reported first)")
    worst = {}
    for D in R198_DOSES:
        post = next(x for x in f72["post_irradiation"]
                    if float(x["T_K"]) == 300.0 and abs(x["dose_rad_SiO2"] / 1e6 - D) < 1e-6)
        V, J = sorted_xy(post["VBE_V"], post["J_A_um2"])
        mm = (V >= R198_WIN[0] - 1e-9) & (V <= R198_WIN[1] + 1e-9)
        jbp = c.jb_device(c.sweep(D), L_SMALL)
        fs = [factor(interp_ln(v, c.V, jbp), j) for v, j in zip(V[mm], J[mm])]
        worst[D] = max(fs)
        rr = [interp_ln(v, c.V, jbp) / j for v, j in zip(V[mm], J[mm])]
        print(f"[R198] {D} Mrad: model JB / Fig. 72 at {int(mm.sum())} traced VBE in {R198_WIN}: {min(rr):.3f} to "
              f"{max(rr):.3f}, worst factor {worst[D]:.3f} (limit {R198_F})")
        assert mm.sum() > 5
    assert max(worst.values()) <= R198_F, worst


# ---------------------------------------------------------------------------------------------- R199 (77 K)
@pytest.mark.slow
@pytest.mark.parametrize("Tk", [77.0, 300.0])
def test_R199_fig73(request, Tk):
    c = request.getfixturevalue("fwd77" if Tk == 77.0 else "fwd300")
    D, val = _series_mrad("prakash2006.fig73", f"{Tk:.1f}")
    fs = []
    for d, v in zip(D, val):
        mod, v0, v1 = c.ratio_at_jc(d, L_SMALL, JC_77)
        fs.append(factor(mod, v))
        print(f"[R199] {Tk:.0f} K {d:.4g} Mrad, JC 10 nA/um^2 (VBE pre {v0}, post {v1}): model {mod:.4g} against "
              f"Fig. 73 {v:.4g}: factor {fs[-1]:.3f} (limit {R199_F})"
              + ("; the contract expects a FAIL at 77 K" if Tk == 77.0 else ""))
    assert max(fs) <= R199_F, fs


@pytest.mark.slow
def test_R199_fig72_77K(fwd77):
    c = fwd77
    f72 = C().figure("sutton2009.fig72")[0]
    pre = next(x for x in f72["curves"] if float(x["T_K"]) == 77.0)
    vstar = first_crossing(*sorted_xy(pre["VBE_V_C"], pre["JC_A_um2"]), JC_77)
    jb0 = interp_ln(vstar, pre["VBE_V_B"], pre["JB_A_um2"])
    fs = []
    for D in R198_DOSES:
        post = next(x for x in f72["post_irradiation"]
                    if float(x["T_K"]) == 77.0 and abs(x["dose_rad_SiO2"] / 1e6 - D) < 1e-6)
        meas = interp_ln(vstar, post["VBE_V"], post["J_A_um2"]) / jb0 - 1.0
        mod, v0, v1 = c.ratio_at_jc(D, L_SMALL, JC_77)
        fs.append(factor(mod, meas))
        print(f"[R199] Fig. 72 77 K {D} Mrad at JC 10 nA/um^2 (measured V* {vstar:.4f} V; model {v0}, {v1}): model "
              f"{mod:.4g} against {meas:.4g} (JB_post / JB_pre - 1): factor {fs[-1]:.3f} (limit {R199_F}); "
              f"the contract expects a FAIL at 77 K")
    assert max(fs) <= R199_F, fs
