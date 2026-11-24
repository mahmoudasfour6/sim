"""Contract amendments A23, A24 and A23b (24 Nov 2026): validation data for the IBM 5AM SiGe HBT, the 5AM device at
300 K and from 393 K to 43 K, and the revised device (a finite hole recombination velocity at the emitter contact,
collector phosphorus, mesh). Acceptance checks R160 to R175 (A23.4, A24.2, re-run on the A23b device as A23b.4
says) and R180 to R183 (A23b.4).

Written by the checker from the contract text (A23 items 0 to 7, A24 items 0 to 4, A23b items 0 to 6, the
build-machine results section), sections 2, 6 and 7 of docs/ARCHITECTURE_V1.md, simws/spec/schema.py, simws/measured.py,
simws/compare.py and simws/hub/*. The new code is called only through the public names A23.7(j) and A23b.6(d)
list (simws.validation.catalog, simws.validation.ibm5am, simws.spec.library.ibm5am_hbt / ibm5am_scale /
IBM5AM_RANGES, simws.engines.devsim_engine.grade_values) and the data files' documented content. Every gated
comparison (window, interpolation, crossing, peak, line fit, ratio) is computed here from the model's raw arrays and
the traced points; the lead's own target functions (luo_targets, early_voltage_measured) are only printed next to
ours. Every tolerance is the contract's (A23.4, A24.2, A23b.4); the A23b device is the stored calibration
(ibm5am.knobs()); A23's device results are history (A23b.5) and are not tested.

Tiers. Fast (not slow): R160, R168, R180, R181, R182, unit checks of ionization_integral, vbe_at_current and
grade_values, and R167's film (no solver). Slow: everything that runs the 2D 5AM device (R161 to R167, R169 to R175,
R183); each expensive quantity is computed once in a module fixture.

Readings of the contract made here (stated so the lead can object):
- "within x %" is |model / measured - 1| <= x; "within a factor F" is max(model / measured, measured / model) <= F.
- "at every traced VBE in a window": the measured trace's own VBE values inside the window (ends included); the model
  is evaluated on a VBE grid and interpolated onto them linearly in ln(I) (never extrapolated). R161's IB =
  IC / beta: at every traced VBE of Fig. 4.7 in 0.70-0.78 V, IC of Fig. 5.7(a) at 300 K interpolated there (ln IC).
- The model's grid at 300 K: 0.60 to 0.80 V in 20 mV steps plus 0.65 V (R167). R166 compares the 0.70 V point of that
  sweep with a mesh-scale-0.5 run at 0.70 V alone.
- R163: V_A of a straight line IC = a + b VCE (least squares over the points with 0.6 <= VCE <= 2.0 V) is a / b,
  the classical extrapolation to IC = 0 (the contract's "IC / slope" with IC the line's value at VCE = 0). The model
  uses the same fit on early_voltage()'s VCE / IC arrays. (early_voltage_measured() is printed: it takes the line's
  IC at the window's centre, 1.3 V, so it is 1.3 V larger; the gate's verdict does not depend on this.)
- R164 / R173: the measured M - 1 at a VCB is the trace interpolated linearly in log10(M - 1) (Luo calls the spikes near
  4 V at 93 and 162 K measurement errors; the contract keeps 4 V, so they are not removed here).
- R165: the peaks are the largest fT of the model's bias points (A23.7(i)) and of the traced points.
- R170: VBE at a current is the first crossing from below, linear in ln(I) between the two points around it.
- R171: the peak beta is the largest IC / IB of the model's points (gummel_window(T)) and the largest traced beta.
- R172: the measured RsBi at 223, 162, 100 and 75 K is the Fig. 4.1 trace interpolated linearly in log10(RsBi) in T.
  "Reported down to the trace's lowest temperature": rsbi() at the trace's temperatures 70, 60, 50, 40, 30 and
  20 K (every other point below 75 K), printed with any failure.
- R174: JC = IC / (0.5 x 1.0 um^2) (A23.7(h)'s drawn area); the model's VBE grid is the trace's VBE at 10 nA/um^2
  +- 80 mV in 20 mV steps.
- R175: Luo's ideality is the slope of his Fig. 5.7(a) trace between 1 uA and 100 uA, n = dVBE / (Vt ln 100) (Fig.
  5.7(b) is not traced); the model's the same, from gummel_window(T). Reported, never gated (a solver failure is
  printed as its category).
- R169: the model result is device.devsim's dc.I.collector (A/m, half cell) for ibm5am_hbt(**knobs) at 300 K, VBE
  0.60-0.78 V, x 2 L (A23.2) to one device in A; it is compared through simws.compare.compare with the ingested
  curve (the catalog's role as the label) and the comparison is recorded for the two result ids. Per A23.7(e) only
  the IC curves are recorded. The expected trust follows A23.4 R169 and sections 6.1 / A1.10: "validated" exactly
  when a "validation" comparison passed and no measurement comparison failed (a failed R161 comparison also keeps
  the result below "validated").
- R160 "ingest labels as the catalog says": every point of an ingested curve carries its own catalog role, so a
  "validation" quantity of Fig. 4.1 (RsBi vs T) must not hold the 300 K point the catalog marks "calibration", and
  a "calibration" quantity must hold no point of another temperature.
- R180: the closed form is A23b.1's J_p plus the p side's own electron term q n_p0 D_n / W_p (e^{V/Vt} - 1) (exact
  for this short p side and <= 1 % of the total here), W' = n side - x_n with x_n of the depletion approximation
  at Vbi - V. The free choices (p side 3 um, mu_n 10, mu_p 400 cm^2/Vs, 2001 nodes) keep the electron term small.
  The diode current is the current at the contact that carries S (the cathode): the anode sits in 1e20 holes whose
  drift and diffusion cancel to ~1e-7 A/cm^2 (the ohmic diode's Kirchhoff residual), which at S = 1e2 cm/s
  (3.8e-6 A/cm^2) is about 2 %; the anode is printed.
- R181: the 2D diode is the 1D one laid along y (2D DEVSIM contacts sit on y_min / y_max) and extruded 0.5 um in x,
  the same mesh along the diode; its current / 0.5 um is compared with the 1D current at every S of R180.
- R182 "every library problem without S keeps its id": the ids of the library problems (all builders, and
  ibm5am_hbt without k_P / S_p) computed by this tree equal those computed by the tree of commit 79419a6 (the A23b
  contract, before any A23b code), run in a subprocess from `git archive`; skipped when git or the commit is
  missing. A round-trip check runs always.
- R168 "a changed data file changes the id": a copy of simws in a temporary directory with luo2011.json's Fig. 1.2
  boron x 1.01, the id computed there in a subprocess.
- Unit checks (no contract tolerance exists for them): ionization_integral against the closed form for a constant
  field, 1e-4 relative (a quadrature of a smooth exponential on 4001 nodes); the contract does not say whether it
  returns the integral 1 - 1/M or M - 1, so either is accepted and printed; vbe_at_current against Vt ln(I/I0) on an
  exact exponential, 1e-9 V; grade_values exact (1e-12).
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
K_B_EV = 8.617333262e-5         # eV/K
EPS0_F_CM = 8.8541878128e-14    # F/cm

# ------------------------------------------------------------------------------------------- the contract's numbers
# A23.3 / A23b.3 knob ranges
RANGES = {"s_um": (0.3, 1.2), "k_B": (0.5, 2.0), "k_E": (0.3, 3.0), "k_P": (0.1, 2.0), "S_p": (1e3, 1e7)}
A23B_S_UM, A23B_K_E = 0.80, 1.0                         # A23b.2
BVCBO_MEAS_V = 10.5                                     # A23b.3 (Lourenco Table 6.1)
# A23.4
R161_IC_TOL, R161_IB_TOL, R161_RS_TOL = 0.15, 0.20, 0.10
R161_IC_WIN, R161_IB_WIN = (0.60, 0.78), (0.70, 0.78)
R162_TOL, R162_WIN = 0.25, (0.60, 0.78)
R163_FACTOR, R163_VCE, R163_IB_UA = 1.5, (0.6, 2.0), (6.0, 12.0)
R164_FACTOR, R164_VCB = 2.0, (2.0, 3.0, 4.0, 5.0)
R165_PEAK_TOL, R165_JE_FACTOR = 0.25, 2.0
R165_VBE = [round(0.80 + 0.02 * i, 2) for i in range(10)]      # A23.7(i): 0.80 to 0.98 V in 20 mV steps, VCB 0
R166_TOL, R166_VBE, R166_SCALE = 0.03, 0.70, 0.5
R167_VBE = (0.60, 0.65)
SUTTON_FILM_NM = (90.0, 150.0)
# A24.2
R170_TOL_V, R170_T, R170_I = 0.020, (393.0, 223.0, 162.0, 93.0), (1e-6, 1e-4)
R171 = {223.0: ("rel", 0.30), 162.0: ("rel", 0.30), 93.0: ("factor", 2.0)}
R172 = {223.0: ("rel", 0.30), 162.0: ("rel", 0.30), 100.0: ("rel", 0.30), 75.0: ("factor", 2.0)}
R172_XFAIL = {100.0: "A23b result: RsBi 0.468 x Luo (the base freezes out too slowly in the model)",
              75.0: "A23b result: RsBi 0.290 x Luo (factor 3.4 > 2)"}
R173_FACTOR, R173_T, R173_VCB = 2.0, (162.0, 93.0), (2.0, 3.0, 4.0)
R174_TOL_V, R174_T, R174_J = 0.030, (300.0, 77.0), 10e-9 / 1.0      # 10 nA/um^2 in A/um^2
R174_AREA_UM2, R174_L_UM = 0.5 * 1.0, 1.0
# A23b.4
R180_TOL, R180_OHMIC_TOL, R180_S = 0.01, 0.005, (1e2, 1e3, 1e4, 1e5)
R180_S_HIGH = 1e8
R181_TOL = 0.001
R183_TOL = 0.10
R180_V = 0.5

GRID300 = [0.60, 0.62, 0.64, 0.65, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80]
R169_GRID = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78]

PRE_A23B_COMMIT = "79419a6"          # A23b contract commit: no A23b code yet (code as at the A23 / A24 commit)
PDF_DIRS = [Path("/tmp/claude-0/-home-claude/dcddde37-0a9f-534f-8d49-446206203045/scratchpad/dl"),
            Path("/mnt/user-data/attachments/dcddde37-0a9f-534f-8d49-446206203045")]

CONTRACT_IDS = ["luo2011.fig1_2", "luo2011.fig5_7a", "luo2011.fig4_7", "luo2011.fig4_1", "luo2011.fig4_2",
                "luo2011.fig3_7a", "pellish2008.fig3", "pellish2008.fig4", "sutton2009.fig72"]
ROLES = ("calibration", "validation", "trend", "input")


# ------------------------------------------------------------------------------------------------ small helpers
def C():
    from simws.validation import catalog
    return catalog


def M():
    from simws.validation import ibm5am
    return ibm5am


def rel(m, ref):
    return abs(m / ref - 1.0)


def factor(m, ref):
    if m <= 0 or ref <= 0:
        return math.inf
    return max(m / ref, ref / m)


def gate(kind, tol, m, ref):
    """(value, passed) for "within tol" (relative) or "within a factor tol"."""
    v = rel(m, ref) if kind == "rel" else factor(m, ref)
    return v, v <= tol


def sorted_xy(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.argsort(x, kind="stable")
    return x[o], y[o]


def interp_log(xq, x, y):
    """y at xq, linear in ln|y| between points; nan outside [x0, xn] (never extrapolated)."""
    x, y = sorted_xy(x, y)
    xq = np.atleast_1d(np.asarray(xq, float))
    out = np.exp(np.interp(xq, x, np.log(np.abs(y))))
    out[(xq < x[0]) | (xq > x[-1])] = np.nan
    return out


def my_vbe_at(V, I, target):
    """First VBE where I reaches target from below, linear in ln I; None if never reached."""
    V, I = np.asarray(V, float), np.abs(np.asarray(I, float))
    for i in range(1, len(V)):
        if I[i - 1] < target <= I[i]:
            a, b = math.log(I[i - 1]), math.log(I[i])
            return float(V[i - 1] + (V[i] - V[i - 1]) * (math.log(target) - a) / (b - a))
    if len(I) and I[0] == target:
        return float(V[0])
    return None


def line_va(vce, ic, lo=R163_VCE[0], hi=R163_VCE[1]):
    """R163 (reading above): V_A = a / b of the least-squares line IC = a + b VCE over lo <= VCE <= hi."""
    vce, ic = np.asarray(vce, float), np.asarray(ic, float)
    m = (vce >= lo - 1e-12) & (vce <= hi + 1e-12)
    b, a = np.polyfit(vce[m], ic[m], 1)
    return float(a / b), int(m.sum())


def vt(T):
    return K_B * T / Q_E


# ------------------------------------------------------------------------------------------------ measured data
def luo_ic(T):
    d = C().load("luo2011.fig5_7a", T_K=T)
    return sorted_xy(d["VBE_V"], np.asarray(d["IC_mA"], float) * 1e-3)        # V, A


def luo_beta(T):
    d = C().load("luo2011.fig4_7", T_K=T)
    return sorted_xy(d["VBE_V"], d["beta"])


def luo_rsbi():
    d = C().load("luo2011.fig4_1")
    return sorted_xy(d["T_K"], d["RsBi_ohm_sq"])


def luo_rsbi_at(T):
    t, r = luo_rsbi()
    return float(interp_log([T], t, r)[0])


def luo_m1(T):
    d = C().load("luo2011.fig4_2", T_K=T)
    return sorted_xy(d["VCB_V"], d["M_minus_1"])


def luo_ft():
    d = C().load("luo2011.fig3_7a")
    return np.asarray(d["JE_mA_um2"], float), np.asarray(d["fT_GHz"], float)


def pellish_gummel():
    d = C().load("pellish2008.fig3")
    V, IC = sorted_xy(d["VBE_V"], d["IC_A"])
    _, IB = sorted_xy(d["VBE_V"], d["IB_A"])
    return V, IC, IB


def pellish_out(IB_uA):
    d = C().load("pellish2008.fig4")
    curves = d["curves"]
    c = curves.get(f"{IB_uA:.1f}") or curves.get(str(IB_uA)) or curves[IB_uA]
    return sorted_xy(c["VCE_V"], np.asarray(c["IC_mA"], float) * 1e-3)


def sutton_jc(T):
    d = C().load("sutton2009.fig72", T_K=T)
    return sorted_xy(d["VBE_V_C"], d["JC_A_um2"])


def in_window(x, win):
    x = np.asarray(x, float)
    return (x >= win[0] - 1e-12) & (x <= win[1] + 1e-12)


# ================================================================================================== FAST TIER
# ---------------------------------------------------------------------------------------------- R160 (data)
def test_R160_catalog_lists_every_dataset_with_source_and_role():
    cat = C().catalog()
    assert set(cat["roles"]) == set(ROLES), cat["roles"]
    ids = [d["id"] for d in cat["datasets"]]
    assert len(ids) == len(set(ids)), "dataset ids must be unique"
    for i in CONTRACT_IDS:                                        # A23.7(j): the traced ids
        assert i in ids, i
        e = C().entry(i)
        assert e["file"], (i, "a traced dataset names its file")
    for d in cat["datasets"]:
        for k in ("id", "paper", "read_from", "device", "conditions", "role"):
            assert str(d.get(k, "")).strip(), (d["id"], k)
        assert d["role"] in ROLES, d
        assert C().role(d["id"]) == d["role"]
        for T, r in (d.get("curve_roles") or {}).items():
            assert r in ROLES and C().role(d["id"], T_K=float(T)) == r, (d["id"], T, r)
    assert C().role("luo2011.fig1_2") == "input"
    for i in ("luo2011.fig5_7a", "luo2011.fig4_7", "luo2011.fig4_1"):       # A23.3: the fit's three sets at 300 K
        assert C().role(i, T_K=300.0) == "calibration", i
    for i in ("pellish2008.fig3", "pellish2008.fig4", "luo2011.fig4_2", "luo2011.fig3_7a", "sutton2009.fig72"):
        assert C().role(i) == "validation", i                     # held out (A23.4 R162-R165, A24 R174)
    assert C().role("luo2011.fig5_7a", T_K=223.0) == "validation"
    with pytest.raises(KeyError):
        C().entry("no_such.dataset")
    print(f"[R160] catalog: {len(ids)} datasets, roles {sorted(set(d['role'] for d in cat['datasets']))}")


def test_R160_every_traced_dataset_loads_with_units_source_and_method():
    cat = C().catalog()
    n = 0
    for d in cat["datasets"]:
        if not d.get("file"):
            with pytest.raises(ValueError):                       # a later stage's row: no data to load
                C().load(d["id"])
            continue
        block, src = C().figure(d["id"])
        assert str(src.get("cite", "")).strip(), d["id"]
        assert str(src.get("url", "")).startswith("http"), d["id"]
        sha = str(src.get("pdf_sha256", ""))
        assert len(sha) == 64 and all(ch in "0123456789abcdef" for ch in sha), (d["id"], sha)
        assert str(block.get("method", "")).startswith("vector"), (d["id"], block.get("method"))   # A23.7(a)
        assert "precision" in block["method"] or "width" in block["method"], d["id"]   # the trace error (A23.1)
        axes = block.get("axes") or {}
        assert axes, d["id"]
        for name, ax in axes.items():
            assert len(ax["value"]) == 2 and len(ax["page_pt"]) == 2 and ax["scale"] in ("linear", "log10"), \
                (d["id"], name)
        loaded = C().load(d["id"])
        assert isinstance(loaded, dict) and loaded.get("source", {}).get("pdf_sha256") == sha, d["id"]
        n += 1
    # the curves: every array finite and the paired arrays of one length; the per-temperature loads
    for T in (393.0, 300.0, 262.0, 223.0, 192.0, 162.0, 136.0, 110.0, 93.0, 76.0, 60.0, 43.0):
        V, I = luo_ic(T)
        assert len(V) > 10 and np.all(np.isfinite(V)) and np.all(I > 0), T
    for T in (300.0, 223.0, 162.0, 93.0, 43.0):
        V, b = luo_beta(T)
        assert len(V) > 10 and np.all(b > 0), T
        x, m1 = luo_m1(T)
        assert len(x) > 10 and np.all(m1 > 0), T
    t, r = luo_rsbi()
    assert t.min() < 25 and t.max() == pytest.approx(300.0, abs=0.5) and np.all(r > 0)
    for T in (300.0, 77.0):
        V, J = sutton_jc(T)
        assert len(V) > 10 and np.all(J > 0), T
    V, IC, IB = pellish_gummel()
    assert len(V) == len(IC) == len(IB) and np.all(IC > 0) and np.all(IB > 0)
    for ib in (6.0, 7.5, 9.0, 10.5, 12.0):
        vce, ic = pellish_out(ib)
        assert len(vce) > 50 and vce.min() < 0.6 and vce.max() > 2.0, ib
    je, ft = luo_ft()
    assert len(je) == len(ft) > 10
    print(f"[R160] {n} traced datasets load; every one has a source (cite, url, 64-hex pdf_sha256), axes with units "
          f"and scales, and a vector method")


def test_R160_sha256_matches_the_pdfs_read():
    found = {}
    for d in PDF_DIRS:
        if d.is_dir():
            for p in d.glob("*.pdf"):
                found[hashlib.sha256(p.read_bytes()).hexdigest()] = p
    if not found:
        pytest.skip("the source PDFs are not on this machine (the fields are checked in the test above)")
    files = sorted({C().entry(i)["file"] for i in CONTRACT_IDS})
    for f in files:
        ds = next(i for i in CONTRACT_IDS if C().entry(i)["file"] == f)
        sha = C().figure(ds)[1]["pdf_sha256"]
        ok = sha in found
        print(f"[R160] {f}: pdf_sha256 {sha[:12]}... {'matches ' + found[sha].name if ok else 'NO MATCH'}")
        assert ok, (f, sha)


def _ingest_args(dataset):
    """axis and one column per traced dataset, in the data's own units (the key's suffix)."""
    return {
        "luo2011.fig5_7a": (("VBE", "VBE_V", "V"), {"IC": ("IC_mA", "mA")}),
        "luo2011.fig4_7": (("VBE", "VBE_V", "V"), {"beta": ("beta", "dimensionless")}),
        "luo2011.fig4_1": (("T", "T_K", "K"), {"RsBi": ("RsBi_ohm_sq", "ohm")}),
        "luo2011.fig4_2": (("VCB", "VCB_V", "V"), {"M_minus_1": ("M_minus_1", "dimensionless")}),
        "luo2011.fig3_7a": (("JE", "JE_mA_um2", "mA/um**2"), {"fT": ("fT_GHz", "GHz")}),
        "pellish2008.fig3": (("VBE", "VBE_V", "V"), {"IC": ("IC_A", "A"), "IB": ("IB_A", "A")}),
        "sutton2009.fig72": (("VBE", "VBE_V_C", "V"), {"JC": ("JC_A_um2", "A/um**2")}),
    }[dataset]


@pytest.mark.parametrize("dataset,temps", [
    ("luo2011.fig5_7a", (393.0, 300.0, 223.0, 162.0, 93.0, 43.0)),
    ("luo2011.fig4_7", (300.0, 223.0, 162.0, 93.0, 43.0)),
    ("luo2011.fig4_2", (300.0, 162.0, 93.0)),
    ("sutton2009.fig72", (300.0, 77.0)),
    ("luo2011.fig3_7a", (None,)),
    ("pellish2008.fig3", (None,)),
])
def test_R160_ingest_labels_as_the_catalog_says(hub, dataset, temps):
    from simws.hub.trust import assess_stored
    axis, cols = _ingest_args(dataset)
    for T in temps:
        r = C().ingest(hub, dataset, axis=axis, columns=cols, T_K=T)
        want = C().role(dataset, T_K=T)
        assert r.status == "ok" and "measured" in r.flags, (dataset, T)
        assert assess_stored(hub, r.id).level == "measured"
        src = C().figure(dataset)[1]
        d = C().load(dataset, T_K=T)
        for q in r.quantities:
            assert q.label == want, (dataset, T, q.name, q.label, want)
            assert q.source and src["cite"][:30] in q.source, (dataset, q.source)
            assert q.unit == cols[q.name][1]
            np.testing.assert_allclose(q.array(), np.asarray(d[cols[q.name][0]], float), rtol=0, atol=0)
            np.testing.assert_allclose(q.axes[0].values, np.asarray(d[axis[1]], float), rtol=0, atol=0)
        assert r.provenance["measured"].get("pdf_sha256") == src["pdf_sha256"]
        print(f"[R160] ingest {dataset} T={T}: label {want}, {len(r.quantities[0].axes[0].values)} points")


def test_R160_ingest_rsbi_curve_keeps_each_points_role(hub):
    """Fig. 4.1 is one curve over T whose 300 K point the catalog marks "calibration" (the others "validation"):
    a quantity labelled "validation" must not carry the 300 K point, nor one labelled "calibration" a cold point."""
    axis, cols = _ingest_args("luo2011.fig4_1")
    seen = []
    for T in (None, 300.0, 162.0):
        try:
            r = C().ingest(hub, "luo2011.fig4_1", axis=axis, columns=cols, T_K=T)
        except ValueError as e:                                   # a refusal is an acceptable answer
            seen.append((T, "refused", str(e)[:80]))
            continue
        for q in r.quantities:
            temps = np.asarray(q.axes[0].values, float) if q.axes else np.asarray([T or 300.0])
            bad = []
            for t in temps:
                want = C().role("luo2011.fig4_1", T_K=float(round(t, 6))) if abs(t - 300.0) < 0.5 else "validation"
                if want != q.label:
                    bad.append(float(t))
            seen.append((T, q.label, len(temps), bad[:3], len(bad)))
            print(f"[R160] ingest luo2011.fig4_1 T={T}: label {q.label}, {len(temps)} points, "
                  f"{len(bad)} of them of another role")
            assert not bad, (T, q.label, f"{len(bad)} points of another catalog role, e.g. {bad[:3]}")


def test_R160_ingest_refuses_trend_and_input(hub):
    refused = []
    for d in C().catalog()["datasets"]:
        if d["role"] in ("trend", "input"):
            with pytest.raises(ValueError):
                C().ingest(hub, d["id"], axis=("u", "u", "dimensionless"), columns={"x": ("B", "cm**-3")})
            refused.append(d["id"])
    assert "luo2011.fig1_2" in refused and any(C().role(i) == "trend" for i in refused)
    print(f"[R160] ingest refused for {len(refused)} trend / input datasets: {refused}")


# ---------------------------------------------------------------------------------------------- R168 (refusals, ids)
def _lib():
    from simws.spec import library
    return library


def test_R168_ranges_are_the_contracts():
    R = _lib().IBM5AM_RANGES
    for k, (lo, hi) in RANGES.items():
        assert tuple(R[k]) == pytest.approx((lo, hi)), (k, R[k])
    assert M().A23B_FIXED["s_um"] == pytest.approx(A23B_S_UM) and M().A23B_FIXED["k_E"] == pytest.approx(A23B_K_E)
    assert M().BVCBO_MEASURED_V == pytest.approx(BVCBO_MEAS_V)


BASE = dict(s_um=A23B_S_UM, k_B=0.63, k_E=A23B_K_E, k_P=0.3, S_p=3e4)


@pytest.mark.parametrize("knob", list(RANGES))
def test_R168_knobs_outside_their_ranges_refused(knob):
    lo, hi = RANGES[knob]
    for bad in (lo * 0.99, hi * 1.01):
        kw = dict(BASE, **{knob: bad})
        with pytest.raises(ValueError, match="device:"):
            _lib().ibm5am_hbt(**kw)
    for ok in (lo, hi):                                        # the ends belong to the range
        _lib().ibm5am_hbt(**dict(BASE, **{knob: ok}))
    print(f"[R168] {knob}: {lo * 0.99:g} and {hi * 1.01:g} refused ('device:'), {lo:g} and {hi:g} accepted")


def test_R168_id_fixed_by_knobs_L_and_T():
    f = _lib().ibm5am_hbt
    a = f(**BASE)
    assert a.problem_id() == f(**BASE).problem_id()
    assert a.model_dump(mode="json") == f(**BASE).model_dump(mode="json")
    changes = {"s_um": 0.82, "k_B": 0.70, "k_E": 1.2, "k_P": 0.5, "S_p": 1e5, "L_um": 1.0, "T": 162.0}
    ids = {a.problem_id()}
    for k, v in changes.items():
        pid = f(**dict(BASE, **{k: v})).problem_id()
        assert pid != a.problem_id(), k
        ids.add(pid)
    assert len(ids) == len(changes) + 1
    assert a.models.temperature_K == 300.0 and f(**dict(BASE, T=162.0)).models.temperature_K == 162.0
    print(f"[R168] the id is fixed by the knobs, L and T: {len(ids)} distinct ids for {len(changes)} changes + base")


_ID_SCRIPT = r"""
import json, sys
from simws.spec import library as L
out = {}
for name in ("pn_diode_1d", "n_bar_1d", "npn_hbt_2d", "implant_si", "see_si_slab", "rc_lowpass", "divider", "diode_iv"):
    out[name] = getattr(L, name)().problem_id()
out["pn_diode_1d_V"] = L.pn_diode_1d(V=[0.1, 0.2, 0.3]).problem_id()
out["ibm5am_a23"] = L.ibm5am_hbt(0.8256017267242715, 0.6256271624154321, 2.7841601866942445).problem_id()
out["ibm5am_cold"] = L.ibm5am_hbt(0.8, 0.63, 1.0, T=93.0, VBE=[0.9, 1.0], L_um=1.0).problem_id()
out["ibm5am_ii"] = L.ibm5am_hbt(0.8, 0.63, 1.0, VBE=[0.7], VCB=2.0, impact_ionization=True).problem_id()
print("IDS " + json.dumps(out))
"""


def _ids_in(tree: Path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(tree)
    p = subprocess.run([sys.executable, "-c", _ID_SCRIPT], cwd=str(tree), env=env, capture_output=True, text=True,
                       timeout=300)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("IDS ")]
    assert p.returncode == 0 and line, p.stderr[-2000:]
    return json.loads(line[-1][4:])


def _copy_simws(dst: Path):
    shutil.copytree(REPO / "simws", dst / "simws", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def test_R168_changed_data_file_changes_the_id(tmp_path):
    a, b = tmp_path / "same", tmp_path / "changed"
    _copy_simws(a)
    _copy_simws(b)
    f = b / "simws" / "data" / "validation" / "luo2011.json"
    d = json.loads(f.read_text())
    d["fig1_2"]["B"]["conc_cm3"] = [c * 1.01 for c in d["fig1_2"]["B"]["conc_cm3"]]
    f.write_text(json.dumps(d))
    ia, ib = _ids_in(a), _ids_in(b)
    here = _lib().ibm5am_hbt(0.8256017267242715, 0.6256271624154321, 2.7841601866942445).problem_id()
    print(f"[R168] ibm5am id {ia['ibm5am_a23'][:12]} (copy) / {here[:12]} (here); boron x 1.01 in the data file: "
          f"{ib['ibm5am_a23'][:12]}")
    assert ia["ibm5am_a23"] == here
    assert ib["ibm5am_a23"] != ia["ibm5am_a23"]
    assert ib["pn_diode_1d"] == ia["pn_diode_1d"]                 # other problems do not read the file


# ---------------------------------------------------------------------------------------------- R180 / R181 (contact)
WP_UM, WN_UM, MU_N, MU_P, NODES = 3.0, 2.0, 10.0, 400.0, 2001
NA, ND = 1e20, 1e17


def _diode_1d(S):
    from simws.spec.library import pn_diode_1d
    from simws.spec.schema import Problem
    p = pn_diode_1d(NA_cm3=NA, ND_cm3=ND, junction_um=WP_UM, length_um=WP_UM + WN_UM, mu_n=MU_N, mu_p=MU_P,
                    nodes=NODES, V=[R180_V])
    p = p.with_changes({"models.srh": "off", "models.tau_n_s": None, "models.tau_p_s": None})   # no recombination
    d = p.model_dump(mode="json")
    if S is not None:
        d["device"]["contacts"][1]["hole_recombination_velocity_cm_s"] = S
    p = Problem.model_validate(d)
    assert p.device.contacts[1].name == "cathode" and p.models.statistics == "boltzmann"
    assert p.models.band_gap_narrowing == "off" and p.models.mobility == "constant" and not p.models.auger
    return p


def _diode_2d(S, wx_um=0.5):
    """The R180 diode along y (2D DEVSIM contacts sit on y_min / y_max), extruded wx_um in x, the same mesh in y."""
    from simws.spec.schema import Box, Contact, DeviceSpec, DopingTerm, MeshHints, Problem, Region
    p1 = _diode_1d(S)
    d1 = p1.device
    dev = DeviceSpec(
        name="pn_diode_2d_extruded", dimension=2,
        regions=[Region(name="si", material="silicon", box=Box(x=(0.0, wx_um), y=d1.regions[0].box.x))],
        contacts=[Contact(name="anode", region="si", side="y_min"),
                  Contact(name="cathode", region="si", side="y_max", hole_recombination_velocity_cm_s=S)],
        doping=[DopingTerm(species=t.species, concentration_cm3=t.concentration_cm3, box=Box(x=(None, None), y=t.box.x),
                           include_upper=t.include_upper, label=t.label) for t in d1.doping],
        mesh=MeshHints(x_lines=[(0.0, wx_um / 4), (wx_um, wx_um / 4)], y_lines=list(d1.mesh.x_lines)),
        description="the R180 diode along y, extruded in x")
    return Problem(device=dev, models=p1.models, analysis=p1.analysis, numerics=p1.numerics)


def _closed_form(S):
    """A23b.1's J_p plus the p side's electron term (reading above), A/cm^2 into the anode."""
    from simws.materials import load
    from simws.materials.semiconductor import intrinsic_density
    mat = load("silicon")
    ni = intrinsic_density(mat, 300.0)
    Vt = vt(300.0)
    eps = mat.get("eps_r", "dimensionless") * EPS0_F_CM
    vbi = Vt * math.log(NA * ND / ni ** 2)
    xn = math.sqrt(2 * eps * (vbi - R180_V) * NA / (Q_E * ND * (NA + ND)))          # cm
    xp = xn * ND / NA
    Wn = WN_UM * 1e-4 - xn
    Wp = WP_UM * 1e-4 - xp
    Dp, Dn = MU_P * Vt, MU_N * Vt
    boltz = math.exp(R180_V / Vt) - 1.0
    jp = Q_E * (ni ** 2 / ND) * boltz / (Wn / Dp + (0.0 if S is None else 1.0 / S))
    jn = Q_E * (ni ** 2 / NA) * Dn / Wp * boltz
    return jp + jn, jp, jn, ni


_DIODE = {}


def _run(problem):
    pid = problem.problem_id()
    if pid not in _DIODE:
        from simws.engines import devsim_engine
        _DIODE[pid] = devsim_engine.run(problem)
    return _DIODE[pid]


def _j_1d(S):
    r = _run(_diode_1d(S))
    assert r.current_unit == "A/cm**2"
    return -r.currents["cathode"][0], r.currents["anode"][0], r


def test_R180_contact_closed_form_1d():
    pytest.importorskip("devsim")
    worst = 0.0
    for S in R180_S:
        j, ja, r = _j_1d(S)
        cf, jp, jn, ni = _closed_form(S)
        ratio = j / cf
        worst = max(worst, abs(ratio - 1))
        print(f"[R180] S = {S:.0e} cm/s: model {j:.5e} A/cm^2 (anode {ja:.5e}), closed form {cf:.5e} "
              f"(J_p {jp:.4e} + J_n {jn:.2e}); model / closed form {ratio:.4f} (limit 1 +- {R180_TOL}); "
              f"engine n_i {r.diagnostics.get('used', {}).get('n_i_cm3')} vs {ni:.4g}")
    for S in R180_S:
        j, _, _ = _j_1d(S)
        assert abs(j / _closed_form(S)[0] - 1) <= R180_TOL, S
    print(f"[R180] worst |model / closed form - 1| {worst:.4f} (limit {R180_TOL})")


def test_R180_high_S_equals_ohmic():
    pytest.importorskip("devsim")
    jo, jao, _ = _j_1d(None)
    jh, jah, _ = _j_1d(R180_S_HIGH)
    d = abs(jh / jo - 1)
    print(f"[R180] S = 1e8: {jh:.6e} against ohmic {jo:.6e} A/cm^2: {d * 100:.4f} % (limit {R180_OHMIC_TOL * 100} %); "
          f"ohmic / closed form {jo / _closed_form(None)[0]:.4f} (reported)")
    assert d <= R180_OHMIC_TOL


def test_R180_S_changes_the_current():
    """The contact acts: the current rises monotonically with S (a sanity check of the same runs, no tolerance)."""
    pytest.importorskip("devsim")
    js = [_j_1d(S)[0] for S in R180_S] + [_j_1d(None)[0]]
    assert all(b > a for a, b in zip(js, js[1:])), js


def test_R181_2d_diode_gives_the_1d_current():
    pytest.importorskip("devsim")
    wy = 0.5                                                      # the extrusion width (x), um
    worst = 0.0
    for S in R180_S:
        r = _run(_diode_2d(S, wy))
        assert r.current_unit == "A/cm"
        j2 = -r.currents["cathode"][0] / (wy * 1e-4)
        j1, _, _ = _j_1d(S)
        d = abs(j2 / j1 - 1)
        worst = max(worst, d)
        print(f"[R181] S = {S:.0e}: 2D {j2:.6e} against 1D {j1:.6e} A/cm^2: {d * 100:.4f} % "
              f"(limit {R181_TOL * 100} %); 2D anode {r.currents['anode'][0] / (wy * 1e-4):.6e}")
        assert d <= R181_TOL, S
    print(f"[R181] worst {worst * 100:.4f} % (limit {R181_TOL * 100} %)")


# ---------------------------------------------------------------------------------------------- R182 (refusals, ids)
def test_R182_contact_refusals():
    from pydantic import ValidationError

    from simws.spec.schema import Contact
    for bad in (0.0, -1.0, -1e5):
        with pytest.raises((ValueError, ValidationError)):
            Contact(name="e", region="si", side="x_min", hole_recombination_velocity_cm_s=bad)
    with pytest.raises((ValueError, ValidationError)):
        Contact(name="e", region="si", side="x_min", kind="schottky", barrier_eV=0.7,
                hole_recombination_velocity_cm_s=1e4)
    c = Contact(name="e", region="si", side="x_min", hole_recombination_velocity_cm_s=1e4)
    assert c.hole_recombination_velocity_cm_s == 1e4
    # refused through a whole description too
    from simws.spec.schema import Problem
    d = _diode_1d(None).model_dump(mode="json")
    d["device"]["contacts"][1]["hole_recombination_velocity_cm_s"] = 0.0
    with pytest.raises((ValueError, ValidationError)):
        Problem.model_validate(d)
    print("[R182] S <= 0 and S on a schottky contact refused (contact and whole description)")


def test_R182_charon_refuses_S_and_devsim_takes_it(charon_problem):
    from simws import engines
    from simws.spec.schema import Problem
    p = charon_problem()
    base = engines.unsupported("charon", p)
    d = p.model_dump(mode="json")
    d["device"]["contacts"][1]["hole_recombination_velocity_cm_s"] = 1e4
    q = Problem.model_validate(d)
    reasons = engines.unsupported("charon", q)
    new = [r for r in reasons if r not in base]
    print(f"[R182] charon reasons without S {base}; with S, new: {new}")
    assert new and any("recombination velocity" in r for r in new), reasons
    assert engines.unsupported("devsim", q) == engines.unsupported("devsim", p)


def test_R182_no_S_leaves_the_description_unchanged():
    from simws.spec.schema import Problem
    for p in (_diode_1d(None), _lib().npn_hbt_2d(), _lib().pn_diode_1d(), _lib().n_bar_1d(),
              _lib().ibm5am_hbt(**dict(BASE, k_P=1.0, S_p=None))):
        d = p.model_dump(mode="json")
        for c in d["device"]["contacts"]:
            assert "hole_recombination_velocity_cm_s" not in c, (p.device.name, c)
        assert Problem.model_validate(d).problem_id() == p.problem_id()
        assert Problem.model_validate_json(p.model_dump_json()).problem_id() == p.problem_id()
    assert _diode_1d(1e4).problem_id() != _diode_1d(None).problem_id()
    assert _diode_1d(1e4).problem_id() != _diode_1d(1e5).problem_id()
    print("[R182] without S no key is written and the id round-trips; S changes the id")


def test_R182_device_with_k_P_and_no_S_p():
    """A23b.2: ibm5am_hbt(..., k_P=1.0, S_p=None): S_p is optional (None = the ohmic emitter contact) for any k_P."""
    p = _lib().ibm5am_hbt(**dict(BASE, S_p=None))
    e = p.device.contact("emitter")
    assert e.hole_recombination_velocity_cm_s is None
    assert "hole_recombination_velocity_cm_s" not in p.model_dump(mode="json")["device"]["contacts"][0]
    assert p.problem_id() != _lib().ibm5am_hbt(**dict(BASE, k_P=1.0, S_p=None)).problem_id()   # k_P acts
    q = _lib().ibm5am_hbt(**BASE)
    assert q.device.contact("emitter").hole_recombination_velocity_cm_s == pytest.approx(BASE["S_p"])


def test_R182_library_ids_unchanged_from_before_A23b(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    ok = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", PRE_A23B_COMMIT + "^{commit}"],
                        capture_output=True)
    if ok.returncode != 0:
        pytest.skip(f"commit {PRE_A23B_COMMIT} not in this repository")
    old = tmp_path / "old"
    old.mkdir()
    arch = subprocess.run(["git", "-C", str(REPO), "archive", PRE_A23B_COMMIT, "simws"], capture_output=True)
    assert arch.returncode == 0, arch.stderr[-500:]
    subprocess.run(["tar", "-x", "-C", str(old)], input=arch.stdout, check=True)
    before = _ids_in(old)
    now = tmp_path / "now"
    _copy_simws(now)
    after = _ids_in(now)
    diff = sorted(k for k in before if before[k] != after.get(k))
    print(f"[R182] {len(before)} library problems: ids before A23b (commit {PRE_A23B_COMMIT}) and now "
          f"{'all equal' if not diff else 'DIFFER for ' + str(diff)}")
    assert not diff, diff


# ---------------------------------------------------------------------------------------------- unit checks
def test_unit_ionization_integral_constant_field():
    """1 - 1/M = int_0^W a_n exp(-int_0^y (a_n - a_p)) dy (A23.7(f)) with constant a_n, a_p:
    a_n / (a_n - a_p) (1 - exp(-(a_n - a_p) W)); van Overstraeten-de Man at T with gamma(T) (section 4 models)."""
    from simws.materials import load
    from simws.tcad.models import alpha_numpy
    mat = load("silicon")
    hw = mat.get("ii_hbar_omega", "eV")
    for T, E, W_um in ((300.0, 3.0e5, 0.3), (300.0, 4.0e5, 0.1), (162.0, 3.0e5, 0.2), (93.0, 2.5e5, 0.3)):
        gamma = math.tanh(hw / (2 * K_B_EV * 300.0)) / math.tanh(hw / (2 * K_B_EV * T))
        an, ap = (float(np.asarray(v, float).ravel()[0]) for v in alpha_numpy(mat, np.asarray([E]), gamma))
        W = W_um * 1e-4
        d = an - ap
        exact = an / d * (1 - math.exp(-d * W)) if abs(d) > 0 else an * W
        y = np.linspace(0.0, W, 4001)
        got = float(M().ionization_integral(y, np.full_like(y, E), T))
        m1 = 1.0 / (1.0 - exact) - 1.0
        # the contract does not say whether the function returns the integral (1 - 1/M) or M - 1 (what R164 gates);
        # either is accepted, the closed form decides which
        which = "1 - 1/M" if got == pytest.approx(exact, rel=1e-4) else (
            "M - 1" if got == pytest.approx(m1, rel=1e-4) else "neither")
        print(f"[unit] ionization_integral T {T:.0f} K, E {E:.1e} V/cm, W {W_um} um: {got:.6e}; closed form "
              f"1 - 1/M {exact:.6e}, M - 1 {m1:.6e}: matches {which}")
        assert which != "neither", (T, E, got, exact, m1)
    y = np.linspace(0.0, 1e-5, 101)
    assert float(M().ionization_integral(y, np.zeros_like(y), 300.0)) == pytest.approx(0.0, abs=1e-12)


def test_unit_vbe_at_current_on_an_exponential():
    T = 300.0
    I0 = 1e-15
    V = np.round(np.arange(0.40, 0.90001, 0.02), 4)
    I = I0 * np.exp(V / vt(T))
    for target in (1e-8, 1e-6, 3.3e-5, 1e-4):                     # inside I(0.40 V) = 5e-9 .. I(0.90 V)
        want = vt(T) * math.log(target / I0)
        got = M().vbe_at_current(V, I, target)
        assert got == pytest.approx(want, abs=1e-9), (target, got, want)
        assert my_vbe_at(V, I, target) == pytest.approx(want, abs=1e-9)
    # a grid point exactly
    assert M().vbe_at_current(V, I, float(I[5])) == pytest.approx(float(V[5]), abs=1e-9)


def test_unit_grade_values_several_terms_half_open_last_closed():
    from simws.engines.devsim_engine import grade_values
    from simws.spec.schema import GradeTerm
    g = [GradeTerm(axis="y", start_um=0.0, stop_um=0.1, x_start=0.0, x_stop=0.10),
         GradeTerm(axis="y", start_um=0.1, stop_um=0.2, x_start=0.05, x_stop=0.08),       # a jump at 0.1
         GradeTerm(axis="y", start_um=0.25, stop_um=0.3, x_start=0.02, x_stop=0.04)]      # a gap 0.2-0.25
    y = np.array([-0.01, 0.0, 0.05, 0.1 - 1e-9, 0.1, 0.15, 0.2, 0.2 + 1e-9, 0.22, 0.25, 0.275, 0.3, 0.31])
    want = np.array([0.0, 0.0, 0.05, 0.1 - 1e-9, 0.05, 0.065, 0.0, 0.0, 0.0, 0.02, 0.03, 0.04, 0.0])
    # 0.2 is the stop of a term that is not the last: half-open there (the next term starts at 0.25), so 0
    got = np.asarray(grade_values(g, y), float)
    print(f"[unit] grade_values {np.round(got, 6).tolist()}")
    np.testing.assert_allclose(got, want, atol=1e-12)
    # adjacent terms that join continuously: no doubling at the shared point
    h = [GradeTerm(axis="y", start_um=0.0, stop_um=0.1, x_start=0.0, x_stop=0.1),
         GradeTerm(axis="y", start_um=0.1, stop_um=0.2, x_start=0.1, x_stop=0.02)]
    np.testing.assert_allclose(np.asarray(grade_values(h, np.array([0.1, 0.2, 0.15])), float), [0.1, 0.02, 0.06],
                               atol=1e-12)


def test_unit_grade_values_one_term_as_before():
    """A23.7(b): one term is translated as before: the description's own evaluator (both ends included)."""
    from simws.engines.devsim_engine import grade_values
    from simws.spec.schema import GradeTerm
    p = _lib().npn_hbt_2d()
    assert len(p.device.grading) == 1
    gt = p.device.grading[0]
    c = np.linspace(-0.1, 1.6, 3401)
    c = np.concatenate([c, [gt.start_um, gt.stop_um]])
    if gt.axis == "y":
        ref = p.device.mole_fraction_at(np.zeros_like(c), c)
    else:
        ref = p.device.mole_fraction_at(c, np.zeros_like(c))
    np.testing.assert_allclose(np.asarray(grade_values([gt], c), float), ref, atol=1e-12)
    one = GradeTerm(axis="y", start_um=0.0, stop_um=0.1, x_start=0.0, x_stop=0.1)
    np.testing.assert_allclose(np.asarray(grade_values([one], np.array([0.0, 0.1, 0.1 + 1e-9])), float),
                               [0.0, 0.1, 0.0], atol=1e-12)


def test_unit_5am_ge_grading_matches_the_traced_curve():
    """A23.2: Ge from the traced curve, linear GradeTerms between its points, depth u at y = s (u - u_poly)."""
    from simws.engines.devsim_engine import grade_values
    k = dict(BASE)
    p = _lib().ibm5am_hbt(**k)
    g = p.device.grading
    assert len(g) > 1 and all(t.element == "Ge" and t.axis == "y" for t in g)
    assert len({t.mode for t in g}) == 1
    assert all(a.stop_um <= b.start_um + 1e-12 for a, b in zip(g, g[1:])), "in order, not overlapping"
    fig = C().load("luo2011.fig1_2")
    u = np.asarray(fig["Ge"]["u"], float)
    x = np.asarray(fig["Ge"]["fraction"], float)
    yy = k["s_um"] * (u - fig["u_poly_mono"])
    m = (yy > g[0].start_um + 1e-6) & (yy < g[-1].stop_um - 1e-6)
    got = np.asarray(grade_values(g, yy[m]), float)
    np.testing.assert_allclose(got, x[m], atol=1e-9)
    print(f"[unit] 5AM Ge: {len(g)} grading terms reproduce {int(m.sum())} traced points")


def test_R167_sige_film_reported():
    """R167 (reported): the SiGe film (Ge above 1 %) of the stored device, next to Sutton's 90 to 150 nm."""
    k = M().knobs()
    p = _lib().ibm5am_hbt(**k)
    y = np.linspace(0.0, 0.6, 600001)
    ge = p.device.mole_fraction_at(np.zeros_like(y), y)
    on = y[ge > 0.01]
    film = (on.max() - on.min()) * 1e3 if on.size else 0.0
    print(f"[R167] SiGe film (Ge > 1 %) {film:.1f} nm at s {k['s_um']:.4f} um, against Sutton's "
          f"{SUTTON_FILM_NM[0]:.0f} to {SUTTON_FILM_NM[1]:.0f} nm (reported)")
    assert film > 0


# ================================================================================================== SLOW TIER
@pytest.fixture(scope="module")
def k5():
    """The stored A23b calibration (A23b.3): k_P and S_p fitted, s and k_E fixed (A23b.2)."""
    k = M().knobs()
    missing = [n for n in ("k_P", "S_p") if n not in k]
    if missing:
        pytest.fail(f"the stored calibration is not A23b's (no {missing}): {k}")
    assert k["s_um"] == pytest.approx(A23B_S_UM) and k["k_E"] == pytest.approx(A23B_K_E), k
    for n, (lo, hi) in RANGES.items():
        assert lo <= k[n] <= hi, (n, k[n])
    return k


@pytest.fixture(scope="module")
def g300(k5):
    g = M().gummel(k5, GRID300)
    V = np.asarray(g["VBE_V"], float)
    assert np.allclose(V, GRID300), V
    return {"V": V, "IC": np.asarray(g["IC_A"], float), "IB": np.asarray(g["IB_A"], float),
            "RsBi": float(g["RsBi_ohm_sq"])}


_COLD = {}


def cold(k, T):
    """gummel over gummel_window(T) at T, cached; an exception is kept and reported as the failure."""
    if T not in _COLD:
        try:
            win = list(M().gummel_window(T))
            g = M().gummel(k, win, T=T)
            _COLD[T] = {"V": np.asarray(g["VBE_V"], float), "IC": np.asarray(g["IC_A"], float),
                        "IB": np.asarray(g["IB_A"], float), "RsBi": float(g["RsBi_ohm_sq"]), "error": None}
        except Exception as e:  # noqa: BLE001
            _COLD[T] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    return _COLD[T]


def need_ok(res, what):
    if res.get("error"):
        pytest.fail(f"{what}: no converged solution ({res['error']})")
    return res


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="A23b result: IC 1.053 to 1.175 x Luo (worst 17.5 % > 15 %, at the low-VBE end); IB and RsBi pass")
def test_R161_calibration_fit(g300):
    V, I = luo_ic(300.0)
    m = in_window(V, R161_IC_WIN)
    ic_model = interp_log(V[m], g300["V"], g300["IC"])
    r_ic = ic_model / I[m]
    vb, beta = luo_beta(300.0)
    mb = in_window(vb, R161_IB_WIN)
    ib_meas = interp_log(vb[mb], V, I) / beta[mb]
    r_ib = interp_log(vb[mb], g300["V"], g300["IB"]) / ib_meas
    rs_meas = luo_rsbi_at(300.0)
    r_rs = g300["RsBi"] / rs_meas
    w_ic, w_ib, w_rs = np.max(np.abs(r_ic - 1)), np.max(np.abs(r_ib - 1)), abs(r_rs - 1)
    print(f"[R161] IC model / Luo 300 K at {m.sum()} traced VBE in {R161_IC_WIN}: {r_ic.min():.3f} to "
          f"{r_ic.max():.3f} (worst {w_ic * 100:.1f} %, limit {R161_IC_TOL * 100:.0f} %)")
    print(f"[R161] IB = IC / beta at {mb.sum()} traced VBE in {R161_IB_WIN}: model / measured {r_ib.min():.3f} to "
          f"{r_ib.max():.3f} (worst {w_ib * 100:.1f} %, limit {R161_IB_TOL * 100:.0f} %)")
    print(f"[R161] RsBi(300 K) model {g300['RsBi']:.0f} against {rs_meas:.0f} ohm/sq: {r_rs:.3f} "
          f"(limit {R161_RS_TOL * 100:.0f} %)")
    assert np.all(np.isfinite(r_ic)) and np.all(np.isfinite(r_ib))
    assert w_ic <= R161_IC_TOL and w_ib <= R161_IB_TOL and w_rs <= R161_RS_TOL, (w_ic, w_ib, w_rs)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="A23b result: IC 1.152 to 1.257 x Pellish (worst 25.7 % > 25 %)")
def test_R162_pellish_chip(g300):
    V, IC, IB = pellish_gummel()
    m = in_window(V, R162_WIN)
    r = interp_log(V[m], g300["V"], g300["IC"]) / IC[m]
    worst = float(np.max(np.abs(r - 1)))
    rb = interp_log(V[m], g300["V"], g300["IB"]) / IB[m]
    beta_model = g300["IC"] / g300["IB"]
    mm = in_window(g300["V"], R162_WIN)
    print(f"[R162] IC model / Pellish at {m.sum()} traced VBE in {R162_WIN}: {r.min():.3f} to {r.max():.3f} "
          f"(worst {worst * 100:.1f} %, limit {R162_TOL * 100:.0f} %)")
    print(f"[R162] reported: IB model / Pellish {rb.min():.2f} to {rb.max():.2f}; model beta {beta_model[mm].min():.0f} "
          f"to {beta_model[mm].max():.0f}, Pellish beta {np.min(IC[m] / IB[m]):.0f} to {np.max(IC[m] / IB[m]):.0f}")
    assert np.all(np.isfinite(r)) and worst <= R162_TOL, worst


@pytest.fixture(scope="module")
def early(k5):
    return {ib: M().early_voltage(k5, ib * 1e-6) for ib in R163_IB_UA}


@pytest.mark.slow
@pytest.mark.parametrize("ib_uA", R163_IB_UA)
@pytest.mark.xfail(strict=True, reason="A23b result: V_A 22 / 24 V against 139 / 267 V (factors 6.2, 11.2): the reconstructed base-collector profile")
def test_R163_early_voltage(early, ib_uA):
    e = early[ib_uA]
    va_m, n_m = line_va(e["VCE_V"], e["IC_A"])
    vce, ic = pellish_out(ib_uA)
    va_meas, n_meas = line_va(vce, ic)
    f = factor(va_m, va_meas)
    ib_at_1 = float(np.interp(1.0, np.asarray(e["VCE_V"], float), np.asarray(e["IB_A"], float)))
    print(f"[R163] IB {ib_uA} uA: model V_A {va_m:.1f} V ({n_m} points, VBE {e.get('VBE_V')}, IB at 1 V "
          f"{ib_at_1 * 1e6:.2f} uA; early_voltage() says {e.get('V_A')}) against Pellish {va_meas:.1f} V "
          f"({n_meas} points; early_voltage_measured() {M().early_voltage_measured(ib_uA):.1f}): factor {f:.2f} "
          f"(limit {R163_FACTOR})")
    assert f <= R163_FACTOR, f


@pytest.fixture(scope="module")
def aval300(k5):
    return M().avalanche(k5, T=300.0, VCB=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0))


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="A23b result: M - 1 factors 21.9 / 7.3 / 3.8 / 2.8 at VCB 2 to 5 V: the collector field peak too high at low VCB")
def test_R164_avalanche_300K(aval300):
    a = aval300
    vcb = np.asarray(a["VCB_V"], float)
    m1 = np.asarray(a["M_minus_1"], float)
    x, y = luo_m1(300.0)
    ic0 = np.asarray(a["IC_A"], float)[int(np.argmin(np.abs(vcb)))] if "IC_A" in a else float("nan")
    fs = []
    for v in R164_VCB:
        mod = float(m1[int(np.argmin(np.abs(vcb - v)))])
        assert abs(vcb[int(np.argmin(np.abs(vcb - v)))] - v) < 1e-9, (v, vcb)
        meas = float(interp_log([v], x, y)[0])
        fs.append(factor(mod, meas))
        print(f"[R164] VCB {v:.0f} V: model M - 1 {mod:.4g} against Luo 300 K {meas:.4g}: factor {fs[-1]:.2f} "
              f"(limit {R164_FACTOR})")
    print(f"[R164] VBE {a.get('VBE_V')} V, IC at VCB 0 {ic0 * 1e6:.2f} uA (A23.4: 10 uA)")
    assert max(fs) <= R164_FACTOR, fs


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="A23b result: peak fT 69.7 GHz against 46.9 GHz (+48 % > 25 %)")
def test_R165_ft_peak(k5):
    f = M().ft_curve(k5, R165_VBE)
    ft, je = np.asarray(f["fT_GHz"], float), np.asarray(f["JE_mA_um2"], float)
    i = int(np.nanargmax(ft))
    je_m, ft_m = luo_ft()
    j = int(np.argmax(ft_m))
    r_pk = rel(ft[i], ft_m[j])
    f_je = factor(je[i], je_m[j])
    print(f"[R165] peak fT model {ft[i]:.1f} GHz at JE {je[i]:.3f} mA/um^2 (VBE {f['VBE_V'][i]}) against Luo "
          f"{ft_m[j]:.1f} GHz at {je_m[j]:.3f}: {r_pk * 100:+.1f} % (limit {R165_PEAK_TOL * 100:.0f} %), JE factor "
          f"{f_je:.2f} (limit {R165_JE_FACTOR}); {'peak at the last bias point' if i == len(ft) - 1 else ''}")
    assert r_pk <= R165_PEAK_TOL and f_je <= R165_JE_FACTOR, (r_pk, f_je)


@pytest.mark.slow
def test_R166_mesh(k5, g300):
    i = GRID300.index(R166_VBE)
    g = M().gummel(k5, [R166_VBE], mesh_scale=R166_SCALE)
    ic, ib = float(g["IC_A"][0]), float(g["IB_A"][0])
    dic, dib = ic / g300["IC"][i] - 1, ib / g300["IB"][i] - 1
    print(f"[R166] VBE {R166_VBE} V, mesh scale {R166_SCALE}: IC {dic * 100:+.2f} %, IB {dib * 100:+.2f} % "
          f"(limit {R166_TOL * 100:.0f} %)")
    assert abs(dic) < R166_TOL and abs(dib) < R166_TOL, (dic, dib)


@pytest.mark.slow
def test_R167_moll_ross_reported(k5, g300):
    mr = [float(v) for v in M().moll_ross(k5, list(R167_VBE))]
    for v, ref in zip(R167_VBE, mr):
        ic = float(g300["IC"][GRID300.index(v)])
        print(f"[R167] VBE {v} V: IC {ic:.4e} A, Moll-Ross / Kroemer {ref:.4e} A: {ic / ref:.3f} (reported)")
    assert all(np.isfinite(mr)) and all(v > 0 for v in mr)


# ---------------------------------------------------------------------------------------------- R169 (trust)
@pytest.fixture(scope="module")
def trust_setup(k5, tmp_path_factory):
    from simws.hub.jobs import Hub
    from simws.spec.library import ibm5am_hbt, ibm5am_scale
    hub = Hub(tmp_path_factory.mktemp("a23_trust"), mode="inprocess")
    p = ibm5am_hbt(**k5, VBE=R169_GRID)
    r = hub.submit("device.devsim", params={"problem": p.model_dump(mode="json")})
    assert r.status == "ok", f"{r.status}: {r.message[-500:]}"
    res = hub.result(r.id)
    q = res.q("dc.I.collector")
    f_m = ibm5am_scale(2.5) * 1e-2            # cm of depth per device (2 x L, A23.2) -> m: A/m x m = A
    assert q.unit == "A/m", q.unit
    qd = q.model_copy(update={"value": (q.array() * f_m).tolist(), "unit": "A"})
    luo = C().ingest(hub, "luo2011.fig5_7a", axis=("VBE", "VBE_V", "V"), columns={"IC": ("IC_mA", "mA")}, T_K=300.0)
    pel = C().ingest(hub, "pellish2008.fig3", axis=("VBE", "VBE_V", "V"), columns={"IC": ("IC_A", "A")})
    return hub, res, qd, luo, pel


def _cmp(hub, res, qd, meas, tol, win):
    from simws.compare import compare, record
    c = compare(qd, meas.q("IC"), tolerance=tol, metric="relative", window=win, interpolation="log")
    c = c.model_copy(update={"a": res.id, "b": meas.id, "note": "model dc.I.collector x 2 L (one device)"})
    record(hub, c)
    return c


@pytest.mark.slow
def test_R169_trust(trust_setup):
    from simws.hub.trust import assess, assess_stored
    hub, res, qd, luo, pel = trust_setup
    assert luo.quantities[0].label == "calibration" and pel.quantities[0].label == "validation"
    lv0 = assess_stored(hub, res.id).level
    c_cal = _cmp(hub, res, qd, luo, R161_IC_TOL, R161_IC_WIN)
    lv1 = assess_stored(hub, res.id).level
    assert c_cal.kind == "measurement" and c_cal.measurement_label == "calibration"
    assert lv1 != "validated", "R161 is recorded against calibration data and validates nothing"
    c_val = _cmp(hub, res, qd, pel, R162_TOL, R162_WIN)
    t = assess_stored(hub, res.id)
    assert c_val.kind == "measurement" and c_val.measurement_label == "validation"
    expect = "validated" if (c_val.passed and c_cal.passed) else None
    print(f"[R169] 300 K result {res.id[:12]}: before {lv0}; after R161 (calibration, worst {c_cal.worst:.3f}, "
          f"{'pass' if c_cal.passed else 'FAIL'}) {lv1}; after R162 (validation, worst {c_val.worst:.3f}, "
          f"{'pass' if c_val.passed else 'FAIL'}) {t.level}; reasons {t.reasons}")
    if expect:
        assert t.level == "validated", t
    else:
        assert t.level != "validated", t
    # the rules themselves on this envelope: a passed calibration comparison alone never validates; a passed
    # validation comparison alone does; one failed measurement comparison keeps it below "validated"
    cal, val = c_cal.model_dump(), c_val.model_dump()
    assert assess(res, [dict(cal, passed=True, inconclusive=False)]).level != "validated"
    assert assess(res, [dict(val, passed=True, inconclusive=False)]).level == "validated"
    assert assess(res, [dict(val, passed=True, inconclusive=False),
                        dict(cal, passed=False, inconclusive=False)]).level != "validated"


# ---------------------------------------------------------------------------------------------- A24 (cold)
@pytest.mark.slow
@pytest.mark.parametrize("T", R170_T)
def test_R170_vbe_at_1uA_and_100uA(k5, T):
    g = need_ok(cold(k5, T), f"R170 at {T:.0f} K")
    V, I = luo_ic(T)
    tg = M().luo_targets(T)
    out = []
    for target in R170_I:
        vm, vmeas = my_vbe_at(g["V"], g["IC"], target), my_vbe_at(V, I, target)
        assert vmeas is not None, (T, target)
        d = None if vm is None else vm - vmeas
        out.append(d)
        print(f"[R170] {T:.0f} K, IC {target * 1e6:.0f} uA: model VBE {vm} against Luo {vmeas:.4f} V "
              f"(luo_targets {tg.get('VBE_1uA' if target == 1e-6 else 'VBE_100uA')}): "
              f"{'no crossing in the window' if d is None else f'{d * 1e3:+.1f} mV'} (limit {R170_TOL_V * 1e3:.0f} mV)")
    assert all(d is not None and abs(d) <= R170_TOL_V for d in out), out


@pytest.mark.slow
@pytest.mark.parametrize("T", list(R171))
def test_R171_peak_beta(k5, T):
    g = need_ok(cold(k5, T), f"R171 at {T:.0f} K")
    beta = g["IC"] / g["IB"]
    i = int(np.argmax(beta))
    vb, b = luo_beta(T)
    j = int(np.argmax(b))
    kind, tol = R171[T]
    v, ok = gate(kind, tol, beta[i], b[j])
    edge = " (at the window's edge)" if i in (0, len(beta) - 1) else ""
    print(f"[R171] {T:.0f} K: peak beta model {beta[i]:.1f} at {g['V'][i]:.3f} V{edge} against Luo {b[j]:.1f} at "
          f"{vb[j]:.3f} V: {'ratio' if kind == 'rel' else 'factor'} {beta[i] / b[j]:.3f} "
          f"(limit {'+-' + str(int(tol * 100)) + ' %' if kind == 'rel' else 'factor ' + str(tol)})")
    assert ok, (beta[i], b[j])


@pytest.mark.slow
@pytest.mark.parametrize("T", [T if T not in R172_XFAIL else pytest.param(T, marks=pytest.mark.xfail(
    strict=True, reason=R172_XFAIL[T])) for T in R172])
def test_R172_rsbi(k5, T):
    if T in (223.0, 162.0):
        rs = need_ok(cold(k5, T), f"R172 at {T:.0f} K")["RsBi"]
    else:
        try:
            rs = float(M().rsbi(k5, T))
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"R172 at {T:.0f} K: no result ({type(e).__name__}: {str(e)[:200]})")
    meas = luo_rsbi_at(T)
    kind, tol = R172[T]
    v, ok = gate(kind, tol, rs, meas)
    print(f"[R172] {T:.0f} K: RsBi model {rs:.0f} against Luo {meas:.0f} ohm/sq: {rs / meas:.3f} "
          f"(limit {'+-' + str(int(tol * 100)) + ' %' if kind == 'rel' else 'factor ' + str(tol)})")
    assert ok, (rs, meas)


@pytest.mark.slow
def test_R172_reported_to_the_lowest_temperature(k5):
    t, r = luo_rsbi()
    low = sorted({float(t[int(np.argmin(np.abs(t - x)))]) for x in (70.0, 60.0, 50.0, 40.0, 30.0)} |
                 {float(t.min())}, reverse=True)                      # the trace's points near 70 ... 30 K, and 20 K
    for T in low:
        meas = luo_rsbi_at(T)
        try:
            rs = float(M().rsbi(k5, float(T)))
            print(f"[R172] reported {T:.1f} K: model {rs:.4g} against Luo {meas:.4g} ohm/sq: {rs / meas:.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"[R172] reported {T:.1f} K: no result ({type(e).__name__}: {str(e)[:120]}); Luo {meas:.4g}")


@pytest.mark.slow
@pytest.mark.parametrize("T", R173_T)
@pytest.mark.xfail(strict=True, reason="A23b result: M - 1 factors 20.3 to 3.1 (162 K), 16.0 to 3.1 (93 K), as R164")
def test_R173_avalanche_cold(k5, T):
    try:
        a = M().avalanche(k5, T=T, VCB=(0.0, 1.0) + R173_VCB)
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"R173 at {T:.0f} K: no result ({type(e).__name__}: {str(e)[:200]})")
    vcb = np.asarray(a["VCB_V"], float)
    m1 = np.asarray(a["M_minus_1"], float)
    x, y = luo_m1(T)
    fs = []
    for v in R173_VCB:
        i = int(np.argmin(np.abs(vcb - v)))
        assert abs(vcb[i] - v) < 1e-9
        meas = float(interp_log([v], x, y)[0])
        fs.append(factor(float(m1[i]), meas))
        print(f"[R173] {T:.0f} K, VCB {v:.0f} V: model M - 1 {m1[i]:.4g} against Luo {meas:.4g}: factor {fs[-1]:.2f} "
              f"(limit {R173_FACTOR})")
    assert max(fs) <= R173_FACTOR, fs


@pytest.mark.slow
@pytest.mark.parametrize("T", R174_T)
def test_R174_sutton_chip(k5, T):
    V, J = sutton_jc(T)
    vmeas = my_vbe_at(V, J, R174_J)
    assert vmeas is not None
    c = round(vmeas, 2)
    grid = [round(c + 0.02 * i, 2) for i in range(-4, 5)]
    try:
        g = M().gummel(k5, grid, T=T, L_um=R174_L_UM)
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"R174 at {T:.0f} K: no converged solution ({type(e).__name__}: {str(e)[:200]})")
    jc = np.asarray(g["IC_A"], float) / R174_AREA_UM2
    vm = my_vbe_at(np.asarray(g["VBE_V"], float), jc, R174_J)
    d = None if vm is None else vm - vmeas
    print(f"[R174] {T:.0f} K, JC 10 nA/um^2 (L 1.0 um): model VBE {vm} against Sutton {vmeas:.4f} V: "
          f"{'no crossing in ' + str(grid[0]) + '-' + str(grid[-1]) + ' V' if d is None else f'{d * 1e3:+.1f} mV'} "
          f"(limit {R174_TOL_V * 1e3:.0f} mV)")
    assert d is not None and abs(d) <= R174_TOL_V, d


def _ideality(V, I, T):
    v1, v2 = my_vbe_at(V, I, 1e-6), my_vbe_at(V, I, 1e-4)
    if v1 is None or v2 is None:
        return None
    return (v2 - v1) / (vt(T) * math.log(100.0))


@pytest.mark.slow
def test_R175_reported(k5):
    for T in (93.0, 43.0):
        g = cold(k5, T)
        V, I = luo_ic(T)
        n_luo = _ideality(V, I, T)
        if g.get("error"):
            print(f"[R175] {T:.0f} K: no result ({g['error']}); Luo's ideality (Fig. 5.7(a) slope, 1 to 100 uA) "
                  f"{n_luo}")
            continue
        n_m = _ideality(g["V"], g["IC"], T)
        beta = g["IC"] / g["IB"]
        print(f"[R175] {T:.0f} K: model ideality {n_m} against Luo's {n_luo} (his Fig. 5.7(b): 1.35 at 43 K); "
              f"peak beta {beta.max():.1f}; RsBi {g['RsBi']:.4g} ohm/sq")


# ---------------------------------------------------------------------------------------------- R183 (BVCBO)
@pytest.mark.slow
def test_R183_bvcbo(k5):
    b = M().bvcbo(k5)
    bv = float(b["BVCBO_V"])
    r = rel(bv, BVCBO_MEAS_V)
    print(f"[R183] BVCBO model {bv:.2f} V against {BVCBO_MEAS_V} V (Lourenco Table 6.1): {r * 100:+.1f} % "
          f"(limit {R183_TOL * 100:.0f} %); knobs {k5}")
    assert r <= R183_TOL, bv
