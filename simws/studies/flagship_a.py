"""Flagship study A: SiGe HBT temperature window (spec: simws/studies/specs/flagship_a.yaml).

    from simws.studies.flagship_a import run
    res = run(outdir="results/flagship_a")          # submits every job through the hub, then analyses

Every number comes from a committed hub result (job key, flags, lineage in the knowledge graph); the analysis only
interpolates and fits. Results outside the trusted window keep their 'extrapolated' flag in every table row.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

from simws.units import K_B_EV

SPEC = Path(__file__).with_name("specs") / "flagship_a.yaml"


def load_spec(path=None) -> dict:
    return yaml.safe_load(Path(path or SPEC).read_text())


def _interp_log(x, y, x0):
    """y at x0 by linear interpolation of log(y) in x (None outside the range)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not (x.min() <= x0 <= x.max()):
        return None
    return float(np.exp(np.interp(x0, x, np.log(y))))


def _at_current(IC, other, I0):
    """other at IC = I0 by interpolation in log IC (None outside the range)."""
    IC, other = np.asarray(IC, float), np.asarray(other, float)
    o = np.argsort(IC)
    if not (IC.min() <= I0 <= IC.max()):
        return None
    return float(np.interp(math.log(I0), np.log(IC[o]), other[o]))


def graded_form(T, A, dE):
    x = dE / (K_B_EV * np.asarray(T, float))
    return A * x / (1.0 - np.exp(-x))


def fit_models(T, ratio):
    """Least-squares fits of ln(ratio): graded base A x/(1-e^-x) and abrupt exp form B exp(dE/kT)."""
    T, r = np.asarray(T, float), np.asarray(ratio, float)
    out = {}
    # abrupt: ln r = ln B + dE/kT  (linear)
    X = np.column_stack([np.ones_like(T), 1.0 / (K_B_EV * T)])
    c, *_ = np.linalg.lstsq(X, np.log(r), rcond=None)
    pred = X @ c
    out["exp_form"] = {"B": float(np.exp(c[0])), "dE_eV": float(c[1]),
                       "rms_log_resid": float(np.sqrt(np.mean((pred - np.log(r)) ** 2)))}
    # graded: scan dE, A from the mean log residual
    best = None
    for dE in np.linspace(0.005, 0.4, 4000):
        g = np.log(graded_form(T, 1.0, dE))
        lnA = float(np.mean(np.log(r) - g))
        rms = float(np.sqrt(np.mean((g + lnA - np.log(r)) ** 2)))
        if best is None or rms < best[2]:
            best = (math.exp(lnA), float(dE), rms)
    out["graded_form"] = {"A": best[0], "dE_grade_eV": best[1], "rms_log_resid": best[2]}
    return out


def run(outdir, spec=None, hub=None, log=print) -> dict:
    from simws.hub.jobs import Hub
    s = load_spec(spec) if not isinstance(spec, dict) else spec
    hub = hub or Hub()
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    sw = s["sweep"]
    V = np.round(np.arange(sw["VBE_V"]["start"], sw["VBE_V"]["stop"] + 1e-9, sw["VBE_V"]["step"]), 6)
    lo, hi = s["validity"]["trusted_T_K"]
    rows, raw = [], {}
    for T in sw["T_K"]:
        per = {}
        for dev, d in s["devices"].items():
            r = hub.submit(d["adapter"], params={"case": d["case"], "VBE": (V, "V"), "T": (float(T), "K"),
                                                 "keep_partial": True}, priority="batch")
            per[dev] = r
            log(f"  T={T:>5} K {dev:9s} {r.status:15s} {', '.join(r.flags)[:110]}")
        raw[T] = per
        row = {"T_K": T, "trusted": bool(lo <= T <= hi), "status": {k: v.status for k, v in per.items()},
               "flags": sorted({f for v in per.values() for f in v.flags}),
               "result_ids": {k: v.id for k, v in per.items()}}
        for dev, r in per.items():
            if r.status != "ok":
                continue
            VB = np.asarray(r.q("hbt.IC").axes[0].values)
            IC, beta = r.q("hbt.IC").array(), r.q("hbt.beta").array()
            I0 = s["compare_at"]["JC_A_per_m"]
            row[f"{dev}.beta_at_JC"] = _at_current(IC, beta, I0)
            row[f"{dev}.VBE_at_JC"] = _at_current(IC, VB, I0)
            row[f"{dev}.VBE_range"] = [float(VB.min()), float(VB.max())]
        if all(v.status == "ok" for v in per.values()):
            a, b = per["sige_hbt"], per["si_bjt"]
            va, vb = np.asarray(a.q("hbt.IC").axes[0].values), np.asarray(b.q("hbt.IC").axes[0].values)
            v0 = s["compare_at"]["VBE_V"]
            common_hi = min(va.max(), vb.max())
            common_lo = max(va.min(), vb.min())
            # low injection: the requested VBE if both devices have it, else the lowest common bias above the
            # current floor (cold devices); never the top of the sweep, where high injection changes the ratio
            vc = v0 if common_lo <= v0 <= common_hi else common_lo
            ra, rb = _interp_log(va, a.q("hbt.IC").array(), vc), _interp_log(vb, b.q("hbt.IC").array(), vc)
            if ra and rb:
                row["IC_ratio"] = ra / rb
                row["IC_ratio_VBE"] = float(vc)
        rows.append(row)
    trusted = [r for r in rows if r["trusted"] and "IC_ratio" in r]
    fits = fit_models([r["T_K"] for r in trusted], [r["IC_ratio"] for r in trusted]) if len(trusted) >= 3 else {}
    allr = [r for r in rows if "IC_ratio" in r]
    if fits:
        g = fits["graded_form"]
        for r in allr:
            r["IC_ratio_graded_model"] = float(graded_form(r["T_K"], g["A"], g["dE_grade_eV"]))
    res = {"spec": s["id"], "version": s["version"], "fidelity": s["fidelity"], "rows": rows, "fits_trusted_window": fits,
           "graph": {"nodes": hub.graph.to_networkx().number_of_nodes()}}
    (out / "flagship_a.json").write_text(json.dumps(res, indent=1, default=float))
    _table(rows, out / "flagship_a_table.txt")
    try:
        _figure(rows, fits, (lo, hi), out / "flagship_a.png")
    except Exception as e:  # noqa: BLE001  the figure is optional
        log(f"  figure skipped: {e}")
    return res


def _fmt(v, f="{:.3g}"):
    return "-" if v is None else f.format(v)


def _table(rows, path):
    hdr = (f"{'T (K)':>6} {'trust':>5} {'beta Si':>8} {'beta SiGe':>9} {'VBE Si':>7} {'VBE SiGe':>8} "
           f"{'IC ratio':>9} {'@VBE':>5}  flags")
    lines = [hdr]
    for r in rows:
        lines.append(f"{r['T_K']:>6} {'yes' if r['trusted'] else 'no':>5} {_fmt(r.get('si_bjt.beta_at_JC')):>8} "
                     f"{_fmt(r.get('sige_hbt.beta_at_JC')):>9} {_fmt(r.get('si_bjt.VBE_at_JC'), '{:.3f}'):>7} "
                     f"{_fmt(r.get('sige_hbt.VBE_at_JC'), '{:.3f}'):>8} {_fmt(r.get('IC_ratio')):>9} "
                     f"{_fmt(r.get('IC_ratio_VBE'), '{:.2f}'):>5}  {'; '.join(r['flags'])[:90]}")
    Path(path).write_text("\n".join(lines) + "\n")


def _figure(rows, fits, window, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c_si, c_sg, ink, grid = "#2a6fdb", "#d9822b", "#333333", "#e6e6e6"
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a in ax:
        a.axvspan(window[0], window[1], color="#f3f3f3", zorder=0)
        a.grid(color=grid, lw=0.6); a.set_xlabel("T (K)", color=ink)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    for dev, col, mk, lab in (("si_bjt", c_si, "o", "Si BJT"), ("sige_hbt", c_sg, "s", "SiGe HBT")):
        pts = [(r["T_K"], r.get(f"{dev}.beta_at_JC"), r["trusted"]) for r in rows if r.get(f"{dev}.beta_at_JC")]
        if pts:
            T, b, t = zip(*pts)
            ax[0].plot(T, b, color=col, lw=2, label=lab)
            for x, y, tr in pts:
                ax[0].plot(x, y, mk, ms=8, mfc=col if tr else "white", mec=col, mew=2)
    ax[0].set_ylabel("current gain at 1 A/m", color=ink); ax[0].legend(frameon=False)
    ax[0].set_title("Gain (hollow = outside the trusted window)", fontsize=10, color=ink)
    pts = [(r["T_K"], r["IC_ratio"], r["trusted"]) for r in rows if "IC_ratio" in r]
    if pts:
        T, q, t = zip(*pts)
        ax[1].plot(T, q, color=c_sg, lw=2, label="DEVSIM")
        for x, y, tr in pts:
            ax[1].plot(x, y, "s", ms=8, mfc=c_sg if tr else "white", mec=c_sg, mew=2)
        if fits:
            g, e = fits["graded_form"], fits["exp_form"]
            Tm = np.linspace(min(T), max(T), 200)
            ax[1].plot(Tm, e["B"] * np.exp(e["dE_eV"] / (K_B_EV * Tm)), "-", color=ink, lw=1.0,
                       label=f"B exp(dE/kT), dE = {1e3 * e['dE_eV']:.0f} meV (rms {e['rms_log_resid']:.3f})")
            ax[1].plot(Tm, graded_form(Tm, g["A"], g["dE_grade_eV"]), "--", color=ink, lw=1.0,
                       label=f"graded-only x/(1-e^-x) (rms {g['rms_log_resid']:.2f})")
        ax[1].set_yscale("log"); ax[1].legend(frameon=False)
    ax[1].set_ylabel("IC(SiGe) / IC(Si) at fixed VBE", color=ink)
    ax[1].set_title("Collector-current enhancement (fits: shaded 250-400 K only)", fontsize=10, color=ink)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
