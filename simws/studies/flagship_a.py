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


def fit_edge_plus_graded(T, ratio, dEg_edge):
    """H1 follow-up: the ratio is exp(dEg_edge/kT) (Ge already present at the emitter edge of the NEUTRAL base,
    taken from the device at each T, not fitted) times the graded-base factor A x/(1 - e^-x) of the remaining
    grade. Fits A and the remaining grade dE; a small residual with one dE at every T supports the explanation."""
    T, r, e = (np.asarray(v, float) for v in (T, ratio, dEg_edge))
    y = np.log(r) - e / (K_B_EV * T)
    best = None
    for dE in np.linspace(0.001, 0.4, 4000):
        g = np.log(graded_form(T, 1.0, dE))
        lnA = float(np.mean(y - g))
        rms = float(np.sqrt(np.mean((g + lnA - y) ** 2)))
        if best is None or rms < best[2]:
            best = (math.exp(lnA), float(dE), rms)
    return {"A": best[0], "dE_grade_remaining_eV": best[1], "rms_log_resid": best[2],
            "dEg_edge_eV": e.tolist()}


def _submit_one(store_root, adapter, params, fidelity):
    """Worker thread: its own Hub (SQLite connections must not cross threads); every job still runs in its own
    adapter process, and commits are serialised by the store's flock."""
    from simws.hub.jobs import Hub
    return Hub(store_root).submit(adapter, params=params, priority="batch", fidelity=fidelity).id


def run(outdir, spec=None, hub=None, log=print, workers: int = 1) -> dict:
    """workers > 1 submits the (T, device) jobs from that many threads at once (each job is its own isolated
    adapter process anyway); results are then read back from the store, so the analysis is identical."""
    from simws.hub.jobs import Hub
    s = load_spec(spec) if not isinstance(spec, dict) else spec
    hub = hub or Hub()
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    sw = s["sweep"]
    V = np.round(np.arange(sw["VBE_V"]["start"], sw["VBE_V"]["stop"] + 1e-9, sw["VBE_V"]["step"]), 6)
    lo, hi = s["validity"]["trusted_T_K"]
    rows, raw = [], {}
    fid = s.get("fidelity", "screening")

    def params_for(T, d):
        p = {"case": d["case"], "VBE": (V, "V"), "T": (float(T), "K"), "keep_partial": True}
        if s.get("physics", {}).get("cryo"):
            p.update(cryo=True, neutral_base_edge=True)       # 0.9a cryogenic core (spec version 2)
        if sw.get("IC_max_A_per_m"):
            p["IC_max_A_per_m"] = float(sw["IC_max_A_per_m"])
        return p

    done = {}
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        # coldest first: they take longest
        todo = [(T, dev, d) for T in sorted(sw["T_K"]) for dev, d in s["devices"].items()]
        with ThreadPoolExecutor(workers) as ex:
            futs = {(T, dev): ex.submit(_submit_one, str(hub.store.root), d["adapter"], params_for(T, d), fid)
                    for T, dev, d in todo}
            for k, f in futs.items():
                done[k] = hub.store.envelope(f.result())
    for T in sw["T_K"]:
        per = {}
        for dev, d in s["devices"].items():
            r = done.get((T, dev)) or hub.submit(d["adapter"], params=params_for(T, d), priority="batch", fidelity=fid)
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
            if "hbt.dEg_ge_nb_edge" in {q.name for q in a.quantities}:
                row["sige_hbt.dEg_ge_nb_edge_eV"] = float(np.interp(vc, va, a.q("hbt.dEg_ge_nb_edge").array()))
                row["sige_hbt.ge_nb_edge"] = float(np.interp(vc, va, a.q("hbt.ge_nb_edge").array()))
                row["sige_hbt.y_nb_edge_um"] = float(np.interp(vc, va, a.q("hbt.y_nb_edge").array()))
        rows.append(row)
    trusted = [r for r in rows if r["trusted"] and "IC_ratio" in r]
    fits = fit_models([r["T_K"] for r in trusted], [r["IC_ratio"] for r in trusted]) if len(trusted) >= 3 else {}
    allr = [r for r in rows if "IC_ratio" in r]
    if fits:
        g = fits["graded_form"]
        for r in allr:
            r["IC_ratio_graded_model"] = float(graded_form(r["T_K"], g["A"], g["dE_grade_eV"]))
    fit_lo, fit_hi = s["validity"].get("fit_T_K", [lo, hi])
    wide = [r for r in allr if fit_lo <= r["T_K"] <= fit_hi]
    fits_wide = fit_models([r["T_K"] for r in wide], [r["IC_ratio"] for r in wide]) if len(wide) >= 3 else {}
    edge = [r for r in wide if "sige_hbt.dEg_ge_nb_edge_eV" in r]
    if len(edge) >= 3:
        fits_wide["edge_plus_graded"] = fit_edge_plus_graded([r["T_K"] for r in edge], [r["IC_ratio"] for r in edge],
                                                             [r["sige_hbt.dEg_ge_nb_edge_eV"] for r in edge])
    res = {"spec": s["id"], "version": s["version"], "fidelity": s["fidelity"], "rows": rows, "fits_trusted_window": fits,
           "fits_fit_window": fits_wide, "fit_window_T_K": [fit_lo, fit_hi],
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
