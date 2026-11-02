"""Publication figures and the methods/parameter table, straight from results/ (run in the tcad env).

    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/make_figures.py [--results results] [--out results/figures]

Each figure is written as PDF (vector, for the paper) and PNG (300 dpi, for slides), with a JSON sidecar holding
the exact numbers plotted and the provenance of the source result, so every panel traces to a commit. A figure
whose inputs are missing is skipped and listed. Style: thin marks, one y-axis per panel, recessive grid, ink-
coloured text; categorical slots 1-3 for code-vs-code comparisons, a validated 5-step blue ramp for ordered
series (LET, bias), direct labels where a colour is below 3:1 contrast.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# validated with the dataviz palette checker (light surface): categorical slots 1-3 pass all gates; the ordinal
# ramp passes monotone lightness, step gaps and light-end contrast for up to 5 steps
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e6e5e0", "#ffffff"


def ramp(n):
    if n > 5:
        raise ValueError("ordered series are capped at 5 (validated ramp); subsample")
    idx = np.round(np.linspace(0, 4, n)).astype(int) if n > 1 else [2]
    return [RAMP5[i] for i in idx]


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.5, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
        "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    return plt


class Figs:
    def __init__(self, results: Path, out: Path):
        self.R, self.out = results, out
        self.made, self.skipped = [], []
        out.mkdir(parents=True, exist_ok=True)

    def j(self, rel):
        return json.loads((self.R / rel).read_text())

    def save(self, fig, name, data, sources):
        prov = {}
        for s in sources:
            try:
                pv = self.j(s).get("provenance", {})
                prov[s] = {k: pv.get(k) for k in ("git_commit", "git_dirty", "created_utc")}
            except Exception:  # noqa: BLE001
                prov[s] = None
        fig.savefig(self.out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(self.out / f"{name}.png", dpi=300, bbox_inches="tight")
        (self.out / f"{name}.json").write_text(json.dumps({"data": data, "sources": prov}, indent=1,
                                                          default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        self.made.append(name)

    def run(self, name, fn):
        try:
            fn()
        except FileNotFoundError as e:
            self.skipped.append(f"{name}: missing {Path(e.filename).relative_to(self.R) if e.filename else e}")
        except Exception as e:  # noqa: BLE001
            self.skipped.append(f"{name}: {type(e).__name__}: {e}")


def fig_particle_strike(F, plt, gold_csv):
    sim = np.loadtxt(F.R / "particle_strike/devsim_currents.csv", delimiter=",", skiprows=1)
    gold = np.loadtxt(gold_csv, delimiter=",", skiprows=1)
    cmp_ = F.j("particle_strike/comparison.json")["results"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(gold[:, 0] * 1e6, gold[:, 1], color=CAT[1], label="Charon (reference)")
    ax.plot(sim[:, 0] * 1e6, sim[:, 1], color=CAT[0], linestyle=(0, (4, 2)), label="DEVSIM (simws)")
    ax.set_xlim(0, 1500)                         # the pulse; the DC tail continues to 10 ms unchanged
    ax.set_xlabel("time (µs)"); ax.set_ylabel("cathode current (A/cm)")
    ax.set_title(f"Particle strike: codes agree within {100 * cmp_['max_abs_dev_in_pulse_frac_of_peak']:.1f} % of peak",
                 loc="left")
    ax.legend(loc="upper right")
    F.save(fig, "fig_particle_strike", {"t_us": sim[:, 0] * 1e6, "devsim": sim[:, 1], "charon_t_us": gold[:, 0] * 1e6,
                                        "charon": gold[:, 1]}, ["particle_strike/comparison.json"])
    plt.close(fig)


def fig_axi_charge(F, plt):
    s = F.j("axisymmetric/summary.json")["results"]
    runs = sorted(s["all_lets"], key=lambda r: r["case"]["let"])
    L = np.array([r["case"]["let"] for r in runs])
    qd = np.array([r["deposited_charge_fC"] for r in runs]); qc = np.array([r["collected_charge_drain_fC"] for r in runs])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.loglog(L, qd, color=CAT[1], marker="o", ms=4, label="deposited")
    ax.loglog(L, qc, color=CAT[0], marker="o", ms=4, label="collected at drain (1 ns)")
    data = {"let": L, "deposited_fC": qd, "collected_fC": qc}
    try:
        ser = F.j("ser/summary.json")["results"]
        if ser.get("critical_let"):
            ax.axhline(ser["q_crit_fC"], color=MUTED, lw=0.9, ls=":")
            ax.axvline(ser["critical_let"], color=MUTED, lw=0.9, ls=":")
            ax.annotate(f"Q_crit {ser['q_crit_fC']:g} fC → LET_crit {ser['critical_let']:.2g}",
                        (ser["critical_let"], ser["q_crit_fC"]), xytext=(8, 5), textcoords="offset points",
                        fontsize=7.5, color=INK2)
            data.update(q_crit_fC=ser["q_crit_fC"], critical_let=ser["critical_let"])
    except FileNotFoundError:
        pass
    ax.set_xlabel("LET (MeV cm²/mg)"); ax.set_ylabel("charge (fC)")
    ax.set_title("Axisymmetric strike, n+/p diode at 1 V (1e20 / 1e17 cm⁻³)", loc="left")
    ax.legend(loc="upper left")
    F.save(fig, "fig_axisymmetric_charge_vs_let", data, ["axisymmetric/summary.json"])
    plt.close(fig)


def fig_axi_transients(F, plt):
    files = sorted((F.R / "axisymmetric").glob("axi_let*.csv"), key=lambda p: float(p.stem[7:]))
    if not files:
        raise FileNotFoundError(str(F.R / "axisymmetric/axi_let*.csv"))
    if len(files) > 5:
        files = [files[i] for i in np.round(np.linspace(0, len(files) - 1, 5)).astype(int)]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    data = {}
    for f, c in zip(files, ramp(len(files))):
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        let = float(f.stem[7:])
        ax.semilogx(d[:, 0] * 1e12, d[:, 1] * 1e3, color=c, label=f"LET {let:g}")
        data[f"let_{let:g}"] = {"t_ps": d[:, 0] * 1e12, "i_mA": d[:, 1] * 1e3}
    ax.set_xlim(0.2, 1000)
    ax.set_xlabel("time (ps)"); ax.set_ylabel("drain current (mA)")
    ax.set_title("Strike current pulses (LET in MeV cm²/mg)", loc="left")
    ax.legend(loc="upper right")
    F.save(fig, "fig_axisymmetric_transients", data, ["axisymmetric/summary.json"])
    plt.close(fig)


def fig_event_spread(F, plt):
    s = F.j("geant4_events/seu_summary.json")["results"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    data = {}
    for i, (tag, r) in enumerate(s.items()):
        q = np.array(r["collected_fC_events"])
        jitter = (np.random.default_rng(i).random(q.size) - 0.5) * 0.25
        ax.scatter(np.full(q.size, i) + jitter, q, s=14, color=CAT[0], alpha=0.85, linewidths=0,
                   label="Geant4 events" if i == 0 else None)
        g = r["gaussian_reference"]["collected_charge_drain_fC"]
        ax.plot([i - 0.3, i + 0.3], [g, g], color=CAT[1], lw=1.5, label="Gaussian track, same LET" if i == 0 else None)
        ax.annotate(f"{r['collected_fC_mean']:.0f} ± {r['collected_fC_std']:.0f} fC", (i + 0.32, r["collected_fC_mean"]),
                    fontsize=7.5, color=INK2, va="center")
        data[tag] = {"events_fC": q, "gaussian_fC": g, "mean_let": r["mean_let"]}
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels([f"{k.split('_')[1][1:]}{k.split('_')[0]}\nLET {v['mean_let']:.1f}".replace("Z17", "Cl").replace("Z36", "Kr")
                        for k, v in s.items()])
    ax.set_xlim(-0.6, len(s) - 0.1)
    ax.set_ylabel("collected charge (fC)")
    ax.set_title("Ion-to-ion spread from Geant4 track structure", loc="left")
    ax.legend(loc="upper left")
    F.save(fig, "fig_event_spread", data, ["geant4_events/seu_summary.json"])
    plt.close(fig)


def fig_seu_node(F, plt):
    files = sorted((F.R / "mixed_mode").glob("seu_let*.csv"), key=lambda p: float(p.stem[7:]))
    if not files:
        raise FileNotFoundError(str(F.R / "mixed_mode/seu_let*.csv"))
    files = files[:5]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    data = {}
    for f, c in zip(files, ramp(len(files))):
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        let = float(f.stem[7:])
        ax.semilogx(np.maximum(d[:, 0], 1e-13) * 1e12, d[:, 1], color=c, label=f"LET {let:g}")
        data[f"let_{let:g}"] = {"t_ps": d[:, 0] * 1e12, "v": d[:, 1]}
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=":")
    ax.text(0.12, 0.53, "V_DD/2", fontsize=7.5, color=INK2)
    ax.set_xlim(0.1, 1000)
    ax.set_xlabel("time (ps)"); ax.set_ylabel("struck node voltage (V)")
    ax.set_title("Mixed-mode node upset (planar 2D)", loc="left")
    ax.legend(loc="lower left")
    F.save(fig, "fig_mixed_mode_node", data, ["mixed_mode/summary.json"])
    plt.close(fig)


def fig_sobol(F, plt):
    s = F.j("sweeps/electrothermal_sobol/summary.json")["results"]["T_max_K"]["sobol"]
    names = list(s)[::-1]
    st = np.array([s[n]["ST"] for n in names]); ci = np.array([s[n]["ST_conf"] for n in names])
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.barh(range(len(names)), st, height=0.55, color=CAT[0])
    ax.errorbar(st, range(len(names)), xerr=ci, fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([n.replace("_", " ") for n in names])
    ax.set_xlim(0, 1); ax.grid(axis="y", visible=False)
    ax.set_xlabel("total-order Sobol index of T_max (95 % CI)")
    ax.set_title("Doping dominates peak-temperature uncertainty", loc="left")
    F.save(fig, "fig_sobol_tmax", {"names": names, "ST": st, "ST_conf": ci}, ["sweeps/electrothermal_sobol/summary.json"])
    plt.close(fig)


def fig_convergence(F, plt):
    r = F.j("verification/electrothermal_mesh.json")["results"]
    lv, v = np.array(r["levels"]), np.array(r["values"])
    err = np.abs(v - r["richardson"])
    h = 2.0 ** -lv
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.loglog(h[err > 0], err[err > 0], color=CAT[0], marker="o", ms=4, label="|T_max − Richardson|")
    ref = err[0] * (h / h[0]) ** 2
    ax.loglog(h, ref, color=MUTED, lw=0.9, ls=":", label="second order")
    ax.invert_xaxis()
    ax.minorticks_off()
    ax.set_xticks(h); ax.set_xticklabels(["1" if x == 1 else f"1/{int(round(1 / x))}" for x in h])
    ax.set_xlabel("relative mesh size h"); ax.set_ylabel("error in T_max (K)")
    ax.set_title(f"Mesh convergence: observed order {r['order']:.2f}", loc="left")
    ax.legend(loc="lower left")
    F.save(fig, "fig_mesh_convergence", {"h": h, "err_K": err, "order": r["order"]}, ["verification/electrothermal_mesh.json"])
    plt.close(fig)


def fig_geant4_radial(F, plt):
    from simws.radiation import geant4
    d = geant4.load(F.R / "geant4_track/deposition.csv")
    frac = np.concatenate([[0.0], d.radial_fraction()]); r = d.r_edges_um * 1e3          # curve starts at the axis
    dr = float(np.diff(d.r_edges_um).mean() * 1e3)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(r, frac, color=CAT[0])
    r90 = float(np.interp(0.9, frac, r))
    ax.plot([r90], [0.9], "o", ms=4, color=CAT[0])
    ax.annotate(f"90 % within {r90:.0f} nm", (r90, 0.9), xytext=(8, -12), textcoords="offset points", fontsize=7.5,
                color=INK2)
    ax.set_xlim(0, max(10 * r90, 200)); ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"radius from ion path (nm; bins of {dr:.0f} nm)"); ax.set_ylabel("cumulative energy fraction")
    ax.set_title(f"Geant4 track core, {int(d.meta['ion_A'])}O at {d.meta['energy_MeV']:g} MeV", loc="left")
    F.save(fig, "fig_geant4_radial", {"r_nm": r, "fraction": frac, "r90_nm": r90}, ["geant4_track/summary.json"])
    plt.close(fig)


def fig_calibration(F, plt):
    s = F.j("calibration/selftest.json")["results"]
    c = s["curves"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5))
    a1.semilogy(c["iv_V"], c["iv_meas_A"], "o", ms=4, color=CAT[1], label="synthetic data (5 % noise)")
    a1.semilogy(c["iv_V"], c["iv_fit_A"], color=CAT[0], label="fit")
    a1.set_xlabel("forward bias (V)"); a1.set_ylabel("current (A)"); a1.legend(loc="upper left")
    a1.set_title("I–V", loc="left")
    a2.plot(c["cv_V"], np.array(c["cv_meas_F"]) * 1e12, "o", ms=4, color=CAT[1], label="synthetic data (2 % noise)")
    a2.plot(c["cv_V"], np.array(c["cv_fit_F"]) * 1e12, color=CAT[0], label="fit")
    a2.set_xlabel("bias (V)"); a2.set_ylabel("capacitance (pF)"); a2.legend(loc="upper left")
    p = s["parameters"]
    a2.set_title("C–V   " + ", ".join(f"{k}: {v['fit']:.3g} (true {v['true']:.3g})" for k, v in p.items()), loc="left",
                 fontsize=7.5)
    F.save(fig, "fig_calibration_selftest", c, ["calibration/selftest.json"])
    plt.close(fig)


def fig_mos_cv(F, plt):
    d = np.loadtxt(F.R / "mos/cv_ideal.csv", delimiter=",", skiprows=1)
    s = F.j("mos/summary.json")["results"]
    cox = s["C_ox_F_cm2"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(d[:, 0], d[:, 1] / cox, color=CAT[0])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("gate voltage (V)"); ax.set_ylabel("C / C_ox")
    ax.set_title("Quasi-static C–V, 10 nm SiO₂ on p-Si 1e17 cm⁻³", loc="left")
    F.save(fig, "fig_mos_cv", {"V": d[:, 0], "C_over_Cox": d[:, 1] / cox}, ["mos/summary.json"])
    plt.close(fig)


def fig_tid(F, plt):
    s = F.j("mos/summary.json")["results"]["tid"]
    keys = sorted(s, key=lambda k: float(k.split("_")[1][:-2]))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    data = {}
    for k, c in zip(keys, ramp(len(keys))):
        r = s[k]
        D = np.array(r["doses_krad"]); dv = -np.array(r["dV_mg"])
        sel = D > 0
        ax.loglog(D[sel], dv[sel], color=c, marker="o", ms=3.5, label=f"t_ox {k.split('_')[1].replace('nm', ' nm')}")
        data[k] = {"dose_krad": D, "minus_dV_mg_V": dv, "E_ox_MV_cm": r["E_ox_MV_cm"]}
    ax.set_xlabel("dose (krad(SiO₂))"); ax.set_ylabel("−ΔV_midgap (V)")
    ax.set_title("Oxide-trapped charge shift grows ~ t_ox²", loc="left")
    ax.legend(loc="upper left")
    F.save(fig, "fig_tid_midgap_shift", data, ["mos/summary.json"])
    plt.close(fig)


def fig_mos_variability(F, plt):
    s = F.j("mos_variability/summary.json")["results"]
    outs = [o for o in ("V_fb", "V_inv", "C_min_over_C_ox") if o in s]
    names = list(s[outs[0]]["sobol"])
    fig, axes = plt.subplots(1, len(outs), figsize=(6.8, 2.2), sharey=True)
    data = {}
    for ax, o in zip(np.atleast_1d(axes), outs):
        st = [s[o]["sobol"][n]["ST"] for n in names]
        ax.barh(range(len(names)), st, height=0.55, color=CAT[0])
        ax.set_xlim(0, 1); ax.grid(axis="y", visible=False)
        unit = "" if o.startswith("C_") else " V"
        label = {"V_fb": "V_fb", "V_inv": "V_inv", "C_min_over_C_ox": "C_min / C_ox"}.get(o, o)
        ax.set_title(f"{label}: σ = {s[o]['std']:.2g}{unit}", loc="left", fontsize=8.5)
        ax.set_xlabel("total-order Sobol index")
        data[o] = {"names": names, "ST": st, "std": s[o]["std"], "loo_q2": s[o]["loo_q2"]}
    np.atleast_1d(axes)[0].set_yticks(range(len(names)))
    np.atleast_1d(axes)[0].set_yticklabels([n.replace("_", " ") for n in names])
    fig.suptitle(f"MOS process variation (polynomial-chaos surrogate, {s['n_train']} runs)", x=0.01, y=1.06, ha="left",
                 fontsize=9.5)
    F.save(fig, "fig_mos_variability", data, ["mos_variability/summary.json"])
    plt.close(fig)


def fig_sige_gummel(F, plt):
    s = F.j("sige/devices.json")["results"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    data = {}
    for key, c in (("SiGe", CAT[0]), ("Si", CAT[1])):
        g = s["gummel"][key]
        V, IC, IB = np.array(g["VBE"]), np.array(g["IC_A"]), np.array(g["IB_A"])
        a1.semilogy(V, IC, color=c, label=f"I_C {key}")
        a1.semilogy(V, IB, color=c, linestyle=(0, (4, 2)),
                    label=f"I_B {key}" + (" (same as SiGe)" if key == "Si" else ""))
        a2.semilogx(IC, IC / IB, color=c, label=key)
        data[key] = {"VBE": V, "IC_A_per_um": IC, "IB_A_per_um": IB, "V_A": s["early"][key]["V_A"]}
    a1.set_xlabel("V_BE (V)"); a1.set_ylabel("current (A/µm)")
    a1.set_title(f"Gummel plot, Ge graded 0→{s['ge_peak']:g}", loc="left")
    a1.legend(loc="upper left", fontsize=7)
    a2.set_xlabel("I_C (A/µm)"); a2.set_ylabel("β = I_C / I_B")
    va = s["early"]
    a2.set_title(f"Gain (V_A: Si {va['Si']['V_A']:.0f} V, SiGe {va['SiGe']['V_A']:.0f} V)", loc="left")
    a2.legend(loc="upper left")
    F.save(fig, "fig_sige_gummel", data, ["sige/devices.json"])
    plt.close(fig)


def fig_sige_buried_channel(F, plt):
    s = F.j("sige/devices.json")["results"]["mos"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    sg, si = s["x_0.25"], s["x_0"]
    V = -np.array(sg["V_sheets"])
    a1.semilogy(V, sg["well_cm2"], color=CAT[0], label="SiGe well")
    a1.semilogy(V, sg["cap_cm2"], color=CAT[1], label="Si cap (surface)")
    a1.semilogy(V, np.array(si["well_cm2"]) + np.array(si["cap_cm2"]), color=CAT[2], linestyle=(0, (4, 2)),
                label="same stack, no Ge")
    a1.set_ylim(1e8, None)
    a1.set_xlabel("−V_gate (V)"); a1.set_ylabel("hole sheet (cm⁻²)")
    a1.set_title("Buried channel: well fills first", loc="left")
    a1.legend(loc="lower right", fontsize=7)
    a2.plot(sg["V_cv"], sg["C_over_Cox"], color=CAT[0], label="Si₀.₇₅Ge₀.₂₅ well")
    a2.plot(si["V_cv"], si["C_over_Cox"], color=CAT[1], label="no Ge")
    a2.set_ylim(0, 1.05)
    a2.set_xlabel("V_gate (V)"); a2.set_ylabel("C / C_ox")
    a2.set_title("Quasi-static C–V: holes arrive earlier", loc="left")
    a2.legend(loc="center left")
    F.save(fig, "fig_sige_buried_channel", {"SiGe_x0.25": sg, "Si": si}, ["sige/devices.json"])
    plt.close(fig)


def fig_hbt_tid(F, plt):
    t = F.j("sige/devices.json")["results"]["tid"]
    D = np.array(t["dose_krad"]); V = np.array(t["VBE"]); IB = np.array(t["IB_A"]); IC = np.array(t["IC_A"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    a1.semilogy(V, IC[0], color=INK2, label="I_C (all doses)")
    a1.semilogy(V, IB[0], color=MUTED, linestyle=(0, (4, 2)), label="I_B pre-rad")
    for d, row, c in zip(D[1:], IB[1:], ramp(len(D) - 1)):
        a1.semilogy(V, row, color=c, label=f"I_B {d:g} krad")
    a1.set_xlabel("V_BE (V)"); a1.set_ylabel("current (A/µm)")
    a1.set_title("TID: excess base current, I_C unchanged", loc="left")
    fig.legend(*a1.get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    i5 = int(np.argmin(np.abs(V - 0.5)))
    sel = D > 0
    beta = IC[:, i5] / IB[:, i5]
    a2.semilogx(D[sel], beta[sel] / beta[0], color=CAT[0], marker="o", ms=4)
    a2.set_ylim(0, 1.05)
    a2.set_xlabel("dose (krad(SiO₂))"); a2.set_ylabel("β / β_pre at V_BE = 0.5 V")
    a2.set_title("Gain degradation (uncalibrated oxide)", loc="left")
    F.save(fig, "fig_hbt_tid", {"dose_krad": D, "VBE": V, "IB_A": IB, "IC_A": IC, "N_it": t["N_it"], "N_ot": t["N_ot"],
                                "assumptions": t["assumptions"]}, ["sige/devices.json"])
    plt.close(fig)


def fig_hbt_strike(F, plt):
    s = F.j("sige/devices.json")["results"]["strike"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    data = {}
    for key, c in (("SiGe", CAT[0]), ("Si", CAT[1])):
        L = np.array([r["let"] for r in s[key]]); q = np.array([r["Q_collector_fC"] for r in s[key]])
        ax.loglog(L, q, color=c, marker="o", ms=4, label=f"collected, {key} HBT")
        data[key] = {"LET": L, "Q_collector_fC": q}
    qd = np.array([r["Q_deposited_fC"] for r in s["SiGe"]])
    ax.loglog(L, qd, color=INK2, linestyle=(0, (4, 2)), label="deposited (1.5 µm track)")
    ax.set_xlabel("LET (MeV cm²/mg)"); ax.set_ylabel("charge (fC per µm, planar 2D)")
    ax.set_title("Ion strike through the emitter, V_C = 2 V", loc="left")
    ax.legend(loc="upper left", fontsize=7)
    data["deposited_fC"] = qd
    F.save(fig, "fig_hbt_strike", data, ["sige/devices.json"])
    plt.close(fig)


def fig_dft_vs_measured(F, plt):
    c = F.j("dft/checks.json")["results"]["parameters"]
    label = {"m_l": "m_l", "m_t": "m_t", "m_hh_100": "m_hh [100]", "m_hh_111": "m_hh [111]", "m_lh_100": "m_lh [100]",
             "m_lh_111": "m_lh [111]", "m_so_100": "m_so", "Delta_so": "Δ_so", "C11": "C11", "C12": "C12", "C44": "C44"}
    keys = [k for k in label if k in c]
    fig, ax = plt.subplots(figsize=(3.6, 0.28 * len(keys) + 0.8))
    y = np.arange(len(keys))[::-1]
    r = np.array([c[k]["dft"] / c[k]["measured"] for k in keys])
    e = np.array([c[k]["sigma"] / c[k]["measured"] for k in keys])
    tol = np.array([c[k]["tol"] for k in keys])
    for yi, t in zip(y, tol):
        ax.plot([1 - t, 1 + t], [yi, yi], color=GRID, lw=6, solid_capstyle="butt", zorder=0)
    ax.errorbar(r, y, xerr=e, fmt="o", ms=4, color=CAT[0], ecolor=CAT[0], elinewidth=1.2, capsize=0)
    ax.axvline(1.0, color=INK2, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([label[k] for k in keys]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("DFT / measured (bar: ± tolerance)")
    ax.set_title("Si band edges and elastic constants from DFT", loc="left")
    F.save(fig, "fig_dft_vs_measured", {k: c[k] for k in keys}, ["dft/checks.json"])
    plt.close(fig)


def fig_dft_intrinsic(F, plt):
    rows = F.j("dft/checks.json")["results"]["intrinsic"]["rows"]
    p = F.j("dft/params.json")["results"]["parameters"]
    T = np.array([r["T"] for r in rows])
    dft = np.array([r["dft"] for r in rows]); base = np.array([r["base_model"] for r in rows])
    mis = np.array([r["measured_fit"] for r in rows])
    rel = 0.5 * np.hypot(p["Nc_300"]["sigma"] / p["Nc_300"]["value"], p["Nv_300"]["sigma"] / p["Nv_300"]["value"])
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.fill_between(T, dft / mis * (1 - rel), dft / mis * (1 + rel), color=CAT[0], alpha=0.18, lw=0)
    ax.plot(T, dft / mis, color=CAT[0], marker="o", ms=4, label="DFT N_c, N_v + measured E_g")
    ax.plot(T, base / mis, color=CAT[1], marker="o", ms=4, linestyle=(0, (4, 2)), label="previous model (T^1.5)")
    ax.axhline(1.0, color=INK2, lw=0.8)
    ax.set_ylim(0.8, 1.2)
    ax.set_xlabel("temperature (K)"); ax.set_ylabel("n_i / measured (Misiakos 1993)")
    ax.set_title("Intrinsic density from first principles", loc="left")
    ax.legend(loc="upper left", fontsize=7)
    F.save(fig, "fig_dft_intrinsic", {"T": T, "ni_dft": dft, "ni_base": base, "ni_measured_fit": mis, "rel_sigma": rel},
           ["dft/checks.json", "dft/params.json"])
    plt.close(fig)


def fig_sige_critical_thickness(F, plt):
    ct = F.j("dft/params.json")["results"]["raw"]["critical_thickness"]
    x = np.array(ct["x"])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.semilogy(x, ct["measured"]["h_MB_nm"], color=CAT[0], label="Matthews–Blakeslee (stable)")
    ax.semilogy(x, ct["measured"]["h_PB_nm"], color=CAT[1], label="People–Bean (metastable)")
    if "dft_Si_elastic" in ct:
        ax.semilogy(x, ct["dft_Si_elastic"]["h_MB_nm"], color=CAT[0], linestyle=(0, (2, 2)), lw=1.0,
                    label="MB with DFT C_ij for Si (overlaps)")
    for name, mk in (("hbt_base", "s"), ("mos_buried_well", "D")):
        d = ct["devices"][name]
        ax.plot([d["x_eff"]], [d["thickness_nm"]], mk, ms=6, color=INK, zorder=5)
        ax.annotate({"hbt_base": "HBT base", "mos_buried_well": "pMOS well"}[name] + f" ({d['state']})",
                    (d["x_eff"], d["thickness_nm"]), textcoords="offset points", xytext=(6, 4), fontsize=7, color=INK2)
    ax.set_xlabel("Ge fraction x"); ax.set_ylabel("critical thickness (nm)")
    ax.set_title("SiGe on Si(001): strained-layer limits", loc="left")
    ax.legend(loc="upper right", fontsize=7)
    F.save(fig, "fig_sige_critical_thickness", ct, ["dft/params.json"])
    plt.close(fig)


def methods_table(F):
    """Every material parameter the simulations use, with value, unit, status and reference (Markdown + LaTeX)."""
    from simws.materials import available, load
    md = ["| Material | Parameter | Value | Unit | Status | Reference |", "|---|---|---|---|---|---|"]
    tex = [r"\begin{longtable}{llrlll}", r"\toprule", r"Material & Parameter & Value & Unit & Status & Reference \\",
           r"\midrule", r"\endhead"]
    esc = lambda t: (str(t).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
                     .replace("_", r"\_").replace("#", r"\#").replace("^", r"\^{}").replace("~", r"\~{}"))
    for name in available():
        m = load(name)
        for k, p in sorted(m.parameters.items()):
            md.append(f"| {name} | `{k}` | {p.value:.6g} | {p.unit} | {p.status} | {p.ref} |")
            tex.append(f"{esc(name)} & {esc(k)} & {p.value:.6g} & {esc(p.unit)} & {esc(p.status)} & {esc(p.ref)} \\\\")
    tex += [r"\bottomrule", r"\end{longtable}"]
    (F.out / "methods_parameters.md").write_text("\n".join(md) + "\n")
    (F.out / "methods_parameters.tex").write_text("\n".join(tex) + "\n")
    F.made.append("methods_parameters (.md, .tex)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--gold", default=str(ROOT / "03_tcad_devices/particle_strike_ref/gold_currents.csv"))
    a = ap.parse_args()
    R = Path(a.results)
    F = Figs(R, Path(a.out) if a.out else R / "figures")
    plt = style()
    F.run("particle strike", lambda: fig_particle_strike(F, plt, a.gold))
    F.run("axisymmetric charge", lambda: fig_axi_charge(F, plt))
    F.run("axisymmetric transients", lambda: fig_axi_transients(F, plt))
    F.run("event spread", lambda: fig_event_spread(F, plt))
    F.run("mixed-mode node", lambda: fig_seu_node(F, plt))
    F.run("sobol", lambda: fig_sobol(F, plt))
    F.run("mesh convergence", lambda: fig_convergence(F, plt))
    F.run("geant4 radial", lambda: fig_geant4_radial(F, plt))
    F.run("calibration", lambda: fig_calibration(F, plt))
    F.run("mos cv", lambda: fig_mos_cv(F, plt))
    F.run("tid", lambda: fig_tid(F, plt))
    F.run("mos variability", lambda: fig_mos_variability(F, plt))
    F.run("sige gummel", lambda: fig_sige_gummel(F, plt))
    F.run("sige buried channel", lambda: fig_sige_buried_channel(F, plt))
    F.run("hbt tid", lambda: fig_hbt_tid(F, plt))
    F.run("hbt strike", lambda: fig_hbt_strike(F, plt))
    F.run("dft vs measured", lambda: fig_dft_vs_measured(F, plt))
    F.run("dft intrinsic", lambda: fig_dft_intrinsic(F, plt))
    F.run("sige critical thickness", lambda: fig_sige_critical_thickness(F, plt))
    F.run("methods table", lambda: methods_table(F))
    (F.out / "INDEX.json").write_text(json.dumps({"made": F.made, "skipped": F.skipped}, indent=1))
    print(f"figures in {F.out}: {len(F.made)} made")
    for s in F.skipped:
        print("  skipped", s)


if __name__ == "__main__":
    main()
