"""One-screen summary of results/ (used by the installer and the nightly log)."""
import json
from pathlib import Path

R = Path("results")


class NotRun(Exception):
    pass


def get(path, *keys):
    p = R / path
    if not p.exists():
        raise NotRun
    d = json.loads(p.read_text())["results"]
    for k in keys:
        d = d[k]
    return d


def line(label, fn):
    try:
        v = fn()
        print(f"  {label:34s} {v if v is not None else '(not run)'}")
    except NotRun:
        print(f"  {label:34s} (not run)")
    except Exception as e:  # noqa: BLE001
        print(f"  {label:34s} error: {e}")


line("QE Si SCF [Ry]", lambda: get("qe_si_scf/summary.json", "total_energy_Ry"))
line("particle strike vs Charon", lambda: f"{100 * get('particle_strike/comparison.json', 'max_abs_dev_in_pulse_frac_of_peak'):.2f} % of peak")
line("electrothermal 5 V", lambda: (lambda p: f"I = {p['current_A_per_cm']:.4f} A/cm, T_max = {p['T_max_K']:.2f} K")(get("electrothermal/summary.json", "points")[-1]))
line("device-model validation", lambda: (lambda r: f"pass={r['pass']}, BV DEVSIM {r['breakdown_V_devsim']:.2f} V vs integral {r['breakdown_V_ionization_integral']:.2f} V")(get("models/validation.json")))
line("mixed-mode RC check", lambda: f"{get('mixed_mode/summary.json', 'rc_check', 'worst_abs_dev_frac_of_vinf'):.1e} of V_inf")
line("SEU node dips", lambda: ", ".join(f"LET {s['case']['let']:g}: {s['v_node_initial_V']:.2f}->{s['v_node_min_V']:.2f} V" for s in get("mixed_mode/summary.json", "seu")))
line("critical LET (planar 2D)", lambda: (lambda v: f"{v:.3g} MeV cm2/mg" if v else "not bracketed")(get("mixed_mode/summary.json", "critical_let_MeV_cm2_mg")))
line("physics model checks", lambda: (lambda r: f"pass={r['pass']}: equilibrium J/scale {r['equilibrium_bgn']['current_over_scale']:.1e}, mobility {r['mobility_resistors']['worst_rel_diff']:.1e}, lifetime {r['lifetime_bars']['worst_rel_diff']:.1e}")(get("models/physics_checks.json")))
line("axisymmetric SEU (LET 10)", lambda: (lambda r: f"collected {r['collected_charge_drain_fC']:.1f} of {r['deposited_charge_fC']:.1f} fC deposited, peak {1e3 * r['peak_drain_current_A']:.2f} mA")(get("axisymmetric/summary.json")))
line("Geant4 event-driven strikes", lambda: "; ".join(f"{k.split('_')[0]}{k.split('_')[1]}: {v['collected_fC_mean']:.0f} +- {v['collected_fC_std']:.1f} fC ({v['n_events']} ions), Gaussian-track bias {100 * v['gaussian_bias_in_efficiency']:+.1f} %" for k, v in get("geant4_events/seu_summary.json").items()))
line("upset rate inputs", lambda: (lambda r: f"Q_crit {r['q_crit_fC']:g} fC -> LET_crit {r['critical_let']:.3g} MeV cm2/mg, engine check {r['engine_verification']['rel_diff']:+.1e}, rate/bit/day {r['rate_per_bit_per_day'] if r['rate_per_bit_per_day'] is not None else '(needs a spectrum file)'}" if r["critical_let"] else "LET_crit not bracketed")(get("ser/summary.json")))
line("paper figures", lambda: (lambda r: f"{len(r['made'])} made, {len(r['skipped'])} skipped -> results/figures")(json.loads((R / "figures/INDEX.json").read_text()) if (R / "figures/INDEX.json").exists() else (_ for _ in ()).throw(NotRun())))
line("MOS checks + total dose", lambda: (lambda r: f"checks pass={r['checks']['pass']}; dV_midgap at 1 Mrad: " + ", ".join(f"{k.split('_')[1]} {v['dV_mg'][-1]:+.3f} V" for k, v in r["tid"].items()))(get("mos/summary.json")))
line("MOS process variation", lambda: (lambda r: "; ".join(f"{o} sigma {r[o]['std']:.3g} (Q2 {r[o]['loo_q2']:.4f})" for o in ("V_fb", "V_inv", "C_min_over_C_ox")))(get("mos_variability/summary.json")))
line("4H-SiC checks", lambda: (lambda r: f"pass={r['pass']}; N 1e18 ionized {100 * r['devsim_ionization']['cases']['N 1e18']['ionized_fraction']:.0f} %, Al 1e18 {100 * r['devsim_ionization']['cases']['Al 1e18']['ionized_fraction']:.1f} %; BV(2e17) DEVSIM {r['breakdown']['BV_devsim']:.0f} V vs integral {r['breakdown']['BV_ionization_integral']:.0f} V")(get("sic/sic_checks.json")))
line("SiGe checks", lambda: (lambda r: f"pass={r['pass']}; HBT/BJT current ratio vs Moll-Ross worst {r['collector_enhancement']['worst_rel_diff']:.1e}; buried-channel onset shift {r['buried_channel']['shift_V']:.3f} V (dEv_eff {r['buried_channel']['dEv_eff_eV']:.3f} eV); TID dIB vs surface integral {r['tid_surface']['dIB_vs_surface_integral_worst']:.1e}; strike charge imbalance {r['strike_conservation']['imbalance_rel']:.1e}")(get("sige/sige_checks.json")))
line("SiGe devices", lambda: (lambda r: f"IC SiGe/Si at 0.6 V {r['IC_ratio_at_0.6V']:.2f}; beta peak Si {r['beta_peak']['Si']:.0f}, SiGe {r['beta_peak']['SiGe']:.0f}; V_A Si {r['V_A']['Si']:.0f} V, SiGe {r['V_A']['SiGe']:.0f} V; Q_C(LET 10) Si {r['Q_collected_fC']['Si'].get('LET_10', float('nan')):.1f} fC, SiGe {r['Q_collected_fC']['SiGe'].get('LET_10', float('nan')):.1f} fC")(get("sige/devices.json", "summary")))
line("DFT parameters (Si)", lambda: (lambda c, p: f"checks pass={c['pass']}; m_l {p['m_l']['value']:.3f}+-{p['m_l']['sigma']:.3f}, m_t {p['m_t']['value']:.3f}+-{p['m_t']['sigma']:.3f}; Nc {p['Nc_300']['value']:.2e}, Nv {p['Nv_300']['value']:.2e}; n_i(T) vs measured worst {100 * c['intrinsic']['worst_rel_dev']:.1f} %; HSE gap {p['Eg0_hybrid']['value']:.3f} eV")(get("dft/checks.json"), get("dft/params.json", "parameters")))
line("calibration self-test", lambda: (lambda r: "recovered " + ", ".join(f"{k} {v['fit']:.3g} (true {v['true']:.3g}, in 95%: {v['inside_95']})" for k, v in r['parameters'].items()))(get("calibration/selftest.json")))
line("Geant4 O-16 100 MeV LET", lambda: (lambda r: f"{r['meta']['entrance_LET_MeV_cm2_per_mg']:.3f} MeV cm2/mg, 90 % energy within {r['radius_90pct_energy_um']:.3f} um")(get("geant4_track/summary.json")))
line("Sobol T_max (total order)", lambda: ", ".join(f"{k} {v['ST']:.2f}" for k, v in get("sweeps/electrothermal_sobol/summary.json", "T_max_K", "sobol").items()))
line("mesh convergence T_max", lambda: (lambda r: f"p = {r['order']:.2f}, GCI {100 * r['gci_fine']:.1e} %")(get("verification/electrothermal_mesh.json")))
line("time-step convergence strike", lambda: (lambda r: f"p = {r['order']:.2f}, GCI {100 * r['gci_fine']:.1e} %" if r["order"] else (f"noise-level, bound {100 * r['gci_fine']:.1e} %" if r["gci_fine"] else "; ".join(r["notes"])))(get("verification/particle_strike_dt.json")))
