"""One-screen summary of results/ (used by the installer and the nightly log)."""
import json
from pathlib import Path

R = Path("results")


def get(path, *keys):
    p = R / path
    if not p.exists():
        return None
    d = json.loads(p.read_text())["results"]
    for k in keys:
        d = d[k]
    return d


def line(label, fn):
    try:
        v = fn()
        print(f"  {label:34s} {v if v is not None else '(not run)'}")
    except Exception as e:  # noqa: BLE001
        print(f"  {label:34s} error: {e}")


line("QE Si SCF [Ry]", lambda: get("qe_si_scf/summary.json", "total_energy_Ry"))
line("particle strike vs Charon", lambda: f"{100 * get('particle_strike/comparison.json', 'max_abs_dev_in_pulse_frac_of_peak'):.2f} % of peak")
line("electrothermal 5 V", lambda: (lambda p: f"I = {p['current_A_per_cm']:.4f} A/cm, T_max = {p['T_max_K']:.2f} K")(get("electrothermal/summary.json", "points")[-1]))
line("device-model validation", lambda: (lambda r: f"pass={r['pass']}, BV DEVSIM {r['breakdown_V_devsim']:.2f} V vs integral {r['breakdown_V_ionization_integral']:.2f} V")(get("models/validation.json")))
line("mixed-mode RC check", lambda: f"{get('mixed_mode/summary.json', 'rc_check', 'worst_abs_dev_frac_of_vinf'):.1e} of V_inf")
line("SEU node dips", lambda: ", ".join(f"LET {s['case']['let']:g}: {s['v_node_initial_V']:.2f}->{s['v_node_min_V']:.2f} V" for s in get("mixed_mode/summary.json", "seu")))
line("critical LET (planar 2D)", lambda: get("mixed_mode/summary.json", "critical_let_MeV_cm2_mg"))
line("Geant4 O-16 100 MeV LET", lambda: (lambda r: f"{r['meta']['entrance_LET_MeV_cm2_per_mg']:.3f} MeV cm2/mg, 90 % energy within {r['radius_90pct_energy_um']:.3f} um")(get("geant4_track/summary.json")))
line("Sobol T_max (total order)", lambda: ", ".join(f"{k} {v['ST']:.2f}" for k, v in get("sweeps/electrothermal_sobol/summary.json", "T_max_K", "sobol").items()))
line("mesh convergence T_max", lambda: (lambda r: f"p = {r['order']:.2f}, GCI {100 * r['gci_fine']:.1e} %")(get("verification/electrothermal_mesh.json")))
line("time-step convergence strike", lambda: (lambda r: f"p = {r['order']:.2f}, GCI {100 * r['gci_fine']:.1e} %" if r["order"] else (f"noise-level, bound {100 * r['gci_fine']:.1e} %" if r["gci_fine"] else "; ".join(r["notes"])))(get("verification/particle_strike_dt.json")))
