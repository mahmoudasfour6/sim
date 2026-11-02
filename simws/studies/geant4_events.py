"""Event-by-event single-event simulation: each Geant4 ion's own deposition drives an axisymmetric DEVSIM run.

Answers two questions a constant-LET Gaussian track cannot:
  * spread: how much collected charge varies from ion to ion (energy-loss straggling, delta-ray halo)
  * bias:   whether the ad-hoc Gaussian track (radius a) over- or under-predicts collection compared with the
            physical radial dose profile at the same deposited energy
Stages are top-level functions so simws.sweep.run_design can run events on parallel worker processes.
"""
from __future__ import annotations

import numpy as np


def event_stage(p: dict) -> dict:
    """p: events_csv, event (int), V_drain. One Geant4 event -> one axisymmetric SEU run."""
    from simws.radiation.geant4 import load_events
    from simws.tcad.axisym import AxiSEUCase, seu_axisymmetric
    meta, evs = load_events(p["events_csv"])
    dep = evs[int(p["event"])]
    case = AxiSEUCase(V_drain=p.get("V_drain", 1.0), height_um=meta["thickness_um"], track_length_um=meta["thickness_um"])
    r = seu_axisymmetric(case, deposition=dep)
    return {k: r[k] for k in ("deposited_charge_fC", "collected_charge_drain_fC", "collected_charge_sub_fC",
                              "peak_drain_current_A", "t_peak_current_s", "track_mesh_correction")}


def gaussian_stage(p: dict) -> dict:
    """Constant-LET Gaussian track with the same mean deposited energy per um as the events (reference)."""
    from simws.tcad.axisym import AxiSEUCase, seu_axisymmetric
    case = AxiSEUCase(let=p["let"], V_drain=p.get("V_drain", 1.0), height_um=p["thickness_um"],
                      track_length_um=p["thickness_um"], track_radius_um=p.get("radius_um", 0.05))
    r = seu_axisymmetric(case)
    return {k: r[k] for k in ("deposited_charge_fC", "collected_charge_drain_fC", "collected_charge_sub_fC",
                              "peak_drain_current_A", "t_peak_current_s", "track_mesh_correction")}


def dispatch_stage(p: dict) -> dict:
    """One design mixing event runs ({'kind': 'event', ...}) and Gaussian references ({'kind': 'gauss', ...})."""
    return event_stage(p) if p["kind"] == "event" else gaussian_stage(p)


def let_stage(p: dict) -> dict:
    """Gaussian constant-LET strike at p['let'] with the default AxiSEUCase; transients returned as lists."""
    from simws.tcad.axisym import AxiSEUCase, seu_axisymmetric
    r = seu_axisymmetric(AxiSEUCase(let=p["let"]))
    for k in ("time_s", "i_drain_A", "i_sub_A"):
        r[k] = np.asarray(r[k]).tolist()
    return r


def summarize(event_rows, gaussian_row):
    q = np.array([r["collected_charge_drain_fC"] for r in event_rows])
    qd = np.array([r["deposited_charge_fC"] for r in event_rows])
    ip = np.array([r["peak_drain_current_A"] for r in event_rows])
    eff = q / qd
    g_eff = gaussian_row["collected_charge_drain_fC"] / gaussian_row["deposited_charge_fC"]
    return {"n_events": len(q), "collected_fC_events": q.tolist(), "deposited_fC_events": qd.tolist(),
            "collected_fC_mean": float(q.mean()), "collected_fC_std": float(q.std(ddof=1)) if len(q) > 1 else 0.0,
            "collected_fC_min": float(q.min()), "collected_fC_max": float(q.max()),
            "deposited_fC_mean": float(qd.mean()), "deposited_fC_rel_std": float(qd.std(ddof=1) / qd.mean()) if len(q) > 1 else 0.0,
            "collection_efficiency_mean": float(eff.mean()),
            "peak_current_A_mean": float(ip.mean()), "peak_current_rel_std": float(ip.std(ddof=1) / ip.mean()) if len(q) > 1 else 0.0,
            "gaussian_reference": gaussian_row, "gaussian_collection_efficiency": float(g_eff),
            "gaussian_bias_in_efficiency": float(g_eff / eff.mean() - 1)}
