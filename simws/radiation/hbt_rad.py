"""Radiation response of the SiGe HBT (simws.tcad.hbt): single-event ion strikes and total-dose base current.

    from simws.radiation.hbt_rad import strike, excess_base_current, dose_response
    s = strike(HBTCase(), let=1.0, V_C=2.0)          # collected charge (fC per um of out-of-plane depth)
    e = excess_base_current(HBTCase(), N_it=[0, 1e11, 1e12], VBE=np.linspace(0.4, 0.8, 9))

Single event (planar 2D, simws.radiation.track.IonTrack): a vertical track through the emitter centre over the
whole device depth, emitter and base grounded, collector at V_C. The track charge is spread over W = 1 um out
of plane (a line of ions, not one round track): use it to compare designs and LETs, not for absolute cross
sections. The collected charge is the time integral of the collector current change. At low LET it is below the
deposited charge (base grounded: no gain); from about 3 MeV cm2/mg here the dense track shorts emitter
to collector through the thin base and the emitter feeds extra current (the ion-shunt effect), so the collector
charge exceeds the deposited charge. The effect is real but the planar track (a sheet, not a cylinder)
exaggerates it.
This device has no substrate: the collector-substrate junction, which dominates the collected charge of real
bulk HBTs, is not modelled (deep trenches, SOI and substrate taps are layout specific).

Total dose: the dominant TID mechanism in SiGe HBTs is interface traps (and oxide charge) in the oxide over the
emitter-base junction periphery (EB spacer / shallow trench edge), which add a surface recombination current to
the base current; the collector current barely changes, so the gain drops. Here the free top surface between
the emitter and base contacts carries
    U_s = s (n p - n_ie^2) / (n + p + 2 n_ie),   s = sigma v_th N_it            (midgap traps, sigma_n = sigma_p)
as a nodal term on the surface row (exact surface integral on the mesh), and the oxide-trapped charge N_ot as a
fixed positive sheet there. dose_response maps dose to (N_ot, N_it) with simws.radiation.tid.buildup, whose
hydrogen-model parameters come from MOS gate oxides: the spacer oxide field during irradiation, the capture
cross-section and the oxide parameters are technology dependent (ELDRS and annealing are not modelled), so
calibrate against measured delta-I_B before quoting absolute numbers. The shape (delta-I_B ideality near 2 and
linear in N_it at low trap density, unchanged I_C) is model independent.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from simws.tcad.hbt import HBTCase, UM, build, gummel
from simws.tcad.physics import contact_current

SIGMA_IT_CM2 = 1e-15     # interface-trap capture cross-section (typical midgap Pb centre order; provisional)
V_TH_CM_S = 1e7          # thermal velocity


def _surface_factor(case: HBTCase):
    """Per-node (surface length / NodeVolume) on the top row over w_e <= x <= x_bc, else 0 (1/cm). A nodal rate
    U_s * factor integrates to the surface integral of U_s exactly."""
    from devsim import get_node_model_values
    g = lambda k: np.array(get_node_model_values(device="d", region="si", name=k))
    x, y, vol = g("x"), g("y"), g("NodeVolume")
    f = np.zeros_like(x)
    top = np.where(np.abs(y) < 1e-12)[0]
    o = top[np.argsort(x[top])]
    xs = x[o]
    mid = np.concatenate(([xs[0]], 0.5 * (xs[1:] + xs[:-1]), [xs[-1]]))
    lo, hi = case.w_e * UM, case.x_bc * UM
    seg = np.clip(np.minimum(mid[1:], hi) - np.maximum(mid[:-1], lo), 0, None)
    f[o] = np.where(vol[o] > 0, seg / vol[o], 0.0)
    return f


def _surface_pre_setup(case):
    def pre():
        from devsim import set_node_values
        from devsim.python_packages.model_create import CreateSolution
        CreateSolution("d", "si", "SurfFactor")
        set_node_values(device="d", region="si", name="SurfFactor", values=list(_surface_factor(case)))
    return pre


def excess_base_current(case: HBTCase, N_it, VBE, N_ot=None, VCB=0.0, sigma_cm2=SIGMA_IT_CM2):
    """Gummel plots with interface traps N_it (cm^-2, list) and oxide charge N_ot (cm^-2, same length or None)
    over the emitter-base periphery. Returns IC, IB (A/um) per N_it, delta-IB and the surface recombination
    current q * int U_s dl computed from the same solutions (the part of delta-IB it accounts for)."""
    from devsim import get_node_model_values
    from devsim.python_packages.model_create import CreateNodeModel
    from simws.units import Q_E
    N_it = list(N_it)
    N_ot = list(N_ot) if N_ot is not None else [0.0] * len(N_it)
    VBE = np.asarray(VBE, float)
    out = {"VBE": VBE, "N_it": np.array(N_it), "N_ot": np.array(N_ot), "IC_A": [], "IB_A": [], "I_surf_A": []}
    for nit, nox in zip(N_it, N_ot):
        s = sigma_cm2 * V_TH_CM_S * nit
        Us = f"{s}*(Electrons*Holes - n_ie^2)/(Electrons + Holes + 2*n_ie)"
        c = dataclasses.replace(case, extra_recombination=f"SurfFactor*{Us}" if nit > 0 else None)
        xc = (lambda n_expr, q=nox: f"SurfFactor*{q}") if nox > 0 else None
        IC, IB, IS = [], [], []

        from simws.tcad.solve import ramp_bias
        build(c, pre_setup=_surface_pre_setup(c), extra_charge=xc)
        opts = dict(absolute_error=1e10, relative_error=1e-8, maximum_iterations=40)
        CreateNodeModel("d", "si", "SurfU", f"SurfFactor*{Us}" if nit > 0 else "0")
        if VCB:
            ramp_bias("d", "collector", VCB, step=0.25, **opts)
        for v in VBE:
            ramp_bias("d", "collector", v + VCB, step=0.1, **opts)
            ramp_bias("d", "base", v, step=0.05, **opts)
            IC.append(abs(contact_current("d", "collector")) * UM)
            IB.append(abs(contact_current("d", "base")) * UM)
            g = lambda k: np.array(get_node_model_values(device="d", region="si", name=k))
            IS.append(Q_E * float(np.sum(g("SurfU") * g("NodeVolume"))) * UM)
        out["IC_A"].append(IC); out["IB_A"].append(IB); out["I_surf_A"].append(IS)
    for k in ("IC_A", "IB_A", "I_surf_A"):
        out[k] = np.array(out[k])
    out["dIB_A"] = out["IB_A"] - out["IB_A"][0]
    out["beta"] = out["IC_A"] / out["IB_A"]
    return out


def dose_response(case: HBTCase, doses_krad, VBE, t_spacer_nm=100.0, E_ox_MV_cm=0.1, ox=None):
    """delta-I_B and gain vs dose: (N_ot, N_it) from simws.radiation.tid.buildup for the spacer oxide
    (thickness t_spacer_nm, field during irradiation E_ox_MV_cm: both technology dependent; see module doc)."""
    from simws.radiation.tid import buildup
    D = np.asarray(doses_krad, float) * 1e3
    N_ot, N_it = buildup(D, t_spacer_nm, E_ox_MV_cm, ox)
    r = excess_base_current(case, N_it, VBE, N_ot=N_ot)
    r["dose_krad"] = np.asarray(doses_krad, float)
    r["assumptions"] = {"t_spacer_nm": t_spacer_nm, "E_ox_MV_cm": E_ox_MV_cm, "sigma_it_cm2": SIGMA_IT_CM2,
                        "note": "MOS hydrogen-model parameters; no ELDRS, no annealing: calibrate before use"}
    return r


@dataclasses.dataclass
class StrikeSetup:
    let: float = 1.0              # MeV cm2/mg
    x0_um: float = 0.25           # track position (emitter centre by default)
    V_C: float = 2.0              # collector bias, emitter and base grounded
    radius_um: float = 0.05
    t_peak_s: float = 4e-12
    t_sigma_s: float = 1e-12
    dt_fine_s: float = 0.5e-12
    t_end_s: float = 2e-9
    max_step_splits: int = 6


def strike(case: HBTCase, setup: StrikeSetup = StrikeSetup(), material=None):
    """Transient ion strike; returns time, terminal currents (A/um), deposited and collected charges (fC per um
    of depth) and the end-of-run imbalance sum_k int I_k dt (should vanish once the device has relaxed)."""
    from devsim import get_node_model_values, set_node_values, set_parameter, solve
    from devsim.python_packages.model_create import CreateSolution
    from simws.materials import load
    from simws.radiation.track import IonTrack
    from simws.tcad.mixed_mode import _time_grid
    from simws.tcad.solve import ramp_bias
    mat = material or load("silicon")
    c = dataclasses.replace(case, extra_generation="Gscale*TrackFootprint")

    def pre():
        set_parameter(device="d", region="si", name="Gscale", value=0.0)
        CreateSolution("d", "si", "TrackFootprint")

    fine = max(setup.radius_um / 2.5, 0.005)
    xl = [(max(setup.x0_um - 4 * setup.radius_um, 0.0), fine), (setup.x0_um, fine),
          (min(setup.x0_um + 4 * setup.radius_um, case.width_um), fine)]
    build(c, material, pre_setup=pre, x_lines=xl)
    g = lambda k: np.array(get_node_model_values(device="d", region="si", name=k))
    track = IonTrack(let=setup.let, x0_um=setup.x0_um, length_um=case.depth_um, radius_um=setup.radius_um,
                     t_peak_s=setup.t_peak_s, t_sigma_s=setup.t_sigma_s, depth_um=1.0)
    foot, mesh_factor = track.footprint(mat, g("x") / UM, g("y") / UM, g("NodeVolume"))
    set_node_values(device="d", region="si", name="TrackFootprint", values=list(foot))
    opts = dict(absolute_error=1e10, relative_error=1e-7, maximum_iterations=40)
    ramp_bias("d", "collector", setup.V_C, step=0.25, **opts)

    def cur():
        return [contact_current("d", k) * UM for k in ("collector", "base", "emitter")]

    i0 = cur()
    solve(type="transient_dc", **opts)
    rows = [(0.0, *i0)]
    splits = 0

    def advance(t0, t1, level=0):
        nonlocal splits
        set_parameter(device="d", region="si", name="Gscale", value=track.temporal(t1))
        try:
            solve(type="transient_bdf1", tdelta=t1 - t0, charge_error=1e3, **opts)
        except Exception:
            if level >= setup.max_step_splits:
                raise RuntimeError(f"HBT strike: no convergence at t = {t1:.3e} s")
            splits += 1
            tm = 0.5 * (t0 + t1)
            advance(t0, tm, level + 1); advance(tm, t1, level + 1)
            return
        rows.append((t1, *cur()))

    t_prev = 0.0
    for t in _time_grid(setup.t_end_s, setup.t_peak_s + 6 * setup.t_sigma_s, setup.dt_fine_s):
        advance(t_prev, t); t_prev = t
    r = np.array(rows)
    dI = r[:, 1:] - np.array(i0)
    Q = np.trapezoid(dI, r[:, 0], axis=0) * 1e15          # fC per um of depth
    q_dep = track.total_charge_fC(mat)
    return {"time_s": r[:, 0], "IC_A": r[:, 1], "IB_A": r[:, 2], "IE_A": r[:, 3],
            "Q_collector_fC": float(abs(Q[0])), "Q_base_fC": float(abs(Q[1])), "Q_emitter_fC": float(abs(Q[2])),
            "Q_signed_fC": [float(q) for q in Q],
            "Q_deposited_fC": q_dep, "imbalance_fC": float(Q.sum()), "track_mesh_correction": mesh_factor,
            "time_step_splits": splits, "let": setup.let, "V_C": setup.V_C}
