"""DEVSIM adapters (env: tcad). Engine imports stay inside run().

devsim.hbt_gummel   SiGe HBT / Si BJT forward Gummel sweep at temperature T
devsim.mos_cv       quasi-static C-V of a MOS capacitor
devsim.strike_hbt   transient ion strike on the HBT: collected charges and terminal currents
devsim.mms_poisson  code verification: manufactured-solution convergence order of the production Poisson equation

Units at the boundary: DEVSIM works in cm and A per cm of depth (2D); the adapters convert to canonical SI
(A/m per metre of emitter length, F/m^2, C/m^2). DEVSIM's contact_current is positive into the device, which is
the canonical passive sign (checked: forward-active npn gives IC > 0, IB > 0, IE < 0).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from simws.hub import guards as G
from simws.hub.adapter import Adapter, InvalidInput, NotConverged
from simws.hub.schema import Manifest, OutputSpec, PhysicsEdge

A_PER_UM_TO_A_PER_M = 1e6
CUR = {"terminal_current": "into_device+"}

# HBTCase / MOSCase field units (their native units); quantities given for these fields are converted to them
_HBT_UNITS = {"width_um": "um", "depth_um": "um", "w_e": "um", "t_e": "um", "w_b": "um", "y_sc": "um", "x_xb": "um",
              "y_xb": "um", "x_bc": "um", "N_E": "cm**-3", "N_B": "cm**-3", "N_C": "cm**-3", "N_SC": "cm**-3",
              "ge_peak": "dimensionless", "ge_start": "dimensionless", "tau": "s", "T": "K"}
_MOS_UNITS = {"t_ox_nm": "nm", "NA": "cm**-3", "si_um": "um", "gate_offset_V": "V", "Q_f": "cm**-2",
              "D_it": "1/cm**2/eV", "cap_nm": "nm", "well_nm": "nm", "ge_x": "dimensionless"}


def _case(cls, units, ctx, extra=None):
    raw = dict(ctx.param("case", {}) or {})
    raw.update(extra or {})
    kw = {}
    names = {f.name for f in dataclasses.fields(cls)}
    for k, v in raw.items():
        if k not in names:
            raise InvalidInput(f"{cls.__name__} has no field '{k}'")
        if isinstance(v, dict) and "unit" in v:
            from simws.units import Q_
            v = float(Q_(v["value"], v["unit"]).to(units[k]).magnitude)
        kw[k] = v
    return cls(**kw)


def _devsim_quiet():
    import os
    os.environ.setdefault("DEVSIM_QUIET", "1")


# ---------------------------------------------------------------------------------------------------- HBT Gummel
class HbtGummel(Adapter):
    manifest = Manifest(
        name="devsim.hbt_gummel", version="1.1", env="tcad", engine="devsim", engine_version="env-locked",
        description="Forward Gummel sweep of the 2D npn SiGe HBT (simws.tcad.hbt) at temperature T",
        inputs=["VBE"],
        outputs=[OutputSpec(name="hbt.IC", unit="A/m", constraint="positive", description="collector current per m of emitter length"),
                 OutputSpec(name="hbt.IB", unit="A/m", constraint="positive"),
                 OutputSpec(name="hbt.IE", unit="A/m"),
                 OutputSpec(name="hbt.beta", unit="dimensionless", constraint="positive")],
        native_conventions=CUR,
        # mobility T laws are checked over 200-500 K and n_i(T) over 250-400 K (simws.materials.laws)
        validity={"T": (250.0, 400.0)},
        physics_edges=[PhysicsEdge(output="hbt.IC", input="mat.n_i", mechanism="minority-carrier injection ~ n_ie^2"),
                       PhysicsEdge(output="hbt.IC", input="sige.dEv", mechanism="Ge band offset raises n_ie in the base"),
                       PhysicsEdge(output="hbt.IB", input="mat.tau", mechanism="SRH recombination in the emitter/base")],
        citations=["devsim", "masetti1983", "rieger1993"],
    )

    def run(self, job, ctx):
        _devsim_quiet()
        from simws.tcad.hbt import UM, HBTCase, build, _currents
        from simws.tcad.solve import ramp_bias
        extra = {}
        if job.key.params.get("T") is not None:
            extra["T"] = float(ctx.param("T", unit="K"))
        case = _case(HBTCase, _HBT_UNITS, ctx, extra)
        VBE = np.atleast_1d(ctx.param("VBE", unit="V")).astype(float)
        VCB = float(ctx.param("VCB", 0.0, unit="V") or 0.0)
        want_jn = bool(ctx.param("intrinsic_jn", False))
        opts = dict(absolute_error=1e10, relative_error=1e-8, maximum_iterations=40)
        rows, jn = [], []
        try:
            build(case)                     # includes the equilibrium solve, which can fail for cold devices
        except Exception as e:
            if "converge" in str(e).lower():
                raise NotConverged(f"equilibrium at T = {case.T:g} K: {e}") from e
            raise
        try:
            if VCB:
                ramp_bias("d", "collector", VCB, step=0.25, **opts)
            for v in VBE:
                ramp_bias("d", "collector", v + VCB, step=0.1, **opts)
                ramp_bias("d", "base", v, step=0.05, **opts)
                rows.append([c * UM for c in _currents()])      # A per um of emitter length, into the device
                if want_jn:
                    from simws.validation.sige_checks import _jn_intrinsic
                    jn.append(_jn_intrinsic(case))
        except Exception as e:  # DEVSIM raises a generic error on Newton failure
            if "converge" not in str(e).lower():
                raise
            # partial sweeps are kept when asked (the converged bias range is itself a result), else not_converged
            if not ctx.param("keep_partial", False) or len(rows) < 2:
                raise NotConverged(str(e)) from e
            ctx.flags.append(f"partial_sweep:no convergence at VBE={VBE[len(rows)]:g} V")
            ctx.diagnostics["converged_points"] = len(rows)
        ctx.diagnostics["case"] = dataclasses.asdict(case)
        n = len(rows)
        return {"VBE": VBE[:n], "I": np.array(rows), "jn": np.array(jn), "T": case.T}

    # currents below this are at the Newton-tolerance noise floor (absolute_error 1e10 particles/s per cm of depth in
    # the continuity equations is ~2e-7 A/m; terminal sums of order 1e-9 A/m are seen): dropped, not reported
    I_FLOOR_A_PER_M = 1e-8
    # low-injection window used for the ideality guard
    IDEALITY_WINDOW_A_PER_M = (1e-6, 10.0)

    def collect(self, job, ctx, raw):
        I = raw["I"] * A_PER_UM_TO_A_PER_M
        V = raw["VBE"]
        keep = (I[:, 0] > self.I_FLOOR_A_PER_M) & (I[:, 1] > self.I_FLOOR_A_PER_M)
        if not keep.all():
            ctx.flags.append(f"dropped {int((~keep).sum())} bias point(s) below the {self.I_FLOOR_A_PER_M:g} A/m current floor")
            ctx.diagnostics["dropped_VBE"] = V[~keep].tolist()
        if keep.sum() < 1:
            from simws.hub.adapter import OutsideValidity
            raise OutsideValidity("no bias point above the current noise floor: raise VBE")
        I, V = I[keep], V[keep]
        ax = [{"name": "VBE", "unit": "V", "values": V.tolist()}]
        qs = [ctx.quantity("hbt.IC", I[:, 0], "A/m", axes=ax, conventions=CUR),
              ctx.quantity("hbt.IB", I[:, 1], "A/m", axes=ax, conventions=CUR),
              ctx.quantity("hbt.IE", I[:, 2], "A/m", axes=ax, conventions=CUR),
              ctx.quantity("hbt.beta", I[:, 0] / I[:, 1], "dimensionless", axes=ax)]
        if len(raw["jn"]):
            qs.append(ctx.quantity("hbt.Jn_intrinsic", np.asarray(raw["jn"])[keep] * 1e4, "A/m**2", axes=ax,
                                   flags=["magnitude"]))
        return qs

    def self_check(self, job, ctx, qs):
        IC, IB, IE = (qs[i].array() for i in range(3))
        V = np.asarray(qs[0].axes[0].values)
        lo, hi = self.IDEALITY_WINDOW_A_PER_M
        win = (IC > lo) & (IC < hi)
        T = ctx.diagnostics["case"]["T"]
        return G.all_of(G.positive("IC", IC), G.positive("IB", IB),
                        G.kirchhoff([IC, IB, IE], rtol=1e-4, atol=1e-8),
                        G.monotonic("IC(VBE)", IC[np.argsort(V)], strict=True) if len(V) > 1 else [],
                        G.ideality("IC", V[win], IC[win], n_min=0.95, n_max=1.3, T=T))


# ---------------------------------------------------------------------------------------------------- MOS C-V
class MosCv(Adapter):
    manifest = Manifest(
        name="devsim.mos_cv", version="1.0", env="tcad", engine="devsim", engine_version="env-locked",
        description="Quasi-static C-V of a MOS capacitor (simws.tcad.mos)",
        inputs=["V"],
        outputs=[OutputSpec(name="mos.C", unit="F/m**2", constraint="positive"),
                 OutputSpec(name="mos.psi_s", unit="V"),
                 OutputSpec(name="mos.Q_gate", unit="C/m**2"),
                 OutputSpec(name="mos.C_ox", unit="F/m**2", constraint="positive")],
        physics_edges=[PhysicsEdge(output="mos.C", input="ox.eps_r", mechanism="series oxide capacitance")],
        citations=["devsim"],
    )

    def run(self, job, ctx):
        _devsim_quiet()
        from simws.tcad.mos import MOSCase, cv
        case = _case(MOSCase, _MOS_UNITS, ctx)
        V = np.atleast_1d(ctx.param("V", unit="V")).astype(float)
        return cv(case, V)

    def collect(self, job, ctx, r):
        ax = [{"name": "V_gate", "unit": "V", "values": np.asarray(r["V"]).tolist()}]
        qs = [ctx.quantity("mos.C", np.asarray(r["C_F_cm2"]) * 1e4, "F/m**2", axes=ax),
              ctx.quantity("mos.psi_s", r["psi_s_V"], "V", axes=ax),
              ctx.quantity("mos.Q_gate", np.asarray(r["Q_gate_C_cm2"]) * 1e4, "C/m**2", axes=ax),
              ctx.quantity("mos.C_ox", r["C_ox_F_cm2"] * 1e4, "F/m**2")]
        if r.get("V_fb") is not None:
            qs.append(ctx.quantity("mos.V_fb", r["V_fb"], "V"))
        return qs

    def self_check(self, job, ctx, qs):
        C, cox = qs[0].array(), float(qs[3].array())
        return G.all_of(G.positive("C", C), G.bounded("C/C_ox", C / cox, hi=1.0 + 1e-3))


# ---------------------------------------------------------------------------------------------------- HBT strike
class StrikeHbt(Adapter):
    manifest = Manifest(
        name="devsim.strike_hbt", version="1.0", env="tcad", engine="devsim", engine_version="env-locked",
        description="Transient heavy-ion strike through the HBT emitter (simws.radiation.hbt_rad.strike)",
        inputs=["LET"],
        outputs=[OutputSpec(name="strike.Q_collector", unit="C/m", constraint="nonnegative",
                            description="collected charge per m of device depth"),
                 OutputSpec(name="strike.Q_base", unit="C/m", constraint="nonnegative"),
                 OutputSpec(name="strike.Q_emitter", unit="C/m", constraint="nonnegative"),
                 OutputSpec(name="strike.Q_deposited", unit="C/m", constraint="positive"),
                 OutputSpec(name="strike.IC_t", unit="A/m")],
        native_conventions=CUR,
        physics_edges=[PhysicsEdge(output="strike.Q_collector", input="strike.Q_deposited",
                                   mechanism="drift + diffusion collection of track charge")],
        citations=["devsim"],
    )

    def run(self, job, ctx):
        _devsim_quiet()
        from simws.radiation.hbt_rad import StrikeSetup, strike
        from simws.tcad.hbt import HBTCase
        case = _case(HBTCase, _HBT_UNITS, ctx)
        kw = {"let": float(ctx.param("LET", unit="MeV*cm**2/mg"))}
        if job.key.params.get("V_C") is not None:
            kw["V_C"] = float(ctx.param("V_C", unit="V"))
        for k in ("x0_um", "radius_um"):
            if job.key.params.get(k) is not None:
                kw[k] = float(ctx.param(k))
        try:
            return strike(case, StrikeSetup(**kw))
        except RuntimeError as e:
            if "convergence" in str(e):
                raise NotConverged(str(e)) from e
            raise

    def collect(self, job, ctx, r):
        fC_um = 1e-15 / 1e-6                                   # fC per um of depth -> C per m
        ctx.diagnostics.update({k: r[k] for k in ("imbalance_fC", "track_mesh_correction", "time_step_splits")})
        ax = [{"name": "time", "unit": "s", "values": np.asarray(r["time_s"]).tolist()}]
        return [ctx.quantity("strike.Q_collector", r["Q_collector_fC"] * fC_um, "C/m"),
                ctx.quantity("strike.Q_base", r["Q_base_fC"] * fC_um, "C/m"),
                ctx.quantity("strike.Q_emitter", r["Q_emitter_fC"] * fC_um, "C/m"),
                ctx.quantity("strike.Q_deposited", r["Q_deposited_fC"] * fC_um, "C/m"),
                ctx.quantity("strike.IC_t", np.asarray(r["IC_A"]) * A_PER_UM_TO_A_PER_M, "A/m", axes=ax,
                             conventions=CUR)]

    def self_check(self, job, ctx, qs):
        q_dep = float(qs[3].array())
        imb = abs(ctx.diagnostics["imbalance_fC"]) * 1e-9
        return G.all_of(G.charge_neutral(imb, q_dep, rtol=1e-3),
                        G.bounded("collected fraction", float(qs[0].array()) / q_dep, lo=0.0, hi=1.0))


# ---------------------------------------------------------------------------------------------------- MMS
class MmsPoisson(Adapter):
    manifest = Manifest(
        name="devsim.mms_poisson", version="1.0", env="tcad", engine="devsim", engine_version="env-locked",
        description="Manufactured-solution order of the production Poisson-Boltzmann discretisation",
        outputs=[OutputSpec(name="mms.observed_order", unit="dimensionless", constraint="positive"),
                 OutputSpec(name="mms.err_max", unit="V", constraint="positive")],
        verification={"code": "observed order within 0.1 of 2"},
        citations=["roache2002", "devsim"],
    )

    def run(self, job, ctx):
        _devsim_quiet()
        from simws.tcad.mms import poisson_order
        return poisson_order(levels=int(ctx.param("levels", 4)), n0=int(ctx.param("n0", 20)))

    def collect(self, job, ctx, r):
        from simws.hub import verification as V
        ctx.verification["code"] = V.code(r["observed_order"], r["expected_order"], orders=r["order"])
        ax = [{"name": "h", "unit": "cm", "values": r["h_cm"]}]
        # the order estimate's own uncertainty: spread of the successive-level orders
        spread = float(np.ptp(r["order"][-2:])) if len(r["order"]) > 1 else None
        return [ctx.quantity("mms.observed_order", r["observed_order"], "dimensionless",
                             sigma=spread if spread else None),
                ctx.quantity("mms.err_max", r["err_max_V"], "V", axes=ax)]

    def self_check(self, job, ctx, qs):
        return [] if ctx.verification["code"]["passed"] else [f"observed order {qs[0].value:.3f} is not 2"]
