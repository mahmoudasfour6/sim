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
from simws.units import K_B_EV

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


def _no_convergence(e) -> bool:
    m = str(e).lower()
    return "converge" in m or "stalled" in m


def _devsim_quiet():
    import os
    os.environ.setdefault("DEVSIM_QUIET", "1")


# ---------------------------------------------------------------------------------------------------- HBT Gummel
class HbtGummel(Adapter):
    manifest = Manifest(
        name="devsim.hbt_gummel", version="1.2", physics_version="1.1", env="tcad", engine="devsim", engine_version="env-locked",
        description="Forward Gummel sweep of the 2D npn SiGe HBT (simws.tcad.hbt) at temperature T",
        inputs=["VBE"],
        outputs=[OutputSpec(name="hbt.IC", unit="A/m", constraint="positive", description="collector current per m of emitter length"),
                 OutputSpec(name="hbt.IB", unit="A/m", constraint="positive"),
                 OutputSpec(name="hbt.IE", unit="A/m"),
                 OutputSpec(name="hbt.beta", unit="dimensionless", constraint="positive")],
        native_conventions=CUR,
        # mobility T laws are checked over 200-500 K and n_i(T) over 250-400 K (simws.materials.laws)
        validity={"T": (250.0, 400.0)},
        verification={"mesh": "paper fidelity: GCI per bias point over 3 meshes (spacings x1, 1/sqrt2, 1/2)"},
        physics_edges=[PhysicsEdge(output="hbt.IC", input="mat.n_i", mechanism="minority-carrier injection ~ n_ie^2"),
                       PhysicsEdge(output="hbt.IC", input="sige.dEv", mechanism="Ge band offset raises n_ie in the base"),
                       PhysicsEdge(output="hbt.IB", input="mat.tau", mechanism="SRH recombination in the emitter/base")],
        citations=["devsim", "masetti1983", "rieger1993"],
    )
    # cryo=True (0.9a): Altermatt freeze-out, Fermi-Dirac statistics (Nilsson inverse), Klaassen mobility with an
    # outer Picard iteration after every bias point, FD-consistent band-gap narrowing and recombination
    CRYO_CITATIONS = ["altermatt2006a", "altermatt2006b", "klaassen1992a", "klaassen1992b", "nilsson1973", "green1990"]

    # paper fidelity: solution verification by mesh refinement (every spacing x 1, 1/sqrt2, 1/2; nodes x1, x2, x4)
    MESH_RATIO = 2 ** 0.5
    MESH_LEVELS = 3
    P_MIN, P_MAX = 0.5, 4.0     # plausible observed orders for the GCI (second-order scheme; corners lower it)

    def run(self, job, ctx):
        _devsim_quiet()
        from simws.tcad.hbt import HBTCase
        extra = {}
        if job.key.params.get("T") is not None:
            extra["T"] = float(ctx.param("T", unit="K"))
        if job.key.params.get("cryo") is not None:
            extra["cryo"] = bool(ctx.param("cryo"))
        case = _case(HBTCase, _HBT_UNITS, ctx, extra)
        VBE = np.atleast_1d(ctx.param("VBE", unit="V")).astype(float)
        VCB = float(ctx.param("VCB", 0.0, unit="V") or 0.0)
        want_jn = bool(ctx.param("intrinsic_jn", False))
        study = job.fidelity == "paper" or bool(ctx.param("mesh_study", False))
        if not study:
            rows, jn = self._sweep(ctx, case, VBE, VCB, want_jn, record=True)
            ctx.diagnostics["case"] = dataclasses.asdict(case)
            return {"VBE": VBE[:len(rows)], "I": np.array(rows), "jn": np.array(jn), "T": case.T}
        # refinement study: coarse -> fine; reported values are the finest mesh's, sigma_num from its GCI. The 0.8 mesh
        # (2 nm at the abrupt emitter-base step) is outside the asymptotic range, so paper fidelity grades the mesh at
        # the junctions (HBTCase.eb_refine) unless the case sets it explicitly
        if "eb_refine" not in (ctx.param("case", {}) or {}):
            case = dataclasses.replace(case, eb_refine=True)
            ctx.flags.append("mesh:eb_refine")
        levels = []
        for k in range(self.MESH_LEVELS):
            c = dataclasses.replace(case, mesh_scale=case.mesh_scale / self.MESH_RATIO ** k)
            rows, jn = self._sweep(ctx, c, VBE, VCB, want_jn and k == self.MESH_LEVELS - 1,
                                   record=k == self.MESH_LEVELS - 1, partial_ok=False)
            levels.append(np.array(rows))
        n = min(len(L) for L in levels)                    # a current-bounded sweep can stop one point apart
        levels = [L[:n] for L in levels]
        ctx.diagnostics["case"] = dataclasses.asdict(dataclasses.replace(
            case, mesh_scale=case.mesh_scale / self.MESH_RATIO ** (self.MESH_LEVELS - 1)))
        return {"VBE": VBE[:n], "I": levels[-1], "jn": np.array(jn)[:n], "T": case.T, "I_levels": levels}

    def _sweep(self, ctx, case, VBE, VCB, want_jn, record=True, partial_ok=True):
        from simws.tcad.hbt import UM, build, _currents
        from simws.tcad.physics import relax_mobility
        from simws.tcad.solve import ramp_bias
        rows, jn = [], []
        ic_max = ctx.param("IC_max_A_per_m", None)
        ic_max = float(ic_max) if ic_max is not None else None
        try:
            used = build(case) or {}        # includes the equilibrium solve, which can fail for cold devices
        except Exception as e:
            if _no_convergence(e):
                raise NotConverged(f"equilibrium at T = {case.T:g} K: {e}") from e
            raise
        # Einstein-relation / detailed-balance guard: at zero bias every terminal current must vanish (an
        # inconsistent D/mu, e.g. Boltzmann diffusion with Fermi-Dirac densities, drives a spurious current)
        i0 = np.array(_currents()) * UM * A_PER_UM_TO_A_PER_M
        ctx.diagnostics.setdefault("equilibrium_currents_A_per_m", []).append(i0.tolist())
        tol = float(used.get("equilibrium_rel_tol", 1e-8))
        opts = dict(absolute_error=1e10, relative_error=tol, maximum_iterations=40)
        cb = None
        if case.cryo:
            cb = lambda _v: relax_mobility("d", "si", opts)
            if record:
                ctx.citations += [c for c in self.CRYO_CITATIONS if c not in ctx.citations]
                ctx.diagnostics["cryo"] = {k: used.get(k) for k in ("Nc_eff_cm3", "Nv_eff_cm3", "Eg_eV",
                                                                    "extended_precision", "equilibrium_rel_tol")}
                if tol > 1e-8:
                    ctx.flags.append(f"newton_rel_tol:{tol:g} (minority-carrier stall at T = {case.T:g} K)")
                if case.T < 250.0:
                    ctx.flags.append("cryo_unvalidated: below 250 K checked against closed forms and Charon only, "
                                     "not against measured devices")
        try:
            if VCB:
                ramp_bias("d", "collector", VCB, step=0.25, callback=cb, retry_callback=True, **opts)
            for v in VBE:
                ramp_bias("d", "collector", v + VCB, step=0.1, callback=cb, retry_callback=True, **opts)
                ramp_bias("d", "base", v, step=0.05, callback=cb, retry_callback=True, **opts)
                rows.append([c * UM for c in _currents()])      # A per um of emitter length, into the device
                if want_jn:
                    from simws.validation.sige_checks import _jn_intrinsic
                    jn.append(_jn_intrinsic(case))
                if record and ctx.param("neutral_base_edge", False):
                    from simws.validation.sige_checks import _neutral_base_edge
                    ctx.diagnostics.setdefault("_edge", []).append(_neutral_base_edge(case))
                if ic_max is not None and len(rows) < len(VBE):
                    # the sweep is bounded by current, not voltage: beyond ic_max the device is in high injection
                    # / Kirk effect, where self-heating (not modelled here) would dominate anyway. Stop when this
                    # point, or the next one extrapolated with the ideal slope (a 0.05 V step is x1800 at 77 K),
                    # exceeds ic_max
                    ic = rows[-1][0] * A_PER_UM_TO_A_PER_M
                    nxt = ic * np.exp(min((VBE[len(rows)] - v) / (K_B_EV * case.T), 700.0))
                    if ic > ic_max or nxt > 10 * ic_max:
                        if record:
                            ctx.flags.append(f"sweep_stopped:IC {ic:.3g} A/m at VBE={v:g} V (limit {ic_max:g} A/m)")
                        break
        except Exception as e:  # DEVSIM raises a generic error on Newton failure
            if not _no_convergence(e):
                raise
            # partial sweeps are kept when asked (the converged bias range is itself a result), else not_converged
            if not partial_ok or not ctx.param("keep_partial", False) or len(rows) < 2:
                raise NotConverged(f"mesh_scale {case.mesh_scale:.3g}: {e}" if not partial_ok else str(e)) from e
            ctx.flags.append(f"partial_sweep:no convergence at VBE={VBE[len(rows)]:g} V")
            ctx.diagnostics["converged_points"] = len(rows)
        return rows, jn

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
        sig = [None] * 4
        if "I_levels" in raw:
            sig = self._mesh_uncertainty(ctx, [L[keep] * A_PER_UM_TO_A_PER_M for L in raw["I_levels"]], V)
        ax = [{"name": "VBE", "unit": "V", "values": V.tolist()}]
        qs = [ctx.quantity("hbt.IC", I[:, 0], "A/m", axes=ax, conventions=CUR, sigma=sig[0]),
              ctx.quantity("hbt.IB", I[:, 1], "A/m", axes=ax, conventions=CUR, sigma=sig[1]),
              ctx.quantity("hbt.IE", I[:, 2], "A/m", axes=ax, conventions=CUR, sigma=sig[2]),
              ctx.quantity("hbt.beta", I[:, 0] / I[:, 1], "dimensionless", axes=ax, sigma=sig[3])]
        for q in qs:
            if q.uncertainty.kind == "sigma":
                q.uncertainty.components["numerical"] = q.uncertainty.sigma
        edge = ctx.diagnostics.pop("_edge", None)
        if edge:
            e = np.asarray(edge)[keep]
            # uncertainty: the edge is located by interpolation inside one mesh cell; half the change across that
            # cell, taken as a uniform interval (/sqrt3), bounds the discretisation error
            u = lambda j: np.abs(e[:, j + 3]) / (2 * 3 ** 0.5)
            qs += [ctx.quantity("hbt.y_nb_edge", e[:, 0], "um", axes=ax, sigma=u(0)),
                   ctx.quantity("hbt.ge_nb_edge", e[:, 1], "dimensionless", axes=ax, sigma=u(1)),
                   ctx.quantity("hbt.dEg_ge_nb_edge", e[:, 2], "eV", axes=ax, sigma=u(2))]
        if len(raw["jn"]):
            qs.append(ctx.quantity("hbt.Jn_intrinsic", np.asarray(raw["jn"])[keep] * 1e4, "A/m**2", axes=ax,
                                   flags=["magnitude"]))
        return qs

    def _mesh_uncertainty(self, ctx, levels, V):
        """Per-bias GCI (simws.verify.analyse) for IC, IB, IE and beta; returns standard uncertainties (absolute).
        The GCI is used only when the observed order is plausible (P_MIN <= p <= P_MAX) and the three meshes are
        near the asymptotic range; otherwise sigma is the conservative bound Fs |f1 - f2| / (r^P_MIN - 1) / k
        (an order no better than P_MIN assumed), the record says so and the result is flagged
        'mesh_not_asymptotic'."""
        from simws.hub import verification as Vf
        from simws.hub.uncertainty import GCI_COVERAGE_K
        names = ["hbt.IC", "hbt.IB", "hbt.IE", "hbt.beta"]
        series = [[L[:, 0] for L in levels], [L[:, 1] for L in levels], [L[:, 2] for L in levels],
                  [L[:, 0] / L[:, 1] for L in levels]]
        out, rec = [], {"kind": "solution", "ratio": self.MESH_RATIO, "levels": self.MESH_LEVELS,
                        "mesh_scales": [self.MESH_RATIO ** -k for k in range(self.MESH_LEVELS)], "quantities": {}}
        ok = True
        for name, ser in zip(names, series):
            sig, per = [], []
            for j in range(len(V)):
                vals = [float(abs(s[j])) for s in ser]
                r = Vf.solution_from_values(list(range(len(vals))), vals, ratio=self.MESH_RATIO)
                p, ar = r.get("order"), r.get("asymptotic_ratio")
                good = (r["sigma_num"] is not None and p is not None and self.P_MIN <= p <= self.P_MAX
                        and ar is not None and 0.5 < ar < 2.0) or (r["sigma_num"] == 0.0)
                if good:
                    sg = float(r["sigma_num"])
                else:
                    sg = 1.25 * abs(vals[-1] - vals[-2]) / (self.MESH_RATIO ** self.P_MIN - 1) / GCI_COVERAGE_K
                    ok = False
                sig.append(sg)
                per.append({"VBE": float(V[j]), "values_coarse_to_fine": vals, "order": p, "gci_fine": r.get("gci_fine"),
                            "asymptotic_ratio": ar, "conservative_bound": not good,
                            "rel_sigma": sg / vals[-1] if vals[-1] else None})
            rec["quantities"][name] = per
            out.append(np.array(sig))
        rec["passed"] = ok
        ctx.verification["mesh"] = rec
        if not ok:
            ctx.flags.append("mesh_not_asymptotic: conservative mesh bound used for some bias points "
                             "(verification['mesh'])")
        return out

    def self_check(self, job, ctx, qs):
        IC, IB, IE = (qs[i].array() for i in range(3))
        V = np.asarray(qs[0].axes[0].values)
        lo, hi = self.IDEALITY_WINDOW_A_PER_M
        win = (IC > lo) & (IC < hi)
        T = ctx.diagnostics["case"]["T"]
        i0 = np.abs(np.asarray(ctx.diagnostics.get("equilibrium_currents_A_per_m", [[0.0]]))).max()
        detailed_balance = ([] if i0 < self.I_FLOOR_A_PER_M else
                            [f"zero-bias terminal current {i0:.3g} A/m above the {self.I_FLOOR_A_PER_M:g} A/m floor "
                             "(Einstein relation / statistics inconsistent)"])
        return G.all_of(detailed_balance, G.positive("IC", IC), G.positive("IB", IB),
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


# ---------------------------------------------------------------------------------------------------- HBT AC (0.9b)
class HbtAc(Adapter):
    manifest = Manifest(
        name="devsim.hbt_ac", version="1.0", env="tcad", engine="devsim", engine_version="env-locked",
        description="Small-signal AC of the 2D HBT (simws.tcad.small_signal): f_T (h21, single-pole fit), f_max "
                    "(Mason's U), charge-control transit time, low-frequency collector/base current noise",
        inputs=["VBE"],
        outputs=[OutputSpec(name="hbt.fT", unit="Hz", constraint="positive"),
                 OutputSpec(name="hbt.fmax", unit="Hz", constraint="positive"),
                 OutputSpec(name="hbt.tau_charge", unit="s", constraint="positive",
                            description="dQ_B/dI_C at fixed V_CE (quasi-static charge control)"),
                 OutputSpec(name="hbt.IC", unit="A/m", constraint="positive"),
                 OutputSpec(name="hbt.IB", unit="A/m", constraint="positive")],
        native_conventions=CUR,
        validity={"T": (250.0, 400.0)},
        physics_edges=[PhysicsEdge(output="hbt.fT", input="hbt.tau_charge", mechanism="f_T = 1/(2 pi tau)")],
        citations=["devsim"],
    )

    def run(self, job, ctx):
        _devsim_quiet()
        from simws.tcad.hbt import HBTCase, build
        from simws.tcad import small_signal as SS
        extra = {}
        if job.key.params.get("T") is not None:
            extra["T"] = float(ctx.param("T", unit="K"))
        case = _case(HBTCase, _HBT_UNITS, ctx, extra)
        VBE = np.atleast_1d(ctx.param("VBE", unit="V")).astype(float)
        VCE = float(ctx.param("VCE", 1.0, unit="V"))
        noise = bool(ctx.param("noise", False))
        try:
            build(case, circuit=True)
        except Exception as e:
            if _no_convergence(e):
                raise NotConverged(str(e)) from e
            raise
        rows = []
        cb = SS.mobility_callback(case)
        if cb:
            ctx.flags.append("ac:frozen_mobility_linearisation")
        try:
            for v in VBE:
                SS.bias(float(v), VCE, cb=cb)
                if cb:
                    cb(float(v))
                dc = SS.dc_currents()
                cc = SS.charge_control(float(v), VCE, cb=cb)
                f_lo = 1e-4 * cc["fT_charge_Hz"]
                beta0 = SS.figures(SS.y_params(f_lo), f_lo)["h21"]
                f = 0.1 * cc["fT_charge_Hz"]
                fig = SS.figures(SS.y_params(f), f, beta0)
                row = {"VBE": float(v), "IC": dc["collector"], "IB": dc["base"], "fT": fig["fT_single_pole"],
                       "fmax": fig["fmax_est"], "tau": cc["tau_s"], "fT_charge": cc["fT_charge_Hz"]}
                if noise:
                    row["S_IC"] = SS.current_noise("VC.I")["S_A2_per_Hz"]
                    row["S_IB"] = SS.current_noise("VB.I")["S_A2_per_Hz"]
                rows.append(row)
        except Exception as e:
            if _no_convergence(e):
                raise NotConverged(str(e)) from e
            raise
        ctx.diagnostics["case"] = dataclasses.asdict(case)
        ctx.diagnostics["fT_ac_vs_charge"] = [r["fT"] / r["fT_charge"] - 1 for r in rows]
        return rows

    def collect(self, job, ctx, rows):
        V = [r["VBE"] for r in rows]
        ax = [{"name": "VBE", "unit": "V", "values": V}]
        per_m = 100.0                                  # per cm of emitter length -> per m
        qs = [ctx.quantity("hbt.fT", [r["fT"] for r in rows], "Hz", axes=ax),
              ctx.quantity("hbt.fmax", [r["fmax"] for r in rows], "Hz", axes=ax),
              ctx.quantity("hbt.tau_charge", [r["tau"] for r in rows], "s", axes=ax),
              ctx.quantity("hbt.IC", [r["IC"] * per_m for r in rows], "A/m", axes=ax, conventions=CUR),
              ctx.quantity("hbt.IB", [r["IB"] * per_m for r in rows], "A/m", axes=ax, conventions=CUR)]
        if "S_IC" in rows[0]:
            qs += [ctx.quantity("hbt.S_IC", [r["S_IC"] * per_m for r in rows], "A**2/Hz/m", axes=ax),
                   ctx.quantity("hbt.S_IB", [r["S_IB"] * per_m for r in rows], "A**2/Hz/m", axes=ax)]
        return qs

    def self_check(self, job, ctx, qs):
        # the AC f_T and the quasi-static charge-control f_T must agree (0.9b gate: 5 %)
        d = ctx.diagnostics.get("fT_ac_vs_charge", [])
        bad = [x for x in d if abs(x) > 0.05]
        return [f"AC f_T differs from charge-control f_T by {100 * max(map(abs, bad)):.1f} %"] if bad else []
