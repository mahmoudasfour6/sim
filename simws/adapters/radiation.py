"""Radiation adapters.

radiation.tid_mos    total-ionizing-dose buildup (N_ot, N_it) and the resulting MOS voltage shifts (env: tcad)
radiation.ser_rate   single-event upset rate from an LET spectrum and a Weibull cross section (env: workflow)

Dose crosses the boundary in Gy with the material named (convention 'dose': Gy(SiO2)); the engine works in krad:
1 krad(SiO2) = 10 Gy(SiO2).
"""
from __future__ import annotations

import numpy as np

from simws import conventions as cv
from simws.hub import guards as G
from simws.hub.adapter import Adapter, InvalidInput
from simws.hub.schema import Manifest, OutputSpec, PhysicsEdge

DOSE = {"dose": "Gy(SiO2)"}


class TidMos(Adapter):
    manifest = Manifest(
        name="radiation.tid_mos", version="1.0", env="tcad", engine="devsim", engine_version="env-locked",
        description="TID oxide/interface-trap buildup and MOS V_fb, V_mg, V_inv shifts (simws.radiation.tid)",
        inputs=["dose", "V_irr"],
        outputs=[OutputSpec(name="tid.N_ot", unit="1/m**2", constraint="nonnegative"),
                 OutputSpec(name="tid.N_it", unit="1/m**2", constraint="nonnegative"),
                 OutputSpec(name="tid.dV_fb", unit="V"), OutputSpec(name="tid.dV_mg", unit="V"),
                 OutputSpec(name="tid.dV_inv", unit="V"), OutputSpec(name="tid.E_ox", unit="V/m")],
        native_conventions={"dose": "rad(SiO2)"},
        physics_edges=[PhysicsEdge(output="tid.dV_mg", input="tid.N_ot", mechanism="positive trapped oxide charge"),
                       PhysicsEdge(output="tid.dV_inv", input="tid.N_it", mechanism="charged interface traps"),
                       PhysicsEdge(output="tid.N_ot", input="tid.E_ox", mechanism="field-dependent charge yield")],
        citations=["mclean1987", "dozier1987", "esqueda2020"],
    )

    def run(self, job, ctx):
        from simws.radiation.tid import dose_response
        from simws.tcad.mos import MOSCase
        if ctx.param("dose_material", "SiO2") != "SiO2":
            raise InvalidInput("radiation.tid_mos takes dose in Gy(SiO2)")
        D_Gy = np.atleast_1d(ctx.param("dose", unit="Gy")).astype(float)
        D_krad = cv.from_canonical("dose", D_Gy, "rad(SiO2)") / 1e3
        case = MOSCase(t_ox_nm=float(ctx.param("t_ox", unit="nm")), NA=float(ctx.param("NA", 1e17, unit="cm**-3")))
        r = dose_response(case, V_irr=float(ctx.param("V_irr", unit="V")), doses_krad=D_krad.tolist(),
                          centroid_nm=float(ctx.param("centroid", 1.0, unit="nm")))
        r["dose_Gy"] = D_Gy
        return r

    def collect(self, job, ctx, r):
        ax = [{"name": "dose", "unit": "Gy", "values": np.asarray(r["dose_Gy"]).tolist()}]
        flags = ["provisional_parameters"]          # f_ot, N_sat and the hydrogen set are technology dependent
        ctx.diagnostics["yield"] = r["yield"]
        mk = lambda n, v, u: ctx.quantity(n, v, u, axes=ax, conventions=DOSE, flags=flags)
        return [mk("tid.N_ot", np.asarray(r["N_ot"]) * 1e4, "1/m**2"),
                mk("tid.N_it", np.asarray(r["N_it"]) * 1e4, "1/m**2"),
                mk("tid.dV_fb", r["dV_fb"], "V"), mk("tid.dV_mg", r["dV_mg"], "V"), mk("tid.dV_inv", r["dV_inv"], "V"),
                ctx.quantity("tid.E_ox", r["E_ox_MV_cm"] * 1e8, "V/m")]

    def self_check(self, job, ctx, qs):
        D = np.asarray(qs[0].axes[0].values)
        o = np.argsort(D)
        return G.all_of(G.monotonic("N_ot(D)", qs[0].array()[o]), G.monotonic("N_it(D)", qs[1].array()[o]))


class SerRate(Adapter):
    manifest = Manifest(
        name="radiation.ser_rate", version="1.0", env="workflow", engine="simws.radiation.ser", engine_version="0.8",
        description="Isotropic-environment upset rate: LET spectrum x Weibull cross section (simws.radiation.ser)",
        inputs=["spectrum_let", "spectrum_flux", "weibull"],
        outputs=[OutputSpec(name="ser.rate", unit="1/s", constraint="nonnegative")],
        verification={"code": "power-law spectrum x step cross section vs closed form"},
        physics_edges=[PhysicsEdge(output="ser.rate", input="ser.L_crit", mechanism="cross-section onset")],
        citations=["petersen1992"],
    )

    def run(self, job, ctx):
        from simws.radiation.ser import LETSpectrum, WeibullCrossSection, rate_isotropic, verify
        L = np.asarray(ctx.param("spectrum_let", unit="MeV*cm**2/mg"), float)
        F = np.asarray(ctx.param("spectrum_flux", unit="1/cm**2/s/sr"), float)    # integral flux F(>L)
        if L.shape != F.shape or L.size < 3:
            raise InvalidInput("spectrum_let and spectrum_flux need the same length (>= 3)")
        o = np.argsort(L)
        w = ctx.param("weibull")
        u = lambda k, unit: float(_q(w[k], unit))
        xs = WeibullCrossSection(l0=u("l0", "MeV*cm**2/mg"), w=u("w", "MeV*cm**2/mg"), s=float(_q(w["s"], None)),
                                 sat_cm2=u("sat", "cm**2"))
        rate = rate_isotropic(LETSpectrum(L[o], F[o], str(ctx.param("spectrum_source", "user"))), xs)
        return {"rate": rate, "verify": verify()}

    def collect(self, job, ctx, r):
        from simws.hub import verification as V
        v = r["verify"]                                  # tolerance as in the 0.4 unit test of simws.radiation.ser
        ctx.verification["code"] = V.closed_form(v["numerical"], v["analytic"], rtol=2e-3)
        return [ctx.quantity("ser.rate", r["rate"], "1/s", flags=["spectrum_user_supplied"])]

    def self_check(self, job, ctx, qs):
        return [] if ctx.verification["code"]["passed"] else ["SER integration failed its closed-form check"]


def _q(v, unit):
    """Plain number, or a canonical quantity dict converted to `unit`."""
    if isinstance(v, dict) and "unit" in v:
        from simws.units import Q_
        return Q_(v["value"], v["unit"]).to(unit or "dimensionless").magnitude
    return v
