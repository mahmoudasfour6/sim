"""materials.intrinsic: n_i(T) and Eg(T) of a material set (pure Python; the smallest real adapter).

params: material (str), profile (str or null), T (temperature quantity, scalar or list)
outputs: mat.n_i [m^-3], mat.Eg [eV] on a T axis
Validity: 250-400 K (the n_i(300 K) T^1.5 scaling and the Varshni fit are checked there; outside -> 'extrapolated').
"""
from __future__ import annotations

import numpy as np

from simws.hub.adapter import Adapter, InvalidInput
from simws.hub.schema import Manifest, OutputSpec, PhysicsEdge


class Intrinsic(Adapter):
    manifest = Manifest(
        name="materials.intrinsic", version="1.0", env="workflow", engine="simws.materials", engine_version="0.8",
        description="Intrinsic carrier density and band gap versus temperature from the material set",
        inputs=["material", "T"],
        outputs=[OutputSpec(name="mat.n_i", unit="cm**-3", constraint="positive"),
                 OutputSpec(name="mat.Eg", unit="eV", constraint="positive")],
        validity={"T": (250.0, 400.0)},
        physics_edges=[PhysicsEdge(output="mat.n_i", input="mat.Eg", mechanism="thermal excitation across the gap",
                                   method="declared")],
        citations=["varshni1967"],
    )

    def run(self, job, ctx):
        from simws.materials import load
        from simws.materials.semiconductor import intrinsic_density
        try:
            mat = load(ctx.param("material"), ctx.param("profile"))
        except (FileNotFoundError, KeyError) as e:
            raise InvalidInput(str(e)) from e
        T = np.atleast_1d(ctx.param("T", unit="K")).astype(float)
        if np.any(T <= 0):
            raise InvalidInput("temperature must be positive")
        Eg0, a, b = mat.get("varshni_Eg0", "eV"), mat.get("varshni_alpha", "eV/K"), mat.get("varshni_beta", "K")
        return {"T": T, "ni_cm3": np.array([intrinsic_density(mat, t) for t in T]),
                "Eg": Eg0 - a * T ** 2 / (T + b), "provisional": mat.provisional()}

    def collect(self, job, ctx, raw):
        axes = [{"name": "T", "unit": "K", "values": raw["T"].tolist()}]
        flags = ["provisional_parameters"] if raw["provisional"] else []
        val = {"T": (250.0, 400.0)}
        ctx.diagnostics["provisional"] = raw["provisional"]
        return [ctx.quantity("mat.n_i", raw["ni_cm3"] * 1e6, "1/m**3", axes=axes, flags=flags, validity=val),
                ctx.quantity("mat.Eg", raw["Eg"], "eV", axes=axes, flags=flags, validity=val)]

    def self_check(self, job, ctx, qs):
        ni = qs[0].array()
        T = np.asarray(qs[0].axes[0].values)
        if len(T) > 1 and np.any(np.diff(ni[np.argsort(T)]) <= 0):
            return ["n_i must increase with temperature"]
        return []
