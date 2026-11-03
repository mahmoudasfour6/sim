"""DFT adapters (env: materials, Quantum ESPRESSO pw.x).

dft.si_bandedge   Si band-edge parameters for ONE functional (LDA | PBE | PBEsol) at the measured lattice constant:
                  electron masses, conduction-minimum position, DFT gap, N_c(T) and N_v(T) power laws.
                  Run it once per functional with member="functional=<name>"; simws.hub.uncertainty.Ensemble then
                  gives the functional spread (model-form uncertainty) from correlated members.

Same calculation as simws.studies.dft_params (bands section of _one_functional), so the 0.7 golden values
(simws.dft.PBE.m_l, ...) are the reproduction gate. Quantum ESPRESSO scratch files never enter the store.
"""
from __future__ import annotations

import dataclasses
import shutil
import tempfile


from simws.hub import guards as G
from simws.hub.adapter import Adapter, InvalidInput
from simws.hub.schema import Manifest, OutputSpec, PhysicsEdge

FUNCTIONALS = ("PBE", "LDA", "PBEsol")


def _dft_env() -> str:
    """The conda env with pw.x (config dft.env, as for the 0.7 rule)."""
    try:
        from simws.config import load
        return (load().get("dft") or {}).get("env", "materials") or "materials"
    except Exception:  # noqa: BLE001
        return "materials"


class SiBandedge(Adapter):
    manifest = Manifest(
        name="dft.si_bandedge", version="1.0", env=_dft_env(), engine="quantum-espresso", engine_version="env-locked",
        description="Si band-edge parameters from pw.x band energies for one exchange-correlation functional",
        # relinking after an environment upgrade: one canary (an hour of pw.x); MPI reductions are not bitwise
        # reproducible, so the canary must agree to 1e-6 rather than 1e-9
        cost_model={"relink_canaries": 1, "relink_rtol": 1e-6},
        inputs=["functional"],
        outputs=[OutputSpec(name="dft.m_l", unit="dimensionless", constraint="positive", description="in m0"),
                 OutputSpec(name="dft.m_t", unit="dimensionless", constraint="positive"),
                 OutputSpec(name="dft.k0_frac_GX", unit="dimensionless", constraint="positive",
                            description="conduction minimum as a fraction of Gamma-X"),
                 OutputSpec(name="dft.Eg_dft", unit="eV", constraint="positive"),
                 OutputSpec(name="dft.Nc_300", unit="1/m**3", constraint="positive"),
                 OutputSpec(name="dft.Nv_300", unit="1/m**3", constraint="positive"),
                 OutputSpec(name="dft.Nc_T_exponent", unit="dimensionless"),
                 OutputSpec(name="dft.Nv_T_exponent", unit="dimensionless")],
        validity={"T": (200.0, 500.0)},
        physics_edges=[PhysicsEdge(output="dft.Nc_300", input="dft.m_l", mechanism="conduction DOS mass"),
                       PhysicsEdge(output="dft.Nc_300", input="dft.m_t", mechanism="conduction DOS mass")],
        citations=["giannozzi2009", "giannozzi2017", "hamann2013", "vansetten2018"],
    )

    def resolve(self, params):
        """Pin the pseudopotential set, the files' SHA-256 and the cutoff into the job key."""
        from simws.hub.canonical import sha256_file
        from simws.studies.dft_params import Config
        cfg = Config.from_workspace(pseudo_set=params.get("pseudo_set"))
        fname = params.get("functional")
        if fname not in FUNCTIONALS:
            return params                                  # run() reports invalid_input
        sc, _, fr, _ = cfg.pseudos_for(fname)
        from pathlib import Path
        out = dict(params)
        out["pseudo_set"] = cfg.pseudo_set
        out.setdefault("ecutwfc", (cfg.ecutwfc, "Ry"))
        out["pseudo_sha256"] = {f: sha256_file(Path(cfg.pseudo_dir) / f)[:16] for f in (sc, fr) if f}
        if cfg.flags:
            out["pseudo_flags"] = list(cfg.flags)
        return out

    def run(self, job, ctx):
        from simws.dft import bandedge
        from simws.hub.canonical import sha256_file
        from simws.studies.dft_params import A_EXP_SI, Config, _power_law
        fname = ctx.param("functional")
        if fname not in FUNCTIONALS:
            raise InvalidInput(f"functional must be one of {FUNCTIONALS}")
        cfg = Config.from_workspace(ranks=job.budget.ranks, pseudo_set=ctx.param("pseudo_set"))
        if job.key.params.get("ecutwfc") is not None:
            cfg = dataclasses.replace(cfg, ecutwfc=round(float(ctx.param("ecutwfc", unit="Ry")), 9))
        sc, fo, fr, fo_fr = cfg.pseudos_for(fname)
        from pathlib import Path
        for f, h in (ctx.param("pseudo_sha256") or {}).items():
            if sha256_file(Path(cfg.pseudo_dir) / f)[:16] != h:
                raise InvalidInput(f"pseudopotential {f} changed since the job was keyed")
        ctx.flags += [f for f in cfg.flags]
        if cfg.pseudo_set != "pseudodojo":
            ctx.flags.append(f"pseudos:{cfg.pseudo_set}")
            ctx.citations = [c for c in ctx.citations if c not in ("hamann2013", "vansetten2018")]
        s = bandedge.Setup(pseudo=sc, soc_pseudo=fr, a_angstrom=A_EXP_SI, functional=fo, ecutwfc=cfg.ecutwfc,
                           ranks=cfg.ranks, pseudo_dir=cfg.pseudo_dir)
        wd = tempfile.mkdtemp(prefix="qe_", dir=ctx.staging)
        try:
            from pathlib import Path
            wd = Path(wd)
            if fr and fo_fr != fo:
                b = bandedge.compute(dataclasses.replace(s, soc_pseudo=None), wd / "bands", dos=False)
                bs = bandedge.compute(dataclasses.replace(s, pseudo=fr, functional=fo_fr), wd / "bands_fr")
                b["holes_soc"], b["dos"] = bs["holes_soc"], bs["dos"]
                b["Eg_dft_soc_eV"] = b["Eg_dft_eV"] - bs["holes_soc"]["Delta_so_eV"] / 3
            else:
                b = bandedge.compute(s, wd / "bands")
        finally:
            shutil.rmtree(wd, ignore_errors=True)
        d = b["dos"]
        sel = [i for i, T in enumerate(d["T_K"]) if 200 <= T <= 500]
        T = [d["T_K"][i] for i in sel]
        Nc300, pc, rc = _power_law(T, [d["Nc_cm3"][i] for i in sel])
        Nv300, pv, rv = _power_law(T, [d["Nv_cm3"][i] for i in sel])
        ctx.diagnostics.update({"pseudo_set": cfg.pseudo_set, "pseudo": sc, "pseudo_fr": fr, "input_dft": fo, "ecutwfc": cfg.ecutwfc,
                                "pw_version": b.get("pw_version"), "notes": b.get("notes", []),
                                "powerlaw_resid": max(rc, rv)})
        e = b["electrons"]
        return {"m_l": e["m_l"], "m_t": e["m_t"], "k0": e["k0_2pi_a"],
                "Eg": b.get("Eg_dft_soc_eV", b["Eg_dft_eV"]), "Nc300": Nc300, "pc": pc, "Nv300": Nv300, "pv": pv}

    def collect(self, job, ctx, r):
        val = {"T": (200.0, 500.0)}
        q = ctx.quantity
        return [q("dft.m_l", r["m_l"], "dimensionless"), q("dft.m_t", r["m_t"], "dimensionless"),
                q("dft.k0_frac_GX", r["k0"], "dimensionless"), q("dft.Eg_dft", r["Eg"], "eV"),
                q("dft.Nc_300", r["Nc300"] * 1e6, "1/m**3", validity=val),
                q("dft.Nv_300", r["Nv300"] * 1e6, "1/m**3", validity=val),
                q("dft.Nc_T_exponent", r["pc"], "dimensionless", validity=val),
                q("dft.Nv_T_exponent", r["pv"], "dimensionless", validity=val)]

    def self_check(self, job, ctx, qs):
        v = {x.name: float(x.array()) for x in qs}
        return G.all_of(G.bounded("k0", v["dft.k0_frac_GX"], 0.7, 0.95),            # Si: Delta valleys near 0.85 X
                        G.bounded("m_l/m_t", v["dft.m_l"] / v["dft.m_t"], 2.0, 8.0),
                        G.bounded("Eg_dft", v["dft.Eg_dft"], 0.2, 1.3),
                        G.bounded("power-law residual", ctx.diagnostics["powerlaw_resid"], hi=0.05))
