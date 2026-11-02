"""First-principles material parameters for silicon with uncertainty, written as a 'dft' material profile.

    from simws.studies.dft_params import Config, run
    res = run(Config.from_workspace(), outdir="results/dft")      # -> results/dft/silicon.dft.yaml + params.json

For each functional in the ensemble (LDA, PBE, PBEsol):
  equation of state (a0, B0) -> elastic constants at a0 -> band edges at the MEASURED lattice constant (masses,
  spin-orbit splitting, N_c(T), N_v(T)) -> deformation potentials Xi_u, b.
Once (PBE base): HSE06 gap correction (EXX q-mesh = k-mesh), and numerical-convergence variants (finer DOS grid,
higher cutoff) for the numerical part of the uncertainty.
Aggregation per parameter: value = ensemble mean; sigma = sqrt(sigma_functional^2 + sigma_numerical^2), with
sigma_functional the standard deviation over functionals and sigma_numerical the largest change in the convergence
variants. The functional spread is a working estimate of the DFT model error, not a rigorous bound; every value
is also compared with its measurement (simws.validation.dft_checks).
Functionals other than a pseudopotential's own are applied through input_dft (recorded in the notes): PBEsol
always, LDA for spin-orbit when only a PBE fully-relativistic pseudopotential is available.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from simws.dft import bandedge, qe, strain

A_EXP_SI = 5.431
BUNDLED = Path(__file__).resolve().parents[1] / "dft" / "pseudo"


@dataclasses.dataclass
class Config:
    pseudo_dir: str
    pbe: str                                  # scalar (or fully-relativistic) PBE pseudopotential for Si
    pbe_fr: Optional[str] = None              # fully-relativistic PBE (spin-orbit); None -> no SOC quantities
    lda: Optional[str] = None                 # LDA pseudopotential; None -> PBE pseudopotential + input_dft
    ecutwfc: float = 40.0
    ranks: int = 1
    functionals: tuple = ("PBE", "LDA", "PBEsol")
    hse_kgrids: tuple = ((4, 4, 4),)          # add (6, 6, 6) for the HSE convergence check (slow)
    convergence: bool = True
    chgnet: bool = True

    @classmethod
    def from_workspace(cls, ranks=None):
        """Settings from config.yaml (dft: pseudo_dir, pseudos, ecutwfc, ranks, hse_kgrids). Without dft.pseudo_dir
        the pseudopotentials shipped in simws/dft/pseudo are used (the validated set); with it, the files named in
        dft.pseudos, or ones picked by name from that directory."""
        from simws.config import load as cfg
        c = cfg()
        d = c.get("dft", {}) or {}
        ps = d.get("pseudos", {}) or {}
        ranks = int(ranks or d.get("ranks", c.get("threads", {}).get("mpi_ranks", 1)))
        common = dict(ecutwfc=float(d.get("ecutwfc", 40.0)), ranks=ranks,
                      hse_kgrids=tuple(tuple(k) for k in d.get("hse_kgrids", [[4, 4, 4]])))
        if not d.get("pseudo_dir"):
            return cls(str(BUNDLED), "Si.pbe-rrkj.UPF", "Si.rel-pbe-rrkj.UPF", "Si.pz-vbc.UPF", **common)
        pdir = str(Path(os.path.expanduser(d["pseudo_dir"])).resolve())
        names = sorted(p.name for p in Path(pdir).iterdir() if p.is_file())
        pick = lambda f: next((n for n in names if f(n.lower())), None)
        fr = lambda n: n.startswith("si") and any(t in n for t in ("rel", "_r.", "fr"))
        pbe = ps.get("Si_pbe") or pick(lambda n: n.startswith("si") and "pbe" in n and "sol" not in n and not fr(n))
        pbe_fr = ps.get("Si_pbe_fr") or pick(lambda n: fr(n) and "sol" not in n)
        lda = ps.get("Si_lda") or pick(lambda n: n.startswith("si") and ("pz" in n or "lda" in n) and not fr(n))
        if not pbe:
            raise FileNotFoundError(f"no PBE pseudopotential for Si in {pdir}; set dft.pseudos.Si_pbe")
        return cls(pdir, pbe, pbe_fr, lda, **common)

    def pseudos_for(self, functional):
        """(scalar pseudo, functional override, fully-relativistic pseudo, its override)."""
        if functional == "PBE":
            return self.pbe, None, self.pbe_fr, None
        if functional == "LDA":
            if self.lda:
                return self.lda, None, self.pbe_fr, ("pz" if self.pbe_fr else None)
            return self.pbe, "pz", self.pbe_fr, ("pz" if self.pbe_fr else None)
        if functional == "PBEsol":
            return self.pbe, "pbesol", self.pbe_fr, ("pbesol" if self.pbe_fr else None)
        raise ValueError(functional)


def _one_functional(cfg: Config, fname, wd: Path):
    sc, fo, fr, fo_fr = cfg.pseudos_for(fname)
    P = {"Si": sc}
    res = {"functional": fname, "pseudo": sc, "pseudo_fr": fr, "input_dft": fo, "input_dft_fr": fo_fr}
    eos = qe.equation_of_state("Si", 5.45, P, functional=fo, ecutwfc=cfg.ecutwfc, kgrid=(8, 8, 8),
                               workdir=wd / "eos", ranks=cfg.ranks, pseudo_dir=cfg.pseudo_dir)
    res["eos"] = {k: v for k, v in eos.items() if k not in ("sigma",)}
    res["elastic"] = strain.elastic_constants("Si", eos["a0_A"], P, ecutwfc=cfg.ecutwfc, kgrid=(8, 8, 8),
                                              workdir=wd / "elastic", ranks=cfg.ranks, functional=fo,
                                              pseudo_dir=cfg.pseudo_dir)
    # band edges: the scalar pseudo with its functional; the SOC part uses the fully-relativistic one
    s = bandedge.Setup(pseudo=sc, soc_pseudo=fr, a_angstrom=A_EXP_SI, functional=fo, ecutwfc=cfg.ecutwfc,
                       ranks=cfg.ranks, pseudo_dir=cfg.pseudo_dir)
    if fr and fo_fr != fo:
        # different overrides for the two runs: do them separately
        res["bands"] = bandedge.compute(dataclasses.replace(s, soc_pseudo=None), wd / "bands", dos=False)
        s_fr = dataclasses.replace(s, pseudo=fr, functional=fo_fr)
        res["bands_soc"] = bandedge.compute(s_fr, wd / "bands_fr")
        res["bands"]["holes_soc"] = res["bands_soc"]["holes_soc"]
        res["bands"]["dos"] = res["bands_soc"]["dos"]
        res["bands"]["Eg_dft_soc_eV"] = res["bands"]["Eg_dft_eV"] - res["bands_soc"]["holes_soc"]["Delta_so_eV"] / 3
    else:
        res["bands"] = bandedge.compute(s, wd / "bands")
    res["deformation"] = strain.deformation_potentials("Si", A_EXP_SI, P, soc_pseudos=({"Si": fr} if fr else None),
                                                       ecutwfc=cfg.ecutwfc, workdir=wd / "deformation",
                                                       ranks=cfg.ranks, functional=fo, pseudo_dir=cfg.pseudo_dir,
                                                       k0=res["bands"]["electrons"]["k0_2pi_a"], soc_functional=fo_fr)
    return res


def _power_law(T, N):
    T, N = np.asarray(T, float), np.asarray(N, float)
    p, lnA = np.polyfit(np.log(T / 300.0), np.log(N), 1)
    fit = np.exp(lnA) * (T / 300.0) ** p
    return float(np.exp(lnA)), float(p), float(np.max(np.abs(fit / N - 1)))


def _extract(r):
    """Flat dict of the parameters taken from one functional's results."""
    b, e, el, dp = r["bands"], r["bands"]["electrons"], r["elastic"], r["deformation"]
    h = b.get("holes_soc") or b["holes_scalar"]
    d = b["dos"]
    sel = [i for i, T in enumerate(d["T_K"]) if 200 <= T <= 500]
    T = [d["T_K"][i] for i in sel]
    Nc300, pc, rc = _power_law(T, [d["Nc_cm3"][i] for i in sel])
    Nv300, pv, rv = _power_law(T, [d["Nv_cm3"][i] for i in sel])
    out = {"a0": r["eos"]["a0_A"], "B0": r["eos"]["B0_GPa"], "C11": el["C11_GPa"], "C12": el["C12_GPa"],
           "C44": el["C44_GPa"], "m_l": e["m_l"], "m_t": e["m_t"], "k0": e["k0_2pi_a"],
           "Nc_300": Nc300, "Nc_T_exponent": pc, "Nv_300": Nv300, "Nv_T_exponent": pv,
           "m_dn_300": float(np.interp(300.0, d["T_K"], d["m_dn_T"])),
           "m_dh_300": float(np.interp(300.0, d["T_K"], d["m_dh_T"])),
           "Xi_u": dp["Xi_u_eV"], "Eg_dft": b.get("Eg_dft_soc_eV", b["Eg_dft_eV"]),
           "_powerlaw_resid": max(rc, rv)}
    for nm in ("hh", "lh", "so"):
        for dd in ("100", "111", "dos"):
            out[f"m_{nm}_{dd}"] = h[f"m_{nm}_{dd}"]
    if "Delta_so_eV" in h:
        out["Delta_so"] = h["Delta_so_eV"]
    if "b_eV" in dp:
        out["b"] = dp["b_eV"]
    return out


UNITS = {"a0": "angstrom", "B0": "GPa", "C11": "GPa", "C12": "GPa", "C44": "GPa", "Nc_300": "cm**-3",
         "Nv_300": "cm**-3", "Xi_u": "eV", "b": "eV", "Delta_so": "eV", "Eg_dft": "eV", "Eg0_hybrid": "eV",
         "n_i": "cm**-3"}


def aggregate(per_func: Dict[str, dict], numerical: Dict[str, float]):
    keys = sorted(set.intersection(*[set(v) for v in per_func.values()]) - {"_powerlaw_resid"})
    agg = {}
    for k in keys:
        vals = np.array([per_func[f][k] for f in per_func], float)
        sf = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        sn = float(numerical.get(k, 0.0))
        agg[k] = {"value": float(vals.mean()), "sigma": float(math.hypot(sf, sn)), "sigma_functional": sf,
                  "sigma_numerical": sn, "by_functional": {f: float(per_func[f][k]) for f in per_func}}
    return agg


def run(cfg: Config, outdir, workdir=None, log=print):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    wd = Path(workdir or out / "runs")
    raw, per = {}, {}
    for f in cfg.functionals:
        log(f"[dft] {f}: equation of state, elastic constants, band edges, deformation potentials")
        raw[f] = _one_functional(cfg, f, wd / f)
        per[f] = _extract(raw[f])
        log(f"[dft] {f}: a0 {per[f]['a0']:.4f} A, m_l {per[f]['m_l']:.3f}, m_t {per[f]['m_t']:.3f}, "
            f"Nc {per[f]['Nc_300']:.3e}, Nv {per[f]['Nv_300']:.3e}")
    numerical = {}
    if cfg.convergence:
        log("[dft] convergence variants (PBE): finer DOS grid, +10 Ry cutoff")
        sc, fo, fr, fo_fr = cfg.pseudos_for("PBE")
        base = bandedge.Setup(pseudo=sc, soc_pseudo=fr, a_angstrom=A_EXP_SI, ecutwfc=cfg.ecutwfc, ranks=cfg.ranks,
                              pseudo_dir=cfg.pseudo_dir)
        variants = {"h_dos_0.01": dataclasses.replace(base, h_dos=0.01, R_val=0.28, R_con_t=0.17, kx_con_min=0.57),
                    "ecut_plus10": dataclasses.replace(base, ecutwfc=cfg.ecutwfc + 10)}
        raw["convergence"] = {}
        for name, s in variants.items():
            rb = bandedge.compute(s, wd / "convergence" / name)
            raw["convergence"][name] = rb
            v = _extract({**raw["PBE"], "bands": rb})
            for k, val in v.items():
                if k.startswith("_"):
                    continue
                numerical[k] = max(numerical.get(k, 0.0), abs(val - per["PBE"][k]))
    agg = aggregate(per, numerical)
    # hybrid-functional gap: PBE(+SOC) band-edge gap + (HSE - PBE) scissor
    hyb = []
    sc, fo, fr, _ = cfg.pseudos_for("PBE")
    for kg in cfg.hse_kgrids:
        log(f"[dft] HSE06 gap correction, k = q = {kg}")
        h = bandedge.hybrid_gap(bandedge.Setup(pseudo=sc, a_angstrom=A_EXP_SI, ecutwfc=cfg.ecutwfc, ranks=cfg.ranks,
                                               pseudo_dir=cfg.pseudo_dir), wd / f"hse_{kg[0]}", kgrid=kg)
        hyb.append(h)
    sc_vals = [h["scissor_eV"] for h in hyb]
    eg_pbe = per["PBE"]["Eg_dft"]
    agg["Eg0_hybrid"] = {"value": eg_pbe + sc_vals[-1],
                         "sigma": float(math.hypot(max(h["scissor_spread_eV"] for h in hyb),
                                                   (max(sc_vals) - min(sc_vals)) if len(sc_vals) > 1 else 0.0)),
                         "scissor_eV": sc_vals, "Eg_pbe_soc_eV": eg_pbe}
    # n_i(300 K) from the DFT densities of states and the MEASURED gap (Varshni, Thurmond 1975)
    from simws.materials import load
    si = load("silicon")
    Eg300 = si.get("varshni_Eg0", "eV") - si.get("varshni_alpha", "eV/K") * 300.0 ** 2 / (300.0 + si.get("varshni_beta", "K"))
    kT = 8.617333262e-5 * 300.0
    Nc, Nv = agg["Nc_300"], agg["Nv_300"]
    ni = math.sqrt(Nc["value"] * Nv["value"]) * math.exp(-Eg300 / (2 * kT))
    rel = 0.5 * math.hypot(Nc["sigma"] / Nc["value"], Nv["sigma"] / Nv["value"])
    agg["n_i"] = {"value": ni, "sigma": ni * rel, "note": "sqrt(Nc Nv) exp(-Eg/2kT), measured Eg(300 K)"}
    if cfg.chgnet:
        try:
            log("[dft] CHGNet benchmark")
            raw["chgnet"] = chgnet_benchmark()
        except ImportError as exc:
            raw["chgnet"] = {"skipped": f"CHGNet not installed in this environment ({exc})"}
    raw["critical_thickness"] = critical_thickness_study(agg)
    res = {"parameters": agg, "per_functional": per, "hybrid": hyb, "raw": raw,
           "config": {k: v for k, v in dataclasses.asdict(cfg).items()}}
    write_profile(out / "silicon.dft.yaml", agg, res)
    return res


def chgnet_benchmark(x=(0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0), n_configs=4):
    from simws.dft import mlip
    lat = mlip.alloy_lattice(x=x, n_configs=n_configs)
    el = {e: mlip.elastic_constants(e) for e in ("Si", "Ge")}
    return {"alloy_lattice": lat, "elastic": el}


# measured elastic constants (GPa) at room temperature: McSkimin & Andreatch, J. Appl. Phys. 35, 2161 (1964) (Si),
# 34, 651 (1963) (Ge)
C_EXP = {"Si": {"C11_GPa": 165.7, "C12_GPa": 63.9, "C44_GPa": 79.6},
         "Ge": {"C11_GPa": 128.5, "C12_GPa": 48.3, "C44_GPa": 66.8}}
HBT_BASE = (0.0, 0.15, 50.0)  # Ge grade start, peak and base width (nm) of simws.tcad.hbt.HBTCase() (unit-tested)
THETA_DISMUKES_A = 0.027      # a(x) = 5.431 + 0.20 x + 0.027 x^2 (Dismukes et al. 1964) -> below Vegard by 0.027 x(1-x)


def critical_thickness_study(agg=None):
    """Matthews-Blakeslee and People-Bean critical thicknesses vs x with the measured a(x) and elastic constants
    (DFT Si elastic constants as a sensitivity case), and the stability of the simws SiGe devices."""
    from simws.dft import mlip
    xs = np.round(np.arange(0.05, 0.501, 0.05), 3)
    cases = {"measured": (C_EXP["Si"], C_EXP["Ge"])}
    if agg and "C11" in agg:
        si_dft = {"C11_GPa": agg["C11"]["value"], "C12_GPa": agg["C12"]["value"]}
        cases["dft_Si_elastic"] = (si_dft, C_EXP["Ge"])
    out = {"x": xs.tolist()}
    for name, (es, eg) in cases.items():
        out[name] = {"h_MB_nm": [mlip.critical_thickness(x, THETA_DISMUKES_A, es, eg, "matthews-blakeslee") for x in xs],
                     "h_PB_nm": [mlip.critical_thickness(x, THETA_DISMUKES_A, es, eg, "people-bean") for x in xs]}
    es, eg = cases["measured"]
    ge_start, ge_peak, w_b_nm = HBT_BASE
    grade = np.linspace(ge_start, ge_peak, 101)
    out["devices"] = {
        "hbt_base": mlip.layer_stability(grade, w_b_nm, THETA_DISMUKES_A, es, eg),
        "mos_buried_well": mlip.layer_stability([0.25], 10.0, THETA_DISMUKES_A, es, eg)}
    return out


def write_profile(path, agg, res):
    """The 'dft' material profile: parameters with value, unit, sigma, status 'computed' and a provenance ref."""
    import yaml
    try:
        from simws.provenance import stamp
        pv = stamp()
        commit = pv.get("git_commit", "?")
    except Exception:  # noqa: BLE001
        commit = "?"
    ref = f"simws.studies.dft_params, LDA/PBE/PBEsol ensemble (commit {commit})"
    entries = {}

    def put(key, name, unit, note="", scale=1.0):
        if name in agg:
            a = agg[name]
            entries[key] = {"value": float(a["value"] * scale), "unit": unit, "sigma": float(a["sigma"] * scale),
                            "ref": ref + (f"; {note}" if note else ""), "status": "computed"}

    put("m_l", "m_l", "dimensionless", "curvature at the conduction minimum")
    put("m_t", "m_t", "dimensionless", "curvature at the conduction minimum")
    for nm in ("hh", "lh", "so"):
        for dd in ("100", "111", "dos"):
            put(f"m_{nm}_{dd}", f"m_{nm}_{dd}", "dimensionless", "spin-orbit bands at Gamma" if "Delta_so" in agg else "")
    put("Delta_so", "Delta_so", "eV")
    put("Nc_300", "Nc_300", "cm**-3", "Boltzmann sum over DFT band energies (non-parabolic)")
    put("Nv_300", "Nv_300", "cm**-3", "Boltzmann sum over DFT band energies (non-parabolic, spin-orbit)")
    put("Nc_T_exponent", "Nc_T_exponent", "dimensionless", "power-law fit 200-500 K")
    put("Nv_T_exponent", "Nv_T_exponent", "dimensionless", "power-law fit 200-500 K")
    put("n_i", "n_i", "cm**-3", "from the DFT Nc, Nv and the measured Eg(300 K)")
    put("Eg0_hybrid", "Eg0_hybrid", "eV", "HSE06 scissor on PBE+SOC, 0 K, no zero-point renormalization; "
        "device runs keep the measured Varshni gap")
    put("C11", "C11", "GPa"); put("C12", "C12", "GPa"); put("C44", "C44", "GPa", "Kleinman relaxed")
    put("deformation_Xi_u", "Xi_u", "eV"); put("deformation_b", "b", "eV")
    put("a0_dft", "a0", "angstrom", "zero-pressure DFT lattice constant (bands use the measured one)")
    doc = {"silicon": {"dft": entries}}
    Path(path).write_text("# Generated by simws.studies.dft_params. Review, then promote with\n"
                          "#   python tools/promote_computed.py " + str(path) + "\n" + yaml.safe_dump(doc, sort_keys=False))
    return path
