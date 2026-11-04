"""0.9c cross-engine agreement: each physics has a second, independent open-source engine, run on the same reference
problems. The tolerances below were fixed on 2026-11-04, before any cross-engine run, from what the two methods
are expected to share and where they legitimately differ; they are not tuned to the results.

dft    Quantum ESPRESSO (norm-conserving Si.pbe-rrkj) vs GPAW (PAW, gpaw-data setups), Si diamond, PBE:
       a0 (Birch-Murnaghan on the same 7-point a grid)         0.5 %   pseudopotential vs PAW frozen core; published
                                                                        Delta-gauge spreads for Si are ~0.1-0.3 %
       B0                                                       5 %     curvature of the same fit, more sensitive
       PBE band energies at a = 5.431 A (G, X, L, G-X line):    50 meV  both relative to the VBM at Gamma
         indirect gap, direct gap at Gamma, L-point gap
fem    scikit-fem vs FEniCSx (DOLFINx), P1 Lagrange on the same gmsh mesh (both read the same .msh):
       two-region variable-coefficient Poisson, nodal values    1e-9 (relative to max|u|)  identical discretisation:
                                                                        only solver round-off may differ
       nonlinear heat k(T) = k300 (300/T)^a, T_max              1e-8 relative
dd     DEVSIM vs Solcore (its drift-diffusion engine, Sesame), 1D Si pn diode, the same parameters
       (N_c, N_v, E_g, mobilities, SRH lifetimes, no BGN, no Auger, ohmic contacts):
       dark current density at 0.3-0.6 V                        2 %     same model; different Scharfetter-Gummel
                                                                        variants and meshes
       short-circuit current under uniform generation           1 %

Each agreement also produces a model-form standard uncertainty: the code-to-code range over sqrt(12) (uniform over
the alternatives, as simws.hub.uncertainty does for every model_form component), written to the store's model-form
registry (simws.hub.model_form) so that paper-fidelity results carry it in their budgets.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

FIXED_ON = "2026-11-04"
TOLERANCES = {
    "dft.a0": 0.005, "dft.B0": 0.05, "dft.band_eV": 0.050,
    "fem.poisson_nodal": 1e-9, "fem.heat_Tmax": 1e-8,
    "dd.dark_J": 0.02, "dd.Jsc": 0.01,
}
SQRT12 = math.sqrt(12.0)

PSEUDO = Path(__file__).resolve().parents[1] / "dft" / "pseudo"


def _mf(a, b):
    """Relative model-form standard uncertainty of a two-engine pair: |a - b| / mean / sqrt(12)."""
    a, b = float(a), float(b)
    return abs(a - b) / max(abs(0.5 * (a + b)), 1e-300) / SQRT12


# ================================================================================================= DFT: QE vs GPAW
def _klist():
    s = [0.75, 0.80, 0.825, 0.85, 0.875, 0.90]
    return np.array([[0, 0, 0], [0, 0, 1.0], [0.5, 0.5, 0.5]] + [[0, 0, x] for x in s])


def _edges(eig, nocc):
    vbm = eig[0, :nocc].max()                         # Gamma
    cb = eig[:, nocc:]
    return {"indirect_gap_eV": float(cb.min() - vbm), "direct_gap_G_eV": float(cb[0].min() - vbm),
            "gap_X_eV": float(cb[1].min() - vbm), "gap_L_eV": float(cb[2].min() - vbm),
            "cbm_k_tpiba": float(_klist()[int(np.argmin(cb.min(axis=1))), 2])}


def _dojo_dir():
    """PseudoDojo v0.4 (tools/fetch_pseudodojo.sh) under config data.pseudopotentials."""
    try:
        from simws.config import load as cfg
        root = Path(cfg()["data"]["pseudopotentials"]).expanduser()
    except Exception:  # noqa: BLE001
        root = Path("sim_data/pseudopotentials")
    return (root if root.is_absolute() else Path.cwd() / root) / "pseudodojo_v0.4"


def dft_engine(engine, work, ecut_Ry=40.0, ecut_eV=500.0, kgrid=(8, 8, 8), qe_pseudo="rrkj") -> dict:
    """One engine's side of the DFT pair: EOS on the shared a grid and band energies at a = 5.431 A on the shared k
    list. Cached per setting in <work>/<tag>.json (the pre-registered setting keeps the names qe.json / gpaw.json).
    qe_pseudo: "rrkj" (bundled Si.pbe-rrkj.UPF) or "dojo" (PseudoDojo v0.4 standard Si.pbe.upf)."""
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    kgrid = tuple(int(k) for k in kgrid)
    default = (ecut_Ry, ecut_eV, kgrid, qe_pseudo) == (40.0, 500.0, (8, 8, 8), "rrkj")
    tag = engine if default else (f"qe_{qe_pseudo}_{ecut_Ry:g}Ry_k{kgrid[0]}" if engine == "qe"
                                  else f"gpaw_{ecut_eV:g}eV_k{kgrid[0]}")
    cache = work / f"{tag}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    a_grid = 5.47 * np.linspace(0.98, 1.02, 7)
    wd = work / tag
    if engine == "qe":
        from simws.dft import qe
        if qe_pseudo == "rrkj":
            pseudos, pdir = {"Si": "Si.pbe-rrkj.UPF"}, str(PSEUDO)
        elif qe_pseudo == "dojo":
            pseudos, pdir = {"Si": "Si.pbe.upf"}, str(_dojo_dir())
            if not (Path(pdir) / "Si.pbe.upf").is_file():
                raise FileNotFoundError(f"PseudoDojo Si.pbe.upf not in {pdir}: run bash tools/fetch_pseudodojo.sh")
        else:
            raise ValueError(qe_pseudo)
        eos = qe.equation_of_state("Si", 5.47, pseudos, ecutwfc=ecut_Ry, kgrid=kgrid, npts=7, span=0.02,
                                   workdir=wd / "qe_eos", pseudo_dir=pdir)
        st = qe.Structure.diamond("Si", 5.431)
        qe.run(qe.PW(st, pseudos, ecutwfc=ecut_Ry, kgrid=kgrid, prefix="si", pseudo_dir=pdir), wd / "qe_bands")
        rb = qe.run(qe.PW(st, pseudos, ecutwfc=ecut_Ry, calculation="bands", klist=_klist(), nbnd=8, prefix="si",
                          pseudo_dir=pdir), wd / "qe_bands")
        eig, version = np.asarray(rb.eigenvalues_eV), rb.pw_version
    elif engine == "gpaw":
        from simws.dft import gpaw_engine as G
        eos = G.equation_of_state("Si", a_grid, ecut_eV=ecut_eV, kgrid=kgrid, workdir=wd / "gpaw_eos")
        gb = G.band_energies("Si", 5.431, _klist(), ecut_eV=ecut_eV, kgrid=kgrid, workdir=wd / "gpaw_bands")
        eig, version = np.asarray(gb["eigenvalues_eV"]), eos.get("engine")
    else:
        raise ValueError(engine)
    out = {"engine": engine, "version": version, "a0_A": float(eos["a0_A"]), "B0_GPa": float(eos["B0_GPa"]),
           **_edges(eig, 4), "eigenvalues_eV": eig.tolist(),
           "settings": {"ecut_Ry": ecut_Ry, "ecut_eV": ecut_eV, "kgrid": list(kgrid), "functional": "PBE",
                        "qe_pseudo": qe_pseudo if engine == "qe" else "PAW (gpaw-data)"}}
    cache.write_text(json.dumps(out, indent=1))
    return out


DFT_QUANTITIES = ("a0_A", "B0_GPa", "indirect_gap_eV", "direct_gap_G_eV", "gap_X_eV", "gap_L_eV")


def _compare(q, g) -> dict:
    out = {"engines": {"qe": q["version"], "gpaw": g["version"]}, "settings": {"qe": q["settings"],
           "gpaw": g["settings"]}, "rows": {}, "pass": True,
           "cbm_position_tpiba": {"qe": q["cbm_k_tpiba"], "gpaw": g["cbm_k_tpiba"]}}
    for k in DFT_QUANTITIES:
        a, b = q[k], g[k]
        if k == "a0_A":
            d, tol = abs(a / b - 1), TOLERANCES["dft.a0"]
        elif k == "B0_GPa":
            d, tol = abs(a / b - 1), TOLERANCES["dft.B0"]
        else:
            d, tol = abs(a - b), TOLERANCES["dft.band_eV"]
        ok = bool(d <= tol)
        out["rows"][k] = {"qe": a, "gpaw": b, "difference": d, "tolerance": tol, "pass": ok,
                          "model_form_rel": _mf(a, b)}
        out["pass"] &= ok
    return out


def dft_pair(work, **kw) -> dict:
    """QE (pw.x, qe env) vs GPAW (in-process) at one setting (the pre-registered one by default)."""
    gkw = {k: v for k, v in kw.items() if k != "qe_pseudo"}
    return _compare(dft_engine("qe", work, **kw), dft_engine("gpaw", work, **gkw))


# Convergence and pseudopotential rule, fixed on 2026-11-04 after the pre-registered setting (QE 40 Ry Si.pbe-rrkj,
# GPAW 500 eV, 8x8x8) failed on the L-point gap (61 meV vs 50 meV), before any of the runs below. The tolerances
# do not change.
#   1. Ladders: QE 40/60/80 Ry, GPAW 500/700/900 eV at 8x8x8; then the top cutoffs at 12x12x12.
#   2. Converged = the last step (cutoff, then k) changes every quantity by < 10 % of its tolerance (band 5 meV,
#      a0 0.05 %, B0 0.5 %). The pair is judged at the top setting; a non-converged ladder is reported as such.
#   3. Pass at the top setting -> the pre-registered failure was numerical; the pair uses the top setting.
#   4. Fail at the top setting -> QE with PseudoDojo v0.4 (standard, ONCVPSP) at the same setting. Pass -> PseudoDojo
#      becomes the QE side of the pair (the rrkj disagreement is recorded as a pseudopotential finding). Fail -> the
#      gate fails.
QE_LADDER, GPAW_LADDER, K_TOP = (40.0, 60.0, 80.0), (500.0, 700.0, 900.0), (12, 12, 12)
CONV_FRACTION = 0.10


def _delta(a, b, k):
    return abs(a[k] / b[k] - 1) if k in ("a0_A", "B0_GPa") else abs(a[k] - b[k])


def _ladder_steps(runs):
    """Changes between consecutive settings, as a fraction of each tolerance."""
    tolk = {"a0_A": TOLERANCES["dft.a0"], "B0_GPa": TOLERANCES["dft.B0"]}
    steps = []
    for (la, a), (lb, b) in zip(runs, runs[1:]):
        fr = {k: _delta(b, a, k) / tolk.get(k, TOLERANCES["dft.band_eV"]) for k in DFT_QUANTITIES}
        steps.append({"from": la, "to": lb, "fraction_of_tolerance": fr, "worst": max(fr.values())})
    return steps


def dft_pair_resolved(work, fetch_dojo=None) -> dict:
    work = Path(work)
    pre = dft_pair(work)
    res = {"preregistered": pre, "rule_fixed_on": "2026-11-04 (after the pre-registered failure, before these runs)"}
    if pre["pass"]:
        res.update(final=pre, decision="pre-registered setting passes")
        return res
    q_runs = [(f"{e:g} Ry k8", dft_engine("qe", work, ecut_Ry=e)) for e in QE_LADDER]
    g_runs = [(f"{e:g} eV k8", dft_engine("gpaw", work, ecut_eV=e)) for e in GPAW_LADDER]
    q_top = dft_engine("qe", work, ecut_Ry=QE_LADDER[-1], kgrid=K_TOP)
    g_top = dft_engine("gpaw", work, ecut_eV=GPAW_LADDER[-1], kgrid=K_TOP)
    q_runs.append((f"{QE_LADDER[-1]:g} Ry k12", q_top)); g_runs.append((f"{GPAW_LADDER[-1]:g} eV k12", g_top))
    conv = {"qe_rrkj": _ladder_steps(q_runs), "gpaw": _ladder_steps(g_runs)}
    converged = all(s["worst"] < CONV_FRACTION for steps in conv.values() for s in steps[-2:])
    top = _compare(q_top, g_top)
    res.update(convergence=conv, converged=bool(converged), top_setting=top,
               ladder={"qe_rrkj": {l: {k: r[k] for k in DFT_QUANTITIES} for l, r in q_runs},
                       "gpaw": {l: {k: r[k] for k in DFT_QUANTITIES} for l, r in g_runs}})
    if top["pass"]:
        res.update(final=top, decision="passes at the converged setting: the pre-registered failure was numerical")
        return res
    if fetch_dojo and not (_dojo_dir() / "Si.pbe.upf").is_file():
        fetch_dojo(_dojo_dir())
    q_dojo = dft_engine("qe", work, ecut_Ry=QE_LADDER[-1], kgrid=K_TOP, qe_pseudo="dojo")
    dojo = _compare(q_dojo, g_top)
    res["pseudodojo"] = dojo
    res["rrkj_vs_dojo_L_gap_eV"] = q_top["gap_L_eV"] - q_dojo["gap_L_eV"]
    if dojo["pass"]:
        res.update(final=dojo, decision="fails with Si.pbe-rrkj at the converged setting, passes with PseudoDojo: "
                                        "PseudoDojo is the QE side of the pair; the rrkj L-point error is recorded")
    else:
        res.update(final=dojo, decision="fails at the converged setting with both QE pseudopotentials: gate fails")
    return res


# ================================================================================================= FEM: skfem vs FEniCSx
def _fem_geometry():
    from simws.geometry import Geometry
    g = Geometry("fem2")
    g.rect("si", "Silicon", 0, 1.0, 0, 0.5).rect("ox", "Oxide", 0, 1.0, 0.5, 0.6)
    g.contact("bottom", 0, 1.0, 0, 0).contact("top", 0, 1.0, 0.6, 0.6)
    g.refine(0.5, 0.5, size=0.01, radius=0.05)
    return g


FEM_PROBLEM = {
    # -div(k grad u) = f per region (lengths in um); u = 0 on bottom, u = 1 on top
    "poisson": {"k": {"si": 11.7, "ox": 3.9}, "f": {"si": 50.0, "ox": 0.0}},
    # -div(k(T) grad T) = q in si, k(T) = k300 (300/T)^a; T = 300 on bottom, adiabatic elsewhere
    "heat": {"k300": 1.48, "a": 1.3, "q": {"si": 400.0, "ox": 0.0}, "k_ox": 0.014, "T0": 300.0},
}


def skfem_solve(msh, problem=FEM_PROBLEM) -> dict:
    """Reference problems with scikit-fem on the .msh (coordinates read in cm, solved in um)."""
    import skfem
    from skfem.helpers import dot, grad
    g = _fem_geometry()
    mesh, sub, bnd = g.to_skfem(msh)
    mesh = skfem.MeshTri(mesh.p / 1e-4, mesh.t)
    basis = skfem.Basis(mesh, skfem.ElementTriP1())
    b0 = skfem.Basis(mesh, skfem.ElementTriP0())
    reg = np.zeros(mesh.t.shape[1], int)
    reg[sub["ox"]] = 1

    def p0(vals):
        return b0.interpolate(np.where(reg == 0, vals[0], vals[1]))

    P = problem["poisson"]
    kq, fq = p0((P["k"]["si"], P["k"]["ox"])), p0((P["f"]["si"], P["f"]["ox"]))

    @skfem.BilinearForm
    def a(u, v, w):
        return w["k"] * dot(grad(u), grad(v))

    @skfem.LinearForm
    def L(v, w):
        return w["f"] * v

    K, F = skfem.asm(a, basis, k=kq), skfem.asm(L, basis, f=fq)
    bot = np.unique(mesh.facets[:, bnd["bottom"]].ravel())
    top = np.unique(mesh.facets[:, bnd["top"]].ravel())
    x = np.zeros(mesh.p.shape[1]); x[top] = 1.0
    u = skfem.solve(*skfem.condense(K, F, x=x, D=np.concatenate([bot, top])))
    H = problem["heat"]
    # one-point (centroid) quadrature for the nonlinear heat problem in both engines, so that k(T) is sampled at the
    # same points (the stiffness and the piecewise-constant load are exact with it)
    # (scikit-fem's intorder=1 still gives a 3-point rule, so the centroid rule is passed explicitly; DOLFINx's
    # quadrature_degree 1 is the centroid)
    centroid = (np.array([[1.0 / 3.0], [1.0 / 3.0]]), np.array([0.5]))
    basis = skfem.Basis(mesh, skfem.ElementTriP1(), quadrature=centroid)
    b0 = skfem.Basis(mesh, skfem.ElementTriP0(), quadrature=centroid)
    qq = p0((H["q"]["si"], H["q"]["ox"]))
    T = np.full(mesh.p.shape[1], H["T0"])
    Fq = skfem.asm(L, basis, f=qq)
    for _ in range(100):
        Ti = basis.interpolate(T).value
        kT = np.where(reg[:, None] == 0, H["k300"] * (300.0 / Ti) ** H["a"], H["k_ox"])
        Kt = skfem.asm(a, basis, k=kT)
        Tn = skfem.solve(*skfem.condense(Kt, Fq, x=np.full_like(T, H["T0"]), D=bot))
        if np.max(np.abs(Tn - T)) < 1e-12 * H["T0"]:
            T = Tn
            break
        T = Tn
    return {"x": mesh.p[0].tolist(), "y": mesh.p[1].tolist(), "u": u.tolist(), "T": T.tolist()}


def fem_pair(work, fenicsx_python=None) -> dict:
    """Mesh once, solve with scikit-fem here and with DOLFINx in the fenicsx environment (simws.fem.dolfinx_engine)."""
    work = Path(work).resolve(); work.mkdir(parents=True, exist_ok=True)
    g = _fem_geometry()
    msh = g.mesh(work / "fem2.msh", size=0.05)
    a = skfem_solve(msh)
    py = fenicsx_python or _env_python("fenicsx")
    if py is None:
        return {"status": "not_installed", "note": "FEniCSx runs in its own conda env (fenicsx); none found",
                "pass": None}
    out = work / "dolfinx.json"
    envv = dict(os.environ)
    repo = str(Path(__file__).resolve().parents[2])
    envv["PYTHONPATH"] = repo + (os.pathsep + envv["PYTHONPATH"] if envv.get("PYTHONPATH") else "")
    # DOLFINx compiles its forms at run time (FFCx -> cffi): use the env's own conda compiler, not a system gcc
    bindir = Path(py).parent
    envv["PATH"] = str(bindir) + os.pathsep + envv.get("PATH", "")
    cc = bindir / "x86_64-conda-linux-gnu-cc"
    if cc.exists():
        envv["CC"], envv["LDSHARED"] = str(cc), f"{cc} -shared"
    subprocess.run([py, "-m", "simws.fem.dolfinx_engine", str(msh), str(out)], check=True, env=envv, timeout=1800,
                   cwd=str(work))                        # FFCx/cffi build their modules in the working directory
    b = json.loads(out.read_text())
    # match nodes by coordinates (both meshes are the same file; node order may differ)
    ka = {(round(x, 9), round(y, 9)): i for i, (x, y) in enumerate(zip(a["x"], a["y"]))}
    idx = np.array([ka[(round(x, 9), round(y, 9))] for x, y in zip(b["x"], b["y"])])
    ua, ub = np.asarray(a["u"])[idx], np.asarray(b["u"])
    Ta, Tb = np.asarray(a["T"])[idx], np.asarray(b["T"])
    du = float(np.max(np.abs(ua - ub)) / np.max(np.abs(ua)))
    dT = abs(Ta.max() / Tb.max() - 1)
    res = {"engines": {"skfem": _version("skfem"), "dolfinx": b.get("version")}, "n_nodes": int(len(idx)),
           "poisson_nodal_rel": du, "heat_Tmax": {"skfem": float(Ta.max()), "dolfinx": float(Tb.max()),
                                                   "rel": dT, "model_form_rel": _mf(Ta.max(), Tb.max())},
           "tolerances": {"poisson": TOLERANCES["fem.poisson_nodal"], "heat": TOLERANCES["fem.heat_Tmax"]}}
    res["pass"] = bool(du <= TOLERANCES["fem.poisson_nodal"] and dT <= TOLERANCES["fem.heat_Tmax"])
    return res


# ================================================================================================= DD: DEVSIM vs Solcore
DD = dict(NA=1e17, ND=1e16, xj_um=1.0, L_um=4.0, mu_n=1000.0, mu_p=400.0, tau=1e-7, T=300.0)


def _devsim_diode(Vs, G=0.0, n=801):
    import simws.tcad  # noqa: F401
    from devsim import (add_1d_contact, add_1d_mesh_line, add_1d_region, create_1d_mesh, create_device,
                        finalize_mesh, get_contact_current, reset_devsim)
    from simws.materials import load
    from simws.tcad import cryo as C
    from simws.tcad.physics import Physics, setup_region
    from simws.tcad.solve import ramp_bias
    reset_devsim()
    L, xj = DD["L_um"] * 1e-4, DD["xj_um"] * 1e-4
    create_1d_mesh(mesh="m")
    add_1d_mesh_line(mesh="m", pos=0, ps=L / (n - 1), tag="a")
    add_1d_mesh_line(mesh="m", pos=L, ps=L / (n - 1), tag="b")
    add_1d_contact(mesh="m", name="anode", tag="a", material="metal")
    add_1d_contact(mesh="m", name="cathode", tag="b", material="metal")
    add_1d_region(mesh="m", material="Si", region="si", tag1="a", tag2="b")
    finalize_mesh(mesh="m"); create_device(mesh="m", device="d")
    mat = load("silicon")
    phys = Physics(T=DD["T"], bgn=False, srh=True, tau_n=DD["tau"], tau_p=DD["tau"], auger=False, mu_n=DD["mu_n"],
                   mu_p=DD["mu_p"])
    used = setup_region("d", "si", mat, f"ifelse(x < {xj!r}, {-DD['NA']!r}, {DD['ND']!r})",
                        f"{DD['NA'] + DD['ND']!r}", phys, {"anode": False, "cathode": False},
                        extra_generation=(f"{G!r}" if G else None))
    Nc, Nv, _ = C.band_params(mat, DD["T"], used["n_i_cm3"])
    used.update({"Nc_cm3": float(Nc), "Nv_cm3": float(Nv), "eps_r": mat.get("eps_r", "dimensionless")})
    opts = dict(absolute_error=1e10, relative_error=1e-12, maximum_iterations=80)
    J = []
    for V in Vs:
        ramp_bias("d", "anode", V, step=0.05, **opts)
        J.append(get_contact_current(device="d", contact="anode", equation="ElectronContinuityEquation")
                 + get_contact_current(device="d", contact="anode", equation="HoleContinuityEquation"))
    return np.array(J), used


def _solcore_diode(Vs, used, G=0.0, n=4001):
    """Solcore's drift-diffusion path: Solcore materials -> process_structure -> Sesame IVcurve (what
    solar_cell_solver(..., 'iv') runs for kind='sesame_PDD'), with a uniform generation set directly so that no
    optical model enters the comparison."""
    from solcore import material
    from solcore.constants import kb, q
    from solcore.sesame_drift_diffusion.process_structure import process_structure
    from solcore.state import State
    from solcore.structure import Junction, Layer
    from solsesame.solvers import Solver
    T = DD["T"]
    Nc, Nv, ni = used["Nc_cm3"], used["Nv_cm3"], used["n_i_cm3"]
    Eg = -2 * kb * T / q * math.log(ni / math.sqrt(Nc * Nv))          # the gap that reproduces DEVSIM's n_i
    common = dict(T=T, Nc=Nc * 1e6, Nv=Nv * 1e6, band_gap=Eg * q, relative_permittivity=used["eps_r"],
                  electron_mobility=DD["mu_n"] * 1e-4, hole_mobility=DD["mu_p"] * 1e-4,
                  electron_minority_lifetime=DD["tau"], hole_minority_lifetime=DD["tau"], radiative_recombination=0,
                  electron_auger_recombination=0, hole_auger_recombination=0, bulk_recombination_energy=0)
    p = material("Si")(Na=DD["NA"] * 1e6, **common)
    nm = material("Si")(Nd=DD["ND"] * 1e6, **common)
    j = Junction([Layer(DD["xj_um"] * 1e-6, p, role="emitter"),
                  Layer((DD["L_um"] - DD["xj_um"]) * 1e-6, nm, role="base")],
                 kind="sesame_PDD", sn_e=1e8, sn_h=1e8, sp_e=1e8, sp_h=1e8)     # ohmic: equilibrium densities
    j.mesh = np.linspace(0, DD["L_um"] * 1e-6, n)
    opts = State(T=T, sesame_periodic=False)
    process_structure(j, opts)
    s = j.sesame_sys
    if G:
        s.generation(np.full(s.nx, G))
    V = np.concatenate([[0.0], np.asarray(Vs, float)]) if Vs[0] != 0 else np.asarray(Vs, float)
    jj, _ = Solver().IVcurve(s, V, tol=1e-10, maxiter=300, verbose=False)
    jj = np.asarray(jj) * s.scaling.current
    return jj[-len(Vs):]


def dd_pair(work=None) -> dict:
    Vs = [0.3, 0.4, 0.5, 0.6]
    Jd, used = _devsim_diode(Vs)
    Js = _solcore_diode(Vs, used)
    rel = np.abs(np.abs(Js) / np.abs(Jd) - 1)
    G = 1e20
    Jg, _ = _devsim_diode([0.0], G=G)
    Jsg = _solcore_diode([0.0], used, G=G)
    jsc_rel = abs(abs(Jsg[0]) / abs(Jg[0]) - 1)
    out = {"V": Vs, "J_devsim": Jd.tolist(), "J_solcore": np.asarray(Js).tolist(), "dark_rel": rel.tolist(),
           "Jsc": {"devsim": float(Jg[0]), "solcore": float(Jsg[0]), "rel": jsc_rel,
                   "ideal_qGL": 1.602176634e-19 * G * DD["L_um"] * 1e-4},
           "model_form_rel_dark": float(max(_mf(abs(a), abs(b)) for a, b in zip(Jd, Js))),
           "tolerances": {"dark": TOLERANCES["dd.dark_J"], "Jsc": TOLERANCES["dd.Jsc"]}}
    out["pass"] = bool(rel.max() <= TOLERANCES["dd.dark_J"] and jsc_rel <= TOLERANCES["dd.Jsc"])
    return out


# ================================================================================================= helpers
def _env_python(env):
    try:
        from simws.config import env_python
        p = env_python(env)
        return str(p) if p.exists() else None
    except Exception:  # noqa: BLE001
        try:
            import dolfinx  # noqa: F401
            return sys.executable
        except ImportError:
            return None


def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception:  # noqa: BLE001
        return None


def registry_entries(res: dict) -> dict:
    """Model-form components (relative standard uncertainties) for simws.hub.model_form from a run_all result."""
    e = {}
    if res.get("dft", {}).get("rows"):
        r = res["dft"]["rows"]
        e["dft.a0"] = r["a0_A"]["model_form_rel"]
        e["dft.B0"] = r["B0_GPa"]["model_form_rel"]
        e["dft.gap"] = r["indirect_gap_eV"]["model_form_rel"]
    if res.get("dd", {}).get("model_form_rel_dark") is not None:
        e["devsim.current"] = res["dd"]["model_form_rel_dark"]
    if isinstance(res.get("fem", {}).get("heat_Tmax"), dict):
        e["thermal.T"] = res["fem"]["heat_Tmax"]["model_form_rel"]
    return e


def run_all(work=None, parts=("dft", "fem", "dd")) -> dict:
    work = Path(work or tempfile.mkdtemp(prefix="simws_pairs_"))
    work.mkdir(parents=True, exist_ok=True)
    res = {"tolerances_fixed_on": FIXED_ON, "tolerances": TOLERANCES}
    if "dft" in parts:
        res["dft"] = dft_pair(work / "dft")
    if "fem" in parts:
        res["fem"] = fem_pair(work / "fem")
    if "dd" in parts:
        res["dd"] = dd_pair(work / "dd")
    res["model_form"] = registry_entries(res)
    res["pass"] = all(v["pass"] for k, v in res.items() if isinstance(v, dict) and v.get("pass") is not None
                      and k in ("dft", "fem", "dd"))
    return res
