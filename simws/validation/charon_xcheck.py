"""0.9a gate: DEVSIM against Sandia Charon on a cold pn diode (freeze-out + Fermi-Dirac + SG drift-diffusion).

Both codes get IDENTICAL inputs, so the comparison tests the implementations, not parameter choices:
  * our silicon band parameters at T (simws.tcad.cryo.band_params) written into Charon's Effective DOS and Band Gap
    blocks (Nc300 = Nc(T) (300/T)^1.5 with exponent 1.5; gap Eg(T) through Charon's TempDep form with alpha = 0, i.e.
    constant; a Band Gap block without "temperature dependence is on" has no Value and Charon aborts); BGN off;
    eps_r 11.7
  * Charon's incomplete-ionization form in both (constant ionization energy, degeneracy 2 donors / 4 acceptors,
    full ionization above a critical doping): DEVSIM ii_model="charon"; Charon contact approximation III (exact)
  * constant mobilities and SRH lifetimes; Fermi-Dirac statistics on; Charon "drift diffusion cvfem" (CVFEM-SG, the
    same Scharfetter-Gummel flux as DEVSIM's finite volumes) on a 1 x 0.5 um quad mesh with 1.25 nm along x
    (simws.validation.charon_mesh); DEVSIM on a 1D line with the same x nodes

Gate (Build Plan 0.9a): terminal current density within 2 % wherever |J| > 1e-6 A/cm^2, and the neutral-region
majority densities (the freeze-out) within 2 %.

Charon practicalities (found on the laptop, Charon 2.2): "Import State File <bare mesh> at index 1" aborts silently;
the nonlinear-Poisson stage aborts with these material blocks; drift-diffusion with full Newton steps diverges at
77 K. So every case starts drift-diffusion at 0 V (backtracking line search), and the cold cases are reached by
temperature continuation at 0 V from the 300 K solution (CHAIN).

Charon is not in the simws environments; it runs on the laptop (tools/run_charon_xcheck.sh):
    python -m simws.validation.charon_xcheck write   DIR          # decks + runlist_all.txt
    (charonInterpreter + charon_mp.exe on each deck, in order; a deck whose start file is missing is skipped)
    python -m simws.validation.charon_xcheck devsim  DIR
    python -m simws.validation.charon_xcheck compare DIR          # needs exodus3 (tcad env)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

L_UM, XJ_UM, W_UM = 1.0, 0.5, 0.5
# Gate mesh: 1.25 nm along x in both codes (simws.validation.charon_mesh). Charon's own 10 nm pndiode.exo does not
# resolve the SRH peak (width ~ kT/qE): there the two codes' recombination currents differ by up to +-14 % about the
# converged value with opposite signs (node lumping vs quadrature), which is discretisation, not physics.
DATA = Path(__file__).with_name("data")
MESHES = {"fine": ("pndiode_fine.exo", 0.00125), "10nm": ("pndiode_10nm.exo", 0.01), "sk": ("sk_2p5nm.exo", 0.0025)}
DX_UM = MESHES["fine"][1]
GATE = 0.02
J_FLOOR = 1e-6                                        # A/cm^2


def case(T: float, N: float = 1e17) -> dict:
    from simws.materials import load
    from simws.materials.semiconductor import intrinsic_density
    from simws.tcad import cryo as C
    mat = load("silicon")
    n_i = intrinsic_density(mat, T)
    Nc, Nv, Eg = C.band_params(mat, T, n_i)
    return dict(T=float(T), NA=N, ND=N, mu_n=1000.0, mu_p=400.0, tau=1e-7, E_ion=0.045, g_D=2.0, g_A=4.0,
                N_crit=1e18, eps_r=mat.get("eps_r", "dimensionless"), Nc=Nc, Nv=Nv, Eg=Eg, n_i=n_i)


def biases(T: float):
    """0.1 V steps to just below turn-on, then 0.05 V steps through the gated range (|J| > 1e-6 A/cm^2):
    at 77 K that is 0.95-1.05 V (built-in voltage 1.09 V), at 300 K 0.3-0.7 V."""
    if T < 150:
        return [round(float(v), 3) for v in list(np.arange(0.0, 0.85, 0.1)) + [0.9, 0.95, 1.0, 1.05]]
    if T < 250:
        return [round(float(v), 3) for v in list(np.arange(0.0, 0.65, 0.1)) + [0.7, 0.75, 0.8, 0.85]]
    return [round(float(v), 3) for v in np.arange(0.0, 0.7 + 1e-9, 0.1)]


# ------------------------------------------------------------------------------------------------ Charon decks
def _material(c: dict) -> str:
    s = (300.0 / c["T"]) ** 1.5
    return f"""start Material Block siliconParameter
      material name is Silicon
      relative permittivity = {c['eps_r']!r}

      start Effective DOS
            Nc300 = {c['Nc'] * s!r}
            Nv300 = {c['Nv'] * s!r}
            Nc_F = 1.5
            Nv_F = 1.5
      end Effective DOS

      start Band Gap
            temperature dependence is on
            300K gap = {c['Eg']!r}
            alpha = 0.0
            beta = 636.0
            Chi300 = 4.05
      end Band Gap

      start Incomplete Ionization Donor
            critical doping = {c['N_crit']!r}
            degeneracy factor = {c['g_D']!r}
            ionization energy = {c['E_ion']!r}
            approximation is III
      end Incomplete Ionization Donor
      start Incomplete Ionization Acceptor
            critical doping = {c['N_crit']!r}
            degeneracy factor = {c['g_A']!r}
            ionization energy = {c['E_ion']!r}
            approximation is III
      end Incomplete Ionization Acceptor

      start Carrier Lifetime Block
            electron lifetime is constant = {c['tau']!r}
            hole lifetime is constant = {c['tau']!r}
      end Carrier Lifetime Block

      start step junction doping
            acceptor concentration = {c['NA']!r}
            donor concentration = {c['ND']!r}
            junction location = {XJ_UM!r}
            dopant order is PN
            direction is x
      end step junction doping

      fixed mobility for electrons is {c['mu_n']!r}
      fixed mobility for holes is {c['mu_p']!r}
end Material Block siliconParameter
"""


def _deck(c, imp, out, V, ic, srh=True):
    srh_line = "      srh recombination is on\n" if srh else "      srh recombination is off\n"
    # a bare mesh has no stored state: "at index 1" on it makes Charon abort silently (MPI_ABORT 14)
    index = "" if imp.startswith("pndiode") else " at index 1"
    return f"""Import State File {imp}{index}

start output parameters
      Output State File {out}
      Output cell average variables in silicon for scalar "Doping"
      unscale variables is on
end output parameters

tpetra is on
lattice temperature = {c['T']!r}
concentration scaling = {max(c['NA'], c['ND'])!r}

start Physics Block Semiconductor
      geometry block is silicon
      standard discretization type is drift diffusion cvfem
      material model is siliconParameter
      use Fermi Dirac
      donor incomplete ionization is on
      acceptor incomplete ionization is on
{srh_line}      radiative recombination is off
      auger recombination is off
end Physics Block Semiconductor

{_material(c)}
BC is ohmic for anode on silicon fixed at {V!r}
BC is ohmic for cathode on silicon fixed at 0

{ic}

start solver block
  start tpetra block
    problem type is steady state
    verbosity level is high
    maximum number nonlinear iterations = 60
  end
  line search method is backtrack
end solver block
"""


IC_MESH = ("initial conditions for ELECTRIC_POTENTIAL in silicon is Equilibrium Potential\n"
           "Initial Conditions for ELECTRON_DENSITY in silicon is Equilibrium Density\n"
           "Initial Conditions for HOLE_DENSITY in silicon is Equilibrium Density")
IC_FILE = ("initial conditions for ELECTRIC_POTENTIAL in silicon is Exodus File\n"
           "Initial Conditions for ELECTRON_DENSITY in silicon is Exodus File\n"
           "Initial Conditions for HOLE_DENSITY in silicon is Exodus File")

# Comparison cases. Charon's nonlinear-Poisson stage aborts with these material blocks, and its drift-diffusion
# Newton diverges at 77 K from the built-in equilibrium guess, so cold cases are reached by temperature
# continuation at 0 V (300 -> 200 -> 150 -> 110 -> 77 K), each step starting from the previous solution.
# T300nosrh (no SRH in either code) separates the recombination model from transport and statistics.
# T300c10 (diagnostic) repeats T300 on our own writer's 10 nm mesh: it must reproduce the Charon-on-pndiode.exo
# numbers (checks the mesh writer) and the -6 % SRH discrepancy (checks the explanation).
# On the fine mesh Charon's 0 V Newton fails on the 110 -> 77 K step, so the cold end is approached in smaller steps
# (110 -> 95 -> 85 -> 77 K). T77c10 is the same cold chain on the 10 nm mesh (converged there in fix7): at 77 K the
# gated bias points are diffusion-dominated, where the mesh matters little (DEVSIM 1.25 vs 10 nm: 0.06 % at 1.00 V,
# 0.55 % at 1.05 V), so it stands in for the gate's 77 K case only if the fine chain still fails, and says so.
CASES = {"T300": dict(T=300.0, srh=True), "T300nosrh": dict(T=300.0, srh=False),
         "T300c10": dict(T=300.0, srh=True, mesh="10nm"),
         "T150": dict(T=150.0, srh=True), "T77": dict(T=77.0, srh=True),
         "T77c10": dict(T=77.0, srh=True, mesh="10nm")}
CHAIN = [("T300", "mesh"), ("T300nosrh", "mesh"), ("T300c10", "mesh"),
         ("cont_T200", "xc_T300_dd_0000mV"), ("T150", "xc_cont_T200"), ("cont_T110", "xc_T150_dd_0000mV"),
         ("cont_T95", "xc_cont_T110"), ("cont_T85", "xc_cont_T95"), ("T77", "xc_cont_T85"),
         ("c10_T200", "xc_T300c10_dd_0000mV"), ("c10_T150", "xc_c10_T200"), ("c10_T110", "xc_c10_T150"),
         ("T77c10", "xc_c10_T110")]


DIAGNOSTIC = {"T300nosrh", "T300c10", "T77c10"}


def write_decks(outdir) -> list:
    """Write every deck, the meshes and runlist_all.txt (run order); each deck's first line names its start file."""
    import shutil
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    for f, _ in MESHES.values():
        shutil.copy(DATA / f, out / f)
    order = []
    for label, src in CHAIN:
        mesh = MESHES[CASES.get(label, {}).get("mesh", "fine")][0]
        imp = mesh if src == "mesh" else f"{src}.exo"
        ic0 = IC_MESH if src == "mesh" else IC_FILE
        if label.startswith(("cont_", "c10_")):
            T = float(label.split("_T")[1])
            name = f"xc_{label}"
            (out / f"{name}.inp").write_text(_deck(case(T), imp, f"{name}.exo", 0.0, ic0))
            order.append(name)
            continue
        cfg = CASES[label]
        c = case(cfg["T"])
        names, prev = [], imp
        for k, V in enumerate(biases(cfg["T"])):
            name = f"xc_{label}_dd_{int(round(V * 1000)):04d}mV"
            (out / f"{name}.inp").write_text(_deck(c, prev, f"{name}.exo", V, ic0 if k == 0 else IC_FILE, cfg["srh"]))
            names.append(name)
            prev = f"{name}.exo"
        (out / f"runlist_{label}.txt").write_text("\n".join(names) + "\n")
        (out / f"case_{label}.json").write_text(json.dumps({**c, **cfg, "biases": biases(cfg["T"])}, indent=1))
        order += names
    order += write_sk_decks(out)
    (out / "runlist_all.txt").write_text("\n".join(order) + "\n")
    return order


# ------------------------------------------------------------------------------------------------ Schottky diode (0.9b)
# n-Si 1e16 cm^-3, 1 um, Schottky anode (work function 4.75 eV, affinity 4.05 eV: phi_Bn = 0.70 eV) at x = 0, ohmic
# cathode; thermionic emission with A_n* = 112, A_p* = 32 A cm^-2 K^-2 behind drift-diffusion (combined TE-diffusion),
# Boltzmann statistics, full ionization, fixed mobilities, SRH. Charon's Schottky contact (Dirichlet potential, Neumann
# carrier fluxes q v (n - n0B)) has no image-force lowering or tunnelling unless asked, like ours by default.
SK = dict(N=1e16, wf=4.75, chi=4.05, A_n=112.0, A_p=32.0, mu_n=1000.0, mu_p=400.0, tau=1e-7)
SK_CASES = {"SK300": 300.0, "SK200": 200.0}


def sk_biases(T):
    return [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3] if T >= 250 else [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45]


def _sk_deck(T, imp, out, V, ic):
    c = case(T)
    s = (300.0 / T) ** 1.5
    index = "" if imp.startswith("sk_") else " at index 1"
    return f"""Import State File {imp}{index}

start output parameters
      Output State File {out}
      unscale variables is on
end output parameters

tpetra is on
lattice temperature = {T!r}
concentration scaling = {SK['N']!r}

start Physics Block Semiconductor
      geometry block is silicon
      standard discretization type is drift diffusion cvfem
      material model is siliconParameter
      srh recombination is on
      radiative recombination is off
      auger recombination is off
end Physics Block Semiconductor

start Material Block siliconParameter
      material name is Silicon
      relative permittivity = {c['eps_r']!r}
      start Effective DOS
            Nc300 = {c['Nc'] * s!r}
            Nv300 = {c['Nv'] * s!r}
            Nc_F = 1.5
            Nv_F = 1.5
      end Effective DOS
      start Band Gap
            temperature dependence is on
            300K gap = {c['Eg']!r}
            alpha = 0.0
            beta = 636.0
            Chi300 = {SK['chi']!r}
      end Band Gap
      start Carrier Lifetime Block
            electron lifetime is constant = {SK['tau']!r}
            hole lifetime is constant = {SK['tau']!r}
      end Carrier Lifetime Block
      start uniform doping named Function1
            type is donor
            concentration = {SK['N']!r}
      end uniform doping named Function1
      fixed mobility for electrons is {SK['mu_n']!r}
      fixed mobility for holes is {SK['mu_p']!r}
end Material Block siliconParameter

start Schottky BC for anode on silicon
      work function = {SK['wf']!r}
      voltage is fixed at {V!r}
      electron Richardson Constant = {SK['A_n']!r}
      hole Richardson Constant = {SK['A_p']!r}
      type is electron
end Schottky BC for anode on silicon

BC is ohmic for cathode on silicon fixed at 0

{ic}

start solver block
  start tpetra block
    problem type is steady state
    verbosity level is high
    maximum number nonlinear iterations = 60
  end
  line search method is backtrack
end solver block
"""


def write_sk_decks(outdir) -> list:
    out = Path(outdir)
    order = []
    for label, T in SK_CASES.items():
        names, prev = [], MESHES["sk"][0]
        for k, V in enumerate(sk_biases(T)):
            name = f"xc_{label}_dd_{int(round(V * 1000)):04d}mV"
            (out / f"{name}.inp").write_text(_sk_deck(T, prev, f"{name}.exo", V, IC_MESH if k == 0 else IC_FILE))
            names.append(name)
            prev = f"{name}.exo"
        (out / f"runlist_{label}.txt").write_text("\n".join(names) + "\n")
        order += names
    return order


def devsim_schottky(T: float, dx_um: float = None, Vs=None) -> dict:
    import simws.tcad  # noqa: F401
    from devsim import (add_1d_contact, add_1d_mesh_line, add_1d_region, create_1d_mesh, create_device,
                        finalize_mesh, get_contact_current, reset_devsim, solve)
    from simws.materials import load
    from simws.tcad.contacts import schottky_contact
    from simws.tcad.physics import Physics, setup_region
    from simws.tcad.solve import ramp_bias
    dx = (dx_um or MESHES["sk"][1]) * 1e-4
    reset_devsim()
    create_1d_mesh(mesh="m")
    add_1d_mesh_line(mesh="m", pos=0.0, ps=dx, tag="anode")
    add_1d_mesh_line(mesh="m", pos=L_UM * 1e-4, ps=dx, tag="cathode")
    for k in ("anode", "cathode"):
        add_1d_contact(mesh="m", name=k, tag=k, material="metal")
    add_1d_region(mesh="m", material="Si", region="r", tag1="anode", tag2="cathode")
    finalize_mesh(mesh="m"); create_device(mesh="m", device="d")
    mat = load("silicon")
    phys = Physics(T=T, bgn=False, auger=False, srh=True, tau_n=SK["tau"], tau_p=SK["tau"], mu_n=SK["mu_n"],
                   mu_p=SK["mu_p"])
    setup_region("d", "r", mat, f"{SK['N']!r}", f"{SK['N']!r}", phys, {"anode": False, "cathode": False})
    used = schottky_contact("d", "r", "anode", mat, phys, phi_Bn=SK["wf"] - SK["chi"], A_n=SK["A_n"], A_p=SK["A_p"])
    opts = dict(absolute_error=1e10, relative_error=1e-10, maximum_iterations=60)
    solve(type="dc", **opts)
    J, Vs = [], list(Vs or sk_biases(T))
    for V in Vs:
        ramp_bias("d", "anode", V, step=0.05, **opts)
        J.append(abs(sum(get_contact_current(device="d", contact="anode", equation=e)
                         for e in ("ElectronContinuityEquation", "HoleContinuityEquation"))))
    return {"T": T, "V": Vs, "J_A_cm2": J, "used": used, "dx_um": dx / 1e-4}


# ------------------------------------------------------------------------------------------------ DEVSIM side
def devsim_diode(T: float, srh: bool = True, dx_um: float = DX_UM, Vs=None) -> dict:
    import simws.tcad  # noqa: F401
    from devsim import (add_1d_contact, add_1d_mesh_line, add_1d_region, create_1d_mesh, create_device,
                        finalize_mesh, get_node_model_values, reset_devsim)
    from simws.materials import load
    from simws.tcad.physics import Physics, contact_current, setup_region
    from simws.tcad.solve import ramp_bias
    c = case(T)
    reset_devsim()
    L, xj = L_UM * 1e-4, XJ_UM * 1e-4
    create_1d_mesh(mesh="m")
    add_1d_mesh_line(mesh="m", pos=0.0, ps=dx_um * 1e-4, tag="anode")
    add_1d_mesh_line(mesh="m", pos=L, ps=dx_um * 1e-4, tag="cathode")
    for k in ("anode", "cathode"):
        add_1d_contact(mesh="m", name=k, tag=k, material="metal")
    add_1d_region(mesh="m", material="Si", region="r", tag1="anode", tag2="cathode")
    finalize_mesh(mesh="m")
    create_device(mesh="m", device="d")
    phys = Physics(T=T, bgn=False, auger=False, srh=srh, tau_n=c["tau"], tau_p=c["tau"], mu_n=c["mu_n"],
                   mu_p=c["mu_p"], incomplete_ionization=True, fermi_dirac=True, ii_model="charon",
                   ii_N_crit=c["N_crit"], ii_E_const=c["E_ion"])
    # pn step: acceptors for x < x_j (anode side), donors from x_j on
    net = f"ifelse(x < {xj - 1e-12!r}, {-c['NA']!r}, {c['ND']!r})"
    used = setup_region("d", "r", load("silicon"), net, f"ifelse(x < {xj - 1e-12!r}, {c['NA']!r}, {c['ND']!r})",
                        phys, {"anode": False, "cathode": False})
    g = lambda n: np.array(get_node_model_values(device="d", region="r", name=n))
    eq = {"x_um": (g("x") * 1e4).tolist(), "psi": g("Potential").tolist(), "n": g("Electrons").tolist(),
          "p": g("Holes").tolist()}
    opts = dict(absolute_error=1e10, relative_error=used.get("equilibrium_rel_tol", 1e-8), maximum_iterations=60)
    J, Vs = [], list(Vs or biases(T))
    for V in Vs:
        ramp_bias("d", "anode", V, step=0.05, **opts)
        J.append(abs(float(contact_current("d", "anode"))))
    return {"case": c, "equilibrium": eq, "V": Vs, "J_A_cm2": J, "dx_um": dx_um, "used": {k: v for k, v in used.items()
                                                                                 if isinstance(v, (int, float))}}


# ------------------------------------------------------------------------------------------------ Charon results
def read_charon(outdir, label: str) -> dict:
    try:
        import exodus3 as exodus
    except ImportError:
        import exodus
    out = Path(outdir)
    names = (out / f"runlist_{label}.txt").read_text().split()
    res = {"V": [], "J_A_cm2": [], "global_names": None}

    def nodal(e, var):
        k = e.num_times()
        return np.asarray(e.get_node_variable_values(var, k))

    for name in names:
        if not (out / f"{name}.exo").exists():
            continue
        try:
            e = exodus.exodus(str(out / f"{name}.exo"), mode="r", array_type="numpy")
            if e.num_times() < 1:
                raise ValueError("no time step stored")
        except Exception as exc:  # noqa: BLE001  a run that aborted leaves a partial file
            res.setdefault("unreadable", []).append(f"{name}: {exc}")
            continue
        x, y, _ = (np.asarray(v) for v in e.get_coords())
        nv = e.get_node_variable_names()
        if name.endswith("_0000mV"):
            key = "equilibrium"
            ux = np.unique(np.round(x, 9))
            fields = {}
            for v in ("ELECTRIC_POTENTIAL", "ELECTRON_DENSITY", "HOLE_DENSITY"):
                if v in nv:
                    a = nodal(e, v)
                    fields[v] = [float(a[np.isclose(x, xx, atol=1e-9)].mean()) for xx in ux]
            res[key] = {"x_um": ux.tolist(), **fields}
        if "_dd_" in name:
            gn = e.get_global_variable_names()
            res["global_names"] = list(gn)
            cur = [g for g in gn if "anode" in g.lower() and "current" in g.lower()] or \
                  [g for g in gn if "anode" in g.lower()]
            I = float(np.asarray(e.get_global_variable_values(cur[0]))[-1]) if cur else float("nan")
            W_cm = (float(y.max() - y.min()) or W_UM) * 1e-4          # mesh in um; 2D current per cm of depth
            res["V"].append(float(name.rsplit("_", 1)[1][:-2]) / 1000)
            res["J_A_cm2"].append(abs(I) / W_cm)
            res["current_variable"] = cur[0] if cur else None
        e.close()
    return res


def compare(outdir, label: str, dev: dict | None = None) -> dict:
    ch = read_charon(outdir, label)
    cfg = CASES[label]
    T = cfg["T"]
    dev = dev or devsim_diode(T, cfg["srh"], MESHES[cfg.get("mesh", "fine")][1])
    rows, worst = [], 0.0
    for V, Jc in zip(ch["V"], ch["J_A_cm2"]):
        Jd = dev["J_A_cm2"][dev["V"].index(V)] if V in dev["V"] else float("nan")
        gated = max(Jc, Jd) > J_FLOOR
        rel = Jd / Jc - 1 if Jc > 0 else float("nan")
        if gated:
            worst = max(worst, abs(rel))
        rows.append({"V": V, "J_charon": Jc, "J_devsim": Jd, "rel_diff": rel, "gated": gated})
    # freeze-out: majority densities in the neutral regions (0.1 um from each contact)
    eqc, eqd = ch.get("equilibrium"), dev["equilibrium"]
    dens = {}
    if eqc and "ELECTRON_DENSITY" in eqc:
        xc, xd = np.asarray(eqc["x_um"]), np.asarray(eqd["x_um"])
        for lab, xx, key, dk in (("p_anode_side", 0.1, "HOLE_DENSITY", "p"), ("n_cathode_side", 0.9, "ELECTRON_DENSITY", "n")):
            vc = float(np.interp(xx, xc, eqc[key])); vd = float(np.interp(xx, xd, eqd[dk]))
            dens[lab] = {"charon": vc, "devsim": vd, "rel_diff": vd / vc - 1, "ionized_fraction_devsim": vd / 1e17}
            worst = max(worst, abs(vd / vc - 1))
        pc = np.asarray(eqc["ELECTRIC_POTENTIAL"]); pd = np.asarray(eqd["psi"])
        dens["built_in_V"] = {"charon": float(pc[-1] - pc[0]), "devsim": float(pd[-1] - pd[0])}
    rep = {"case": label, "T_K": T, "srh": cfg["srh"], "currents": rows, "densities": dens, "worst_rel_diff": worst, "gate": GATE,
           "charon_unreadable": ch.get("unreadable", []),
           "pass": bool(rows) and worst < GATE, "charon_current_variable": ch.get("current_variable"),
           "charon_global_variables": ch.get("global_names"),
           "note": "J from Charon's anode global variable divided by the mesh width (2D current per cm of depth)"}
    if rows and not rep["pass"]:
        r = [x["rel_diff"] for x in rows if x["gated"]]
        if len(r) > 2 and np.std(np.log1p(r)) < 0.01:
            rep["hint"] = f"constant ratio {1 + np.mean(r):.4g}: likely a current-unit/width convention, not physics"
    return rep


def compare_sk(outdir, label: str, dev: dict | None = None) -> dict:
    ch = read_charon(outdir, label)
    T = SK_CASES[label]
    dev = dev or devsim_schottky(T)
    rows, worst = [], 0.0
    for V, Jc in zip(ch["V"], ch["J_A_cm2"]):
        Jd = dev["J_A_cm2"][dev["V"].index(V)] if V in dev["V"] else float("nan")
        gated = max(Jc, Jd) > J_FLOOR
        rel = Jd / Jc - 1 if Jc > 0 else float("nan")
        if gated:
            worst = max(worst, abs(rel))
        rows.append({"V": V, "J_charon": Jc, "J_devsim": Jd, "rel_diff": rel, "gated": gated})
    return {"case": label, "T_K": T, "currents": rows, "densities": {}, "worst_rel_diff": worst, "gate": GATE,
            "pass": bool(rows) and worst < GATE, "charon_unreadable": ch.get("unreadable", [])}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m simws.validation.charon_xcheck")
    ap.add_argument("action", choices=["write", "compare", "devsim"])
    ap.add_argument("dir")
    ap.add_argument("--cases", nargs="+", default=list(CASES) + list(SK_CASES))
    a = ap.parse_args(argv)
    if a.action == "write":
        print(" ".join(write_decks(a.dir)))
        return 0
    gate, points, warns = {}, {}, []
    for label in a.cases:
        sk = label in SK_CASES
        cfg = {"T": SK_CASES[label], "srh": True} if sk else CASES[label]
        f = Path(a.dir, f"devsim_{label}.json")
        if a.action == "devsim":
            r = devsim_schottky(cfg["T"]) if sk else devsim_diode(cfg["T"], cfg["srh"], MESHES[cfg.get("mesh", "fine")][1])
            f.write_text(json.dumps(r, indent=1))
            print(json.dumps({"case": label, "V": r["V"], "J": r["J_A_cm2"]}))
            continue
        if not Path(a.dir, f"runlist_{label}.txt").exists():
            continue
        dev = json.loads(f.read_text()) if f.exists() else None
        r = compare_sk(a.dir, label, dev) if sk else compare(a.dir, label, dev)
        # the slow test asserts every comparison_T*.json; the SRH-off case is a diagnostic, not part of the gate
        # a case Charon could not run at all is recorded (gate.json) but is not a comparison
        kind = "diagnostic" if label in DIAGNOSTIC or not r["currents"] else "comparison"
        Path(a.dir, f"{kind}_{label}.json").write_text(json.dumps(r, indent=1))
        n_ok = sum(1 for x in r["currents"])
        points[label] = n_ok
        diag = label in DIAGNOSTIC
        verdict = (f"diagnostic, not gated ({'within' if r['pass'] else 'outside'} 2 %)" if diag
                   else ("PASS" if r["pass"] else "FAIL"))
        print(f"\n{label}{' Schottky diode' if sk else ''} (T = {cfg['T']:g} K, SRH {'on' if cfg['srh'] else 'off'}): {n_ok} Charon bias points; "
              f"worst gated difference {100 * r['worst_rel_diff']:.2f} % -> {verdict}")
        # noise-floor margin: the gated points must sit far above Charon's own current noise (its "current" at the
        # lowest biases, where both codes' true currents are far smaller) and above the gate floor
        gated = [x for x in r["currents"] if x["gated"]]
        if gated and not diag:
            noise = max([abs(x["J_charon"]) for x in r["currents"] if x["V"] <= 0.1] or [0.0])
            jmin = min(min(x["J_charon"], x["J_devsim"]) for x in gated)
            margin = jmin / noise if noise > 0 else float("inf")
            r["noise_floor"] = {"charon_low_bias_A_cm2": noise, "smallest_gated_A_cm2": jmin, "margin": margin}
            if margin < 100 or jmin < 10 * J_FLOOR:
                warns.append(f"WARN {label}: smallest gated current {jmin:.3g} A/cm^2 is only {margin:.3g}x Charon's "
                             f"low-bias noise ({noise:.3g}) / {jmin / J_FLOOR:.3g}x the gate floor")
            Path(a.dir, f"{kind}_{label}.json").write_text(json.dumps(r, indent=1))
        for row in r["currents"]:
            print(f"  V={row['V']:.2f}  Charon {row['J_charon']:.4e}  DEVSIM {row['J_devsim']:.4e}  "
                  f"{100 * row['rel_diff']:+.2f}%{'' if row['gated'] else '  (below floor)'}")
        for k, v in r["densities"].items():
            print(f"  {k}: " + ", ".join(f"{kk} {vv:.6g}" for kk, vv in v.items()))
        if r.get("charon_unreadable"):
            print("  unreadable Charon files:", r["charon_unreadable"])
        gate[label] = r["pass"]
    if a.action == "devsim":
        return 0
    # the 10 nm cold case stands in ONLY when Charon could not run the fine-mesh 77 K case at all; a fine-mesh case
    # that ran and missed 2 % fails the gate
    if points.get("T77", 0) > 0:
        cold = gate.get("T77")
    else:
        cold = gate.get("T77c10") and "T77c10 (10 nm; fine-mesh 77 K run did not converge in Charon)"
    gate["gate_pass"] = bool(gate.get("T300") and gate.get("T150") and cold)
    gate["schottky_pass"] = bool(gate.get("SK300") and gate.get("SK200"))   # 0.9b contact cross-check
    gate["cold_case"] = "T77" if points.get("T77", 0) > 0 else cold or None
    gate["warnings"] = warns
    Path(a.dir, "gate.json").write_text(json.dumps(gate, indent=1))
    for w in warns:
        print(w)
    print(f"\nGATE: {'PASS' if gate['gate_pass'] else 'FAIL'} (T300 {gate.get('T300')}, T150 {gate.get('T150')}, "
          f"cold case {gate['cold_case']}); Schottky (0.9b): {'PASS' if gate['schottky_pass'] else 'FAIL'}")
    return 0 if gate["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
