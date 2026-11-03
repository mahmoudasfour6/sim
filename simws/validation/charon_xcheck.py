"""0.9a gate: DEVSIM against Sandia Charon on a cold pn diode (freeze-out + Fermi-Dirac + SG drift-diffusion).

Both codes get IDENTICAL inputs, so the comparison tests the implementations, not parameter choices:
  * our silicon band parameters at T (simws.tcad.cryo.band_params) written into Charon's Effective DOS and Band Gap
    blocks (Nc300 = Nc(T) (300/T)^1.5 with exponent 1.5; constant gap Eg(T)); BGN off; eps_r 11.7
  * Charon's incomplete-ionization form in both (constant ionization energy, degeneracy 2 donors / 4 acceptors,
    full ionization above a critical doping): DEVSIM ii_model="charon"; Charon contact approximation III (exact)
  * constant mobilities and SRH lifetimes; Fermi-Dirac statistics on; Charon "drift diffusion cvfem" (CVFEM-SG, the
    same Scharfetter-Gummel flux as DEVSIM's finite volumes) on Charon's own pndiode.exo mesh (1 x 0.5 um quads,
    10 nm along x); DEVSIM on a 1D line with the same x nodes

Gate (Build Plan 0.9a): terminal current density within 2 % wherever |J| > 1e-6 A/cm^2, and the neutral-region
majority densities (the freeze-out) within 2 %.

Charon is not in the simws environments; it runs on the laptop (tools/run_charon_xcheck.sh):
    python -m simws.validation.charon_xcheck write  DIR --T 77      # decks + run list
    (charonInterpreter + charon_mp.exe on each deck, in order)
    python -m simws.validation.charon_xcheck compare DIR            # needs exodus3 (tcad env)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

L_UM, XJ_UM, DX_UM, W_UM = 1.0, 0.5, 0.01, 0.5        # pndiode.exo: 1 x 0.5 um, 50 + 50 intervals along x
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
            300K gap = {c['Eg']!r}
            alpha = 0.0
            beta = 636.0
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


def _deck(c, imp, out, disc, V, ic):
    srh = "      srh recombination is on\n" if disc != "nlp cvfem" else ""
    return f"""Import State File {imp} at index 1

start output parameters
      Output State File {out}
      Output cell average variables in silicon for scalar "Doping"
      unscale variables is on
end output parameters

tpetra is on
lattice temperature = {c['T']!r}

start Physics Block Semiconductor
      geometry block is silicon
      standard discretization type is {disc}
      material model is siliconParameter
      use Fermi Dirac
      donor incomplete ionization is on
      acceptor incomplete ionization is on
{srh}      radiative recombination is off
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
end solver block
"""


def write_decks(outdir, T: float, mesh="pndiode.exo") -> list:
    """Write the deck chain (nlp equilibrium -> dd at 0 V -> dd at each forward bias); returns the run order."""
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    c = case(T)
    tag = f"T{int(round(T))}"
    order = []
    nlp = f"xc_{tag}_nlp"
    (out / f"{nlp}.inp").write_text(_deck(c, mesh, f"{nlp}.exo", "nlp cvfem", 0.0,
                                          "Initial conditions for ELECTRIC_POTENTIAL in silicon is Equilibrium Potential"))
    order.append(nlp)
    prev, first = f"{nlp}.exo", True
    for V in biases(T):
        name = f"xc_{tag}_dd_{int(round(V * 1000)):04d}mV"
        ic = ("initial conditions for ELECTRIC_POTENTIAL in silicon is Exodus File\n"
              + ("Initial Conditions for ELECTRON_DENSITY in silicon is Equilibrium Density\n"
                 "Initial Conditions for HOLE_DENSITY in silicon is Equilibrium Density" if first else
                 "Initial Conditions for ELECTRON_DENSITY in silicon is Exodus File\n"
                 "Initial Conditions for HOLE_DENSITY in silicon is Exodus File"))
        (out / f"{name}.inp").write_text(_deck(c, prev, f"{name}.exo", "drift diffusion cvfem", V, ic))
        order.append(name)
        prev, first = f"{name}.exo", False
    (out / f"runlist_{tag}.txt").write_text("\n".join(order) + "\n")
    (out / f"case_{tag}.json").write_text(json.dumps({**c, "biases": biases(T)}, indent=1))
    return order


# ------------------------------------------------------------------------------------------------ DEVSIM side
def devsim_diode(T: float) -> dict:
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
    add_1d_mesh_line(mesh="m", pos=0.0, ps=DX_UM * 1e-4, tag="anode")
    add_1d_mesh_line(mesh="m", pos=L, ps=DX_UM * 1e-4, tag="cathode")
    for k in ("anode", "cathode"):
        add_1d_contact(mesh="m", name=k, tag=k, material="metal")
    add_1d_region(mesh="m", material="Si", region="r", tag1="anode", tag2="cathode")
    finalize_mesh(mesh="m")
    create_device(mesh="m", device="d")
    phys = Physics(T=T, bgn=False, auger=False, srh=True, tau_n=c["tau"], tau_p=c["tau"], mu_n=c["mu_n"],
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
    J = []
    for V in biases(T):
        ramp_bias("d", "anode", V, step=0.05, **opts)
        J.append(abs(float(contact_current("d", "anode"))))
    return {"case": c, "equilibrium": eq, "V": biases(T), "J_A_cm2": J, "used": {k: v for k, v in used.items()
                                                                                 if isinstance(v, (int, float))}}


# ------------------------------------------------------------------------------------------------ Charon results
def read_charon(outdir, T: float) -> dict:
    try:
        import exodus3 as exodus
    except ImportError:
        import exodus
    out, tag = Path(outdir), f"T{int(round(T))}"
    names = (out / f"runlist_{tag}.txt").read_text().split()
    res = {"V": [], "J_A_cm2": [], "global_names": None}

    def nodal(e, var):
        k = e.num_times()
        return np.asarray(e.get_node_variable_values(var, k))

    for name in names:
        e = exodus.exodus(str(out / f"{name}.exo"), mode="r", array_type="numpy")
        x, y, _ = (np.asarray(v) for v in e.get_coords())
        nv = e.get_node_variable_names()
        if name.endswith("_nlp") or name.endswith("_0000mV"):
            key = "nlp" if name.endswith("_nlp") else "equilibrium"
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


def compare(outdir, T: float, dev: dict | None = None) -> dict:
    ch = read_charon(outdir, T)
    dev = dev or devsim_diode(T)
    rows, worst = [], 0.0
    for V, Jc in zip(ch["V"], ch["J_A_cm2"]):
        Jd = dev["J_A_cm2"][dev["V"].index(V)] if V in dev["V"] else float("nan")
        gated = max(Jc, Jd) > J_FLOOR
        rel = Jd / Jc - 1 if Jc > 0 else float("nan")
        if gated:
            worst = max(worst, abs(rel))
        rows.append({"V": V, "J_charon": Jc, "J_devsim": Jd, "rel_diff": rel, "gated": gated})
    # freeze-out: majority densities in the neutral regions (0.1 um from each contact)
    eqc, eqd = ch.get("equilibrium") or ch.get("nlp"), dev["equilibrium"]
    dens = {}
    if eqc and "ELECTRON_DENSITY" in eqc:
        xc, xd = np.asarray(eqc["x_um"]), np.asarray(eqd["x_um"])
        for lab, xx, key, dk in (("p_anode_side", 0.1, "HOLE_DENSITY", "p"), ("n_cathode_side", 0.9, "ELECTRON_DENSITY", "n")):
            vc = float(np.interp(xx, xc, eqc[key])); vd = float(np.interp(xx, xd, eqd[dk]))
            dens[lab] = {"charon": vc, "devsim": vd, "rel_diff": vd / vc - 1, "ionized_fraction_devsim": vd / 1e17}
            worst = max(worst, abs(vd / vc - 1))
        pc = np.asarray(eqc["ELECTRIC_POTENTIAL"]); pd = np.asarray(eqd["psi"])
        dens["built_in_V"] = {"charon": float(pc[-1] - pc[0]), "devsim": float(pd[-1] - pd[0])}
    rep = {"T_K": T, "currents": rows, "densities": dens, "worst_rel_diff": worst, "gate": GATE,
           "pass": bool(rows) and worst < GATE, "charon_current_variable": ch.get("current_variable"),
           "charon_global_variables": ch.get("global_names"),
           "note": "J from Charon's anode global variable divided by the mesh width (2D current per cm of depth)"}
    if rows and not rep["pass"]:
        r = [x["rel_diff"] for x in rows if x["gated"]]
        if len(r) > 2 and np.std(np.log1p(r)) < 0.01:
            rep["hint"] = f"constant ratio {1 + np.mean(r):.4g}: likely a current-unit/width convention, not physics"
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m simws.validation.charon_xcheck")
    ap.add_argument("action", choices=["write", "compare", "devsim"])
    ap.add_argument("dir")
    ap.add_argument("--T", type=float, nargs="+", default=[77.0, 300.0])
    a = ap.parse_args(argv)
    ok = True
    for T in a.T:
        if a.action == "write":
            print(" ".join(write_decks(a.dir, T)))
        elif a.action == "devsim":
            r = devsim_diode(T)
            Path(a.dir, f"devsim_T{int(round(T))}.json").write_text(json.dumps(r, indent=1))
            print(json.dumps({"T": T, "V": r["V"], "J": r["J_A_cm2"]}))
        else:
            dev_file = Path(a.dir, f"devsim_T{int(round(T))}.json")
            dev = json.loads(dev_file.read_text()) if dev_file.exists() else None
            r = compare(a.dir, T, dev)
            Path(a.dir, f"comparison_T{int(round(T))}.json").write_text(json.dumps(r, indent=1))
            print(json.dumps({k: r[k] for k in ("T_K", "worst_rel_diff", "pass")}))
            for row in r["currents"]:
                print(f"  V={row['V']:.2f}  Charon {row['J_charon']:.4e}  DEVSIM {row['J_devsim']:.4e}  "
                      f"{100 * row['rel_diff']:+.2f}%{'' if row['gated'] else '  (below floor)'}")
            for k, v in r["densities"].items():
                print(f"  {k}: {v}")
            ok &= r["pass"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
