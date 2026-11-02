"""CHGNet universal machine-learned potential (trained on Materials Project PBE data): SiGe alloy lattice constant
and elastic constants, benchmarked against DFT and experiment; strained-layer critical thickness.

Benchmark result (simws 0.7): CHGNet 0.3 overestimates a(Si) by 0.6 % and a(Ge) by 1.8 %, gives elastic
constants 20-40 % too soft for Si and unusable for Ge (C44 ~ 1 GPa vs 67 measured: the known softening of
universal potentials), and a Vegard bowing of the wrong sign. None of these feed the material database; the
critical thickness uses the measured lattice constants (Dismukes) and DFT or measured elastic constants. Re-run
the benchmark (workflow rule dft_mlip_benchmark) after fine-tuning or with another potential.

    from simws.dft.mlip import alloy_lattice, elastic_constants, critical_thickness
    lat = alloy_lattice(x=[0, 0.25, 0.5, 0.75, 1], n_configs=4)      # random 64-atom alloys, fully relaxed
    el = elastic_constants("Si"); el["C11_GPa"], el["C12_GPa"], el["C44_GPa"]

Because CHGNet inherits PBE's lattice-constant overestimate (about 1 %), absolute lattice constants are reported
as computed and the bowing (deviation from Vegard's law) is what carries over: a(x) = a_Si + (a_Ge - a_Si) x
- theta x (1 - x), with the end points from experiment and theta from CHGNet (relative error cancels).
The critical thickness then uses the misfit f(x) = (a(x) - a_Si) / a_Si and Poisson ratio nu = C12 / (C11 + C12)
(interpolated linearly between Si and Ge): Matthews-Blakeslee (equilibrium, 60-degree dislocations) and
People-Bean (metastable limit for low-temperature growth).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import numpy as np

A_EXP = {"Si": 5.431, "Ge": 5.658}        # 300 K, Angstrom (Dismukes et al., J. Phys. Chem. 68, 3021 (1964))


@lru_cache(maxsize=1)
def _model():
    from chgnet.model import CHGNet
    return CHGNet.load(verbose=False)


def _diamond_conventional(a, symbols=("Si",) * 8):
    from pymatgen.core import Lattice, Structure
    base = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    frac = np.vstack([base, base + 0.25])
    return Structure(Lattice.cubic(a), list(symbols), frac)


def _relax(struct, relax_cell=True, fmax=0.005, steps=600):
    from chgnet.model import StructOptimizer
    opt = StructOptimizer(model=_model())
    r = opt.relax(struct, fmax=fmax, steps=steps, relax_cell=relax_cell, verbose=False)
    return r["final_structure"], float(r["trajectory"].energies[-1])


def alloy_lattice(x: Sequence[float] = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0), n_configs=4,
                  supercell=2, seed=1):
    """Relaxed cubic-equivalent lattice constant (Angstrom) of random Si(1-x)Ge(x), mean and standard error over
    configurations, and the Vegard bowing parameter theta (Angstrom) with its standard error."""
    rng = np.random.default_rng(seed)
    a_si = A_EXP["Si"]
    out = {"x": [], "a_mean_A": [], "a_sem_A": [], "a_all_A": [], "n_atoms": 8 * supercell ** 3}
    for xi in x:
        n = 8 * supercell ** 3
        nge = int(round(xi * n))
        vals = []
        reps = 1 if nge in (0, n) else n_configs
        for _ in range(reps):
            sym = np.array(["Si"] * (n - nge) + ["Ge"] * nge)
            rng.shuffle(sym)
            s0 = _diamond_conventional(a_si * (1 + 0.04 * xi))
            s0.make_supercell([supercell] * 3)
            for i, sp in enumerate(sym):
                s0.replace(i, sp)
            rel, _ = _relax(s0)
            vals.append((rel.volume / supercell ** 3) ** (1 / 3))
        out["x"].append(nge / n)
        out["a_all_A"].append(vals)
        out["a_mean_A"].append(float(np.mean(vals)))
        out["a_sem_A"].append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
    X, A = np.array(out["x"]), np.array(out["a_mean_A"])
    a0, a1 = A[0], A[-1]
    dev = A - (a0 + (a1 - a0) * X)                      # deviation from Vegard between the CHGNet end points
    w = X * (1 - X)
    sel = w > 0
    sig = np.maximum(np.array(out["a_sem_A"])[sel], 1e-5)
    theta = float(-np.sum(w[sel] * dev[sel] / sig ** 2) / np.sum(w[sel] ** 2 / sig ** 2))
    theta_se = float(1 / math.sqrt(np.sum(w[sel] ** 2 / sig ** 2)))
    resid = dev[sel] + theta * w[sel]
    out.update({"a_Si_A": float(a0), "a_Ge_A": float(a1), "theta_A": theta, "theta_se_A": theta_se,
                "theta_fit_rms_A": float(np.sqrt(np.mean(resid ** 2))),
                "a_Si_rel_err": float(a0 / A_EXP["Si"] - 1), "a_Ge_rel_err": float(a1 / A_EXP["Ge"] - 1)})
    return out


def elastic_constants(element="Si", delta=0.005):
    """C11, C12, C44 (GPa) of the diamond crystal from stress-strain at the CHGNet equilibrium lattice constant;
    internal coordinates relaxed under shear (Kleinman relaxation). Central differences in +-delta."""
    s0, _ = _relax(_diamond_conventional(A_EXP[element], (element,) * 8))
    a = s0.lattice.a
    m = _model()

    def stress(eps, relax_atoms):
        from pymatgen.core import Lattice
        s = s0.copy()
        F = np.eye(3) + np.asarray(eps)
        s.lattice = Lattice(s0.lattice.matrix @ F.T)
        if relax_atoms:
            s, _ = _relax(s, relax_cell=False, fmax=0.002)
        return np.asarray(m.predict_structure(s)["s"], float)     # GPa, tensile positive (checked: C11 > 0)

    e1 = np.zeros((3, 3)); e1[0, 0] = delta
    s_p, s_m = stress(e1, False), stress(-e1, False)
    C11 = (s_p[0, 0] - s_m[0, 0]) / (2 * delta)
    C12 = (s_p[1, 1] - s_m[1, 1]) / (2 * delta)
    e4 = np.zeros((3, 3)); e4[0, 1] = e4[1, 0] = delta / 2        # engineering shear strain delta
    t_p, t_m = stress(e4, True), stress(-e4, True)
    C44 = (t_p[0, 1] - t_m[0, 1]) / (2 * delta)
    t_p0, t_m0 = stress(e4, False), stress(-e4, False)
    C44_0 = (t_p0[0, 1] - t_m0[0, 1]) / (2 * delta)
    return {"element": element, "a_A": float(a), "C11_GPa": float(C11), "C12_GPa": float(C12), "C44_GPa": float(C44),
            "C44_unrelaxed_GPa": float(C44_0), "B_GPa": float((C11 + 2 * C12) / 3), "delta": delta}


def misfit(x, theta_A, a_si=A_EXP["Si"], a_ge=A_EXP["Ge"]):
    a = a_si + (a_ge - a_si) * x - theta_A * x * (1 - x)
    return (a - a_si) / a_si, a


def poisson(x, el_si, el_ge):
    c11 = el_si["C11_GPa"] + (el_ge["C11_GPa"] - el_si["C11_GPa"]) * x
    c12 = el_si["C12_GPa"] + (el_ge["C12_GPa"] - el_si["C12_GPa"]) * x
    return c12 / (c11 + c12)


def critical_thickness(x, theta_A, el_si, el_ge, model="matthews-blakeslee"):
    """Critical thickness (nm) of a uniform Si(1-x)Ge(x) layer on Si(001)."""
    f, a = misfit(x, theta_A)
    if f <= 0:
        return float("inf")
    nu = poisson(x, el_si, el_ge)
    if model == "matthews-blakeslee":
        b = a / math.sqrt(2)                       # 60-degree dislocation, Burgers vector a/2<110>
        cos_a, cos_l = 0.5, 0.5
        k = b * (1 - nu * cos_a ** 2) / (8 * math.pi * f * (1 + nu) * cos_l)
        h = 10 * b
        for _ in range(200):
            h_new = k * (math.log(h / b) + 1)
            if abs(h_new - h) < 1e-9 * h:
                break
            h = max(h_new, b * 1.01)
        return h / 10.0
    if model == "people-bean":
        b = 4.0                                    # Angstrom, as in People & Bean, APL 47, 322 (1985)
        k = (1 - nu) / (1 + nu) * b ** 2 / (16 * math.pi * math.sqrt(2) * a) / f ** 2
        h = 100.0
        for _ in range(500):
            h_new = k * math.log(max(h / b, 1.0001))
            if abs(h_new - h) < 1e-9 * h:
                break
            h = h_new
        return h / 10.0
    raise ValueError(model)


def layer_stability(profile_x, thickness_nm, theta_A, el_si, el_ge):
    """Stability of a (possibly graded) layer: the effective Ge fraction is the thickness-weighted mean (strain
    energy ~ integral of f dz); returns its MB and PB critical thicknesses and the ratios t / h_c."""
    xs = np.asarray(profile_x, float)
    x_eff = float(xs.mean())
    hmb = critical_thickness(x_eff, theta_A, el_si, el_ge, "matthews-blakeslee")
    hpb = critical_thickness(x_eff, theta_A, el_si, el_ge, "people-bean")
    return {"x_eff": x_eff, "thickness_nm": thickness_nm, "h_MB_nm": hmb, "h_PB_nm": hpb,
            "t_over_hMB": thickness_nm / hmb, "t_over_hPB": thickness_nm / hpb,
            "state": ("stable" if thickness_nm < 0.9 * hmb else "at the stability limit" if thickness_nm <= hmb
                      else "metastable" if thickness_nm < hpb else "relaxes")}
