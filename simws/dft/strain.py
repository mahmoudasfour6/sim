"""Elastic constants and band deformation potentials of a diamond crystal from Quantum ESPRESSO.

    from simws.dft.strain import elastic_constants, deformation_potentials
    el = elastic_constants("Si", a0, {"Si": "Si.pbe-rrkj.UPF"})                  # at the DFT equilibrium a0
    dp = deformation_potentials("Si", 5.431, {"Si": "..."}, soc_pseudos={"Si": "fully-relativistic UPF"})

Elastic constants (central differences, strain +-delta, at the zero-pressure DFT lattice constant):
  C11, C12 from uniaxial eps_xx (stress sigma_xx, sigma_yy); C44 from the engineering shear eps_xy = eps_yx = delta/2
  with the atoms relaxed (Kleinman internal relaxation; the unrelaxed value is reported too).
  QE writes stress with the opposite sign to the tensile-positive convention; we flip it, and the result is
  checked (C11 > C12 > 0, B = (C11 + 2 C12)/3 vs the equation-of-state bulk modulus).
Deformation potentials (at the measured lattice constant; first order, central differences):
  Xi_u  uniaxial conduction-band potential: E(Delta_z) - E(Delta_x) = Xi_u (eps_zz - eps_xx), strain eps_zz = +-delta;
        each valley minimum is found on a line through its strained position
  b     valence shear potential along [001] (spin-orbit included): the top two Kramers pairs split by
        |2 b (eps_zz - eps_xx)| for strain << Delta_so; reported with the standard negative sign
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from simws.dft.qe import PW, Structure, run


def _stress(st, pseudos, ecutwfc, kgrid, wd, ranks, relax=False, functional=None, pseudo_dir=None):
    r = run(PW(st, pseudos, ecutwfc=ecutwfc, kgrid=kgrid, tstress=True, relax_atoms=relax, functional=functional,
               prefix="el", pseudo_dir=pseudo_dir, conv_thr=1e-11), wd, ranks=ranks)
    return -np.asarray(r.stress_GPa), r


def elastic_constants(element, a0, pseudos, ecutwfc=40.0, kgrid=(10, 10, 10), delta=0.005, workdir="elastic",
                      ranks=1, functional=None, pseudo_dir=None):
    base = Structure.diamond(element, a0)
    wd = Path(workdir)
    out = {"a0_A": a0, "delta": delta}
    s0, _ = _stress(base, pseudos, ecutwfc, kgrid, wd / "ref", ranks, functional=functional, pseudo_dir=pseudo_dir)
    out["residual_stress_GPa"] = float(np.trace(s0) / 3)
    S = {}
    for sgn in (+1, -1):
        e = np.zeros((3, 3)); e[0, 0] = sgn * delta
        S[("xx", sgn)], _ = _stress(base.strained(e), pseudos, ecutwfc, kgrid, wd / f"xx{sgn:+d}", ranks,
                                    functional=functional, pseudo_dir=pseudo_dir)
        e = np.zeros((3, 3)); e[0, 1] = e[1, 0] = sgn * delta / 2
        S[("xy", sgn)], _ = _stress(base.strained(e), pseudos, ecutwfc, kgrid, wd / f"xy{sgn:+d}", ranks, relax=True,
                                    functional=functional, pseudo_dir=pseudo_dir)
        S[("xy0", sgn)], _ = _stress(base.strained(e), pseudos, ecutwfc, kgrid, wd / f"xy0{sgn:+d}", ranks,
                                     functional=functional, pseudo_dir=pseudo_dir)
    d = 2 * delta
    out["C11_GPa"] = float((S[("xx", 1)][0, 0] - S[("xx", -1)][0, 0]) / d)
    out["C12_GPa"] = float((S[("xx", 1)][1, 1] - S[("xx", -1)][1, 1]) / d)
    out["C44_GPa"] = float((S[("xy", 1)][0, 1] - S[("xy", -1)][0, 1]) / d)
    out["C44_unrelaxed_GPa"] = float((S[("xy0", 1)][0, 1] - S[("xy0", -1)][0, 1]) / d)
    out["B_GPa"] = (out["C11_GPa"] + 2 * out["C12_GPa"]) / 3
    out["poisson_001"] = out["C12_GPa"] / (out["C11_GPa"] + out["C12_GPa"])
    return out


def _valley_min(bands_fn, axis, k0, scale, span=0.04, n=9):
    t = np.linspace(k0 - span, k0 + span, n)
    k = np.zeros((n, 3)); k[:, axis] = t * scale
    E = bands_fn(k)
    i = int(np.argmin(E))
    c = np.polyfit(t - t[i], E, 4)
    tt = np.linspace(-span, span, 2001)
    return float(np.min(np.polyval(c, tt)))


def deformation_potentials(element, a, pseudos, soc_pseudos=None, ecutwfc=30.0, kgrid=(8, 8, 8), delta=0.002,
                           k0=0.84, workdir="deformation", ranks=1, functional=None, pseudo_dir=None,
                           soc_functional="same"):
    wd = Path(workdir)
    out = {"a_A": a, "delta": delta}
    nocc = 4
    split = {}
    for sgn in (+1, -1):
        eps = np.diag([0.0, 0.0, sgn * delta])
        st = Structure.diamond(element, a).strained(eps)
        sub = wd / f"zz{sgn:+d}"
        common = dict(ecutwfc=ecutwfc, functional=functional, prefix="dp", pseudo_dir=pseudo_dir)
        run(PW(st, pseudos, kgrid=kgrid, nbnd=8, **common), sub, ranks=ranks)

        def bands(k, st=st, sub=sub, common=common):
            return run(PW(st, pseudos, calculation="bands", klist=k, nbnd=8, **common), sub, ranks=ranks).eigenvalues_eV[:, nocc]

        # reciprocal vector components scale as 1/(1 + eps_ii) for a diagonal strain
        Ez = _valley_min(bands, 2, k0, 1.0 / (1 + sgn * delta))
        Ex = _valley_min(bands, 0, k0, 1.0)
        split[sgn] = Ez - Ex
        if soc_pseudos:
            subs = wd / f"zz{sgn:+d}_soc"
            cs = dict(common, functional=(functional if soc_functional == "same" else soc_functional))
            run(PW(st, soc_pseudos, kgrid=kgrid, nbnd=16, soc=True, **cs), subs, ranks=ranks)
            g = run(PW(st, soc_pseudos, calculation="bands", klist=np.zeros((1, 3)), nbnd=16, soc=True, **cs),
                    subs, ranks=ranks).eigenvalues_eV[0]
            out.setdefault("valence_split_eV", {})[sgn] = float(g[7] - g[5])
    out["Xi_u_eV"] = float((split[1] - split[-1]) / (2 * delta))
    out["Xi_u_linearity"] = float(abs(split[1] + split[-1]) / max(abs(split[1] - split[-1]), 1e-12))
    if soc_pseudos:
        v = out["valence_split_eV"]
        out["b_eV"] = -float((v[1] + v[-1]) / 2 / (2 * delta))
        out["b_asymmetry"] = float(abs(v[1] - v[-1]) / (v[1] + v[-1]))
    return out
