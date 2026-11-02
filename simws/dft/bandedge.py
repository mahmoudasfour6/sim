"""Band-edge parameters of a diamond-structure semiconductor (Si) from DFT band energies.

    from simws.dft.bandedge import Setup, compute
    res = compute(Setup(pseudo="Si.pbe-rrkj.UPF", soc_pseudo="Si.rel-pbe-rrkj.UPF"), workdir="runs/si_bands")

What is computed (k in units of 2 pi / a, energies from pw.x 'bands' runs on the converged SCF density):
  conduction minimum   position k0 on Gamma-X (fit of the lowest conduction band), indirect gap E_g(DFT)
  electron masses      m_l, m_t: curvature of E(k) at k0 along and across [100] (fit E0 + c2 dk^2 + c3 dk^3 + c4 dk^4)
  hole masses          heavy/light (and split-off with spin-orbit) along [100], [110], [111]; DOS masses from the
                       angular average m_dos^(3/2) = < m(n)^(3/2) > over directions (exact for warped parabolic
                       bands); split-off energy Delta_so
  N_c(T), N_v(T)       directly from the band energies: N = (g/(2 pi)^3) int sum_n exp(-|E_n(k) - E_edge|/kT) d^3k
                       on a uniform k grid around the band edge (cubic symmetry used, each state counted once;
                       g = 2 without spin-orbit, 1 with spinor bands). This keeps band non-parabolicity and
                       warping, which band-edge masses miss (the reason measured N_v grows faster than T^1.5).
                       Conduction: the neighbourhood of the three X points, both lowest conduction bands
                       (each X region holds the two Delta valleys that meet there). Valence: a sphere around Gamma.
  DOS masses at T      m_dos(T) = (N(T) / (2 (2 pi m0 k T / h^2)^1.5))^(2/3)  (N_c per valley set: includes 6^(2/3))
Gap: semilocal DFT underestimates it, so a hybrid (HSE06) correction is applied as a scissor: the HSE - PBE
difference of the gaps between the same SCF k-points (Gamma -> the k point nearest k0), see hybrid_gap().
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from simws.dft.qe import PW, Structure, run

HB2_2M = 3.80998212          # hbar^2 / (2 m0) in eV A^2
K_B = 8.617333262e-5         # eV/K
NC_M1_300 = 2.50945e19       # 2 (2 pi m0 k 300K / h^2)^1.5 in cm^-3


@dataclass
class Setup:
    pseudo: str                              # scalar-relativistic or fully-relativistic UPF for the main runs
    soc_pseudo: Optional[str] = None         # fully-relativistic UPF: enables spin-orbit valence quantities
    element: str = "Si"
    a_angstrom: float = 5.431                # experimental lattice constant (bands at the measured volume)
    functional: Optional[str] = None         # None = the pseudopotential's own
    ecutwfc: float = 30.0
    kgrid: tuple = (8, 8, 8)
    ranks: int = 1
    pseudo_dir: Optional[str] = None
    h_fit: float = 0.01                      # k step of the mass stencils (2 pi / a)
    h_dos: float = 0.0125                    # k spacing of the DOS grids
    R_val: float = 0.25                      # radius of the valence DOS sphere around Gamma
    R_con_t: float = 0.15                    # transverse radius of the conduction DOS region around X
    kx_con_min: float = 0.60                 # conduction DOS region: kx from here to the X point (1.0)
    n_dirs: int = 60                         # directions for the angular hole-mass average


def _tpiba(a):
    return 2 * math.pi / a


class _Bands:
    """SCF once, then 'bands' runs for arbitrary k lists in the same directory."""

    def __init__(self, s: Setup, workdir: Path, soc: bool):
        self.s, self.soc = s, soc
        self.wd = Path(workdir) / ("soc" if soc else "scalar")
        pseudo = s.soc_pseudo if soc else s.pseudo
        self.base = dict(structure=Structure.diamond(s.element, s.a_angstrom), pseudos={s.element: pseudo},
                         ecutwfc=s.ecutwfc, functional=s.functional, soc=soc, prefix="be", pseudo_dir=s.pseudo_dir)
        self.nbnd = 16 if soc else 8
        self.nocc = 8 if soc else 4
        self.scf = run(PW(kgrid=s.kgrid, nbnd=self.nbnd, **self.base), self.wd, ranks=s.ranks)
        self.notes = list(self.scf.notes)
        self.vbm_scf, _ = self.scf.band_edges()

    def energies(self, k):
        k = np.atleast_2d(np.asarray(k, float))
        out = []
        for i in range(0, len(k), 4000):                         # pw.x npk limit is 40000; keep runs modest
            r = run(PW(calculation="bands", klist=k[i:i + 4000], nbnd=self.nbnd, **self.base), self.wd,
                    ranks=self.s.ranks)
            out.append(r.eigenvalues_eV)
        return np.vstack(out)


def _fit_curvature(dk, E, odd=False):
    """Least-squares E = E0 + c2 dk^2 (+ c3 dk^3) + c4 dk^4; returns (E0, c2)."""
    cols = [np.ones_like(dk), dk ** 2] + ([dk ** 3] if odd else []) + [dk ** 4]
    if odd:
        cols.insert(1, dk)
    A = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(A, E, rcond=None)
    return c[0], (c[2] if odd else c[1])


def _mass(c2, a):
    """E = c2 dk^2 with dk in 2 pi / a  ->  m* / m0 (positive for minima, negative c2 gives a hole mass > 0)."""
    return HB2_2M * _tpiba(a) ** 2 / abs(c2)


def _fib_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])


def electrons(b: _Bands):
    s, a, h = b.s, b.s.a_angstrom, b.s.h_fit
    kx = np.round(np.arange(0.70, 1.0001, 0.01), 6)
    E = b.energies(np.column_stack([kx, 0 * kx, 0 * kx]))[:, b.nocc]
    i = int(np.argmin(E))
    sel = slice(max(i - 3, 0), min(i + 4, len(kx)))
    c = np.polyfit(kx[sel] - kx[i], E[sel], 4)
    roots = [r.real for r in np.roots(np.polyder(c)) if abs(r.imag) < 1e-9 and abs(r.real) < 0.03]
    k0 = kx[i] + (min(roots, key=lambda r: np.polyval(c, r)) if roots else 0.0)
    d = h * np.arange(-4, 5)
    kl = np.column_stack([k0 + d, 0 * d, 0 * d])
    kt = np.column_stack([np.full_like(d, k0), d, 0 * d])
    El = b.energies(np.vstack([kl, kt]))[:, b.nocc]
    E0l, c2l = _fit_curvature(d, El[:9], odd=True)
    E0t, c2t = _fit_curvature(d, El[9:])
    return {"k0_2pi_a": float(k0), "E_cbm_eV": float(min(E0l, E0t)), "m_l": _mass(c2l, a), "m_t": _mass(c2t, a),
            "scan_kx": kx.tolist(), "scan_E": E.tolist()}


def holes(b: _Bands):
    """Directional heavy/light(/split-off) masses at Gamma and their DOS (angular-average) masses."""
    s, a, h = b.s, b.s.a_angstrom, b.s.h_fit
    t = h * np.arange(1, 5)
    named = {"100": [1, 0, 0], "110": [1, 1, 0], "111": [1, 1, 1]}
    dirs = [np.array(v, float) / np.linalg.norm(v) for v in named.values()] + list(_fib_dirs(s.n_dirs))
    k = np.vstack([[0.0, 0.0, 0.0]] + [np.outer(t, n) for n in dirs])
    E = b.energies(k)
    step = 2 if b.soc else 1                                        # Kramers pairs with spin-orbit
    top = [b.nocc - step * (j + 1) for j in range(3)]       # hh, lh, so (third band; degenerate at Gamma without SOC)
    names = ["hh", "lh", "so"]
    E_gamma = E[0]
    vbm = float(E_gamma[b.nocc - 1])
    m = {nm: [] for nm in names}
    for i in range(len(dirs)):
        blk = E[1 + 4 * i: 5 + 4 * i]
        for nm, j in zip(names, top):
            dk = np.concatenate([-t[::-1], [0.0], t])
            e = np.concatenate([blk[::-1, j], [E_gamma[j]], blk[:, j]])
            _, c2 = _fit_curvature(dk, e)
            m[nm].append(_mass(c2, a))
    out = {"E_vbm_eV": vbm}
    for nm in names:
        arr = np.array(m[nm])
        out[f"m_{nm}_100"], out[f"m_{nm}_110"], out[f"m_{nm}_111"] = (float(x) for x in arr[:3])
        out[f"m_{nm}_dos"] = float(np.mean(arr[3:] ** 1.5) ** (2 / 3))
    if b.soc:
        out["Delta_so_eV"] = float(E_gamma[top[0]] - E_gamma[top[2]])
    use = ("hh", "lh") if b.soc else names          # without spin-orbit all three bands meet at the top
    mv = sum(out[f"m_{nm}_dos"] ** 1.5 for nm in use) ** (2 / 3)
    out["m_dh_band_edge"] = float(mv)
    return out


def _grid_valence(R, h):
    n = int(math.ceil(R / h))
    g = np.arange(-n, n + 1) * h
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    P = P[np.linalg.norm(P, axis=1) <= R + 1e-12]
    key = -np.sort(-np.abs(P), axis=1)                              # Oh: |k| components, sorted descending
    uniq, w = np.unique(np.round(key / h).astype(int), axis=0, return_counts=True)
    return uniq * h, w.astype(float)


def _grid_conduction(kx_min, Rt, h):
    """Around X = (1,0,0): kx in [kx_min, 1] (mirror image through X counted by weight 2, X plane once),
    |k_perp| <= Rt; C4v symmetry in (ky, kz)."""
    n = int(math.ceil(Rt / h))
    nx = int(round((1.0 - kx_min) / h))
    kxs = 1.0 - h * np.arange(nx + 1)
    g = np.arange(-n, n + 1) * h
    Y, Z = np.meshgrid(g, g, indexing="ij")
    T = np.column_stack([Y.ravel(), Z.ravel()])
    T = T[np.linalg.norm(T, axis=1) <= Rt + 1e-12]
    key = -np.sort(-np.abs(T), axis=1)
    ut, wt = np.unique(np.round(key / h).astype(int), axis=0, return_counts=True)
    pts, w = [], []
    for kx in kxs:
        wx = 1.0 if abs(kx - 1.0) < 1e-9 else 2.0
        for (a, b), c in zip(ut * h, wt):
            pts.append([kx, a, b]); w.append(wx * c)
    return np.array(pts), np.array(w)


def thermal_dos(bc: _Bands, bv: _Bands, T=(200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0),
                E_cbm=None, E_vbm=None):
    """N_c(T) and N_v(T) (cm^-3) from Boltzmann sums over the band energies; edges from the mass fits."""
    s = bc.s
    a_cm = s.a_angstrom * 1e-8
    dk3 = (s.h_dos * 2 * math.pi / a_cm) ** 3                      # cm^-3 per grid point
    kc, wc = _grid_conduction(s.kx_con_min, s.R_con_t, s.h_dos)
    kv, wv = _grid_valence(s.R_val, s.h_dos)
    Ec = bc.energies(kc)[:, bc.nocc:bc.nocc + (4 if bc.soc else 2)]
    Ev = bv.energies(kv)[:, bv.nocc - (6 if bv.soc else 3):bv.nocc]
    gc, gv = (1.0 if bc.soc else 2.0), (1.0 if bv.soc else 2.0)
    E_cbm = Ec.min() if E_cbm is None else min(E_cbm, Ec.min())
    E_vbm = Ev.max() if E_vbm is None else max(E_vbm, Ev.max())
    out = {"T_K": list(T), "Nc_cm3": [], "Nv_cm3": [], "m_dn_T": [], "m_dh_T": [], "edge_fraction_c": [],
           "edge_fraction_v": [], "n_points_c": int(len(kc)), "n_points_v": int(len(kv))}
    Rmask_v = np.linalg.norm(kv, axis=1) > s.R_val - 1.5 * s.h_dos
    rperp = np.linalg.norm(kc[:, 1:], axis=1)
    Rmask_c = (rperp > s.R_con_t - 1.5 * s.h_dos) | (kc[:, 0] < s.kx_con_min + 1.5 * s.h_dos)
    for TK in T:
        kT = K_B * TK
        bc_w = np.exp(-(Ec - E_cbm) / kT).sum(axis=1) * wc
        bv_w = np.exp(-(E_vbm - Ev) / kT).sum(axis=1) * wv
        Nc = 3 * gc / (2 * math.pi) ** 3 * dk3 * bc_w.sum()       # three X points
        Nv = gv / (2 * math.pi) ** 3 * dk3 * bv_w.sum()
        ref = NC_M1_300 * (TK / 300.0) ** 1.5
        out["Nc_cm3"].append(Nc); out["Nv_cm3"].append(Nv)
        out["m_dn_T"].append((Nc / ref) ** (2 / 3)); out["m_dh_T"].append((Nv / ref) ** (2 / 3))
        out["edge_fraction_c"].append(float(bc_w[Rmask_c].sum() / bc_w.sum()))
        out["edge_fraction_v"].append(float(bv_w[Rmask_v].sum() / bv_w.sum()))
    return out


def compute(s: Setup, workdir, T=(200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0), dos=True):
    """All band-edge quantities for one functional / pseudopotential set."""
    wd = Path(workdir)
    bs = _Bands(s, wd, soc=False)
    res = {"setup": {k: v for k, v in s.__dict__.items()}, "pw_version": bs.scf.pw_version,
           "functional_used": bs.scf.functional, "notes": bs.notes}
    res["electrons"] = electrons(bs)
    res["holes_scalar"] = holes(bs)
    bv = bs
    if s.soc_pseudo:
        bv = _Bands(s, wd, soc=True)
        res["holes_soc"] = holes(bv)
        res["notes"] += bv.notes
    res["Eg_dft_eV"] = res["electrons"]["E_cbm_eV"] - res["holes_scalar"]["E_vbm_eV"]
    if s.soc_pseudo:   # spin-orbit lifts the valence top by Delta_so/3; the conduction band barely moves in Si
        res["Eg_dft_soc_eV"] = res["Eg_dft_eV"] - res["holes_soc"]["Delta_so_eV"] / 3
    if dos:
        res["dos"] = thermal_dos(bs, bv, T, E_cbm=res["electrons"]["E_cbm_eV"],
                                 E_vbm=(res["holes_soc"] if s.soc_pseudo else res["holes_scalar"])["E_vbm_eV"])
    return res


def hybrid_gap(s: Setup, workdir, kgrid=(4, 4, 4), nqx=None, functional="hse", k_target=0.85):
    """HSE - PBE gap correction on an unshifted SCF grid: gaps from Gamma (VBM) to the grid k-point on Gamma-X
    nearest k_target, plus the direct gaps at Gamma and X for the spread of the scissor.
    nqx defaults to the k grid: a coarser EXX q-mesh is not converged for Si (2x2x2 on a 4x4x4 grid
    over-corrects the gap by about 0.4 eV)."""
    nqx = tuple(nqx or kgrid)
    out = {"kgrid": list(kgrid), "nqx": list(nqx)}
    for f in (s.functional, functional):
        tag = f or "semilocal"
        st = Structure.diamond(s.element, s.a_angstrom)
        r = run(PW(st, {s.element: s.pseudo}, ecutwfc=s.ecutwfc, kgrid=kgrid, kshift=(0, 0, 0), nbnd=8,
                   functional=f, nqx=nqx, prefix="hyb", pseudo_dir=s.pseudo_dir), Path(workdir) / tag, ranks=s.ranks)
        k, E = r.kpoints_tpiba, r.eigenvalues_eV
        # map every k to its cubic-star representative to recognise Gamma-X points stored in other orientations
        rep = -np.sort(-np.abs(k), axis=1)
        gx = np.where((np.abs(rep[:, 1]) < 1e-6) & (np.abs(rep[:, 2]) < 1e-6))[0]
        iG = int(np.argmin(np.linalg.norm(k, axis=1)))
        iX = gx[int(np.argmin(np.abs(rep[gx, 0] - 1.0)))]
        iD = gx[int(np.argmin(np.abs(rep[gx, 0] - k_target)))]
        vbm = E[iG, 3]
        out[tag] = {"Eg_G_to_kD": float(E[iD, 4] - vbm), "kD": float(rep[iD, 0]), "Eg_G_to_X": float(E[iX, 4] - vbm),
                    "Eg_direct_G": float(E[iG, 4] - E[iG, 3]), "wall_s": r.wall_s, "notes": r.notes}
    lo, hi = out[s.functional or "semilocal"], out[functional]
    out["Eg_hybrid_G_to_X_eV"] = hi["Eg_G_to_X"]
    corr = {k: hi[k] - lo[k] for k in ("Eg_G_to_kD", "Eg_G_to_X", "Eg_direct_G")}
    out["scissor_eV"] = corr["Eg_G_to_kD"]
    out["scissor_spread_eV"] = float(max(abs(corr["Eg_G_to_X"] - corr["Eg_G_to_kD"]), 0.01))
    out["corrections"] = corr
    return out
