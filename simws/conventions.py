"""Conventions registry, version 1.0 (interface specification: conventions registry section).

Every adapter declares the native conventions of the quantities it reads or writes; these functions convert at the
adapter boundary, and tests/unit/test_hub_conventions.py round-trips every converter.

    from simws import conventions as cv
    sigma = cv.convert("stress", qe_stress, "compressive+")            # -> tensile positive
    eps_v = cv.convert("strain_shear", voigt, "tensor")                  # -> engineering shear
    I_in  = cv.convert("terminal_current", I_devsim, "out_of_device+")  # -> passive sign
    tag   = cv.tag("stress")                                            # 'stress:tensile+'

Axes with a canonical value and the alternatives it converts from:
    stress            tensile+            | compressive+                      (Quantum ESPRESSO, CHGNet)
    strain_shear      engineering         | tensor                            (factor 2 on Voigt shear slots)
    voigt_order       xx,yy,zz,yz,xz,xy   | xx,yy,zz,xy,xz,yz                 (LAMMPS order)
    terminal_current  into_device+        | out_of_device+
    band_offset       B_minus_A           | A_minus_B
    doping            ND-NA               | NA-ND
    trap_level        Et-Ei               | Ec-Et, Et-Ev                      (needs Eg and Ei - Ev)
    kpoints           crystal             | tpiba                             (needs the cell and alat)
    energy_zero       vbm | vacuum | fermi, shifted by a stated reference energy
    dose              Gy(<material>)      | rad(<material>); untagged dose is refused
    per_cell          per_cell            | per_atom                          (needs the atom count)
Frames: rotations between the crystal frame and the device frame (x lateral, y depth, z out of plane) for
vectors, rank-2 tensors and 6x6 Voigt stiffness matrices.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

VERSION = "1.0"

CANONICAL: Dict[str, str] = {
    "stress": "tensile+",
    "strain_shear": "engineering",
    "voigt_order": "xx,yy,zz,yz,xz,xy",
    "terminal_current": "into_device+",
    "band_offset": "B_minus_A",
    "doping": "ND-NA",
    "trap_level": "Et-Ei",
    "kpoints": "crystal",
    "energy_zero": "vbm",
    "dose": "Gy",
    "per_cell": "per_cell",
}


class ConventionError(ValueError):
    pass


def tag(axis: str, value: str = None) -> str:
    return f"{axis}:{value or CANONICAL[axis]}"


def tags(*axes) -> Dict[str, str]:
    return {a: CANONICAL[a] for a in axes}


# ---------------------------------------------------------------------------------------------------- converters
def _sign(x):
    return -np.asarray(x, dtype=float)




def _perm(order_from: str, order_to: str):
    a, b = order_from.split(","), order_to.split(",")
    norm = lambda s: "".join(sorted(s))
    ia = [norm(x) for x in a]
    return [ia.index(norm(x)) for x in b]


def _voigt_perm(x, frm, to):
    x = np.asarray(x, dtype=float)
    p = _perm(frm, to)
    if x.shape[-1] == 6 and x.ndim == 1:
        return x[p]
    if x.shape == (6, 6):
        return x[np.ix_(p, p)]
    raise ConventionError(f"Voigt reorder needs a 6-vector or 6x6 matrix, got shape {x.shape}")


def _shear(x, frm, to):
    x = np.array(x, dtype=float)
    f = {("tensor", "engineering"): 2.0, ("engineering", "tensor"): 0.5}[(frm, to)]
    x[..., 3:6] *= f
    return x


def _trap(x, frm, to, Eg=None, Ei_minus_Ev=None):
    """Trap level conversions. Ei_minus_Ev: intrinsic level above the valence band (eV)."""
    if Eg is None or Ei_minus_Ev is None:
        raise ConventionError("trap-level conversion needs Eg and Ei_minus_Ev")
    x = np.asarray(x, dtype=float)
    # first to Et - Ev
    if frm == "Et-Ei":
        et_ev = x + Ei_minus_Ev
    elif frm == "Ec-Et":
        et_ev = Eg - x
    elif frm == "Et-Ev":
        et_ev = x
    else:
        raise ConventionError(frm)
    return {"Et-Ei": et_ev - Ei_minus_Ev, "Ec-Et": Eg - et_ev, "Et-Ev": et_ev}[to]


def _kpoints(k, frm, to, cell=None, alat=None):
    """crystal (reduced) <-> tpiba (cartesian, units of 2 pi / alat). cell: 3x3 rows in the same length unit as alat."""
    if cell is None or alat is None:
        raise ConventionError("k-point conversion needs the cell and alat")
    k = np.atleast_2d(np.asarray(k, dtype=float))
    recip = np.linalg.inv(np.asarray(cell, dtype=float)).T * alat      # rows: b_i in 2 pi / alat
    if (frm, to) == ("crystal", "tpiba"):
        return k @ recip
    if (frm, to) == ("tpiba", "crystal"):
        return k @ np.linalg.inv(recip)
    raise ConventionError((frm, to))


def _energy_zero(E, frm, to, reference=None):
    """Shift energies between references. `reference` = energy of the `to` zero measured on the `frm` scale."""
    if reference is None:
        raise ConventionError("energy-zero conversion needs the reference energy on the source scale")
    return np.asarray(E, dtype=float) - reference


def _dose(D, frm, to):
    def parse(t):
        if "(" not in t:
            raise ConventionError(f"dose must name its material, e.g. Gy(SiO2); got '{t}'")
        unit, mat = t.split("(", 1)
        return unit, mat.rstrip(")")
    uf, mf = parse(frm)
    ut, mt = parse(to)
    if mf != mt:
        raise ConventionError(f"dose in {mf} cannot be converted to dose in {mt} without a stopping-power ratio")
    f = {("rad", "Gy"): 0.01, ("Gy", "rad"): 100.0, ("Gy", "Gy"): 1.0, ("rad", "rad"): 1.0}[(uf, ut)]
    return np.asarray(D, dtype=float) * f


def _per_cell(x, frm, to, n_atoms=None):
    if not n_atoms:
        raise ConventionError("per-atom conversion needs the atom count")
    x = np.asarray(x, dtype=float)
    return x * n_atoms if (frm, to) == ("per_atom", "per_cell") else x / n_atoms


_CONVERTERS: Dict[str, Callable] = {
    "stress": lambda x, f, t, **k: _sign(x) if f != t else np.asarray(x, float),
    "terminal_current": lambda x, f, t, **k: _sign(x) if f != t else np.asarray(x, float),
    "band_offset": lambda x, f, t, **k: _sign(x) if f != t else np.asarray(x, float),
    "doping": lambda x, f, t, **k: _sign(x) if f != t else np.asarray(x, float),
    "strain_shear": lambda x, f, t, **k: _shear(x, f, t) if f != t else np.asarray(x, float),
    "voigt_order": lambda x, f, t, **k: _voigt_perm(x, f, t),
    "trap_level": lambda x, f, t, **k: _trap(x, f, t, **k),
    "kpoints": lambda x, f, t, **k: _kpoints(x, f, t, **k) if f != t else np.asarray(x, float),
    "energy_zero": lambda x, f, t, **k: _energy_zero(x, f, t, **k) if f != t else np.asarray(x, float),
    "dose": lambda x, f, t, **k: _dose(x, f, t),
    "per_cell": lambda x, f, t, **k: _per_cell(x, f, t, **k) if f != t else np.asarray(x, float),
}

ALTERNATIVES: Dict[str, Tuple[str, ...]] = {
    "stress": ("tensile+", "compressive+"),
    "terminal_current": ("into_device+", "out_of_device+"),
    "band_offset": ("B_minus_A", "A_minus_B"),
    "doping": ("ND-NA", "NA-ND"),
    "strain_shear": ("engineering", "tensor"),
    "voigt_order": ("xx,yy,zz,yz,xz,xy", "xx,yy,zz,xy,xz,yz"),
    "trap_level": ("Et-Ei", "Ec-Et", "Et-Ev"),
    "kpoints": ("crystal", "tpiba"),
    "per_cell": ("per_cell", "per_atom"),
}


def convert(axis: str, value, frm: str, to: str = None, **ctx):
    """Convert `value` from convention `frm` to `to` (default: canonical) on the given axis."""
    if axis not in _CONVERTERS:
        raise ConventionError(f"unknown convention axis '{axis}'")
    to = to or CANONICAL[axis]
    if axis == "dose" and to == "Gy":
        mat = frm.split("(", 1)[1].rstrip(")") if "(" in frm else None
        if mat is None:
            raise ConventionError("dose must name its material, e.g. rad(SiO2)")
        to = f"Gy({mat})"
    alts = ALTERNATIVES.get(axis)
    if alts and (frm not in alts or to not in alts):
        raise ConventionError(f"{axis}: unknown convention {frm!r} or {to!r}; known: {alts}")
    return _CONVERTERS[axis](value, frm, to, **ctx)


def to_canonical(axis: str, value, native: str, **ctx):
    return convert(axis, value, native, None, **ctx)


def from_canonical(axis: str, value, native: str, **ctx):
    canonical = CANONICAL[axis]
    if axis == "dose":
        canonical = "Gy(" + native.split("(", 1)[1]
    if axis == "energy_zero":
        ref = ctx.pop("reference")
        return np.asarray(value, dtype=float) + ref
    return convert(axis, value, canonical, native, **ctx)


# ---------------------------------------------------------------------------------------------------- frames
def rotation_from_wafer(normal_crystal, flat_crystal) -> np.ndarray:
    """Rotation R with v_device = R @ v_crystal. Device axes: y = depth into the wafer (= -normal, since the normal
    points out of the top surface), x = along the wafer flat (lateral), z = x cross y (out of the 2D plane)."""
    n = np.asarray(normal_crystal, float); n /= np.linalg.norm(n)
    f = np.asarray(flat_crystal, float); f = f - np.dot(f, n) * n
    if np.linalg.norm(f) < 1e-12:
        raise ConventionError("wafer flat must not be parallel to the normal")
    x = f / np.linalg.norm(f)
    y = -n
    z = np.cross(x, y)
    return np.vstack([x, y, z])


def rotate_vector(R, v):
    return np.asarray(R) @ np.asarray(v, float)


def rotate_tensor(R, T):
    R = np.asarray(R, float)
    return R @ np.asarray(T, float) @ R.T


def bond_matrix(R) -> np.ndarray:
    """6x6 Bond matrix M so that C' = M C M^T for stiffness in Voigt notation (order xx,yy,zz,yz,xz,xy)."""
    a = np.asarray(R, float)
    M = np.zeros((6, 6))
    idx = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    for I, (i, j) in enumerate(idx):
        for J, (k, l) in enumerate(idx):
            if J < 3:
                M[I, J] = a[i, k] * a[j, l]
            else:
                M[I, J] = a[i, k] * a[j, l] + a[i, l] * a[j, k]
    return M


def rotate_stiffness(R, C):
    M = bond_matrix(R)
    return M @ np.asarray(C, float) @ M.T


def voigt_to_tensor(v, engineering=True):
    v = np.asarray(v, float)
    s = 0.5 if engineering else 1.0
    return np.array([[v[0], s * v[5], s * v[4]], [s * v[5], v[1], s * v[3]], [s * v[4], s * v[3], v[2]]])


def tensor_to_voigt(T, engineering=True):
    T = np.asarray(T, float)
    s = 2.0 if engineering else 1.0
    return np.array([T[0, 0], T[1, 1], T[2, 2], s * T[1, 2], s * T[0, 2], s * T[0, 1]])
