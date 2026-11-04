"""Kwant transmission through a 1D barrier: the independent engine for simws.validation.contact_transport_checks.
Same chain as contact_transport_checks.tb_transmission: spacing a (cm), onsite U + 2t, hopping -t, both leads at the
last site's potential."""
from __future__ import annotations

import numpy as np

HBAR = 1.054571817e-34
Q_E = 1.602176634e-19


def _import_kwant():
    """conda-forge Kwant links MUMPS built against MPICH: MPI must be initialised before Kwant loads, or the import
    aborts ('MPI routine before initializing'). Importing mpi4py.MPI initialises it."""
    try:
        from mpi4py import MPI  # noqa: F401
    except ImportError:
        pass
    import kwant
    return kwant


def transmission(x, U, E, m, a=1e-9):
    kwant = _import_kwant()
    xs = np.arange(x[0], x[-1], a)
    Us = np.interp(xs, x, U)
    t = (HBAR ** 2 / (2 * m * (a * 1e-2) ** 2)) / Q_E
    lat = kwant.lattice.chain(norbs=1)
    syst = kwant.Builder()
    for i, u in enumerate(Us):
        syst[lat(i)] = u + 2 * t
    syst[lat.neighbors()] = -t
    UL = Us[-1]
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead[lat(0)] = UL + 2 * t
    lead[lat.neighbors()] = -t
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    f = syst.finalized()
    out = []
    for e in np.atleast_1d(E):
        if e <= UL or e >= UL + 4 * t:
            out.append(0.0)
            continue
        out.append(float(kwant.smatrix(f, float(e)).transmission(1, 0)))
    return np.array(out)
