"""DOLFINx side of the 0.9c FEM engine pair (simws.validation.engine_pairs.FEM_PROBLEM), run in the fenicsx env:

    python -m simws.fem.dolfinx_engine mesh.msh out.json

Reads the same gmsh .msh as scikit-fem (coordinates in cm, solved in um), solves the two reference problems with P1
Lagrange elements and writes the nodal solutions with their coordinates. The nonlinear heat problem uses the same
Picard iteration and the same one-point quadrature as the scikit-fem side, so the two discrete problems are identical.
Works with DOLFINx 0.8-0.11 (0.10 renamed gmshio to dolfinx.io.gmsh and returns a MeshData object).
"""
from __future__ import annotations

import json
import sys

import numpy as np


def _problem():
    # kept in sync with simws.validation.engine_pairs.FEM_PROBLEM (imported when available)
    try:
        from simws.validation.engine_pairs import FEM_PROBLEM
        return FEM_PROBLEM
    except Exception:  # noqa: BLE001
        return {"poisson": {"k": {"si": 11.7, "ox": 3.9}, "f": {"si": 50.0, "ox": 0.0}},
                "heat": {"k300": 1.48, "a": 1.3, "q": {"si": 400.0, "ox": 0.0}, "k_ox": 0.014, "T0": 300.0}}


def solve(msh_path):
    import gmsh
    import ufl
    import dolfinx
    from dolfinx import fem
    from dolfinx.fem.petsc import LinearProblem
    try:                                            # DOLFINx 0.10 renamed dolfinx.io.gmshio to dolfinx.io.gmsh
        from dolfinx.io import gmshio
    except ImportError:
        import dolfinx.io.gmsh as gmshio
    from mpi4py import MPI
    P = _problem()
    gmsh.initialize(["-noenv"])
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(str(msh_path))
    names = {gmsh.model.getPhysicalName(d, t): (d, t) for d, t in gmsh.model.getPhysicalGroups()}
    out = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    if isinstance(out, tuple):
        mesh, cell_tags, facet_tags = out[0], out[1], out[2]
    else:                                           # DOLFINx 0.10 MeshData
        mesh, cell_tags, facet_tags = out.mesh, out.cell_tags, out.facet_tags
    mesh.geometry.x[:, :2] *= 1e4                   # cm -> um
    V = fem.functionspace(mesh, ("Lagrange", 1))
    Q = fem.functionspace(mesh, ("DG", 0))

    def piecewise(vals):
        f = fem.Function(Q)
        f.x.array[:] = vals["si"]
        f.x.array[cell_tags.indices[cell_tags.values == names["ox"][1]]] = vals["ox"]
        return f

    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)

    def dofs(name):
        return fem.locate_dofs_topological(V, fdim, facet_tags.find(names[name][1]))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    k, f = piecewise(P["poisson"]["k"]), piecewise(P["poisson"]["f"])
    bcs = [fem.dirichletbc(dolfinx.default_scalar_type(0.0), dofs("bottom"), V),
           fem.dirichletbc(dolfinx.default_scalar_type(1.0), dofs("top"), V)]
    lin = dict(petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    try:
        prob = LinearProblem(k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, f * v * ufl.dx, bcs=bcs,
                             petsc_options_prefix="poisson_", **lin)
    except TypeError:                               # DOLFINx < 0.10 has no petsc_options_prefix
        prob = LinearProblem(k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, f * v * ufl.dx, bcs=bcs, **lin)
    uh = prob.solve()
    if isinstance(uh, tuple):
        uh = uh[0]
    H = P["heat"]
    dx1 = ufl.dx(metadata={"quadrature_degree": 1})
    Tprev = fem.Function(V); Tprev.x.array[:] = H["T0"]
    is_si = piecewise({"si": 1.0, "ox": 0.0})
    q = piecewise(H["q"])
    kT = is_si * H["k300"] * (300.0 / Tprev) ** H["a"] + (1 - is_si) * H["k_ox"]
    bcT = [fem.dirichletbc(dolfinx.default_scalar_type(H["T0"]), dofs("bottom"), V)]
    a_form, L_form = kT * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx1, q * v * dx1
    for it in range(100):
        try:
            pr = LinearProblem(a_form, L_form, bcs=bcT, petsc_options_prefix=f"heat{it}_", **lin)
        except TypeError:
            pr = LinearProblem(a_form, L_form, bcs=bcT, **lin)
        Tn = pr.solve()
        if isinstance(Tn, tuple):
            Tn = Tn[0]
        d = float(np.max(np.abs(Tn.x.array - Tprev.x.array)))
        Tprev.x.array[:] = Tn.x.array
        if d < 1e-12 * H["T0"]:
            break
    x = V.tabulate_dof_coordinates()
    return {"x": x[:, 0].tolist(), "y": x[:, 1].tolist(), "u": uh.x.array.real.tolist(),
            "T": Tprev.x.array.real.tolist(), "picard_iterations": it + 1, "version": dolfinx.__version__}


if __name__ == "__main__":
    res = solve(sys.argv[1])
    with open(sys.argv[2], "w") as fh:
        json.dump(res, fh)
