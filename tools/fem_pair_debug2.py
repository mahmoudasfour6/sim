"""Second FEM-pair diagnostic: same temperature field in both engines, compare k(T) cell by cell and the assembled
conduction energy  E = sum_cells k |grad T|^2 A  (run from ~/sim_workspace, tcad env):
    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/fem_pair_debug2.py"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import skfem

from simws.validation import engine_pairs as E

work = Path("runs/fem_debug").resolve(); work.mkdir(parents=True, exist_ok=True)
msh = E._fem_geometry().mesh(work / "fem2.msh", size=0.05)
s = E.skfem_solve(msh)
H = E.FEM_PROBLEM["heat"]
mesh, sub, _ = E._fem_geometry().to_skfem(msh)
mesh = skfem.MeshTri(mesh.p / 1e-4, mesh.t)
T = np.asarray(s["T"])
reg = np.zeros(mesh.t.shape[1], int); reg[sub["ox"]] = 1
Tc = T[mesh.t].mean(axis=0)
k_s = np.where(reg == 0, H["k300"] * (300.0 / Tc) ** H["a"], H["k_ox"])
basis = skfem.Basis(mesh, skfem.ElementTriP1(), intorder=1)
g = basis.interpolate(T).grad                       # (2, nel, nqp)
A = basis.dx.sum(axis=1)
E_s = float(np.sum(k_s * (g[0, :, 0] ** 2 + g[1, :, 0] ** 2) * A))
cent = mesh.p[:, mesh.t].mean(axis=1)
np.savez(work / "skfem_T.npz", x=mesh.p[0], y=mesh.p[1], T=T)
DRIVER = r'''
import json, sys, numpy as np
import gmsh, ufl, dolfinx
from dolfinx import fem
try:
    from dolfinx.io import gmshio
except ImportError:
    import dolfinx.io.gmsh as gmshio
from mpi4py import MPI
from simws.validation.engine_pairs import FEM_PROBLEM
H = FEM_PROBLEM["heat"]
gmsh.initialize(["-noenv"]); gmsh.option.setNumber("General.Terminal", 0); gmsh.open(sys.argv[1])
names = {gmsh.model.getPhysicalName(d, t): (d, t) for d, t in gmsh.model.getPhysicalGroups()}
out = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2); gmsh.finalize()
mesh, ct = (out[0], out[1]) if isinstance(out, tuple) else (out.mesh, out.cell_tags)
mesh.geometry.x[:, :2] *= 1e4
V = fem.functionspace(mesh, ("Lagrange", 1)); Q = fem.functionspace(mesh, ("DG", 0))
d = np.load(sys.argv[2]); key = {(round(a, 9), round(b, 9)): v for a, b, v in zip(d["x"], d["y"], d["T"])}
X = V.tabulate_dof_coordinates()
Tf = fem.Function(V); Tf.x.array[:] = [key[(round(a, 9), round(b, 9))] for a, b in X[:, :2]]
is_si = fem.Function(Q); is_si.x.array[:] = 1.0
is_si.x.array[ct.indices[ct.values == names["ox"][1]]] = 0.0
kT = is_si * H["k300"] * (300.0 / Tf) ** H["a"] + (1 - is_si) * H["k_ox"]
kq = fem.Function(Q); kq.interpolate(fem.Expression(kT, Q.element.interpolation_points))
dx1 = ufl.dx(metadata={"quadrature_degree": 1})
E_d = fem.assemble_scalar(fem.form(kT * ufl.inner(ufl.grad(Tf), ufl.grad(Tf)) * dx1))
E_dd = fem.assemble_scalar(fem.form(kT * ufl.inner(ufl.grad(Tf), ufl.grad(Tf)) * ufl.dx))
cent = dolfinx.mesh.compute_midpoints(mesh, 2, np.arange(mesh.topology.index_map(2).size_local, dtype=np.int32))
json.dump({"E_deg1": float(E_d), "E_default": float(E_dd), "k": kq.x.array.tolist(),
           "cx": cent[:, 0].tolist(), "cy": cent[:, 1].tolist(),
           "ox_cells": int((ct.values == names["ox"][1]).sum()), "n_cells": int(len(kq.x.array))}, open(sys.argv[3], "w"))
'''
(work / "driver2.py").write_text(DRIVER)
py = E._env_python("fenicsx"); bindir = Path(py).parent
env = dict(os.environ, PYTHONPATH=str(Path.cwd()), PATH=f"{bindir}:{os.environ['PATH']}")
cc = bindir / "x86_64-conda-linux-gnu-cc"
if cc.exists():
    env["CC"], env["LDSHARED"] = str(cc), f"{cc} -shared"
p = subprocess.run([py, str(work / "driver2.py"), str(msh), str(work / "skfem_T.npz"), str(work / "d2.json")],
                   env=env, cwd=work, capture_output=True, text=True)
if p.returncode:
    print(p.stdout[-2000:], p.stderr[-3000:]); sys.exit(1)
r = json.loads((work / "d2.json").read_text())
ck = {(round(a, 6), round(b, 6)): i for i, (a, b) in enumerate(zip(cent[0], cent[1]))}
idx = np.array([ck[(round(a, 6), round(b, 6))] for a, b in zip(r["cx"], r["cy"])])
k_d = np.asarray(r["k"])
rel = np.abs(k_d / k_s[idx] - 1)
print(f"cells: skfem {len(k_s)} (ox {int(reg.sum())}), dolfinx {r['n_cells']} (ox {r['ox_cells']})")
print(f"k per cell, same T field: max rel diff {rel.max():.3e}  (si cells {np.abs(rel[reg[idx] == 0]).max():.3e},"
      f" ox cells {np.abs(rel[reg[idx] == 1]).max():.3e})")
print(f"conduction energy: skfem {E_s:.12e}  dolfinx deg1 {r['E_deg1']:.12e}  dolfinx default {r['E_default']:.12e}")
print(f"  rel deg1 {r['E_deg1'] / E_s - 1:.3e}  rel default {r['E_default'] / E_s - 1:.3e}")
