"""Diagnose the scikit-fem vs DOLFINx heat difference of the 0.9c FEM pair (run from ~/sim_workspace, tcad env):
    PYTHONPATH=. ~/miniforge3/envs/tcad/bin/python tools/fem_pair_debug.py
Solves the heat problem with k(T) exponent a = 0 (linear) and a = 1.3 in both engines, prints T_max and the Picard
history, and checks which quadrature rule DOLFINx used. Read-only: writes only under runs/fem_debug."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from simws.validation import engine_pairs as E

work = Path("runs/fem_debug").resolve(); work.mkdir(parents=True, exist_ok=True)
msh = E._fem_geometry().mesh(work / "fem2.msh", size=0.05)
py = E._env_python("fenicsx")
bindir = Path(py).parent
env = dict(os.environ, PYTHONPATH=str(Path.cwd()), PATH=f"{bindir}:{os.environ['PATH']}")
cc = bindir / "x86_64-conda-linux-gnu-cc"
if cc.exists():
    env["CC"], env["LDSHARED"] = str(cc), f"{cc} -shared"

DRIVER = r'''
import json, sys, numpy as np
import simws.validation.engine_pairs as E
a = float(sys.argv[3])
P = E.FEM_PROBLEM; P["heat"]["a"] = a
import simws.fem.dolfinx_engine as D
D._problem = lambda: P
r = D.solve(sys.argv[1])
import basix, ffcx, dolfinx
q = basix.make_quadrature(basix.CellType.triangle, 1)
r["quad_deg1_points"] = np.asarray(q[0]).tolist(); r["versions"] = [dolfinx.__version__, basix.__version__, ffcx.__version__]
json.dump(r, open(sys.argv[2], "w"))
'''
(work / "driver.py").write_text(DRIVER)
for a in (0.0, 1.3):
    P = copy.deepcopy(E.FEM_PROBLEM); P["heat"]["a"] = a
    s = E.skfem_solve(msh, P)
    out = work / f"dolfinx_a{a}.json"
    p = subprocess.run([py, str(work / "driver.py"), str(msh), str(out), str(a)], env=env, cwd=work,
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-3000:]); sys.exit(1)
    d = json.loads(out.read_text())
    ka = {(round(x, 9), round(y, 9)): i for i, (x, y) in enumerate(zip(s["x"], s["y"]))}
    idx = np.array([ka[(round(x, 9), round(y, 9))] for x, y in zip(d["x"], d["y"])])
    Ts, Td = np.asarray(s["T"])[idx], np.asarray(d["T"])
    print(f"a = {a}: T_max skfem {Ts.max():.10f}  dolfinx {Td.max():.10f}  rel {Td.max() / Ts.max() - 1:.3e}  "
          f"max nodal |dT| {np.abs(Ts - Td).max():.3e} K  dolfinx Picard its {d['picard_iterations']}")
print("dolfinx/basix/ffcx", d["versions"], " degree-1 rule points", d["quad_deg1_points"])
