"""Two-way electro-thermal coupling through preCICE (0.9b): DEVSIM and scikit-fem as separate processes on their own
meshes.

    Electrical  (DEVSIM, the resistor of simws.coupling.electrothermal)  writes Heat  (W/cm per node: an integral
                quantity), reads Temperature
    Thermal     (scikit-fem P1 heat equation on its own gmsh mesh)       reads Heat, writes Temperature

Mappings (both defined on the Thermal mesh's triangles, which preCICE receives as connectivity):
    Heat         Electrical -> Thermal  linear-cell-interpolation, constraint conservative: each DEVSIM node's power
                 is split over the vertices of the thermal triangle that contains it by its barycentric weights, which
                 sum to one, so the total power is conserved to round-off
    Temperature  Thermal -> Electrical  linear-cell-interpolation, constraint consistent: exact for the P1 field

Coupling: serial-implicit, one time window per bias (steady state), Aitken acceleration on Temperature, converged
when max|dT| < tol_K. Each window records the power written, the power read and the heat reaching the sink, so the
transfer's conservation and the thermal energy balance are part of every result.

The same-mesh case (thermal mesh = DEVSIM's mesh) must reproduce the in-process loop of
simws.coupling.electrothermal to the iteration tolerance: that is the bridge's own check.

preCICE runs from a scratch directory (`work`); the participants talk through sockets with an exchange directory
inside it. LD_LIBRARY_PATH must reach libprecice (conda installs it into the env, so nothing is needed there).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CONFIG = """<?xml version="1.0" encoding="UTF-8" ?>
<precice-configuration>
  <log enabled="false" />
  <data:scalar name="Heat" />
  <data:scalar name="Temperature" />
  <mesh name="ElecMesh" dimensions="2">
    <use-data name="Heat" />
    <use-data name="Temperature" />
  </mesh>
  <mesh name="ThermMesh" dimensions="2">
    <use-data name="Heat" />
    <use-data name="Temperature" />
  </mesh>
  <participant name="Electrical">
    <provide-mesh name="ElecMesh" />
    <receive-mesh name="ThermMesh" from="Thermal" />
    <write-data name="Heat" mesh="ElecMesh" />
    <read-data name="Temperature" mesh="ElecMesh" />
    <mapping:linear-cell-interpolation direction="write" from="ElecMesh" to="ThermMesh" constraint="conservative" />
    <mapping:linear-cell-interpolation direction="read" from="ThermMesh" to="ElecMesh" constraint="consistent" />
  </participant>
  <participant name="Thermal">
    <provide-mesh name="ThermMesh" />
    <read-data name="Heat" mesh="ThermMesh" />
    <write-data name="Temperature" mesh="ThermMesh" />
  </participant>
  <m2n:sockets acceptor="Electrical" connector="Thermal" exchange-directory="{exchange}" />
  <coupling-scheme:serial-implicit>
    <participants first="Electrical" second="Thermal" />
    <max-time-windows value="{n_windows}" />
    <time-window-size value="1.0" />
    <max-iterations value="{max_iter}" />
    <exchange data="Heat" mesh="ThermMesh" from="Electrical" to="Thermal" />
    <exchange data="Temperature" mesh="ThermMesh" from="Thermal" to="Electrical" />
    <absolute-convergence-measure limit="{tol_K}" data="Temperature" mesh="ThermMesh" />
    <acceleration:aitken>
      <data name="Temperature" mesh="ThermMesh" />
      <initial-relaxation value="0.5" />
    </acceleration:aitken>
  </coupling-scheme:serial-implicit>
</precice-configuration>
"""


def write_config(work: Path, n_windows: int, tol_K: float, max_iter: int) -> Path:
    work = Path(work)
    (work / "exchange").mkdir(parents=True, exist_ok=True)
    p = work / "precice-config.xml"
    # preCICE's absolute measure is the 2-norm over the mesh; scale the per-node tolerance accordingly by the caller
    p.write_text(CONFIG.format(exchange=str(work / "exchange"), n_windows=n_windows, tol_K=tol_K, max_iter=max_iter))
    return p


# ------------------------------------------------------------------------------------------------- participants
def _emit(work, name, rows):
    (Path(work) / f"{name}.json").write_text(json.dumps(rows, default=float))


def _electrical(work, case_kw):
    os.chdir(work)
    import precice
    from simws.coupling.electrothermal import ResistorCase, build_electrical
    from simws.materials import load
    case = ResistorCase(**case_kw)
    mat = load("silicon")
    e = build_electrical(case, mat)
    part = precice.Participant("Electrical", "precice-config.xml", 0, 1)
    pos = np.column_stack([e.px, e.py])
    vid = part.set_mesh_vertices("ElecMesh", pos)
    part.initialize()
    T = np.full(e.n_nodes, case.T_sink)
    rows, v_prev, k = [], 0.0, 0
    biases = list(case.biases)
    iters = 0
    while part.is_coupling_ongoing():
        if part.requires_writing_checkpoint():
            iters = 0
        V = biases[min(k, len(biases) - 1)]
        if iters == 0 and V != v_prev:
            for vv in np.linspace(v_prev, V, max(2, int(abs(V - v_prev) / 0.25) + 1))[1:]:
                e.set_bias(vv); e.solve()
            v_prev = V
        dt = part.get_max_time_step_size()
        T = np.asarray(part.read_data("ElecMesh", "Temperature", vid, dt), float) if (k or iters) else T
        if not np.all(T > 0):                     # before the first exchange preCICE hands back zeros
            T = np.full(e.n_nodes, case.T_sink)
        e.set_T(T); e.solve()
        P = e.nodal_power()
        part.write_data("ElecMesh", "Heat", vid, P)
        part.advance(dt)
        iters += 1
        if part.requires_reading_checkpoint():
            continue
        if part.is_time_window_complete():
            I = abs(e.contact_current())
            rows.append({"bias_V": V, "current_A_per_cm": I, "power_written_W_per_cm": float(P.sum()),
                         "IV_W_per_cm": I * V, "T_max_elec_K": float(T.max()), "coupling_iterations": iters})
            k += 1
            iters = 0
    part.finalize()
    _emit(work, "Electrical", rows)


def _thermal(work, case_kw, mesh_spec):
    os.chdir(work)
    import precice
    from simws.coupling.electrothermal import ResistorCase, thermal_solver
    from simws.materials import load
    case = ResistorCase(**case_kw)
    mat = load("silicon")
    mesh = _thermal_mesh(case, mesh_spec, work)
    p = mesh.p
    sink = np.where(np.isclose(p[0], case.length_um * 1e-4, rtol=0, atol=1e-12))[0]
    thermal = thermal_solver(mesh, sink, case, mat)
    part = precice.Participant("Thermal", "precice-config.xml", 0, 1)
    vid = part.set_mesh_vertices("ThermMesh", p.T.copy())
    for a, b, c in vid[mesh.t.T]:          # pyprecice 3.2's set_mesh_triangles checks [N x dim], wrong in 2D
        part.set_mesh_triangle("ThermMesh", int(a), int(b), int(c))
    part.initialize()
    T = np.full(p.shape[1], case.T_sink)
    rows = []
    while part.is_coupling_ongoing():
        if part.requires_writing_checkpoint():
            pass
        dt = part.get_max_time_step_size()
        Q = np.asarray(part.read_data("ThermMesh", "Heat", vid, dt), float)
        T, q_sink = thermal(Q, T)
        part.write_data("ThermMesh", "Temperature", vid, T)
        part.advance(dt)
        if part.requires_reading_checkpoint():
            continue
        if part.is_time_window_complete():
            rows.append({"power_read_W_per_cm": float(Q.sum()), "heat_to_sink_W_per_cm": q_sink,
                         "T_max_K": float(T.max()), "n_thermal_nodes": int(p.shape[1])})
    part.finalize()
    _emit(work, "Thermal", rows)


def _thermal_mesh(case, spec, work):
    """'devsim' = the electrical mesh's own nodes/triangles (same-mesh check); else a gmsh mesh of the same rectangle:
    {'kind': 'gmsh', 'size': um, 'structured': bool}."""
    import skfem
    if spec == "devsim":
        from simws.coupling.electrothermal import build_electrical
        from simws.materials import load
        e = build_electrical(case, load("silicon"))
        return e.mesh
    from simws.geometry import Geometry
    g = Geometry("thermal").rect("si", "Silicon", 0, case.length_um, 0, case.height_um)
    path = g.mesh(Path(work) / "thermal.msh", size=spec.get("size", 0.1), structured=spec.get("structured", False))
    mesh, _, _ = g.to_skfem(path)
    return skfem.MeshTri(mesh.p, mesh.t)


def run(case=None, thermal_mesh="devsim", work=None, tol_K=1e-6, max_iter=80, timeout=1800) -> dict:
    """Run both participants in separate processes. Returns per-bias rows with the conservation record."""
    from simws.coupling.electrothermal import ResistorCase
    case = case or ResistorCase(biases=[1.0, 5.0])
    case_kw = {k: v for k, v in case.__dict__.items()}
    import tempfile
    work = Path(work or tempfile.mkdtemp(prefix="simws_precice_")).resolve()   # participants run with cwd=work
    work.mkdir(parents=True, exist_ok=True)
    write_config(work, len(case.biases), tol_K, max_iter)
    # each participant is its own interpreter (python -m simws.coupling.precice_bridge <role> ...), so the caller's
    # __main__ is never re-imported and a crash in one is reported with its log
    repo = str(Path(__file__).resolve().parents[2])
    envv = dict(os.environ)
    envv["PYTHONPATH"] = repo + (os.pathsep + envv["PYTHONPATH"] if envv.get("PYTHONPATH") else "")
    args = {"Electrical": [json.dumps(case_kw)], "Thermal": [json.dumps(case_kw), json.dumps(thermal_mesh)]}
    procs, logs = {}, {}
    t0 = time.time()
    for name, extra in args.items():
        (work / f"{name}.json").unlink(missing_ok=True)
        logs[name] = open(work / f"{name}.log", "w")
        procs[name] = subprocess.Popen([sys.executable, "-m", "simws.coupling.precice_bridge", name, str(work), *extra],
                                       stdout=logs[name], stderr=subprocess.STDOUT, env=envv, cwd=str(work))
    try:
        while any(p.poll() is None for p in procs.values()):
            bad = [n for n, p in procs.items() if p.poll() not in (None, 0)]
            if bad or time.time() - t0 > timeout:
                why = f"participant {bad[0]} exited {procs[bad[0]].returncode}" if bad else "timeout"
                tail = "".join((work / f"{n}.log").read_text()[-1500:] for n in (bad or procs))
                raise RuntimeError(f"preCICE run failed ({why}); logs in {work}:\n{tail}")
            time.sleep(0.2)
        bad = [n for n, p in procs.items() if p.returncode != 0]
        if bad:
            raise RuntimeError(f"preCICE participant {bad[0]} exited {procs[bad[0]].returncode}; log "
                               f"{work / (bad[0] + '.log')}:\n{(work / (bad[0] + '.log')).read_text()[-1500:]}")
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.kill()
        for f in logs.values():
            f.close()
    got = {n: json.loads((work / f"{n}.json").read_text()) for n in procs}
    out = []
    for er, tr in zip(got["Electrical"], got["Thermal"]):
        w, r = er["power_written_W_per_cm"], tr["power_read_W_per_cm"]
        out.append({**er, **tr,
                    "transfer_conservation_rel_error": abs(r - w) / max(abs(w), 1e-300),
                    "energy_balance_rel_error": abs(tr["heat_to_sink_W_per_cm"] - r) / max(abs(r), 1e-300),
                    "joule_vs_IV_rel_error": abs(w - er["IV_W_per_cm"]) / max(er["IV_W_per_cm"], 1e-300)})
    res = {"thermal_mesh": thermal_mesh, "points": out, "wall_s": time.time() - t0, "work": str(work),
           "vt_isothermal": True}
    (work / "result.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    role, wd = sys.argv[1], sys.argv[2]
    if role == "Electrical":
        _electrical(wd, json.loads(sys.argv[3]))
    elif role == "Thermal":
        _thermal(wd, json.loads(sys.argv[3]), json.loads(sys.argv[4]))
    else:
        raise SystemExit(f"unknown participant {role}")
