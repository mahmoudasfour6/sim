"""FiPy anneal engine (architecture v1, contract A16.3): dC/dt = d/dx (K(C) dC/dx) with NIST's FiPy.

Cells of width dx (centres at (k + 1/2) dx) on a Grid1D, DiffusionTerm with K at the faces (FiPy's face
interpolation of the cell values of K), TransientTerm (backward Euler) on the description's time points, sweeps
until the largest relative change is below 1e-10. A fixed surface is a constraint on the left face. Reported at the
faces, x_k = k dx, so both anneal engines report the same points.

J. E. Guyer, D. Wheeler, J. A. Warren, "FiPy: Partial Differential Equations with Python", Computing in Science &
Engineering 11(3), 6-15 (2009).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from simws.engines.base import AnnealCapability, AnnealResult, EngineNotAvailable, Unsupported

FAMILY = "anneal"
CODE = "fipy"
NAME = "fipy"
SWEEP_TOL = 1e-10
MAX_SWEEPS = 50

CAPABILITY = AnnealCapability(
    engine=NAME, env="tcad", in_process=True,
    description="One dopant diffusing in 1D silicon on FiPy (cell-centred finite volume, backward Euler, sweeps): "
                "intrinsic or Fermi-level diffusivity of Plummer, Deal and Griffin",
    dopants=["B", "P", "As", "Sb"], diffusivity=["intrinsic", "fermi"], surfaces=["reflecting", "fixed"],
    tested_temperature_C=(800.0, 1200.0),
    notes="Isothermal steps only; one dopant on a uniform background; no damage (TED), clustering, solubility, "
          "oxidation or segregation (A16.2).")


def available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("fipy") is not None
    except Exception:  # noqa: BLE001
        return False


def unsupported(problem) -> List[str]:
    return CAPABILITY.unsupported(problem)


def run(problem, workdir: Optional[str] = None) -> AnnealResult:
    from simws.process import diffusivity as dif
    reasons = unsupported(problem)
    if reasons:
        raise Unsupported(NAME, reasons)
    try:
        import fipy as fp
    except ImportError as e:
        raise EngineNotAvailable(f"fipy is not installed on this machine ({e})") from e
    M = problem.n_intervals
    dx = problem.numerics.dx_um * 1e-4                                 # cm
    mesh = fp.Grid1D(nx=M, dx=dx)
    xc = np.asarray(mesh.cellCenters[0], float) / 1e-4                 # um
    c0 = problem.initial.at(xc)
    C = fp.CellVariable(mesh=mesh, value=c0, hasOld=True)
    dose0 = float(np.sum(c0) * dx)
    if problem.surface == "fixed":
        C.constrain(float(problem.surface_concentration_cm3), mesh.facesLeft)
    flags = [dif.NOT_MODELLED_FLAG] + ([dif.NI_FLAG] if problem.models.diffusivity == "fermi" else [])
    linear = problem.models.diffusivity == "intrinsic"
    sweeps_max = 0
    for step in problem.steps:
        T = dif.kelvin(step.temperature_C)
        Kc = fp.CellVariable(mesh=mesh, value=dif.K(problem, np.asarray(C.value), T))
        eq = fp.TransientTerm() == fp.DiffusionTerm(coeff=Kc.arithmeticFaceValue)
        t_prev = 0.0
        for t in problem.time_points_s(step):
            dt = float(t - t_prev)
            C.updateOld()
            n = 0
            while True:
                Kc.setValue(dif.K(problem, np.asarray(C.value), T))
                before = np.array(C.value, copy=True)
                eq.sweep(var=C, dt=dt)
                n += 1
                if linear:
                    break
                after = np.asarray(C.value)
                change = float(np.max(np.abs(after - before)) / max(float(np.max(np.abs(after))), 1e-300))
                if change < SWEEP_TOL:
                    break
                if n >= MAX_SWEEPS:
                    from simws.hub.adapter import NotConverged
                    raise NotConverged(f"fipy: {MAX_SWEEPS} sweeps at t = {t:g} s without reaching {SWEEP_TOL:g} "
                                       f"(last change {change:.2e})")
            sweeps_max = max(sweeps_max, n)
            t_prev = float(t)
    faces = np.asarray(C.faceValue, float)
    xf = np.asarray(mesh.faceCenters[0], float) / 1e-4
    order = np.argsort(xf)
    return AnnealResult(engine=NAME, problem_id=problem.problem_id(), depth_um=xf[order].tolist(),
                        concentration_cm3=faces[order].tolist(), initial_dose_cm2=dose0, flags=flags,
                        diagnostics={"cells": int(M), "sweeps_max": sweeps_max,
                                     "final_dose_cells_cm2": float(np.sum(np.asarray(C.value)) * dx)})
