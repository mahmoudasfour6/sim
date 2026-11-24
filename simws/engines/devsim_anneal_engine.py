"""DEVSIM anneal engine (architecture v1, contract A16.3): dC/dt = d/dx (K(C) dC/dx) on a 1D DEVSIM mesh.

Nodes at k dx (cm inside DEVSIM); on each edge the flux K_edge (C0 - C1) / L with K_edge the arithmetic mean of K at
the two nodes; backward Euler (transient_bdf1) on the time points of the description; Newton to relative 1e-10.
A fixed surface is a contact equation C - Cs = 0 at x = 0; otherwise both ends carry no flux.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from simws.engines.base import AnnealCapability, AnnealResult, EngineNotAvailable, Unsupported

FAMILY = "anneal"
CODE = "devsim"
NAME = "devsim_anneal"
UM = 1e-4
REL_TOL = 1e-10
MAX_ITER = 40
PIN_SCALE = 1e40
NEG_TOL = 1e-12

CAPABILITY = AnnealCapability(
    engine=NAME, env="tcad", in_process=True,
    description="One dopant diffusing in 1D silicon on DEVSIM (vertex-centred finite volume, backward Euler, "
                "Newton): intrinsic or Fermi-level diffusivity of Plummer, Deal and Griffin",
    dopants=["B", "P", "As", "Sb"], diffusivity=["intrinsic", "fermi"], surfaces=["reflecting", "fixed"],
    tested_temperature_C=(800.0, 1200.0),
    notes="Isothermal steps only; one dopant on a uniform background; TED only through the effective +1 model "
          "(A22, models.ted); no clustering, solubility, oxidation or segregation (A16.2).")

DEV, REG = "anneal", "si"


def available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("devsim") is not None
    except Exception:  # noqa: BLE001
        return False


def unsupported(problem) -> List[str]:
    return CAPABILITY.unsupported(problem)


def _load():
    import os
    os.environ.setdefault("DEVSIM_QUIET", "1")
    try:
        import simws.tcad  # noqa: F401  (safe reset_devsim)
        import devsim  # noqa: F401
    except ImportError as e:
        raise EngineNotAvailable(f"devsim is not installed on this machine ({e})") from e


def run(problem, workdir: Optional[str] = None) -> AnnealResult:
    from simws.process import diffusivity as dif
    reasons = unsupported(problem)
    if reasons:
        raise Unsupported(NAME, reasons)
    _load()
    import devsim as ds
    ds.reset_devsim()
    M = problem.n_intervals
    L = problem.thickness_um * UM
    ds.create_1d_mesh(mesh="m")
    ds.add_1d_mesh_line(mesh="m", pos=0.0, ps=problem.numerics.dx_um * UM, tag="top")
    ds.add_1d_mesh_line(mesh="m", pos=L, ps=problem.numerics.dx_um * UM, tag="bot")
    ds.add_1d_region(mesh="m", material="Si", region=REG, tag1="top", tag2="bot")
    ds.add_1d_contact(mesh="m", name="top", tag="top", material="metal")
    ds.add_1d_contact(mesh="m", name="bot", tag="bot", material="metal")
    ds.finalize_mesh(mesh="m")
    ds.create_device(mesh="m", device=DEV)
    x = np.asarray(ds.get_node_model_values(device=DEV, region=REG, name="x"), float) / UM
    if x.size != M + 1:
        raise RuntimeError(f"devsim_anneal: mesh has {x.size} nodes, expected {M + 1}")
    c0 = problem.initial.at(x)
    ds.node_solution(device=DEV, region=REG, name="C")
    ds.set_node_values(device=DEV, region=REG, name="C", values=[float(v) for v in c0])
    ds.edge_from_node_model(device=DEV, region=REG, node_model="C")
    vol = np.asarray(ds.get_node_model_values(device=DEV, region=REG, name="NodeVolume"), float)
    dose0 = float(np.sum(vol * c0))                                   # cm^-2 (1D volumes are lengths)
    ds.node_model(device=DEV, region=REG, name="Ctime", equation="C")
    ds.node_model(device=DEV, region=REG, name="Ctime:C", equation="1")
    # DEVSIM's transient needs a transient_dc first (it stores the time data); with no-flux ends that DC problem is
    # singular, so a pin to the initial profile, PIN_SCALE (C - Cinit), is on for that one solve and off after:
    # the stored state equals the initial profile to ~1e-30 relative
    ds.node_solution(device=DEV, region=REG, name="Cinit")
    ds.set_node_values(device=DEV, region=REG, name="Cinit", values=[float(v) for v in c0])
    ds.set_parameter(device=DEV, region=REG, name="pin_scale", value=PIN_SCALE)
    ds.node_model(device=DEV, region=REG, name="Pin", equation="pin_scale*(C - Cinit)")
    ds.node_model(device=DEV, region=REG, name="Pin:C", equation="pin_scale")
    opts = dict(absolute_error=1e30, relative_error=REL_TOL, maximum_iterations=MAX_ITER)
    flags = [dif.NOT_MODELLED_FLAG] + ([dif.NI_FLAG] if problem.models.diffusivity == "fermi" else [])
    newton_max = 0
    started = False
    from simws.process import ted
    sched = ted.schedule(problem)                                    # A22.2: TED end inside a step is a time point
    if problem.models.ted == "plus_one":                             # A22.9: not "TED not modelled" any more
        flags = [ted.NOT_MODELLED_WITH_TED if f == dif.NOT_MODELLED_FLAG else f for f in flags] + [ted.TED_FLAG]
    ds.set_parameter(device=DEV, region=REG, name="ted_factor", value=1.0)
    for step, (points, t_end) in zip(problem.steps, sched):
        T = dif.kelvin(step.temperature_C)
        f_on = ted.enhancement(problem.dopant, T) if problem.models.ted == "plus_one" else 1.0
        kexpr = f"(ted_factor * {dif.K_expression(problem, T, 'C')})"
        ds.node_model(device=DEV, region=REG, name="K", equation=kexpr)
        ds.node_model(device=DEV, region=REG, name="K:C", equation=f"diff({kexpr}, C)")
        ds.edge_from_node_model(device=DEV, region=REG, node_model="K")
        flux = "0.5*(K@n0 + K@n1)*(C@n0 - C@n1)*EdgeInverseLength"
        ds.edge_model(device=DEV, region=REG, name="Flux", equation=flux)
        for nn in ("n0", "n1"):
            ds.edge_model(device=DEV, region=REG, name=f"Flux:C@{nn}", equation=f"diff({flux}, C@{nn})")
        ds.equation(device=DEV, region=REG, name="Diffusion", variable_name="C", time_node_model="Ctime",
                    node_model="Pin", edge_model="Flux", variable_update="default")
        if problem.surface == "fixed":
            ds.set_parameter(device=DEV, name="Cs", value=float(problem.surface_concentration_cm3))
            ds.contact_node_model(device=DEV, contact="top", name="Cbc", equation="C - Cs")
            ds.contact_node_model(device=DEV, contact="top", name="Cbc:C", equation="1")
            ds.contact_equation(device=DEV, contact="top", name="Diffusion", node_model="Cbc")
        if not started:
            ds.solve(type="transient_dc", **opts)
            ds.set_parameter(device=DEV, region=REG, name="pin_scale", value=0.0)
            started = True
        t_prev = 0.0
        for t in points:
            on = t <= t_end * (1 + 1e-12)
            ds.set_parameter(device=DEV, region=REG, name="ted_factor", value=float(f_on if on else 1.0))
            info = ds.solve(type="transient_bdf1", tdelta=float(t - t_prev), charge_error=1e30, info=True, **opts)
            if isinstance(info, dict):
                newton_max = max(newton_max, len(info.get("iterations", [])))
            t_prev = float(t)
    c = np.asarray(ds.get_node_model_values(device=DEV, region=REG, name="C"), float)
    # A16.10: round-off ahead of the front leaves values like -2e-316; values below zero by less than NEG_TOL of the
    # peak are reported as 0 (larger negatives are left for the hub's check to refuse)
    peak = float(np.max(np.abs(c))) if c.size else 0.0
    small_neg = (c < 0) & (c >= -NEG_TOL * peak)
    n_clipped = int(np.count_nonzero(small_neg))
    c = np.where(small_neg, 0.0, c)
    order = np.argsort(x)
    return AnnealResult(engine=NAME, problem_id=problem.problem_id(), depth_um=x[order].tolist(),
                        concentration_cm3=c[order].tolist(), initial_dose_cm2=dose0, flags=flags,
                        diagnostics={"nodes": int(x.size), "newton_iterations_max": newton_max,
                                     "negative_roundoff_set_to_zero": n_clipped,
                                     "final_dose_volume_cm2": float(np.sum(vol * c))})
