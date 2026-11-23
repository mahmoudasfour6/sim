"""DEVSIM translator (architecture v1, section 4.4): Problem -> DEVSIM device -> EngineResult.

The physics is the existing simws.tcad.physics (setup_region); this module only translates the description:
mesh lines, doping / Ge node expressions, contacts, Physics fields, solver options and the bias ramps.
Lengths in the description are micrometres; DEVSIM works in cm. Engine imports happen inside functions.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simws.engines.base import Capability, EngineNotAvailable, EngineResult, Unsupported
from simws.spec.schema import DCSweep, DopingTerm, GradeTerm, Problem, SmallSignal, StrikeTransient

UM = 1e-4                  # micrometre -> cm
DEVICE, MESH = "d", "m"

CAPABILITY = Capability(
    engine="devsim", env="tcad",
    description="DEVSIM drift-diffusion (simws.tcad.physics): 1D and 2D silicon, SiGe grading, Fermi-Dirac, "
                "Masetti / Klaassen mobility, SRH, Auger, Slotboom, freeze-out, impact ionization",
    in_process=True, dimensions=[1, 2], max_regions=1, materials=["silicon"], grading_elements=["Ge"],
    contact_kinds=["ohmic"], statistics=["boltzmann", "fermi_dirac"], mobility=["constant", "masetti", "klaassen"],
    velocity_saturation=[False, True], srh=["off", "constant", "scharfetter"], auger=[False, True],
    band_gap_narrowing=["off", "slotboom"], incomplete_ionization=["off", "altermatt", "constant_energy"],
    impact_ionization=["off", "van_overstraeten"], analyses=["equilibrium", "dc_sweep", "small_signal", "strike"],
    tested_temperature_K=(250.0, 400.0), mesh_study=True,
    notes="2D contacts only on y_min / y_max (pad regions); one grading term; one region starting at the origin")

# the Newton tolerance below which setup_region's equilibrium solve counts as nominal
_NOMINAL_REL_TOL = 1e-8


# ------------------------------------------------------------------------------------------ capability
def unsupported(problem: Problem) -> List[str]:
    out = CAPABILITY.unsupported(problem)
    dev = problem.device
    if dev.dimension == 2:
        for c in dev.contacts:
            if c.side not in ("y_min", "y_max"):
                out.append(f"contacts: 2D DEVSIM contacts must be on y_min or y_max ('{c.name}' is on {c.side})")
    for r in dev.regions:
        ivs = [r.box.x] + ([r.box.y] if r.box.y is not None else [])
        if any(iv[0] != 0 for iv in ivs):
            out.append("regions: the region box must start at 0")
            break
    if problem.analysis.kind == "strike" and dev.dimension != 1:
        out.append("strike: the strike analysis is for 1D devices only (A17.1)")
    if len(dev.grading) > 1:
        out.append(f"grading: {len(dev.grading)} grading terms; the DEVSIM translator takes one")
    if dev.regions and not any(s.startswith("regions:") for s in out):
        r = dev.regions[0]
        for name, iv, lines in (("x", r.box.x, dev.mesh.x_lines),
                                ("y", r.box.y, dev.mesh.y_lines)):
            if iv is not None and lines and (abs(lines[0][0] - iv[0]) > 1e-12 or abs(lines[-1][0] - iv[1]) > 1e-12):
                out.append(f"mesh: the {name} mesh lines must span the region from {iv[0]:g} to {iv[1]:g} um")
    return out


def available() -> bool:
    """True when the devsim package can be found (cheap: nothing is imported)."""
    try:
        import importlib.util
        return importlib.util.find_spec("devsim") is not None
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------------------ expressions
def _num(v: float) -> str:
    return repr(float(v))


def _axis_factor(c: str, lo: Optional[float], hi: Optional[float], include_upper: bool) -> List[str]:
    f = []
    if lo is not None:
        f.append(f"step({c} - {_num(lo * UM)})")
    if hi is not None:
        f.append(f"step({_num(hi * UM)} - {c})" if include_upper else f"(1 - step({c} - {_num(hi * UM)}))")
    return f


def _term_expr(t: DopingTerm) -> str:
    f = _axis_factor("x", t.box.x[0], t.box.x[1], t.include_upper)
    if t.box.y is not None:
        f += _axis_factor("y", t.box.y[0], t.box.y[1], t.include_upper)
    return "*".join([_num(t.concentration_cm3)] + f)


def doping_expressions(problem: Problem) -> Tuple[str, str]:
    """(net, total) node expressions in cm^-3, with x, y in cm."""
    don = [_term_expr(t) for t in problem.device.doping if t.species == "donor"]
    acc = [_term_expr(t) for t in problem.device.doping if t.species == "acceptor"]
    don += [PROFILE_SOLUTION["donor"]] if any(t.species == "donor" for t in problem.device.profiles) else []
    acc += [PROFILE_SOLUTION["acceptor"]] if any(t.species == "acceptor" for t in problem.device.profiles) else []
    nd = "(" + " + ".join(don) + ")" if don else "(0)"
    na = "(" + " + ".join(acc) + ")" if acc else "(0)"
    return f"{nd} - {na}", f"{nd} + {na}"


PROFILE_SOLUTION = {"donor": "ProfileDonors", "acceptor": "ProfileAcceptors"}     # A16.5, cm^-3


def set_profiles(problem: Problem, region: str) -> None:
    """A16.5: the device's ProfileTerms evaluated at every node into node solutions (one per species), which the
    doping expressions add to the boxes."""
    if not problem.device.profiles:
        return
    from devsim import get_node_model_values, node_solution, set_node_values
    x = np.asarray(get_node_model_values(device=DEVICE, region=region, name="x"), float) / UM
    y = (np.asarray(get_node_model_values(device=DEVICE, region=region, name="y"), float) / UM
         if problem.device.dimension == 2 else None)
    for species, name in PROFILE_SOLUTION.items():
        terms = [t for t in problem.device.profiles if t.species == species]
        if not terms:
            continue
        v = sum(t.at(x if t.axis == "x" else y) for t in terms)
        node_solution(device=DEVICE, region=region, name=name)
        set_node_values(device=DEVICE, region=region, name=name, values=[float(a) for a in v])


def grade_expression(g: GradeTerm) -> str:
    c, c0, c1 = g.axis, g.start_um * UM, g.stop_um * UM
    return (f"step({c} - {_num(c0)})*step({_num(c1)} - {c})*({_num(g.x_start)} + "
            f"{_num(g.x_stop - g.x_start)}*({c} - {_num(c0)})/{_num(c1 - c0)})")


def physics_from(problem: Problem):
    """ModelSpec -> simws.tcad.physics.Physics (mapping table of section 4.4)."""
    from simws.tcad.physics import Physics
    m = problem.models
    g = problem.device.grading[0] if problem.device.grading else None
    return Physics(
        T=m.temperature_K, fermi_dirac=m.statistics == "fermi_dirac",
        mobility="klaassen" if m.mobility == "klaassen" else "masetti",
        mu_n=m.mu_n_cm2_Vs if m.mobility == "constant" else None,
        mu_p=m.mu_p_cm2_Vs if m.mobility == "constant" else None,
        field_mobility=bool(m.velocity_saturation), srh=m.srh != "off",
        tau_n=m.tau_n_s if m.srh == "constant" else None, tau_p=m.tau_p_s if m.srh == "constant" else None,
        auger=bool(m.auger), bgn=m.band_gap_narrowing == "slotboom", impact=m.impact_ionization == "van_overstraeten",
        incomplete_ionization=m.incomplete_ionization != "off",
        ii_model="charon" if m.incomplete_ionization == "constant_energy" else "altermatt",
        ii_N_crit=m.ii_critical_doping_cm3, ii_E_const=m.ii_energy_eV,
        donor_species=m.donor_species, acceptor_species=m.acceptor_species,
        extended_precision=m.extended_precision,
        ge=grade_expression(g) if g else None, ge_mode=g.mode if g else "strained")


# ------------------------------------------------------------------------------------------ mesh
def _mesh_1d(problem: Problem, region) -> None:
    from devsim import (add_1d_contact, add_1d_mesh_line, add_1d_region, create_1d_mesh, create_device,
                        finalize_mesh)
    dev, s = problem.device, problem.device.mesh.scale
    lines = dev.mesh.x_lines
    create_1d_mesh(mesh=MESH)
    for i, (pos, h) in enumerate(lines):
        tag = "a" if i == 0 else ("b" if i == len(lines) - 1 else None)
        kw = {"tag": tag} if tag else {}
        add_1d_mesh_line(mesh=MESH, pos=pos * UM, ps=h * s * UM, **kw)
    for c in dev.contacts:
        add_1d_contact(mesh=MESH, name=c.name, tag="a" if c.side == "x_min" else "b", material="metal")
    add_1d_region(mesh=MESH, material="Si", region=region.name, tag1="a", tag2="b")
    finalize_mesh(mesh=MESH)
    create_device(mesh=MESH, device=DEVICE)


def _mesh_2d(problem: Problem, region) -> None:
    """The construction of simws.tcad.mixed_mode._mesh_2d (thin pad regions above and below carry the contacts)
    with the spec's region name, and the contacts of simws.tcad.hbt.build."""
    from devsim import (add_2d_contact, add_2d_mesh_line, add_2d_region, create_2d_mesh, create_device,
                        finalize_mesh)
    dev, s = problem.device, problem.device.mesh.scale
    width, height = region.box.x[1], region.box.y[1]
    create_2d_mesh(mesh=MESH)
    for pos, ps in dev.mesh.x_lines:
        add_2d_mesh_line(mesh=MESH, dir="x", pos=pos * UM, ps=ps * s * UM)
    for pos, ps in dev.mesh.y_lines:
        add_2d_mesh_line(mesh=MESH, dir="y", pos=pos * UM, ps=ps * s * UM)
    add_2d_mesh_line(mesh=MESH, dir="y", pos=-1e-8, ps=1e-8)
    add_2d_mesh_line(mesh=MESH, dir="y", pos=height * UM + 1e-8, ps=1e-8)
    add_2d_region(mesh=MESH, material="Si", region=region.name, yl=0, yh=height * UM)
    add_2d_region(mesh=MESH, material="Si", region="pad_t", yl=-1e-8, yh=0)
    add_2d_region(mesh=MESH, material="Si", region="pad_b", yl=height * UM, yh=height * UM + 1e-8)
    for c in dev.contacts:
        y = 0 if c.side == "y_min" else height * UM
        xl, xh = (0, width * UM) if c.span is None else (c.span[0] * UM, c.span[1] * UM)
        add_2d_contact(mesh=MESH, name=c.name, material="metal", region=region.name, yl=y, yh=y, xl=xl, xh=xh,
                       bloat=1e-10)
    finalize_mesh(mesh=MESH)
    create_device(mesh=MESH, device=DEVICE)


# ------------------------------------------------------------------------------------------ helpers
def _json(x: Any) -> Any:
    """numpy values and tuples -> plain JSON values; non-finite floats become None."""
    if isinstance(x, dict):
        return {str(k): _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if np.isfinite(f) else None
    return x if x is None or isinstance(x, str) else str(x)


def _source(contact: str) -> str:
    """Name of the voltage source on a circuit contact."""
    return f"V_{contact}"


def _node(contact: str) -> str:
    from simws.tcad.physics import _bias
    return _bias(contact)


def _no_convergence(e: BaseException) -> bool:
    m = str(e).lower()
    return "converge" in m or "stalled" in m


def _load_devsim():
    os.environ.setdefault("DEVSIM_QUIET", "1")
    try:
        import simws.tcad  # noqa: F401  (safe reset_devsim; library import order)
        import devsim  # noqa: F401
    except ImportError as e:
        raise EngineNotAvailable(f"devsim is not installed on this machine ({e})") from e


class _State:
    """What a built device needs for later ramps."""

    def __init__(self, problem: Problem, used: dict, region: str, opts: dict, flags: List[str]):
        self.problem, self.used, self.region, self.opts, self.flags = problem, used, region, opts, flags
        self.callback = None
        if problem.models.mobility == "klaassen":
            from simws.tcad.physics import relax_mobility
            self.callback = lambda _v: relax_mobility(DEVICE, region, self.opts)

    def ramp(self, contact: str, target: float, step: float):
        from simws.tcad.solve import ramp_bias
        return ramp_bias(DEVICE, contact, target, step=step, callback=self.callback,
                         retry_callback=self.callback is not None, **self.opts)

    def currents(self) -> Dict[str, float]:
        from simws.tcad.physics import contact_current
        return {c.name: float(contact_current(DEVICE, c.name)) for c in self.problem.device.contacts}


def _build(problem: Problem) -> _State:
    """Everything up to and including the equilibrium solve."""
    from simws.hub.adapter import NotConverged
    reasons = unsupported(problem)
    if reasons:
        raise Unsupported("devsim", reasons)
    _load_devsim()
    from devsim import reset_devsim
    from simws.materials import load
    from simws.tcad.physics import setup_region
    dev = problem.device
    region = dev.regions[0]
    reset_devsim()
    (_mesh_1d if dev.dimension == 1 else _mesh_2d)(problem, region)
    set_profiles(problem, region.name)
    net, tot = doping_expressions(problem)
    phys = physics_from(problem)
    num = problem.numerics
    circuit = isinstance(problem.analysis, SmallSignal)      # A6.4: AC needs every contact on a voltage source
    if circuit:
        from devsim import circuit_element
        for c in dev.contacts:
            circuit_element(name=_source(c.name), n1=_node(c.name), n2="0", value=0.0)
    extra = _strike_footprint(problem, region.name) if isinstance(problem.analysis, StrikeTransient) else None
    try:
        used = setup_region(DEVICE, region.name, load(region.material), net, tot, phys,
                            {c.name: circuit for c in dev.contacts}, extra_generation=extra)
    except RuntimeError as e:
        if _no_convergence(e):
            raise NotConverged(f"equilibrium solve at 0 V: {e}") from e
        raise
    flags: List[str] = []
    tol = num.relative_error
    eq = used.get("equilibrium_rel_tol")
    if eq is not None and eq > _NOMINAL_REL_TOL:          # cold-device fallback of setup_region (legacy cryogenic)
        tol = max(num.relative_error, eq)
        if tol > num.relative_error:
            flags.append(f"newton_rel_tol:{tol:g}")
    opts = dict(absolute_error=num.absolute_error, relative_error=tol, maximum_iterations=num.max_iterations)
    if problem.models.temperature_K < 250.0:             # contract A5.4: the legacy adapter's warning, kept
        flags.append("cryo_unvalidated: below 250 K checked against closed forms and Charon only, "
                     "not against measured devices")
    return _State(problem, used, region.name, opts, flags)


STRIKE_RATE = "strike_rate"            # pairs/cm^3/s inside the segment, set at every time step (A17.2)
STRIKE_FOOTPRINT = "StrikeFootprint"


def _strike_footprint(problem: Problem, region: str) -> str:
    """A17.2: 1 inside the segment, 0 outside, 1/2 at a node exactly on an end, scaled so that its node-volume
    integral equals the segment length; the rate parameter starts at 0. Returns the generation expression."""
    from devsim import get_node_model_values, node_solution, set_node_values, set_parameter
    an = problem.analysis
    x = np.asarray(get_node_model_values(device=DEVICE, region=region, name="x"), float) / UM
    vol = np.asarray(get_node_model_values(device=DEVICE, region=region, name="NodeVolume"), float)
    a, b = an.depth_um
    on_end = np.isclose(x, a, rtol=0, atol=1e-9) | np.isclose(x, b, rtol=0, atol=1e-9)
    f = np.where(on_end, 0.5, np.where((x > a) & (x < b), 1.0, 0.0))
    total = float((f * vol).sum())
    if total <= 0:
        raise Unsupported("devsim", [f"strike: no mesh node in the segment {an.depth_um} um"])
    f *= (b - a) * UM / total
    node_solution(device=DEVICE, region=region, name=STRIKE_FOOTPRINT)
    set_node_values(device=DEVICE, region=region, name=STRIKE_FOOTPRINT, values=[float(v) for v in f])
    set_parameter(device=DEVICE, region=region, name=STRIKE_RATE, value=0.0)
    return f"{STRIKE_RATE}*{STRIKE_FOOTPRINT}"


def build(problem: Problem) -> dict:
    """Build the device in DEVSIM (device 'd', region = the spec's region name) and solve the equilibrium. Returns
    the values setup_region used, so that later analyses (AC, strikes) can start from the built device."""
    return _build(problem).used


# ------------------------------------------------------------------------------------------ run
def run(problem: Problem, workdir: Optional[Path] = None, _post_build=None) -> EngineResult:
    """_post_build(state): a hook run after the equilibrium solve (A11: devsim_eb adds the energy balance there)."""
    from simws.hub.adapter import NotConverged
    st = _build(problem)
    if _post_build is not None:
        _post_build(st)
    contacts = [c.name for c in problem.device.contacts]
    an = problem.analysis
    unit = "A/cm**2" if problem.device.dimension == 1 else "A/cm"
    diag = {"used": _json(st.used), "solver_options": _json(st.opts), "n_contacts": len(contacts)}
    if isinstance(an, SmallSignal):
        return _small_signal(problem, st, contacts, unit, diag)
    if isinstance(an, StrikeTransient):
        return _strike(problem, st, contacts, unit, diag)
    if not isinstance(an, DCSweep):
        i0 = st.currents()
        return EngineResult(engine="devsim", problem_id=problem.problem_id(), analysis="equilibrium",
                            sweep_contact=None, sweep_V=[0.0], currents={k: [v] for k, v in i0.items()},
                            current_unit=unit, flags=st.flags, diagnostics=_json(diag))
    cur: Dict[str, List[float]] = {k: [] for k in contacts}
    reached = 0.0
    try:
        for name, v in an.fixed_V.items():
            st.ramp(name, v, 0.25)
        for v in an.values_V:
            for t in an.tracking:
                st.ramp(t.contact, v + t.offset_V, t.step_V)
            st.ramp(an.contact, v, an.step_V)
            reached = v
            for k, i in st.currents().items():
                cur[k].append(i)
    except RuntimeError as e:
        if not _no_convergence(e):
            raise
        raise NotConverged(f"{an.contact} reached {reached:g} V of {list(an.values_V)} "
                           f"({len(cur[contacts[0]])} points done): {e}") from e
    return EngineResult(engine="devsim", problem_id=problem.problem_id(), analysis="dc_sweep",
                        sweep_contact=an.contact, sweep_V=[float(v) for v in an.values_V], currents=cur,
                        current_unit=unit, flags=st.flags, diagnostics=_json(diag))


def _small_signal(problem: Problem, st: "_State", contacts: List[str], unit: str, diag: dict) -> EngineResult:
    """A6.4: ramp the sources to the operating point, then the legacy Y-parameter procedure
    (simws.tcad.small_signal.y_params) for the two ports."""
    from devsim import circuit_alter, get_circuit_node_value, solve
    from simws.hub.adapter import NotConverged
    from simws.tcad.solve import ramp_source
    an = problem.analysis
    flags = list(st.flags)
    cb = None
    if st.callback is not None:                              # Klaassen: outer iteration after every ramp step
        cb = lambda _v=None: st.callback(_v)                 # noqa: E731
        flags.append("ac:frozen_mobility_linearisation")
    reached = "0 V"
    try:
        for b in an.bias:
            start = get_circuit_node_value(node=_node(b.contact), solution="dcop")
            ramp_source(_source(b.contact), b.value_V, start=start, step=b.step_V, callback=cb, **st.opts)
            reached = f"{b.contact} = {b.value_V:g} V"
        if cb:
            cb()
    except RuntimeError as e:
        if not _no_convergence(e):
            raise
        raise NotConverged(f"operating point: reached {reached}: {e}") from e

    def src_i(contact: str, solution: str) -> float:
        return float(get_circuit_node_value(node=f"{_source(contact)}.I", solution=solution))

    dc = {c: [-src_i(c, "dcop")] for c in contacts}          # current into the device
    yr = {k: [] for k in ("11", "12", "21", "22")}
    yi = {k: [] for k in ("11", "12", "21", "22")}
    for f in an.frequencies_Hz:
        for j, drive in enumerate(an.ports, start=1):
            circuit_alter(name=_source(drive), param="acreal", value=1.0)
            solve(type="ac", frequency=float(f))
            for i, port in enumerate(an.ports, start=1):
                yr[f"{i}{j}"].append(-src_i(port, "ssac_real"))
                yi[f"{i}{j}"].append(-src_i(port, "ssac_imag"))
            circuit_alter(name=_source(drive), param="acreal", value=0.0)
    return EngineResult(engine="devsim", problem_id=problem.problem_id(), analysis="small_signal",
                        sweep_contact=None, sweep_V=[0.0], currents=dc, current_unit=unit, flags=flags,
                        diagnostics=_json(diag), frequencies_Hz=[float(f) for f in an.frequencies_Hz],
                        y_real=yr, y_imag=yi, y_unit="S/cm**2" if problem.device.dimension == 1 else "S/cm")


STRIKE_FLAGS = ["strike:fixed_bias: the contacts are held at their bias; a circuit node's voltage change does not "
                "act back on the collection (no mixed mode)",
                "strike:1d_column: the pairs are spread over the device area (no track radius, no 2D funnelling)"]


def _strike(problem: Problem, st: "_State", contacts: List[str], unit: str, diag: dict) -> EngineResult:
    """A17.2: ramp the biases, store the DC state for the transient, then backward Euler on the description's time
    points with the generation rate at the end of each step."""
    from devsim import set_parameter, solve
    from simws.hub.adapter import NotConverged
    an = problem.analysis
    reached = "0 V"
    try:
        for b in an.bias:
            st.ramp(b.contact, b.value_V, b.step_V)
            reached = f"{b.contact} = {b.value_V:g} V"
    except RuntimeError as e:
        if not _no_convergence(e):
            raise
        raise NotConverged(f"strike bias: reached {reached}: {e}") from e
    from devsim import get_contact_charge

    def charges() -> Dict[str, float]:
        return {c: float(get_contact_charge(device=DEVICE, contact=c, equation="PotentialEquation"))
                for c in contacts}

    cur: Dict[str, List[float]] = {k: [v] for k, v in st.currents().items()}
    part: Dict[str, List[float]] = {k: list(v) for k, v in cur.items()}
    q_prev = charges()
    times = [0.0]
    rate0 = an.pairs_cm2 / ((an.depth_um[1] - an.depth_um[0]) * UM)          # pairs/cm^3 per unit g
    quad = 0.0
    t_prev = 0.0
    try:
        solve(type="transient_dc", **st.opts)
        for t in an.time_points_s():
            g = float(an.g(t))
            set_parameter(device=DEVICE, region=st.region, name=STRIKE_RATE, value=rate0 * g)
            solve(type="transient_bdf1", tdelta=float(t - t_prev), charge_error=1e30, **st.opts)
            quad += an.pairs_cm2 * g * (t - t_prev)
            times.append(float(t))
            q_now = charges()
            for k, v in st.currents().items():                 # A17.4: particle + displacement (backward)
                part[k].append(v)
                cur[k].append(v + (q_now[k] - q_prev[k]) / (t - t_prev))
            q_prev = q_now
            t_prev = float(t)
    except RuntimeError as e:
        if not _no_convergence(e):
            raise
        raise NotConverged(f"strike transient: stopped after t = {t_prev:.4e} s: {e}") from e
    diag = dict(diag, generated_pairs_quadrature_cm2=quad, n_time_points=len(times), particle_currents=part)
    return EngineResult(engine="devsim", problem_id=problem.problem_id(), analysis="strike", sweep_contact=None,
                        sweep_V=[], currents=cur, current_unit=unit, flags=list(st.flags) + STRIKE_FLAGS,
                        diagnostics=_json(diag), time_s=times)
