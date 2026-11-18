"""DEVSIM with an electron energy balance (architecture v1, contract amendment A11): a different-physics engine.

Translates a Problem exactly as simws.engines.devsim does, solves the drift-diffusion equilibrium, adds the
electron temperature equation (simws.tcad.energy_balance; Baccarani and Wordeman 1985, energy-balance form), solves
the equilibrium again and runs the analysis. Same code as `devsim` (CODE = "devsim"), so a comparison of the two
is a model-form check, never a verification.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from simws.engines.base import Capability, EngineResult, Unsupported
from simws.spec.schema import Problem

FAMILY = "device"
CODE = "devsim"
NAME = "devsim_eb"
MODEL = "baccarani_wordeman_1985_energy_balance"
FLAG = "physics:energy_balance_electrons"

CAPABILITY = Capability(
    engine=NAME, env="tcad",
    description="DEVSIM drift-diffusion with an electron energy balance (Baccarani-Wordeman 1985): hot electrons, "
                "velocity overshoot; 1D silicon",
    in_process=True, dimensions=[1], max_regions=1, materials=["silicon"], grading_elements=[],
    contact_kinds=["ohmic"], statistics=["boltzmann"], mobility=["constant", "masetti"],
    velocity_saturation=[False], srh=["off", "constant", "scharfetter"], auger=[False, True],
    band_gap_narrowing=["off", "slotboom"], incomplete_ionization=["off"], impact_ionization=["off"],
    analyses=["equilibrium", "dc_sweep"], tested_temperature_K=(300.0, 300.0), mesh_study=True,
    notes="Electrons only (holes isothermal); no recombination energy, no Tn-driven impact ionization, no "
          "lattice heating. Velocity saturation comes from the energy balance itself.")


def unsupported(problem: Problem) -> List[str]:
    from simws.engines import devsim_engine
    out = CAPABILITY.unsupported(problem)
    return out + [r for r in devsim_engine.unsupported(problem) if r not in out
                  and not r.startswith(("dimensions:", "analyses:"))]


def available() -> bool:
    from simws.engines import devsim_engine
    return devsim_engine.available()


def _post_build(state) -> None:
    """Add the energy balance to the built device and solve the equilibrium again."""
    from devsim import solve
    from simws.engines.devsim_engine import DEVICE
    from simws.hub.adapter import NotConverged
    from simws.materials import load
    from simws.tcad.energy_balance import add_energy_balance
    p = state.problem
    vs = add_energy_balance(DEVICE, state.region, load(p.device.regions[0].material), p.models.temperature_K,
                            [c.name for c in p.device.contacts])
    state.used["eb_vsat_n_cm_s"] = vs
    try:
        solve(type="dc", **state.opts)
    except RuntimeError as e:
        raise NotConverged(f"energy-balance equilibrium: {e}") from e
    state.ramp = lambda contact, target, step: _ramp(state, contact, target, step)


_SOLUTIONS = ("Potential", "Electrons", "Holes", "Tn")


def _ramp(state, contact: str, target: float, step: float, min_step: float = 1e-5) -> float:
    """Bias ramp for the energy balance: like simws.tcad.solve.ramp_bias, but the node solutions are saved before
    each step and put back after a failed one (a diverged Newton step can leave Tn overflowing, and re-solving
    from there never recovers), and the step grows by at most a factor 1.5 up to twice the requested step."""
    from devsim import get_node_model_values, get_parameter, set_node_values, set_parameter, solve
    from devsim.python_packages.simple_physics import GetContactBiasName
    from simws.engines.devsim_engine import DEVICE
    name = GetContactBiasName(contact)
    v = float(get_parameter(device=DEVICE, name=name))
    h, h_max = abs(step), 2 * abs(step)
    while abs(target - v) > 1e-12:
        saved = {n: list(get_node_model_values(device=DEVICE, region=state.region, name=n)) for n in _SOLUTIONS}
        v_try = v + max(-h, min(h, target - v))
        set_parameter(device=DEVICE, name=name, value=v_try)
        try:
            solve(type="dc", **state.opts)
        except Exception:  # noqa: BLE001  (DEVSIM raises RuntimeError; a floating point error may surface too)
            set_parameter(device=DEVICE, name=name, value=v)
            for n, vals in saved.items():
                set_node_values(device=DEVICE, region=state.region, name=n, values=vals)
            h *= 0.5
            if h < min_step:
                err = RuntimeError(f"bias ramp on '{contact}' stalled at {v:.6g} V (target {target} V)")
                err.last_bias = v
                raise err
            continue
        v = v_try
        h = min(h_max, h * 1.5)
    return v


def _tn_max(region: str) -> float:
    from devsim import get_node_model_values
    from simws.engines.devsim_engine import DEVICE
    return float(max(get_node_model_values(device=DEVICE, region=region, name="Tn")))


def run(problem: Problem, workdir: Optional[Path] = None) -> EngineResult:
    from simws.engines import devsim_engine
    reasons = unsupported(problem)
    if reasons:
        raise Unsupported(NAME, reasons)
    holder: Dict[str, object] = {}

    def hook(st):
        _post_build(st)
        holder["st"] = st

    res = devsim_engine.run(problem, workdir, _post_build=hook)
    st = holder["st"]
    diag = dict(res.diagnostics)
    diag.update({"eb_vsat_n_cm_s": float(st.used["eb_vsat_n_cm_s"]), "eb_Tn_max_K": _tn_max(st.region),
                 "eb_model": MODEL})
    return res.model_copy(update={"engine": NAME, "flags": list(res.flags) + [FLAG], "diagnostics": diag})


def profiles(problem: Problem) -> dict:
    """Run `problem` and return its node profiles at the last point: x (um), Tn (K), n (cm^-3), potential (V)."""
    from devsim import get_node_model_values
    from simws.engines.devsim_engine import DEVICE
    run(problem)
    region = problem.device.regions[0].name
    get = lambda name: [float(v) for v in get_node_model_values(device=DEVICE, region=region, name=name)]  # noqa: E731
    x = [v * 1e4 for v in get("x")]
    order = sorted(range(len(x)), key=x.__getitem__)
    out = {"x_um": x, "Tn_K": get("Tn"), "n_cm3": get("Electrons"), "potential_V": get("Potential")}
    return {k: [v[i] for i in order] for k, v in out.items()}
