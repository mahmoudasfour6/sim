"""Process chain (architecture v1, contract A16.6): implant -> anneal -> device -> circuit.

    from simws.chain import process as P
    r_imp = hub.submit("implant.rustbca", params={"problem": implant_si("B", 35, dose_cm2=1e14, ...)})
    ini = P.initial_from_implant(hub.result(r_imp.id), dose_cm2=1e14)
    ap = P.anneal_problem("B", ini, background_cm3=1e16, background_type="donor", steps=[(1000, 1800)],
                          model="fermi")
    r_an = hub.submit("anneal.devsim_anneal", params={"problem": ap})
    dp = P.diode_from_anneal(hub.result(r_an.id), ap, V=[0.3, 0.35, ..., 0.6])
    r_dev = hub.submit("device.devsim", params={"problem": dp})
    fit = chain.diode.fit_shockley(P.forward_V(dp), P.forward_J(hub.result(r_dev.id), dp), 300.0)

The anneal knows nothing of implant damage (no transient enhanced diffusion, A16.2); the as-implanted profile's
counting noise is carried into the anneal unchanged.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np


def gaussian_profile(rp_um: float, sigma_um: float, dose_cm2: float, dx_um: float = 0.001,
                     depth_um: Optional[float] = None, image: bool = True):
    """A Gaussian of dose `dose_cm2` at depth rp_um (with its image in the surface when `image`, so the
    reflecting-surface solution keeps the Gaussian form), tabulated every dx_um to rp + 8 sigma (or depth_um)."""
    from simws.spec.anneal import DepthProfile
    end = depth_um if depth_um is not None else rp_um + 8.0 * sigma_um
    x = np.arange(0.0, end + 0.5 * dx_um, dx_um)
    s = sigma_um * 1e-4
    c = dose_cm2 / (math.sqrt(2.0 * math.pi) * s) * np.exp(-(x - rp_um) ** 2 / (2.0 * sigma_um ** 2))
    if image:
        c = c + dose_cm2 / (math.sqrt(2.0 * math.pi) * s) * np.exp(-(x + rp_um) ** 2 / (2.0 * sigma_um ** 2))
    return DepthProfile(depth_um=x.tolist(), concentration_cm3=c.tolist())


def initial_from_implant(result, dose_cm2: Optional[float] = None):
    """The as-implanted profile of a stored implant.* result as a DepthProfile (um, cm^-3): its
    implant.concentration, or implant.profile x dose_cm2 when the result has no concentration (no dose given)."""
    from simws.spec.anneal import DepthProfile
    names = {q.name for q in result.quantities}
    if "implant.concentration" in names and dose_cm2 is None:
        q = result.q("implant.concentration")
        c = np.asarray(q.array(), float) * 1e-6                       # 1/m^3 -> 1/cm^3
    else:
        if dose_cm2 is None:
            raise ValueError("initial_from_implant: the result has no implant.concentration; give dose_cm2")
        q = result.q("implant.profile")
        c = np.asarray(q.array(), float) * 1e-2 * float(dose_cm2)     # 1/m per ion -> 1/cm per ion, x cm^-2
    x = np.asarray(q.axes[0].values, float) * 1e6                     # m -> um
    return DepthProfile(depth_um=x.tolist(), concentration_cm3=np.clip(c, 0.0, None).tolist())


def anneal_problem(dopant: str, initial, background_cm3: float = 0.0, background_type: str = "acceptor",
                   steps: Sequence[Tuple[float, float]] = ((1000.0, 1800.0),), model: str = "intrinsic",
                   thickness_um: float = 1.0, dx_um: float = 0.001, steps_per_anneal: int = 200,
                   surface: str = "reflecting", surface_concentration_cm3: Optional[float] = None):
    """An AnnealProblem; `steps` are (temperature_C, time_s) pairs."""
    from simws.spec.anneal import AnnealProblem
    ini = initial.model_dump(mode="json") if hasattr(initial, "model_dump") else initial
    return AnnealProblem(dopant=dopant, initial=ini, background_cm3=background_cm3, background_type=background_type,
                         thickness_um=thickness_um, steps=[{"temperature_C": float(T), "time_s": float(t)}
                                                            for T, t in steps],
                         models={"diffusivity": model}, surface=surface,
                         surface_concentration_cm3=surface_concentration_cm3,
                         numerics={"dx_um": dx_um, "steps_per_anneal": steps_per_anneal})


def diode_from_anneal(result, anneal, V: Optional[Sequence[float]] = None, length_um: Optional[float] = None,
                      T_K: float = 300.0, fine_um: float = 0.002, coarse_um: float = 0.05, models=None):
    """A 1D DEVSIM-ready diode: the annealed dopant (anneal.concentration of `result`) as a ProfileTerm on the
    uniform background of `anneal` (an AnnealProblem of the other type), contacts "top" (x = 0) and "back".
    V: forward biases (V >= 0); "top" is swept at +V when the dopant is B (p on top), at -V otherwise.
    Default models: the ModelSpec defaults (Masetti mobility, Scharfetter SRH, Auger, Slotboom BGN)."""
    from simws.spec.schema import (Box, Contact, DCSweep, DeviceSpec, DopingTerm, Equilibrium, MeshHints, ModelSpec,
                                   Numerics, Problem, ProfileTerm, Region)
    own = "acceptor" if anneal.dopant == "B" else "donor"
    if anneal.background_type == own or anneal.background_cm3 <= 0:
        raise ValueError("diode_from_anneal: the anneal needs a background of the other type")
    q = result.q("anneal.concentration")
    x = np.asarray(q.axes[0].values, float) * 1e6
    c = np.asarray(q.array(), float) * 1e-6
    xj = None
    below = np.nonzero(c <= anneal.background_cm3)[0]
    if c[0] > anneal.background_cm3 and below.size:
        xj = float(x[below[0]])
    if xj is None:
        raise ValueError("diode_from_anneal: the annealed profile has no junction with the background")
    L = float(length_um) if length_um is not None else max(float(x[-1]), xj + 2.0)
    fine_end = min(xj + 0.3, L - coarse_um)
    lines = [(0.0, fine_um), (fine_end, fine_um), (L, coarse_um)]
    dev = DeviceSpec(
        name="annealed_diode_1d", dimension=1,
        regions=[Region(name="si", material="silicon", box=Box(x=(0.0, L)))],
        contacts=[Contact(name="top", region="si", side="x_min"), Contact(name="back", region="si", side="x_max")],
        doping=[DopingTerm(species=anneal.background_type, concentration_cm3=anneal.background_cm3, box=Box(),
                           label="background")],
        profiles=[ProfileTerm(species=own, axis="x", position_um=x.tolist(), concentration_cm3=c.tolist(),
                              label=f"annealed {anneal.dopant}")],
        mesh=MeshHints(x_lines=lines),
        description=f"diode from an annealed {anneal.dopant} profile (A16.6), junction near {xj:.3f} um")
    sign = 1.0 if own == "acceptor" else -1.0
    analysis = Equilibrium() if V is None else DCSweep(contact="top", values_V=[sign * float(v) for v in V],
                                                       step_V=0.05)
    return Problem(device=dev, models=models or ModelSpec(temperature_K=T_K), analysis=analysis,
                   numerics=Numerics(relative_error=1e-10, absolute_error=1e10, max_iterations=100))


def forward_V(problem) -> np.ndarray:
    """The forward biases (>= 0) of a diode_from_anneal problem."""
    return np.abs(np.asarray(problem.analysis.values_V, float))


def forward_J(result, problem) -> np.ndarray:
    """Forward current density (A/m^2, > 0) at the top contact of a stored device.* result (dc.I.top)."""
    return np.abs(np.asarray(result.q("dc.I.top").to("A/m**2"), float))
