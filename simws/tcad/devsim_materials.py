"""Apply a simws Material to a DEVSIM region (DEVSIM works in cm, s, V, C)."""
from __future__ import annotations

from simws.units import EPS0_F_PER_CM, K_B_EV, Q_E


def apply(device: str, region: str, mat, T: float = 300.0, mu_n=None, mu_p=None, tau_n=None, tau_p=None):
    """Set DEVSIM's simple_physics parameters from `mat`. Explicit mu/tau arguments (cm^2/Vs, s) override the
    material values; tau defaults to 1e30 s, i.e. SRH effectively off unless the caller sets a lifetime."""
    from devsim import set_parameter
    from devsim.python_packages import simple_physics as sp

    sp.SetSiliconParameters(device, region, T)        # creates the parameter set; values overridden below
    n_i = mat.get("n_i", "cm**-3")
    # simple_physics uses q = 1.6e-19 C; use CODATA values so currents are not biased by 0.14 %
    values = {
        "ElectronCharge": Q_E, "T": T, "kT": K_B_EV * Q_E * T, "V_t": K_B_EV * T,
        "Permittivity": mat.get("eps_r", "dimensionless") * EPS0_F_PER_CM,
        "n_i": n_i, "n1": n_i, "p1": n_i,
        "mu_n": mat.get("mu_n", "cm**2/V/s") if mu_n is None else mu_n,
        "mu_p": mat.get("mu_p", "cm**2/V/s") if mu_p is None else mu_p,
        "taun": 1e30 if tau_n is None else tau_n,
        "taup": 1e30 if tau_p is None else tau_p,
    }
    for name, val in values.items():
        set_parameter(device=device, region=region, name=name, value=val)
    return values
