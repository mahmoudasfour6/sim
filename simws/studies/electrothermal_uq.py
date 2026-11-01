"""Uncertainty of the electro-thermal resistor's peak temperature and current at 5 V from material-parameter
uncertainty (Sobol indices). Ranges are the spread between common literature sources for Si near 300 K."""
from simws.sweep import Space

SPACE = Space({
    "thermal_k": (1.30, 1.60),             # W/cm/K   (Glassbrenner & Slack 1.56; thin-film/doped values lower)
    "thermal_k_T_exponent": (1.1, 1.4),    # k ~ (T/300)^-a
    "mu_n_T_exponent": (2.2, 2.6),         # mu ~ (T/300)^-z (lattice scattering)
    "donors": (8e16, 1.2e17),              # cm^-3, process variation +-20 %
})
OUTPUTS = ("T_max_K", "current_A_per_cm")


def stage(p: dict) -> dict:
    from simws.coupling import electrothermal as et
    from simws.materials import load
    mat = load("silicon").with_overrides(
        thermal_k=(p["thermal_k"], "W/cm/K"),
        thermal_k_T_exponent=(p["thermal_k_T_exponent"], "dimensionless"),
        mu_n_T_exponent=(p["mu_n_T_exponent"], "dimensionless"))
    res = et.run(et.ResistorCase(donors=p["donors"], nx=50, ny=6, biases=[1.0, 3.0, 5.0]), mat)
    last = res["points"][-1]
    return {"T_max_K": last["T_max_K"], "current_A_per_cm": last["current_A_per_cm"]}
