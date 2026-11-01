"""One pint registry for the whole workspace. Never create a second UnitRegistry elsewhere:
quantities from different registries cannot be combined."""
import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# Physical constants in the units TCAD codes use (cm, s, V, C, eV).
Q_E = 1.602176634e-19        # C
K_B_EV = 8.617333262e-5      # eV/K
EPS0_F_PER_CM = 8.8541878128e-14  # F/cm
M0_KG = 9.1093837015e-31     # kg
HBAR = 1.054571817e-34       # J s


def to_magnitude(quantity, unit):
    """Convert a pint Quantity to a float in `unit`, raising if the dimensions disagree."""
    return float(quantity.to(unit).magnitude)
