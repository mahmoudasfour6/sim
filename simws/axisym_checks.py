"""Thin wrapper so the workflow can run the cylindrical-geometry check in the same report."""


def axisym_resistor():
    from simws.tcad.axisym import resistor_check
    return resistor_check()
