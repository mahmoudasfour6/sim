"""Refinement studies (solution verification) for the workflow's own results.

electrothermal_mesh(level): T_max at 5 V on an nx = 25 * 2^level mesh (ny scaled with it)
particle_strike_dt(level):  collected extra charge with the pulse time step dt = 1e-5 / 2^level s
"""
import numpy as np


def electrothermal_mesh(level: int) -> float:
    from simws.coupling import electrothermal as et
    nx = 25 * 2 ** level
    res = et.run(et.ResistorCase(nx=nx, ny=max(4, nx // 10), biases=[1.0, 3.0, 5.0]))
    return res["points"][-1]["T_max_K"]


def particle_strike_dt(level: int) -> float:
    from simws.tcad import particle_strike as ps
    r = ps.run(ps.StrikeCase(dt_pulse=1e-5 / 2 ** level))
    t, i = r["time_s"], r["cathode_A_per_cm"]
    return float(np.trapezoid(i - i[1], t))          # extra (strike-induced) charge, C/cm
