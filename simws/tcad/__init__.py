"""DEVSIM adapters (materials -> DEVSIM parameters) and validated device simulations.

Importing simws.tcad makes devsim.reset_devsim safe to call: some DEVSIM builds clear the global parameters
set at import (direct_solver = "custom" + solver_callback for the bundled UMFPACK), after which every solve
fails with 'Unrecognized "direct_solver" parameter value "unknown"'. The wrapper snapshots DEVSIM's start-up
globals once and restores them after each reset. Only those start-up globals are restored, never parameters a
simulation set later (e.g. contact biases), so the reset still gives a clean state.
"""
import glob as _glob
import os as _os
import sys as _sys

# Linear solver. UMFPACK (DEVSIM's default fallback) is the default here: in tests on this workspace's devices,
# MKL PARDISO failed LU factorization on the electrothermal resistor and did not cure the minority-carrier
# Newton cycling seen in mixed-mode runs. Opt in with SIMWS_DEVSIM_SOLVER=mkl (needs `mkl` in the env).
if _os.environ.get("SIMWS_DEVSIM_SOLVER", "").lower() == "mkl" and "DEVSIM_MATH_LIBS" not in _os.environ:
    _mkl = sorted(_glob.glob(_os.path.join(_sys.prefix, "lib", "libmkl_rt.so*")))
    if _mkl:
        _os.environ["DEVSIM_MATH_LIBS"] = _mkl[-1]

import devsim as _devsim  # noqa: E402

if not getattr(_devsim.reset_devsim, "_simws_safe", False):
    _raw_reset = _devsim.reset_devsim
    _startup = {}
    try:
        _names = _devsim.get_parameter_list()
    except Exception:  # noqa: BLE001 - older DEVSIM without a global listing
        _names = ("direct_solver", "solver_callback", "info", "symbolic_iteration_limit")
    for _n in _names:
        try:
            _startup[_n] = _devsim.get_parameter(name=_n)
        except Exception:  # noqa: BLE001
            pass

    def reset_devsim():
        _raw_reset()
        for name, value in _startup.items():
            try:
                _devsim.set_parameter(name=name, value=value)
            except Exception:  # noqa: BLE001
                pass
        if _devsim.get_parameter(name="direct_solver") == "unknown":
            try:                                   # last resort: DEVSIM's own UMFPACK shim
                from devsim.umfpack import umfshim
                umfshim.reload_umfpack()
            except Exception:  # noqa: BLE001
                pass

    reset_devsim._simws_safe = True
    _devsim.reset_devsim = reset_devsim
