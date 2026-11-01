"""DEVSIM adapters (materials -> DEVSIM parameters) and validated device simulations.

Importing simws.tcad makes devsim.reset_devsim safe to call: some DEVSIM builds clear the global parameters
set at import (direct_solver = "custom" + solver_callback for the bundled UMFPACK), after which every solve
fails with 'Unrecognized "direct_solver" parameter value "unknown"'. The wrapper snapshots DEVSIM's start-up
globals once and restores them after each reset. Only those start-up globals are restored, never parameters a
simulation set later (e.g. contact biases), so the reset still gives a clean state.
"""
import devsim as _devsim

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
