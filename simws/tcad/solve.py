"""Robust DEVSIM bias ramping: step toward the target, halve the step after a failed Newton solve (restoring
the last converged solution), grow it again after successes."""
from __future__ import annotations


def ramp_bias(device, contact, target, step=0.1, min_step=1e-4, max_step=None, callback=None, **solve_opts):
    """Ramp contact `contact` to `target` volts. `callback(v)` runs after every converged point.
    Raises RuntimeError if the step falls below min_step (e.g. at avalanche breakdown); the device is left
    at the last converged bias, which is returned in the exception's `.last_bias` attribute."""
    from devsim import get_parameter, set_parameter, solve
    from devsim.python_packages.simple_physics import GetContactBiasName

    opts = dict(type="dc", absolute_error=1e10, relative_error=1e-8, maximum_iterations=40)
    opts.update(solve_opts)
    name = GetContactBiasName(contact)
    v = float(get_parameter(device=device, name=name))
    max_step = max_step or 4 * abs(step)
    h = abs(step)
    while abs(target - v) > 1e-12:
        v_try = v + max(-h, min(h, target - v))
        set_parameter(device=device, name=name, value=v_try)
        try:
            solve(**opts)
        except Exception:
            set_parameter(device=device, name=name, value=v)
            try:
                solve(**opts)                      # restore the last converged state
            except Exception:
                pass
            h *= 0.5
            if h < min_step:
                err = RuntimeError(f"bias ramp on '{contact}' stalled at {v:.6g} V (target {target} V)")
                err.last_bias = v
                raise err
            continue
        v = v_try
        if callback:
            callback(v)
        h = min(max_step, h * 1.5)
    return v


def ramp_source(source, target, start=0.0, step=0.1, min_step=1e-4, max_step=None, callback=None, **solve_opts):
    """Same adaptive ramp for a circuit voltage source created with circuit_element (mixed-mode)."""
    from devsim import circuit_alter, solve

    opts = dict(type="dc", absolute_error=1e10, relative_error=1e-6, maximum_iterations=40)
    opts.update(solve_opts)
    v, h = float(start), abs(step)
    max_step = max_step or 4 * abs(step)
    while abs(target - v) > 1e-12:
        v_try = v + max(-h, min(h, target - v))
        circuit_alter(name=source, value=v_try)
        try:
            solve(**opts)
        except Exception:
            circuit_alter(name=source, value=v)
            try:
                solve(**opts)
            except Exception:
                pass
            h *= 0.5
            if h < min_step:
                err = RuntimeError(f"source ramp on '{source}' stalled at {v:.6g} V (target {target} V)")
                err.last_bias = v
                raise err
            continue
        v = v_try
        if callback:
            callback(v)
        h = min(max_step, h * 1.5)
    return v
