#!/usr/bin/env python3
"""From Charon's gold file: (1) n_i = sqrt(n*p) at t = 0 (equilibrium-density initial condition, so n*p = n_i^2
at every node), written to ni.txt; (2) the cathode current time series, written to gold_currents.csv."""
import sys
import numpy as np
try:
    import exodus3 as exodus
except ImportError:
    import exodus

e = exodus.exodus(sys.argv[1], mode="r", array_type="numpy")
x = np.asarray(e.get_coords()[0])
n0 = np.asarray(e.get_node_variable_values("ELECTRON_DENSITY", 1))
p0 = np.asarray(e.get_node_variable_values("HOLE_DENSITY", 1))
p_bulk = np.median(p0[x < 0.3])
scale = 1e16 if p_bulk < 1e6 else 1.0          # outputs may be scaled by the 1e16 concentration scaling
ni_nodes = np.sqrt(n0 * p0) * scale
print(f"p-side bulk hole density at t=0: {p_bulk * scale:.4e} cm^-3 (scale factor applied: {scale:g})")
print(f"n_i from sqrt(n*p): median {np.median(ni_nodes):.5e}, min {ni_nodes.min():.5e}, max {ni_nodes.max():.5e} cm^-3")
open("ni.txt", "w").write(f"{np.median(ni_nodes):.6e}\n")
t = np.asarray(e.get_times())
Ic = np.asarray(e.get_global_variable_values("cathode_silicon_Current"))
np.savetxt("gold_currents.csv", np.column_stack([t, Ic]), delimiter=",", header="time_s,cathode_A_per_cm", comments="")
e.close()
