# Legacy pipeline runners (retired 2026-11-01)

These scripts are kept for history only. **Do not use their outputs as results.** An audit found that their
"coupled" stages mostly substitute fixed numbers or closed-form expressions for solver output:

- `run_multiscale_pipeline.py`: the "DEVSIM" stage imports DEVSIM only for its version; currents are closed
  form, and the recombination factor 26.0 checked by the fast tests is 1 + K_tau * Phi * tau_0.
- `run_all_6_workspaces_pipeline.py` / `heavy_sim_runner.py`: the QE energy is a typed constant; the ViennaPS
  trench is simulated but discarded (the contour used is a tanh formula).
- `run_level6_autonomous_suite.py`: the strike current is a prescribed pulse; "avalanche" comes from a Laplace
  solve with no carriers; stress is a closed-form estimate; the "MD" and quantum factors are constants; the
  Sobol / surrogate / optimizer run on an invented analytic function.
- `run_spatial_coupled_pipeline.py`: conductivity model with no carrier transport.

Replacements live in the `simws` package and run through `workflow/Snakefile`:
validated DEVSIM strike transient (`simws.tcad.particle_strike`, checked against Charon),
in-process electro-thermal coupling with exact energy balance (`simws.coupling.electrothermal`),
Si film subbands (`simws.quantum.si_film`), material database (`simws.materials`).

Note: four fast tests in `tests/test_workspace_regression.py` (materials/QE constant, SRAM node, multiscale
ratio) still read outputs these runners produced. They check stored numbers, not physics, and should be
replaced as the corresponding stages are rebuilt.
