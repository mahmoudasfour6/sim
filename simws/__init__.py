"""simws: shared infrastructure for the sim_workspace toolchain.

Submodules
----------
units       pint unit registry and helpers (one registry for the whole workspace)
materials   versioned material-parameter database with units, references and code profiles
config      workspace paths and toolchain locations from config/config.yaml (no hardcoded paths)
provenance  run metadata stamped onto every output (git commit, env, inputs)
io          standard field files (VTU + metadata sidecar) and mesh-to-mesh interpolation
tcad        DEVSIM adapters and validated device simulations
quantum     validated quantum-confinement solvers
coupling    in-process multiphysics coupling (electro-thermal)
"""
__version__ = "0.2.0"
