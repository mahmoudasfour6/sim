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
hub         0.8 core contract: typed envelopes, job keys, content-addressed store, knowledge graph, gatekeeper,
            adapter lifecycle, uncertainty budgets, verification records, citations (simws.hub.jobs.Hub)
adapters    built-in adapters wrapping the engines (DEVSIM, Quantum ESPRESSO, radiation tools, material laws)
conventions sign/ordering/frame conventions registry and converters
queue       overnight job queue (python -m simws.queue)
"""
__version__ = "0.9.0a0"
