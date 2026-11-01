# Unified Linux Multi-Physics & Semiconductor Simulation Workspace

## Directory Structure
- `01_materials_dft_md/` -> Atomistic DFT & Molecular Dynamics (`Quantum ESPRESSO`, `GPAW`, `LAMMPS`, `ASE`, `Pymatgen`)
- `02_process_radiation/` -> Semiconductor Process, Ion Implantation & Radiation (`Geant4`, `OpenMC`, `ViennaPS`, `PySRIM`, `Charon` SEE/TID decks)
- `03_tcad_devices/` -> Device & Circuit TCAD (`Sandia Charon`, `DEVSIM`, `Sandia Dakota`, `Ngspice`)
- `04_fem_multiphysics/` -> Electro-Thermal & Waveguide FEM (`FEniCSx`, `scikit-fem`, `Femwell`, `PyVista`)
- `05_quantum_iii_v/` -> Quantum Transport & III-V Quantum Wells (`Kwant`, `QuTiP`, `Solcore`, `nextnanopy`, `NanoTCAD-NEGF`)
- `06_photonics_spl/` -> Photonic Inverse Design & Layout (`PyMeep`, `GDSFactory`, `KLayout`, `Ceviche`, `GRCWA`)
- `sim_data/` -> Shared Physics Libraries (`potentials/` Tersoff-ZBL, `pseudopotentials/` QE UPF, `openmc_data/`)
- `pipeline/run_multiscale_pipeline.py` -> Cross-Environment Multi-Scale Orchestrator (`materials` -> `process_rad` -> `tcad` -> `fem`)
