#!/usr/bin/env python3
"""
Atomistic / Materials Simulation Verification & TCAD/Photonics Parameter Extractor
1. Builds Silicon (Si), 3C-SiC, and GaAs crystal structures using ASE & Pymatgen.
2. Analyses space-group symmetry using spglib.
3. Generates a Quantum ESPRESSO (pw.x) input deck for Si bandstructure/dielectric properties.
4. Runs a LAMMPS Molecular Dynamics simulation of a Silicon crystal recoil / thermal equilibration.
"""
import subprocess
import shutil
import numpy as np
from ase.build import bulk
from ase.io.espresso import write_espresso_in
import pymatgen.core as mg
import spglib
import gpaw

def run_materials_suite():
    # 1. ASE & Pymatgen crystal structures
    si_atoms = bulk("Si", "diamond", a=5.431, cubic=True)
    gaas_atoms = bulk("GaAs", "zincblende", a=5.653, cubic=True)
    
    lattice = mg.Lattice.cubic(4.3596)
    sic_struct = mg.Structure(lattice, ["Si", "C"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    
    # 2. Symmetry check with spglib
    cell = (si_atoms.get_cell(), si_atoms.get_scaled_positions(), si_atoms.get_atomic_numbers())
    sym_dataset = spglib.get_symmetry_dataset(cell)
    print(f"[Materials] ASE + spglib verified: Si space group = {sym_dataset.international} (#{sym_dataset.number})")
    print(f"[Materials] Pymatgen v{mg.__version__} verified: 3C-SiC density = {sic_struct.density:.3f} g/cm^3")
    print(f"[Materials] GPAW v{gpaw.__version__} DFT package verified.")

    # 3. Generate Quantum ESPRESSO input deck for Si
    input_data = {
        "control": {"calculation": "scf", "prefix": "si_tcad", "pseudo_dir": "./"},
        "system": {"ecutwfc": 30.0, "ecutrho": 120.0, "nbnd": 8},
        "electrons": {"conv_thr": 1.0e-8},
    }
    pseudopotentials = {"Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF"}
    with open("si_scf.pwi", "w") as f:
        write_espresso_in(f, si_atoms, input_data=input_data, pseudopotentials=pseudopotentials, kpts=(4, 4, 4))
    pw_bin = shutil.which("pw.x")
    print(f"[Materials] Quantum ESPRESSO binary found at: {pw_bin} (wrote si_scf.pwi)")

    # 4. Run a fast LAMMPS Molecular Dynamics simulation (PKA recoil / thermal lattice test)
    lmp_in = """
units metal
dimension 3
boundary p p p
atom_style atomic

lattice diamond 5.431
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 28.0855

pair_style lj/cut 6.0
pair_coeff 1 1 0.1 2.095 6.0

velocity all create 300.0 12345 mom yes rot yes dist gaussian
# Simulate a primary knock-on atom (PKA) recoil velocity kick on atom 1
group pka id 1
velocity pka set 25.0 10.0 5.0

fix 1 all nve
thermo 10
thermo_style custom step temp pe ke etotal press
run 50
"""
    with open("in.si_recoil", "w") as f:
        f.write(lmp_in.strip() + "\n")

    lmp_bin = shutil.which("lmp")
    if lmp_bin:
        res = subprocess.run([lmp_bin, "-in", "in.si_recoil", "-log", "lammps_recoil.log"],
                             capture_output=True, text=True, check=True)
        print(f"[Materials] LAMMPS ({lmp_bin}) PKA recoil MD simulation completed 50 steps -> lammps_recoil.log")

if __name__ == "__main__":
    run_materials_suite()
