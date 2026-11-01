#!/usr/bin/env bash
# Tuned MPI launcher for Intel Core Ultra 5 125U (7 physical cores / 14 threads, 26 GB WSL2 RAM)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
NP=${SIM_MPI_RANKS:-6}
exec mpirun -np "$NP" "$@"
