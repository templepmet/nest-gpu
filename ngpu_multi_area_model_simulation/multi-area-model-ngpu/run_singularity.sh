#!/bin/bash -e

# OMP_NUM_THREADS=4 mpirun -np 1 singularity exec --nv --no-home ../../singularity/nestgpu.sif python run_simulation.py

# OMP_NUM_THREADS=4 mpirun -np 1 singularity exec --nv --no-home ../../singularity/nestgpu.sif python run_simulation_init.py
OMP_NUM_THREADS=4 mpirun -np 32 singularity exec --nv --no-home ../../singularity/nestgpu.sif python run_simulation_sim.py
