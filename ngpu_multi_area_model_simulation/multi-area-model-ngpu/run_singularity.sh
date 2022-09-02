#!/bin/bash -e

OMP_NUM_THREADS=4 mpirun -np 1 singularity exec --nv --no-home ../../singularity/nestgpu.sif python run_simulation.py

# cd ../../develop/
# mpirun -np 2 singularity exec --nv --no-home ../singularity/nestgpu.sif python hellompi.py
