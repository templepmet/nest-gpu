#!/bin/bash -e

mpirun -np 1 singularity exec --nv --no-home ../../singularity/nestgpu.sif python run_simulation.py
