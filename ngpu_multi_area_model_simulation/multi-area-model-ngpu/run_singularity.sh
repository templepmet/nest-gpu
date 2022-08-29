#!/bin/bash -e

singularity exec --nv ../../singularity/nestgpu.sif python run_simulation.py
