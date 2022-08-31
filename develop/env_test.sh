#!/bin/bash -eu

mpirun -np 2 singularity exec --nv --no-home ../singularity/nestgpu.sif python hellompi.py
# mpirun -np 2 singularity exec --nv --cleanenv ../singularity/nestgpu.sif ./a.out

mpirun -np 2 singularity exec --nv ../singularity/nestgpu.sif python hellompi.py 
# mpirun -np 2 singularity exec --nv ../singularity/nestgpu.sif ./a.out

# singularity exec --nv ../singularity/nestgpu.sif env > env.txt
