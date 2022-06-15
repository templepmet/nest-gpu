#!/bin/bash -e

source /etc/profile.d/lmod.sh
# source /etc/profile.d/lmod.csh
module load singularity
# -fakeroot is not working
# singularity build -f nest-gpu.sif nest-gpu.def
singularity build nest-gpu.sif nest-gpu.def
singularity shell --nv nest-gpu.sif