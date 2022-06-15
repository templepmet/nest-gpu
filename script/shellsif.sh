#!/bin/bash -e

source /etc/profile.d/lmod.sh
# source /etc/profile.d/lmod.csh
module load singularity
# -fakeroot is not working
singularity shell --nv nest-gpu.sif