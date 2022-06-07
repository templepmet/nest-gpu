#!/bin/bash -e

source /etc/profile.d/lmod.sh
# source /etc/profile.d/lmod.csh
module load singularity
singularity build -f nest-gpu.sif nest-gpu.def
