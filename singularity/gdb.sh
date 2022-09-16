#!/bin/bash -e

PID=$1
CURRENT=$(cd $(dirname $0);pwd)

source /etc/profile.d/lmod.sh
module load singularity
singularity exec $CURRENT/nestgpu.sif gdb -p $PID
