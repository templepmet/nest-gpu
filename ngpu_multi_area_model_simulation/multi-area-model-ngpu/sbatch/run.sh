#!/bin/bash -e

GPUS_PER_NODE=2
HOSTNAME=`hostname`
if [ $HOSTNAME = "c1" ]; then
	export CUDA_VISIBLE_DEVICES=0
else
	export CUDA_VISIBLE_DEVICES=$(( OMPI_COMM_WORLD_RANK % GPUS_PER_NODE ))
fi
singularity exec --nv --no-home ../../singularity/nestgpu.sif python run_simulation.py
