#!/bin/bash -e

GPUS_PER_NODE=2
export CUDA_VISIBLE_DEVICES=$(( OMPI_COMM_WORLD_RANK % GPUS_PER_NODE ))
singularity exec --nv --no-home ../../singularity/nestgpu.sif python run_simulation.py
