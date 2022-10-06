#!/bin/bash -e

GPUS_PER_NODE=8
export CUDA_VISIBLE_DEVICES=$(( OMPI_COMM_WORLD_RANK % GPUS_PER_NODE ))
singularity exec --nv --bind $SINGULARITY_PWD $SINGULARITY_IMAGE python run_simulation.py
