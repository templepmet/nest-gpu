#!/bin/bash -e

singularity build -f --force $WORK_DIR/nestgpu.sif nestgpu.def
