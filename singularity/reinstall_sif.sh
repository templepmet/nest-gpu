#!/bin/bash -e

singularity build -f --force --sandbox $WORK_DIR/nestgpu_sandbox $WORK_DIR/nestgpu.sif
./reinstall_sandbox.sh
singularity build -f --force $WORK_DIR/nestgpu.sif $WORK_DIR/nestgpu_sandbox
