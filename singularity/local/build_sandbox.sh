#!/bin/bash -e

singularity build -f --force --sandbox $WORK_DIR/nestgpu_sandbox nestgpu.def
