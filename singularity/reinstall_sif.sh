#!/bin/bash -e

singularity build -f --force --sandbox nestgpu_sandbox nestgpu.sif
./reinstall_sandbox.sh
singularity build -f --force nestgpu.sif nestgpu_sandbox
