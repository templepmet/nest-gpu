#!/bin/bash -e

sudo singularity build --force --sandbox nestgpu_sandbox nestgpu.sif
./reinstall_sandbox.sh
sudo singularity build --force nestgpu.sif nestgpu_sandbox
