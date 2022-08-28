#!/bin/bash -e

singularity run --nv -w --bind ..:/opt/nestgpu_install nestgpu_sandbox
