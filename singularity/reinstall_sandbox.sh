#!/bin/bash -e

singularity run --nv -w --bind ..:/opt/nestgpu_install,../nest-simulator-3.3:/opt/nest_install nestgpu_sandbox
