#!/bin/bash -e

singularity run -f -w --bind ..:/opt/nestgpu_install,../nest-simulator-3.3:/opt/nest_install $WORK_DIR/nestgpu_sandbox
