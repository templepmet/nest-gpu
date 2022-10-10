#!/bin/bash -e

singularity build -f --force --sandbox nestgpu_sandbox nestgpu.def
