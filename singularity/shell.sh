#!/bin/bash -e

TARGET_SIF=$1
singularity shell --nv $TARGET_SIF
