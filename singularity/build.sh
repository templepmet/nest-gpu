#!/bin/bash -e

SOURCE_DEF=nest-gpu_base.def
TARGET_DEF=nest-gpu.def
TARGET_SIF=nest-gpu.sif
HOST_USER=$USER

./gen_def.sh $SOURCE_DEF $TARGET_DEF
sudo -E ./_build_sif.sh $TARGET_SIF $TARGET_DEF $HOST_USER
