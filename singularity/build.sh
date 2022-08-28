#!/bin/bash -e

SOURCE_DEF=nestgpu.def
TARGET_SIF=nestgpu.sif
HOST_USER=$USER

sudo -E ./_build_sif.sh $TARGET_SIF $SOURCE_DEF $HOST_USER
