#!/bin/bash -e

TARGET_SIF=$1
SOURCE_DEF=$2
HOST_USER=$3

# source /etc/profile.d/lmod.sh
# module load singularity
singularity build --force $TARGET_SIF $SOURCE_DEF
chown $HOST_USER $TARGET_SIF
