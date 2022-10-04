#!/bin/bash -e

TARGET_SANDBOX=$1
SOURCE_DEF=$2
HOST_USER=$3

source /etc/profile.d/lmod.sh
module load singularity
singularity build --force --sandbox $TARGET_SANDBOX $SOURCE_DEF
chown $HOST_USER $TARGET_SANDBOX
