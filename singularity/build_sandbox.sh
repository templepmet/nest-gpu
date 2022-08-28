#!/bin/bash -e

SOURCE_DEF=nestgpu.def
TARGET_SANDBOX=nestgpu_sandbox
HOST_USER=$USER

sudo -E ./_build_sandbox.sh $TARGET_SANDBOX $SOURCE_DEF $HOST_USER
