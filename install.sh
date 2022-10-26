#!/bin/bash -eu

export MPICXX=mpicxx
# alias scorep-mpicxx="scorep-mpicxx -no-pie"
# export MPICXX=scorep-mpicxx

./clean.sh
autoreconf -i
./configure --prefix=$NEST_GPU --with-gpu-arch=sm_80 # for A100
make -j24
make install
