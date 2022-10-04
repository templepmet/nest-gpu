#!/bin/bash -eu

autoreconf -i
./configure --prefix=$NEST_GPU --with-gpu-arch=sm_80 # for A100
make -j24
make install
