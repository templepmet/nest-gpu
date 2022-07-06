#!/bin/bash -eu

autoreconf -i
# ./configure --prefix=${NEST_GPU} --with-gpu-arch=sm_75 # RTX 2080 Ti
./configure --prefix=${NEST_GPU} --with-gpu-arch=sm_61 # GTX 1080
make -j8
make install
