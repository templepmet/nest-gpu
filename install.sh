#!/bin/bash -eu

autoreconf -i
./configure --prefix=${NEST_GPU} --with-gpu-arch=sm_75
make -j8
make install
