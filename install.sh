#!/bin/bash -eu

autoreconf -i
./configure --prefix=$NEST_GPU --with-gpu-arch=sm_61,sm_86 # GTX 1080, RTX A4500
make -j24
make install
