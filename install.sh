#!/bin/bash -eu

autoreconf -i
ARCH_FLAG="-arch=sm_61 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_86,code=sm_86"
./configure --prefix=$NEST_GPU --with-gpu-arch="$ARCH_FLAG"
make -j24
make install
