#!/bin/bash -eu

rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=$NEST \
	-Dwith-python=ON \
	-Dwith-mpi=ON \
	-Dwith-openmp=ON \
	-Dwith-boost=ON \
	-Dwith-readline=ON \
	-Dwith-ltdl=ON \
	-Dwith-gsl=ON \
	..
make -j8
rm -rf $NEST
make install
