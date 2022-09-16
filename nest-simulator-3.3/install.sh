#!/bin/bash -eu

if [ -d build ]; then
	cd build
else
	mkdir -p build
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
fi

make -j24
rm -rf $NEST
make install
