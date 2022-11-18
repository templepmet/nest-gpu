cd osu-micro-benchmarks
mkdir -p build
cd build
../configure CC=mpicc CXX=mpicxx
make
