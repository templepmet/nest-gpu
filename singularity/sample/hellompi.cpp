#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
	int me, nprocs, namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &me);
	MPI_Get_processor_name(processor_name, &namelen);

	printf("Hello World! I'm  process %d of %d on %s\n", me, nprocs, processor_name);

	MPI_Finalize();

	return 0;
}