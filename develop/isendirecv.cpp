#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	int mpi_id;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);

	int sendbuf[3] = {1, 2, 3};
	int recvbuf[3];

	MPI_Request request[2];
	MPI_Status status[2];

	MPI_Isend(sendbuf, 0, MPI_INT, 0, 1, MPI_COMM_WORLD, &request[0]);
	MPI_Irecv(recvbuf, 3, MPI_INT, 0, 1, MPI_COMM_WORLD, &request[1]);
	MPI_Waitall(4, request, status);

	for (int i = 0; i < 3; ++i)
	{
		printf("%d, ", recvbuf[i]);
	}
	puts("");

	MPI_Finalize();

	return 0;
}
