#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	int mpi_id;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);

	int n = 8;
	int *sendbuf = new int[n];
	int *recvbuf = new int[20];
	int sendpls[2];
	int sendcount[2];
	int recvpls[2] = {0, 10};
	int recvcount[2] = {10, 10};

	if (mpi_id == 0)
	{
		sendcount[0] = 1;
		sendcount[1] = 6;
		sendpls[0] = 0;
		sendpls[1] = 1;

		recvcount[0] = 1;
		recvcount[1] = 10;
		recvpls[0] = 0;
		recvpls[1] = 10;
	}
	else
	{
		sendcount[0] = 5;
		sendcount[1] = 1;
		sendpls[0] = 0;
		sendpls[1] = 5;

		recvcount[0] = 10;
		recvcount[1] = 1;
		recvpls[0] = 0;
		recvpls[1] = 10;
	}

	for (int i = 0; i < n; ++i)
	{
		sendbuf[i] = i + 1;
	}

	memset(recvbuf, -1, sizeof(int) * 20);
	MPI_Alltoallv(sendbuf, sendcount, sendpls, MPI_INT,
				  recvbuf, recvcount, recvpls, MPI_INT, MPI_COMM_WORLD);

	// MPI_Alltoall(sendbuf, 4, MPI_INT,
	// 			 recvbuf, 4, MPI_INT, MPI_COMM_WORLD);

	for (int p = 0; p < 2; ++p)
	{
		if (p == mpi_id)
		{
			printf("rank%d: ", p);
			for (int i = 0; i < 20; ++i)
			{
				printf("%d, ", recvbuf[i]);
			}
			puts("");
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	delete sendbuf;
	delete recvbuf;

	MPI_Finalize();

	return 0;
}
