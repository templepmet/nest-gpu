#include <cstdio>
#include <mpi.h>
#include <cuda.h>

int main(int argc, char **argv)
{
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = 4;
    const int p = 2;
    int *d, *h;
    MPI_Status st;
    MPI_Request req;

    cudaMalloc(&d, sizeof(int) * n);
    cudaMallocHost(&h, sizeof(int) * n);

    if (rank == 1) {
       for (int i = 0; i < n; ++i) {
            h[i] = i + 1;
        }
        cudaMemcpy(d, h, sizeof(int) * n, cudaMemcpyHostToDevice);
    }

    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            printf("%d ", h[i]);
        }
        puts("");
    }

    if (rank == 1) {
        MPI_Isend(d, n, MPI_INT, 0, 0, MPI_COMM_WORLD, &req);
        MPI_Waitall(1, &req, &st);
    }
    else if (rank == 0) {
        MPI_Irecv(d, n, MPI_INT, 1, 0, MPI_COMM_WORLD, &req);
        MPI_Waitall(1, &req, &st);
    }

    cudaMemcpy(h, d, sizeof(int) * n, cudaMemcpyDeviceToHost);

    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            printf("%d ", h[i]);
        }
        puts("");
    }

    cudaFree(d);
    cudaFreeHost(h);
    MPI_Finalize();

    return 0;
}
