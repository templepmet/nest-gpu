#include <cuda.h>
#include <mpi.h>

#include <cstdio>
#include <vector>

using namespace std;

int main(int argc, char **argv) {
    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("rank:%d, size:%d\n", rank, size);

    int *d_sendbuf;
    int *d_recvbuf;
    int *h_buf;

    cudaMalloc(&d_sendbuf, sizeof(int) * size * size);
    cudaMalloc(&d_recvbuf, sizeof(int) * size * size);
    cudaMallocHost(&h_buf, sizeof(int) * size * size);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < rank + 1; ++j) {
            h_buf[i * size + j] = (i + 1) * (j + 1);
        }
    }
    cudaMemcpy(d_sendbuf, h_buf, sizeof(int) * size * size,
               cudaMemcpyHostToDevice);

    vector<MPI_Request> req(size * 2);
    vector<MPI_Status> st(size * 2);
    MPI_Request req_ata;
    MPI_Status st_ata;
    vector<int> send_num(size, 0);
    vector<int> recv_num(size, 0);
    vector<int> send_cumul(size + 1, 0);
    vector<int> recv_cumul(size + 1, 0);
    for (int i = 0; i < size; ++i) {
        send_num[i] = rank + 1;
        recv_num[i] = size;
        send_cumul[i + 1] += send_cumul[i] + size;
        recv_cumul[i + 1] += recv_cumul[i] + size;
    }
    recv_num[rank] = rank + 1;
    MPI_Ialltoallv(d_sendbuf, send_num.data(), send_cumul.data(), MPI_INT,
                   d_recvbuf, recv_num.data(), recv_cumul.data(), MPI_INT,
                   MPI_COMM_WORLD, &req_ata);
    MPI_Wait(&req_ata, &st_ata);
    // for (int i = 0; i < size; ++i) {
    //     MPI_Isend(&d_sendbuf[i * size], rank + 1, MPI_INT, i, 0,
    //     MPI_COMM_WORLD,
    //               &req[i]);
    // }
    // for (int i = 0; i < size; ++i) {
    //     MPI_Irecv(&d_recvbuf[i * size], size, MPI_INT, i, 0, MPI_COMM_WORLD,
    //               &req[i + size]);
    // }
    // MPI_Waitall(size * 2, &req[0], &st[0]);

    cudaMemcpy(h_buf, d_recvbuf, sizeof(int) * size * size,
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        if (i == rank) {
            printf("rank%d:", i);
            for (int j = 0; j < size * size; ++j) {
                printf("%d,", h_buf[j]);
            }
            puts("");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    cudaFree(d_sendbuf);
    cudaFree(d_recvbuf);
    cudaFreeHost(h_buf);
    MPI_Finalize();

    return 0;
}
