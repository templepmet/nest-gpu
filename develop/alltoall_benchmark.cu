#include <chrono>
#include <iostream>
#include <vector>

#include <mpi.h>
#include <nccl.h>

#define NCCL_CALL(call)                                                        \
    {                                                                          \
        ncclResult_t ncclStatus = call;                                        \
        if (ncclSuccess != ncclStatus) {                                       \
            fprintf(stderr,                                                    \
                    "ERROR: NCCL call \"%s\" in line %d of file %s failed "    \
                    "with "                                                    \
                    "%s (%d).\n",                                              \
                    #call, __LINE__, __FILE__, ncclGetErrorString(ncclStatus), \
                    ncclStatus);                                               \
            exit(ncclStatus);                                                  \
        }                                                                      \
    }

using namespace std;

int main(int argc, char *argv[]) {
    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // cudaSetDevice(0);

    printf("rank=%d,size=%d\n", rank, size);

    ncclUniqueId nccl_uid;
    if (rank == 0) {
        NCCL_CALL(ncclGetUniqueId(&nccl_uid));
    }
    MPI_Bcast(&nccl_uid, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t nccl_comm;
    NCCL_CALL(ncclCommInitRank(&nccl_comm, size, nccl_uid, rank));

    int *d_sendbuf;
    int *d_recvbuf;
    int *h_sendbuf;
    int *h_recvbuf;
    // int nsize = 1 << 28;  // 256M
    int nsize = 100;

    cudaMalloc(&d_sendbuf, sizeof(int) * nsize * size);
    cudaMalloc(&d_recvbuf, sizeof(int) * nsize * size);
    cudaMallocHost(&h_sendbuf, sizeof(int) * nsize * size);
    cudaMallocHost(&h_recvbuf, sizeof(int) * nsize * size);

    // for (int i = 0; i < size; ++i) {
    //     for (int j = 0; j < rank + 1; ++j) {
    //         h_buf[i * size + j] = (i + 1) * (j + 1);
    //     }
    // }
    // cudaMemcpy(d_sendbuf, h_buf, sizeof(int) * size * size,
    //            cudaMemcpyHostToDevice);
    std::vector<int> send_num(size);
    vector<int> recv_num(size, 0);
    for (int i = 0; i < size; ++i) {
        send_num[i] = nsize - i;
    }
    for (int i = 0; i < size; ++i) {
        recv_num[i] = send_num[rank];
    }
    vector<int> send_cumul(size + 1, 0);
    vector<int> recv_cumul(size + 1, 0);
    for (int i = 0; i < size; ++i) {
        send_cumul[i + 1] += send_cumul[i] + nsize;
        recv_cumul[i + 1] += recv_cumul[i] + nsize;
    }

    std::chrono::system_clock::time_point start, end;
    int iter = 100;

    {  // nccl
        start = std::chrono::system_clock::now();
        for (int k = 0; k < iter; ++k) {
            ncclGroupStart();
            for (int i = 0; i < size; ++i) {
                ncclSend(d_sendbuf + i * nsize, send_num[i], ncclInt, i,
                         nccl_comm, 0);
                // int cnt = (i == rank ? i + 1 : size);
                ncclRecv(d_recvbuf + i * nsize, recv_num[i], ncclInt, i,
                         nccl_comm, 0);
            }
            ncclGroupEnd();
        }

        end = std::chrono::system_clock::now();
        double dt = static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count() /
            1e6);
        if (rank == 0) {
            printf("nccl alltoall: %lf[s]\n", dt);
        }
    }
    {  // cuda-aware-mpi
        start = std::chrono::system_clock::now();
        for (int i = 0; i < iter; ++i) {
            MPI_Alltoallv(d_sendbuf, send_num.data(), send_cumul.data(),
                          MPI_INT, d_recvbuf, recv_num.data(),
                          recv_cumul.data(), MPI_INT, MPI_COMM_WORLD);
        }
        end = std::chrono::system_clock::now();
        double dt = static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count() /
            1e6);
        if (rank == 0) {
            printf("cuda-aware-mpi: %lf[s]\n", dt);
        }
    }
    {  // standard-mpi
        start = std::chrono::system_clock::now();
        for (int i = 0; i < iter; ++i) {
            cudaMemcpy(h_sendbuf, d_sendbuf, sizeof(int) * nsize * size,
                       cudaMemcpyDeviceToHost);
            MPI_Alltoallv(h_sendbuf, send_num.data(), send_cumul.data(),
                          MPI_INT, h_recvbuf, recv_num.data(),
                          recv_cumul.data(), MPI_INT, MPI_COMM_WORLD);
            cudaMemcpy(d_recvbuf, h_recvbuf, sizeof(int) * nsize * size,
                       cudaMemcpyHostToDevice);
        }
        end = std::chrono::system_clock::now();
        double dt = static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count() /
            1e6);
        if (rank == 0) {
            printf("srandard-mpi: %lf[s]\n", dt);
        }
    }

    // ncclはデータサイズが大きい場合に有効：レイテンシが大きいが送受信時間は小さい
    // cuda-aware-mpiは~10MB程度で有効：レイテンシが小さく送受信時間もそこそこ
    // standard-mpiは極小で有効：レイテンシが最も小さいが送受信＋データ転送時間を有する

    // for (int i = 0; i < size; ++i) {
    //     if (i == rank) {
    //         printf("rank%d:", i);
    //         for (int j = 0; j < size * size; ++j) {
    //             printf("%d,", h_buf[j]);
    //         }
    //         puts("");
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    cudaFreeHost(h_recvbuf);
    cudaFreeHost(h_sendbuf);
    cudaFree(d_recvbuf);
    cudaFree(d_sendbuf);

    NCCL_CALL(ncclCommDestroy(nccl_comm));
    MPI_Finalize();

    return 0;
}

// mpic++ nccl.cu -L
// /system/apps/rhel8/gpu/nvhpc/nvhpc22.2/22.2/Linux_x86_64/22.2/comm_libs/nccl/lib
// -lnccl
