#include <cub/cub.cuh>
#include <iostream>

void *d_temp_storage;
size_t temp_storage_bytes;

int main() {
    const int n = 10;
    int *h_in = new int[n];
    int *h_out = new int[n + 1];
    for (int i = 0; i < n; ++i) {
        h_in[i] = i + 1;
    }
    int *d_in;
    int *d_out;

    cudaMalloc(&d_in, sizeof(int) * n);
    cudaMalloc(&d_out, sizeof(int) * (n + 1));
    cudaMemcpy(d_in, h_in, sizeof(int) * n, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in,
                                  d_out, n + 1);
    printf("temp_bytes:%d\n", temp_storage_bytes);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // scan
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in,
                                  d_out, n + 1, stream);
    cudaMemcpyAsync(h_out, d_out, sizeof(int) * (n + 1), cudaMemcpyDeviceToHost,
                    stream);

    cudaStreamSynchronize(stream);

    for (int i = 0; i < n + 1; ++i) {
        printf("%d,", h_out[i]);
    }
    puts("");

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}