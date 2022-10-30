#include <iostream>

using namespace std;

int main() {
	cudaEvent_t start_d;
	cudaEvent_t stop_d;
	cudaEventCreate(&start_d);
	cudaEventCreate(&stop_d);

	cudaEventRecord(start_d);
	cudaEventRecord(stop_d);
	while (cudaEventQuery(stop_d) != cudaSuccess) {
		printf("not finished\n");
	}
	cudaEventSynchronize(stop_d);

	float ms;
	cudaEventElapsedTime(&ms, start_d, stop_d);
	printf("%f\n", ms);

	cudaEventDestroy(start_d);
	cudaEventDestroy(stop_d);

	return 0;
}