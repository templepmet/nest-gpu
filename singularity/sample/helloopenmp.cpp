#include <stdio.h>
#include <omp.h>

int main(int argc, char **argv)
{
#pragma omp parallel num_threads(4)
	{
		int id = omp_get_thread_num();
		printf("id=%d\n", id);
	}

	return 0;
}