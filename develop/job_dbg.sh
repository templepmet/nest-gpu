#!/bin/sh
#----- qsub option -----
#PBS -q DBG
#PBS --group=${GROUP_NAME}
#PBS -l elapstim_req=00:10:00
#PBS -o ./log/job_%s_out.txt
#PBS -e ./log/job_%s_err.txt
#PBS -b 1  # ノード数
#PBS -l cpunum_job=76  # 1ノードあたりのCPUコア数 <=76
#PBS -l gpunum_job=8   # 1ノードあたりのGPU使用台数 <=8
#PBS -v OMP_NUM_THREADS=1
#PBS -T openmpi
#PBS -v NQSV_MPI_MODULE=BaseGPU/2022
#----- Program execution -----

cd $PBS_O_WORKDIR

module load BaseGPU/2022

RESULT_FILE=./log/result.txt

date > $RESULT_FILE
echo "PBS_JOBID: $PBS_JOBID" >> $RESULT_FILE

time \
	mpirun $NQSV_MPIOPTS -np 8 -npernode 8 --map-by core --bind-to core --display-devel-map \
	./wrap_cuda.sh 8 ./a.out \
	>> $RESULT_FILE
