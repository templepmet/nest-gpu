#!/bin/sh
#----- qsub option -----
#PBS -q DBG
#PBS --group=${GROUP_NAME}
#PBS -l elapstim_req=00:10:00
#PBS -o job_%s_out.txt
#PBS -e job_%s_err.txt
#PBS -b 1  # ノード数
#PBS -l cpunum_job=76  # 1ノードあたりのCPUコア数 <=76
#PBS -v OMP_NUM_THREADS=76
#PBS -T openmpi
#PBS -v NQSV_MPI_MODULE=BaseGPU/2022
#----- Program execution -----

cd $PBS_O_WORKDIR

module load BaseGPU/2022

RESULT_FILE=result.txt

date > $RESULT_FILE
echo "PBS_JOBID: $PBS_JOBID" >> $RESULT_FILE

python delay.py >> $RESULT_FILE
