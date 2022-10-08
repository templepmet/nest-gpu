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
#PBS -v OMP_NUM_THREADS=2
#PBS -T openmpi
#PBS -v NQSV_MPI_MODULE=BaseGPU/2022
#----- Program execution -----

cd $PBS_O_WORKDIR

module load BaseGPU/2022

SINGULARITY_PWD=`mpirun -np 1 pwd`
SINGULARITY_IMAGE=$WORK_DIR/nestgpu.sif
RESULT_FILE=./log/result.txt

if ls log/*.txt >/dev/null 2>&1
then
	mv log/*.txt log/old/
fi

date > $RESULT_FILE
echo "PBS_JOBID: $PBS_JOBID" >> $RESULT_FILE

SCALING=0.01
time \
	mpirun $NQSV_MPIOPTS -np 1 --display-devel-map \
	singularity exec --nv --bind $SINGULARITY_PWD $SINGULARITY_IMAGE python run_theory.py $SCALING \
	>> $RESULT_FILE

time \
	mpirun $NQSV_MPIOPTS -np 32 -npernode 32 --map-by core --bind-to core --display-devel-map \
	./wrap_cuda.sh singularity exec --nv --bind $SINGULARITY_PWD $SINGULARITY_IMAGE python run_simulation.py \
	>> $RESULT_FILE

SIM_LABEL=$(awk 'match($0, /"simulation_label": "(.*)"\}/, a){print a[1]}' label_info.json)
if ls $SIM_LABEL/job*.txt >/dev/null 2>&1
then
	mkdir -p $SIM_LABEL/old_job
	mv $SIM_LABEL/job*.txt $SIM_LABEL/old_job
fi
cp ./log/result.txt $SIM_LABEL/
