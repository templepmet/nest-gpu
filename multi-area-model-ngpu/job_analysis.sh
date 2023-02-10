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

if ls log/*.txt >/dev/null 2>&1; then
	mv log/*.txt log/old/
fi

# below "cp result.txt"
LABEL=4nodes_8gpus_N0.5_K0.5_T0.5_0:357504.sqd
RESULT_FILE=./simulation_result/$LABEL/result.txt
python analysis/each_proc/time.py simulation_result/$LABEL >> $RESULT_FILE
python analysis/each_proc/time_overlap.py simulation_result/$LABEL >> $RESULT_FILE
python analysis/each_proc/time_each_label_sum.py simulation_result/$LABEL >> $RESULT_FILE
# python analysis/each_proc/time_comm_wait.py simulation_result/$LABEL >> $RESULT_FILE
python analysis/each_proc/memory.py simulation_result/$LABEL >> $RESULT_FILE
python analysis/each_proc/neuron.py simulation_result/$LABEL >> $RESULT_FILE
python analysis/each_proc/spike.py simulation_result/$LABEL >> $RESULT_FILE
if [ -d ./simulation_result/$LABEL/syndelay ]; then
	python analysis/each_proc/synapse.py simulation_result/$LABEL >> $RESULT_FILE
	python analysis/distributions/delay_connection.py simulation_result/$LABEL >> $RESULT_FILE
	python analysis/distributions/delay_spike.py simulation_result/$LABEL >> $RESULT_FILE
fi
if [ -d ./simulation_result/$LABEL/comm ]; then
	python analysis/distributions/comm_spike.py simulation_result/$LABEL >> $RESULT_FILE
fi
