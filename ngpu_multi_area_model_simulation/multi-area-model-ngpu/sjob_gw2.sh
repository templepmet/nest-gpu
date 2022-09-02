#!/bin/bash

#SBATCH -J nestgpu              # Job name
#SBATCH -o ./joblog/job_%j_out.txt # Name of stdout output file (%j expands to %jobId)
#SBATCH -e ./joblog/job_%j_err.txt # Name of stderr output file (%j expands to %jobId)
#SBATCH -N 16                    # Total number of nodes requested
#SBATCH -n 32                    # Total number of mpi tasks requested
#SBATCH -c 6                     # Each process, number of core
#SBATCH -t 00:30:00             # Run time (hh:mm:ss) - 1.5 hours

echo Runnning on host `hostname`
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Current working directory is `pwd`"

# if [ "$#" -ne 1 ]; then
#     seed=12345
# else
#     seed=$1
# fi

# cat run_simulation.templ | sed "s/__seed__/$seed/g" > run_simulation.py

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# for Infiniband
mpirun --mca btl openib,self,vader singularity exec --nv --no-home ../../singularity/nestgpu.sif python run_simulation.py
# setting CUDA_VISIBLE_DEVICES=

echo ending
echo Time is `date`

sleep 5
