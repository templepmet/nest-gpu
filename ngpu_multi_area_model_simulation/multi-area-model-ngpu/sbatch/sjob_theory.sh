#!/bin/bash

#SBATCH -J nestgpu              # Job name
#SBATCH -o ./log/job_%j_out.txt # Name of stdout output file (%j expands to %jobId)
#SBATCH -e ./log/job_%j_err.txt # Name of stderr output file (%j expands to %jobId)
#SBATCH -N 1                    # Total number of nodes requested
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -c 1                     # Each process, number of core
#SBATCH -t 01:00:00             # Run time (hh:mm:ss) - 1.5 hours

echo Runnning on host `hostname`
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Current working directory is `pwd`"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
singularity exec --nv --no-home ../../singularity/nestgpu.sif python run_theory.py $SCALING

echo ending
echo Time is `date`

sleep 5
