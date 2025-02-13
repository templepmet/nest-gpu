#!/bin/bash

#SBATCH -J nestgpu              # Job name
#SBATCH -o ./log/job_%j_out.txt # Name of stdout output file (%j expands to %jobId)
#SBATCH -e ./log/job_%j_err.txt # Name of stderr output file (%j expands to %jobId)
#SBATCH -N 1                    # Total number of nodes requested
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -c 12                     # Each process, number of core
#SBATCH -t 01:00:00             # Run time (hh:mm:ss) - 1.5 hours

echo Runnning on host `hostname`
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Current working directory is `pwd`"

python distributions/firing_rate.py ../multi-area-model-ngpu/b0de5c560c17c5d6c29a7d808f1f2de4/recordings/

echo ending
echo Time is `date`

sleep 5
