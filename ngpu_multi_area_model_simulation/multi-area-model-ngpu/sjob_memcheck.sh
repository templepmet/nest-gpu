#!/bin/bash

#SBATCH -J nestgpu              # Job name
#SBATCH -o ./joblog/job_%j_out.txt # Name of stdout output file (%j expands to %jobId)
#SBATCH -e ./joblog/job_%j_err.txt # Name of stderr output file (%j expands to %jobId)
#SBATCH -N 2                    # Total number of nodes requested
#SBATCH -n 4                    # Total number of mpi tasks requested
#SBATCH -c 1                     # Each process, number of core
#SBATCH -t 00:05:00             # Run time (hh:mm:ss) - 1.5 hours

echo Runnning on host `hostname`
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Current working directory is `pwd`"

# srun -n $SLURM_NNODES hostname
# srun -n $SLURM_NNODES free -h
mpirun --mca btl openib,self,vader ./run.sh

echo ending
echo Time is `date`

sleep 5
