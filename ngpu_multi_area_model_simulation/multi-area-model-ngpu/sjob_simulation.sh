#!/bin/bash

#SBATCH -J nestgpu              # Job name
#SBATCH -o ./log/job_%j_out.txt # Name of stdout output file (%j expands to %jobId)
#SBATCH -e ./log/job_%j_err.txt # Name of stderr output file (%j expands to %jobId)
#SBATCH -N 16                    # Total number of nodes requested
#SBATCH -n 32                    # Total number of mpi tasks requested
#SBATCH -c 6                     # Each process, number of core
#SBATCH -t 01:00:00             # Run time (hh:mm:ss) - 1.5 hours
# # SBATCH --exclude=c[1-3]

echo Runnning on host `hostname`
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Current working directory is `pwd`"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mpirun --mca btl openib,self,vader ./run.sh >> ./log/result.txt

echo ending
echo Time is `date`

sleep 5
