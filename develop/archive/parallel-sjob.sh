#!/bin/bash

#SBATCH -J nestgpu              # Job name
#SBATCH -o ./log/job_%j_out.txt # Name of stdout output file (%j expands to %jobId)
#SBATCH -e ./log/job_%j_err.txt # Name of stderr output file (%j expands to %jobId)
#SBATCH -N 4                    # Total number of nodes requested
#SBATCH -n 4                    # Total number of mpi tasks requested
#SBATCH -t 00:30:00             # Run time (hh:mm:ss) - 1.5 hours

echo Runnning on host `hostname`
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Current working directory is `pwd`"

# Run the parallel MPI executable "a.out"
# --btl openib,self,vader               Infiniband
# --btl tcp,self,vader                  Gigabit Ethernet
# --mca btl_base_verbose 100            for debug
# execute ompi_info for more details

# for Infiniband
mpirun --mca btl openib,self,vader singularity exec --nv nest-gpu.sif python brunel_mpi_without_remote.py 1000000

# for Gigabit network
# mpirun --mca btl tcp,self,vader singularity exec --nv nest-gpu.sif python brunel_mpi_without_remote.py 1000000
# mpirun --mca btl tcp,self,vader singularity exec --nv nest-gpu.sif python brunel_mpi_without_remote.py 100000 > result/result_${SLURM_JOB_ID}.txt

echo ending
echo Time is `date`

sleep 5

