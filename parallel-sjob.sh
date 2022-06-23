#!/bin/bash

#SBATCH -J nestgpu              # Job name
#SBATCH -o job.%j.out           # Name of stdout output file (%j expands to %jobId)
#SBATCH -e job.%j.err           # Name of stderr output file (%j expands to %jobId)
#SBATCH -N 1                    # Total number of nodes requested
#SBATCH -n 2                    # Total number of mpi tasks requested
#SBATCH -t 00:30:00             # Run time (hh:mm:ss) - 1.5 hours
# # SBATCH --nodelist c16

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
# mpirun --mca btl openib,self,vader --mca btl_base_verbose 100 python hpc_benchmark.py

# for Gigabit network
# mpirun --mca btl tcp,self,vader singularity exec -e --nv nest-gpu.sif python python/examples/brunel_mpi.py 100000
# singularity exec -e --nv nest-gpu.sif python python/examples/example1.py # success
mpirun -np 2 hostname

echo ending
echo Time is `date`

sleep 5

