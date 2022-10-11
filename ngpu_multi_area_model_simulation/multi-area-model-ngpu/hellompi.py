# hello_mpi.py:
# usage: python hello_mpi.py

from mpi4py import MPI
import sys

if __name__ == "__main__":
  size = MPI.COMM_WORLD.Get_size()
  rank = MPI.COMM_WORLD.Get_rank()
  name = MPI.Get_processor_name()

  local_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
  local_size = local_comm.Get_size()
  local_rank = local_comm.Get_rank()
  compute_nodes = size // local_size

  print(f"hostname={name}, rank={rank}, size={size}, local_rank={local_rank}, local_size={local_size}, compute_nodes={compute_nodes}")
