from time import sleep
from mpi4py import MPI
import nest
import sys

def print_hello(rank, size, name):
  msg = "Hello World! I am process {0} of {1} on {2}.\n"
  sys.stdout.write(msg.format(rank, size, name))

if __name__ == "__main__":
	size = MPI.COMM_WORLD.Get_size()
	rank = MPI.COMM_WORLD.Get_rank()
	name = MPI.Get_processor_name()

	print_hello(rank, size, name)

	MPI.COMM_WORLD.Barrier()
	print('after barrier')
	
	if rank == 0:
		print(nest.dict_miss_is_error)
	if rank == 1:
		sleep(5)
		print(nest.dict_miss_is_error)
