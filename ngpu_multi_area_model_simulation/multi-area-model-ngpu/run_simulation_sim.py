"""
This script is used to run a simulation

It initializes the network class and then runs the simulate method of
the simulation class instance.

"""

import json
import os
import sys
import numpy as np
from config import base_path, data_path
from multiarea_model import MultiAreaModel
from mpi4py import MPI
import shutil
from multiarea_model.default_params import nested_update, sim_params

import nestgpu as ngpu

ngpu.ConnectMpiInit()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
print(f'Hello World! I am process {rank} of {size} on {name}.\n')

num_processes = size
local_num_threads = int(os.environ['OMP_NUM_THREADS'])
if rank == 0:
    print(f'Simulation: {num_processes} process, {local_num_threads} thread')

# label = comm.bcast(label, root=0)
label = "d7864cf9b59aa22838dc1d557af5535d"

# fn = os.path.join(data_path,
#                   label,
#                   '_'.join(('custom_params',
#                             label,
#                            str(rank))))
fn = os.path.join(data_path,
                  label, '_'.join(('custom_params', label)))
with open(fn, 'r') as f:
    custom_params = json.load(f)

# os.remove(fn)

network_label = custom_params['network_label']

M = MultiAreaModel(network_label,
                   simulation=True,
                   sim_spec=custom_params['sim_params'])

M.simulation.simulate()
