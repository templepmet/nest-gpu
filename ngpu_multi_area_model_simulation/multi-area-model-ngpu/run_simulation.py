import os
import json

import numpy as np
from mpi4py import MPI

from config import base_path, data_path
from multiarea_model import MultiAreaModel
from multiarea_model.default_params import nested_update
import nestgpu as ngpu


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
print(f'I am process {rank} of {size} on {name}.\n')

num_processes = size
local_num_threads = int(os.environ['OMP_NUM_THREADS'])

with open('label_info.json') as f:
    label_info = json.load(f)
simulation_label = label_info['simulation_label']

fn = os.path.join(data_path, simulation_label, '_'.join(('custom_params', simulation_label)))
with open(fn, 'r') as f:
    custom_params = json.load(f)
network_label = custom_params['network_label']

sim_params = custom_params['sim_params']
update_sim_params = {'num_processes': num_processes,
                  'local_num_threads': local_num_threads}
nested_update(sim_params, update_sim_params)

ngpu.ConnectMpiInit()

M = MultiAreaModel(network_label,
                   simulation=True,
                   sim_spec=custom_params['sim_params'])

M.simulation.simulate()
