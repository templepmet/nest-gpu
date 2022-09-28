import os
import re
import json

import numpy as np
from mpi4py import MPI
import pynvml

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
theory_label = label_info['theory_label']

fn = os.path.join(data_path, theory_label, '_'.join(('custom_params', theory_label)))
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

label_info['simulation_label'] = M.simulation.label
with open('label_info.json', 'w') as f:
    json.dump(label_info, f)

M.simulation.simulate()

def getMemInfo():
	pid = os.getpid()
	memInfo = {}
	with open(f'/proc/{pid}/status') as f:
		text = f.read()
		memData = re.findall('(Vm.+?):\t.*?(\d*)\skB', text)
		for data in memData:
			memInfo[data[0]] = int(data[1])
		return memInfo

print(f'MPI Rank {rank} : Host Memory : ', getMemInfo())

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
nvmlMemInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f'MPI Rank {rank} : GPU Memory : ', nvmlMemInfo)
pynvml.nvmlShutdown()

# free NEST-GPU memory via call deconstructa
