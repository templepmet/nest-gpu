import json
import os
import re

import nestgpu as ngpu
import numpy as np
import pynvml
from config import base_path, data_path
from mpi4py import MPI
from multiarea_model import MultiAreaModel, MultiAreaModelPar
from multiarea_model.default_params import nested_update

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
local_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
local_size = local_comm.Get_size()
local_rank = local_comm.Get_rank()

def print_mpi_info():
    name = MPI.Get_processor_name()
    compute_nodes = size // local_size

    print(
        f"hostname={name}, rank={rank}, size={size}, local_rank={local_rank}, local_size={local_size}, compute_nodes={compute_nodes}"
    )

def print_mem_info():
    def getMemInfo():
        pid = os.getpid()
        memInfo = {}
        with open(f"/proc/{pid}/status") as f:
            text = f.read()
            memData = re.findall("(Vm.+?):\t.*?(\d*)\skB", text)
            for data in memData:
                memInfo[data[0]] = int(data[1])
            return memInfo

    print(f"MPI Rank {rank} : Host Memory : ", getMemInfo())

    pynvml.nvmlInit()
    numGpus = pynvml.nvmlDeviceGetCount()
    handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank % numGpus)
    nvmlMemInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"MPI Rank {rank} : GPU Memory : ", nvmlMemInfo)
    pynvml.nvmlShutdown()

def simulation(label):
    ngpu.ConnectMpiInit()
    # ngpu.AddMode("dump_syndelay")
    # ngpu.AddMode("dump_comm_dist")
    # ngpu.AddMode("comm_throughput")
    # ngpu.AddMode("comm_cuda")
    # ngpu.AddMode("comm_overlap")
    # ngpu.AddMode("comm_persistent")
    # build_par = False
    build_par = True
    if build_par:
        ngpu.AddMode("build_par")
        M = MultiAreaModelPar(label=label, network_spec=label, simulation=True, sim_spec=label)
    else:
        M = MultiAreaModel(label=label, network_spec=label, simulation=True, sim_spec=label)
    M.simulation.simulate()

def main():
    print_mpi_info()
    with open("sim_info.json") as f:
        label = json.load(f)["label"]
    simulation(label)
    print_mem_info()
    # free NEST-GPU memory via call deconstructa

if __name__ == "__main__":
    main()
