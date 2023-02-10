import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import matplotlib; matplotlib.use('agg')

argv = sys.argv
if len(argv) < 2:
    print(f"usage: python {argv[0]} simulation_directory")
    sys.exit(1)
sim_dir = argv[1]
result_file = os.path.join(sim_dir, "result.txt")

time_label = [
    "SpikeBufferUpdate_time",
    "poisson_generator_time",
    "neuron_Update_time",
    "copy_ext_spike_time",
    "SendExternalSpike_time",
    "PackSendSpike_time",
    "SendRecvSpikeRemote_immed_time",
    "SendRecvSpikeRemote_delay_time",
    "UnpackRecvSpike_time",
    "CopySpikeFromRemote_time",
    "MpiBarrier_time",
    "copy_spike_time",
    "ClearGetSpikeArrays_time",
    "NestedLoop_time",
    "GetSpike_time",
    "SpikeReset_time",
    "ExternalSpikeReset_time",
    "RevSpikeBufferUpdate_time",
    "BufferRecSpikeTimes_time",
    "Other_time"
]

# definition dependeny of host/device include time
# definition else


# only last time = sum time
procs = 32
sum_time = defaultdict(lambda: 0)

with open(result_file) as f:
    text = f.read()
for p in range(procs):
    for lab in time_label:
        times = re.findall(f"MPI rank {p} :   {lab}: (.*)\(def\), (.*)\(host\), (.*)\(device\)\n", text)[-1]
        each_time = float(times[0])
        sum_time[lab] += each_time

print(sum_time)
