import re
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

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
    "SendSpikeToRemote_time",
    "RecvSpikeFromRemote_time",
    "CopySpikeFromRemote_time",
    "MpiBarrier_time",
    "copy_spike_time",
    "ClearGetSpikeArrays_time",
    "NestedLoop_time",
    "GetSpike_time",
    "SpikeReset_time",
    "ExternalSpikeReset_time",
    "RevSpikeBufferUpdate_time",
    "BufferRecSpikeTimes_time"
]

# definition dependeny of host/device include time
# definition else


# only last time = sum time
procs = 32
host_time = [None] * procs
device_time = [None] * procs
sum_time = [0] * procs
for i in range(procs):
    host_time[i] = {}
    device_time[i] = {}
build_time = [0] * procs
sim_time = [0] * procs

with open(result_file) as f:
    text = f.read()
for p in range(procs):
    for lab in time_label:
        ret = re.findall(f"MPI rank {p} :   {lab}: (.*), (.*)\n", text)[-1]
        host_time[p][lab] = float(ret[0])
        device_time[p][lab] = float(ret[1])

        build_time[p] = float(
            re.findall(f"MPI rank {p} : Building time: (.*)\n", text)[-1]
        )
        sim_time[p] = float(
            re.findall(f"MPI rank {p} : Simulation time: (.*)\n", text)[-1]
        )
        sum_time[p] += host_time[p][lab]

print(sim_time)
print(sum_time)
print((np.array(sim_time) - np.array(sum_time)) / np.array(sim_time))
# reason for differ
# building time is constant: build_real_time_ - start_real_time_
# simulation time is variable: end_real_time_ - build_real_time_
#  start_real_time_: create constructa real time
#  build_real_time_: first simulation real time
#  end_real_time_: each finish simulation real time
# -> so, end_real_time_ include else processing time not include simulatiobn
#   ex.) transfer GPU->CPU spike record data, python overhead...
# not ignore transfer GPU->CPU GPU->CPU spike record data time !!!!!!!!!

# time
# convert for plot
y = {}
for lab in time_label:
    yt = [0.0] * procs
    for p in range(procs):
        yt[p] = host_time[p][lab]
    y[lab] = yt

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.size"] = 12
plt.figure()
plt.grid()
plt.xlabel("MPI Process")
plt.ylabel("Processing Time [s]")
bottom = np.array([0.0] * procs)
x = [i for i in range(procs)]
idx = 0
for lab in time_label:
    plt.bar(
        x,
        y[lab],
        bottom=bottom,
        align="center",
        label=lab,
        color=matplotlib.cm.tab20(idx),
    )
    bottom += np.array(y[lab])
    idx += 1

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 1), loc="upper left")
plt.savefig(os.path.join(sim_dir, "time.png"), bbox_inches="tight", pad_inches=0.2)
