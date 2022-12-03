import re
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("agg")
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
    "BufferRecSpikeTimes_time",
    "Blocking_time",
]

part = {
    "SpikeBufferUpdate_time": "calc",
    "poisson_generator_time": "calc",
    "neuron_Update_time": "calc",
    "copy_ext_spike_time": "deliv",
    "SendExternalSpike_time": "deliv",
    "SendSpikeToRemote_time": "comm",
    "RecvSpikeFromRemote_time": "comm",
    "CopySpikeFromRemote_time": "comm",
    "MpiBarrier_time": "comm",
    "copy_spike_time": "deliv",
    "ClearGetSpikeArrays_time": "deliv",
    "NestedLoop_time": "deliv",
    "GetSpike_time": "calc",
    "SpikeReset_time": "calc",
    "ExternalSpikeReset_time": "calc",
    "RevSpikeBufferUpdate_time": "calc",
    "BufferRecSpikeTimes_time": "calc",
    "Blocking_time": None,
}

# definition dependeny of host/device include time
# definition else


# only last time = sum time
procs = 32
each_time = [None] * procs
host_time = [None] * procs
dev_time = [None] * procs
sum_time = [0] * procs
for i in range(procs):
    each_time[i] = {}
    host_time[i] = {}
    dev_time[i] = {}
build_time = [0] * procs
sim_time = [0] * procs

with open(result_file) as f:
    text = f.read()
for p in range(procs):
    for lab in time_label:
        times = re.findall(
            f"MPI rank {p} :   {lab}: (.*)\(def\), (.*)\(host\), (.*)\(device\)\n", text
        )[-1]
        each_time[p][lab] = float(times[0])
        host_time[p][lab] = float(times[1])
        dev_time[p][lab] = float(times[2])
        build_time[p] = float(
            re.findall(f"MPI rank {p} : Building time: (.*)\n", text)[-1]
        )
        sim_time[p] = float(
            re.findall(f"MPI rank {p} : Simulation time: (.*)\n", text)[-1]
        )
        sum_time[p] += each_time[p][lab]
    print(f"Rank={p}: {each_time[p]}")
max_time = max(sim_time)

# print(sim_time)
# print(sum_time)
# print((np.array(sim_time) - np.array(sum_time)) / np.array(sim_time))
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
plot_label = ["calc", "comm", "deliv"]
y = {}
for l in plot_label:
    y[l] = [0.0] * procs

# else_y = [0.0] * procs
# y["Else_time"] = 0
for lab in time_label:
    yt = [0.0] * procs
    for p in range(procs):
        yt[p] = each_time[p][lab]
        yt[p] = max(yt[p], 0.0)  # fix
    if part[lab] is not None:
        y[part[lab]] += np.array(yt)

prob = 0.9975
max_p = -1
max_t = 0
max_ylim = 0
for p in range(procs):
    sum_t = (1 - prob) * y["comm"][p] + max(
        prob * y["comm"][p], y["calc"][p] + y["deliv"][p]
    )
    tmp_ylim = y["comm"][p] + y["calc"][p] + y["deliv"][p]
    max_ylim = max(max_ylim, tmp_ylim)
    if max_t < sum_t:
        max_t = sum_t
        max_p = p

max_calc = y["calc"][max_p]
max_deliv = y["deliv"][max_p]
for p in range(procs):
    y["comm"][p] = max(
        (1 - prob) * y["comm"][p], prob * y["comm"][p] - max_calc - max_deliv
    )

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.size"] = 12
plt.figure()
plt.grid()
plt.xlabel("MPI Process")
plt.ylabel("Processing Time [s]")
bottom = np.array([0.0] * procs)
x = [i for i in range(procs)]
idx = 0
for lab in plot_label:
    plt.bar(
        x,
        y[lab],
        bottom=bottom,
        align="center",
        label=lab,
    )
    bottom += np.array(y[lab])
    idx += 1
plt.ylim(0.0, np.max(max_ylim) * 1.1)

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 0), loc="lower left")
plt.savefig(
    os.path.join(sim_dir, "time_3p_effect.png"), bbox_inches="tight", pad_inches=0.2
)

print(max_p)
print(bottom)
print(np.argmax(bottom))
print(max(bottom))
