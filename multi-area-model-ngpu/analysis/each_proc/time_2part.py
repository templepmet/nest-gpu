import os
import re
import sys
from collections import defaultdict

import japanize_matplotlib
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
    "Blocking_time"
]

part = {
    "SpikeBufferUpdate_time": "calc",
    "poisson_generator_time": "calc",
    "neuron_Update_time": "calc",
    "copy_ext_spike_time": "calc",
    "SendExternalSpike_time": "calc",
    "SendSpikeToRemote_time": "comm",
    "RecvSpikeFromRemote_time": "comm",
    "CopySpikeFromRemote_time": "comm",
    "MpiBarrier_time": "comm",
    "copy_spike_time": "calc",
    "ClearGetSpikeArrays_time": "calc",
    "NestedLoop_time": "calc",
    "GetSpike_time": "calc",
    "SpikeReset_time": "calc",
    "ExternalSpikeReset_time": "calc",
    "RevSpikeBufferUpdate_time": "calc",
    "BufferRecSpikeTimes_time": "calc",
    "Blocking_time": None
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
        times = re.findall(f"MPI rank {p} :   {lab}: (.*)\(def\), (.*)\(host\), (.*)\(device\)\n", text)[-1]
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

# time
# convert for plot
plot_label = ["calc", "comm"]
label_table = {"calc": "計算", "comm": "通信"}
plot_color = {"calc":"tab:orange", "comm":"tab:blue"}
y = {}
for l in plot_label:
    y[l] = [0.0] * procs

# else_y = [0.0] * procs
# y["Else_time"] = 0
for lab in time_label:
    yt = [0.0] * procs
    for p in range(procs):
        yt[p] = each_time[p][lab]
        yt[p] = max(yt[p], 0.0) # fix
    if part[lab] is not None:
        y[part[lab]] += np.array(yt)

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.size"] = 16
plt.figure()
plt.grid()
plt.xlabel("GPU ID")
plt.ylabel("処理時間 [s]")
bottom = np.array([0.0] * procs)
x = [i for i in range(procs)]
idx = 0
for lab in plot_label:
    plt.bar(
        x,
        y[lab],
        bottom=bottom,
        # color=plot_color[lab],
        align="center",
        label=label_table[lab],
    )
    bottom += np.array(y[lab])
    idx += 1
plt.ylim(0.0, np.max(bottom) * 1.1)

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
plt.legend(ncol=2, bbox_to_anchor=(1, 1), loc="lower right")
plt.savefig(os.path.join(sim_dir, "time_2p.png"), bbox_inches="tight", pad_inches=0.2)

# print(y["calc"][14])
# print(y["comm"][14])
# print(y["deliver"][14])
# print(max(bottom))
