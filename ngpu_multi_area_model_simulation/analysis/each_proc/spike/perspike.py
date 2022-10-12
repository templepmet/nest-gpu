import os
import subprocess
import re
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

sim_label = "simulated_8nodes_4gpus_1.00scale"
result_file = f"../../../multi-area-model-ngpu/simulation_result/{sim_label}/result.txt"
recording_dir = f"../../../multi-area-model-ngpu/simulation_result/{sim_label}/recordings"
gid_file = os.path.join(recording_dir, "network_gids.txt")

procs = 32
areas_rank = [""] * procs
with open(result_file) as f:
    text = f.read()
for p in range(procs):
    areas_rank[p] = re.findall(f"Rank {p}: created area (.*) with", text)[0]

network = defaultdict(lambda: [])
with open(gid_file) as f:
    for line in f:
        row = line.split(",")
        area = row[0]
        pop = row[1]
        network[area].append(pop)

num_spikes_area = defaultdict(lambda: 0)
nothing_spike = set()
for area in areas_rank:
    for pop in network[area]:
        spike_dat = os.path.join(recording_dir, f"spike_times_{area}_{pop}.dat")
        if os.path.isfile(spike_dat):
            print("processsing:", area, pop)
            # num_spikes = sum(1 for line in open(spike_dat))
            num_spikes = int(subprocess.check_output(f"wc -l {spike_dat}".split()).decode().split()[0])
            num_spikes_area[area] += num_spikes
        else:
            nothing_spike.add(area)

print(num_spikes_area)
print(nothing_spike)
x = np.arange(procs)
y = []
for area in areas_rank:
    if area in nothing_spike:
        y.append(0)
    else:
        y.append(num_spikes_area[area])

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.size"] = 12

plt.figure()
plt.grid()
plt.xlabel("MPI Process")
plt.ylabel("Number of Spikes")
# plt.ylim(0, np.max(host_mem) * 1.2)
width = 0.4
# plt.bar(x - width / 2, host_mem, width=width)
plt.bar(x, y)
# plt.legend(ncol=2)
plt.savefig("spikes.png", bbox_inches="tight", pad_inches=0.2)
