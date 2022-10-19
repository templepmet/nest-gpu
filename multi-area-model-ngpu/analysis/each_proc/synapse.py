import os
import sys
import re
import subprocess
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

argv = sys.argv
if len(argv) < 2:
    print(f"usage: python {argv[0]} simulation_directory")
    sys.exit(1)
sim_dir = argv[1]
result_file = os.path.join(sim_dir, "result.txt")
gid_file = os.path.join(sim_dir, "recordings", "network_gids.txt")
connection_dir = os.path.join(sim_dir, "connection")


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

num_synapse_area = defaultdict(lambda: 0)
for area in areas_rank:
    for pop in network[area]:
        print("processsing:", area, pop)
        connection_txt = os.path.join(connection_dir, f"connection_{area}_{pop}.txt")
        num_synapse = int(
            subprocess.check_output(f"wc -l {connection_txt}".split())
            .decode()
            .split()[0]
        )
        num_synapse_area[area] += num_synapse


x = np.arange(procs)
y = []
for area in areas_rank:
    y.append(num_synapse_area[area])
print(num_synapse_area)
print(f"sum_synapse={np.sum(y)}")

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
plt.savefig(os.path.join(sim_dir, "synapse.png"), bbox_inches="tight", pad_inches=0.2)
