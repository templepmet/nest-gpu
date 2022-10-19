import os
import sys
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

sim_label = "1nodes_8gpus_0.01scale_0:67750.sqd"
sim_dir = f"../../multi-area-model-ngpu/simulation_result/{sim_label}"
result_file = f"{sim_dir}/result.txt"
recording_dir = f"{sim_dir}/recordings"
syndelay_dir = f"{sim_dir}/syndelay"
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

delay_hist = [0] * 100 # max delay = 0.1 * 100 = 10 ms
# areas_rank = ["V1"]
# network["V1"] = ["23E"]
for area in areas_rank:
    for pop in network[area]:
        print("processsing:", area, pop)
        syndelay_txt = os.path.join(syndelay_dir, f"syndelay_{area}_{pop}.txt")
        with open(syndelay_txt) as f:
            delays = re.findall(f".*'delay': (.*), .*\n", f.read())
        for delay_str in delays:
            delay_raw = float(delay_str)
            delay_int = int(round(delay_raw / 0.1))
            delay_float = 0.1 * delay_int
            assert abs(delay_raw - delay_float) < 1e-6
            delay_hist[delay_int] += 1

print(delay_hist)

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.size"] = 12

plt.figure()
plt.grid()
plt.xlabel("Synaptic Delay")
plt.ylabel("Count")
max_delay = len(delay_hist) - 1
for i in range(len(delay_hist)):
    if delay_hist[len(delay_hist) - i - 1] > 0:
        max_delay = len(delay_hist) - i
        break
x = np.arange(max_delay + 1)
y = delay_hist[0:len(x)]

xtmp = np.arange(0, max_delay + 1, 5)
plt.xticks(xtmp, xtmp / 10)
plt.plot(x, y)
plt.savefig(f"delay_{sim_label}.png", bbox_inches="tight", pad_inches=0.2)
