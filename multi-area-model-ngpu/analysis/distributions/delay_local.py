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
syndelay_dir = os.path.join(sim_dir, "syndelay")
procs = 32

sum_hist = []
for p in range(procs):
    with open(os.path.join(syndelay_dir, f"local_{p}.txt")) as f:
        raw = f.readline()
        delay_local = [int(d) for d in raw.split(",")]
    with open(os.path.join(syndelay_dir, f"remote_{p}.txt")) as f:
        raw = f.readline()
        if raw != "":
            delay_remote = [int(d) for d in raw.split(",")]
        else:
            delay_remote = []
    assert len(delay_local) >= len(delay_remote)
    if len(sum_hist) < len(delay_local):
        sum_hist += [0] * (len(delay_local) - len(sum_hist))
    for i in range(len(delay_local)):
        sum_hist[i] += delay_local[i]
    for i in range(len(delay_remote)):
        assert delay_local[i] >= delay_remote[i]
        # sum_hist[i] -= delay_remote[i]

print("---delay_local---")
print("dmin:", sum_hist[0])
print("sum:", sum(sum_hist))
print("dmin_per:", sum_hist[0] / sum(sum_hist))

sum_hist = [0] + sum_hist
y = sum_hist
x = np.arange(len(y))

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.size"] = 12

plt.figure()
plt.grid()
plt.xlabel("Synaptic Delay")
plt.ylabel("Count")

# xtmp = np.arange(0, max_delay + 1, 5)
# plt.xticks(xtmp, xtmp / 10)
plt.bar(x, y)
plt.savefig(
    os.path.join(sim_dir, "delay_local.pdf"), bbox_inches="tight", pad_inches=0.2
)
