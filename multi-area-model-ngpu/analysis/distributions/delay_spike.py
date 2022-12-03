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
iterations = 0
for p in range(procs):
    print(f"Load File: spike_{p}.txt")
    with open(os.path.join(syndelay_dir, f"spike_{p}.txt")) as f:
        for row in f:
            hist = [int(d) for d in row.split(",")]
            if len(sum_hist) < len(hist):
                sum_hist += [0] * (len(hist) - len(sum_hist))
            for i in range(len(hist)):
                sum_hist[i] += hist[i]
            if p == 0:
                iterations += 1

ave_hist = np.array(sum_hist) / iterations


print("---delay_spike---")
print("(sum)dmin:", sum_hist[0])
print("(sum)sum:", sum(sum_hist))
print("(sum)dmin_per:", sum_hist[0] / sum(sum_hist))
print("(ave)dmin:", ave_hist[0])
print("(ave)sum:", sum(ave_hist))
print("(ave)dmin_per:", ave_hist[0] / sum(ave_hist))

ave_hist = [0] + list(ave_hist)
y = ave_hist
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
    os.path.join(sim_dir, "delay_spike.png"), bbox_inches="tight", pad_inches=0.2
)
