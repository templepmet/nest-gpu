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


def add_hist(h1, h2):
    if len(h1) < len(h2):
        h1 += [0] * (len(h2) - len(h1))
    for i in range(len(h2)):
        h1[i] += h2[i]


delay_all = []
delay_remote = []
delay_spike = []
iterations = 0
for p in range(procs):
    print(p)
    with open(os.path.join(syndelay_dir, f"local_{p}.txt")) as f:
        raw = f.readline()
        d_all = [int(d) for d in raw.split(",")]
    with open(os.path.join(syndelay_dir, f"remote_{p}.txt")) as f:
        raw = f.readline()
        if raw != "":
            d_remote = [int(d) for d in raw.split(",")]
        else:
            d_remote = []
    add_hist(delay_all, d_all)
    add_hist(delay_remote, d_remote)

delay_spike = np.array(delay_spike) / iterations
x = np.arange(len(delay_all))

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.size"] = 12

plt.figure()
plt.grid()
plt.xlabel("Synaptic Delay")
plt.ylabel("Count")

# xtmp = np.arange(0, max_delay + 1, 5)
# plt.xticks(xtmp, xtmp / 10)
plt.bar(x, delay_all, label="All")
plt.bar(x, delay_remote, label="Remote")

# plt.plot(x, delay_all, label="All")
# plt.plot(x, delay_remote, label="Remote")
# plt.plot(x, delay_spike, label="Spike")
plt.legend()
plt.savefig(
    os.path.join(sim_dir, "delay_connection.png"), bbox_inches="tight", pad_inches=0.2
)
