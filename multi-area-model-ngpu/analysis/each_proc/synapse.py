import os
import sys

import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

argv = sys.argv
if len(argv) < 2:
    print(f"usage: python {argv[0]} simulation_directory")
    sys.exit(1)
sim_dir = argv[1]
syndelay_dir = os.path.join(sim_dir, "syndelay")
procs=32

synapse = [0] * 32
for p in range(procs):
    with open(os.path.join(syndelay_dir, f"local_{p}.txt")) as f:
        raw = f.readline()
        delay_hist = [int(d) for d in raw.split(",")]
        synapse[p] += sum(delay_hist)
    with open(os.path.join(syndelay_dir, f"remote_{p}.txt")) as f:
        raw = f.readline()
        if raw == "":
            continue
        delay_hist = [int(d) for d in raw.split(",")]
        synapse[p] -= sum(delay_hist)

print("sum_synapse:", sum(synapse))

y = synapse
x = np.arange(procs)

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.size"] = 12

plt.figure()
plt.grid()
plt.xlabel("MPI Process")
plt.ylabel("Number of Target Synpase")
# width = 0.4
# plt.bar(x - width / 2, host_mem, width=width)
plt.bar(x, y)
# plt.legend(ncol=2)
plt.savefig(os.path.join(sim_dir, "synapse.png"), bbox_inches="tight", pad_inches=0.2)
