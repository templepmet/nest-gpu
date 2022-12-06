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


# only last time = sum time
procs = 32
host_mem = [0] * procs
gpu_mem = [0] * procs

with open(result_file) as f:
    text = f.read()
for p in range(procs):
    host_mem[p] = int(re.findall(f"MPI Rank {p} : Host Memory :.*'VmPeak': (.+?),.*\n", text)[0])
    gpu_mem[p] = int(re.findall(f"MPI Rank {p} : GPU Memory :.*used: (.*) B\)\n", text)[0])

host_mem = np.array(host_mem) / 1e6
gpu_mem = np.array(gpu_mem) / 1e9

print("host_mem sum:", sum(host_mem))
print("gpu_mem sum:", sum(gpu_mem))

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.size"] = 12

# memory
plt.figure()
plt.grid()
plt.xlabel("MPI Process")
plt.ylabel("Memory Size [GB]")
plt.ylim(0, np.max(host_mem) * 1.2)
x = np.arange(32)
width = 0.4
plt.bar(x - width / 2, host_mem, width=width, label="Host VmPeak")
plt.bar(x + width / 2, gpu_mem, width=width, label="GPU Used")
plt.legend(ncol=2)
plt.savefig(os.path.join(sim_dir, "memory.png"), bbox_inches="tight", pad_inches=0.2)
