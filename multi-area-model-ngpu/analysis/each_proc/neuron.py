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


procs = 32
neurons = [0] * procs

with open(result_file) as f:
    text = f.read()
for p in range(procs):
    neurons[p] = int(
        re.findall(f"Rank {p}: created area .* with (.*) local nodes\n", text)[0]
    )

print(neurons)
sum_neuron = np.sum(neurons)
print("sum_neuron:", sum_neuron)

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.size"] = 12
plt.figure()
plt.grid()
plt.xlabel("MPI Process")
plt.ylabel("Number of Neurons")
# plt.ylim(0, np.max(host_mem) * 1.2)
x = np.arange(32)
width = 0.4
# plt.bar(x - width / 2, host_mem, width=width)
plt.bar(x, neurons)
# plt.legend(ncol=2)
plt.savefig(os.path.join(sim_dir, "neuron.png"), bbox_inches="tight", pad_inches=0.2)
