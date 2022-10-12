import re
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

sim_label = "simulated_8nodes_4gpus_1.00scale"
result_file = f"../../multi-area-model-ngpu/simulation_result/{sim_label}/result.txt"
synapse_json = "synapse_full.json"

with open(synapse_json) as f:
    synapse = json.load(f)

num_target_synapse = defaultdict(lambda: 0)
num_source_synapse = defaultdict(lambda: 0)
for target in synapse:
    for tpop in synapse[target]:
        for source in synapse[target][tpop]:
            for spop in synapse[target][tpop][source]:
                num = synapse[target][tpop][source][spop]
                num_target_synapse[target] += num
                if source != "external":
                    num_source_synapse[source] += num

print(num_target_synapse)
print(num_source_synapse)

procs = 32
areas_name = [""] * procs
with open(result_file) as f:
    text = f.read()
for p in range(procs):
    areas_name[p] = re.findall(f"Rank {p}: created area (.*) with", text)[0]

print(areas_name)
print(list(num_target_synapse.keys()))
print(list(num_source_synapse.keys()))

sum_synapse = np.sum(list(num_source_synapse.values()))
print("sum_synapse:", sum_synapse) # 24.1 x 10^9

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.size"] = 12

plt.figure()
plt.grid()
plt.xlabel("MPI Process")
plt.ylabel("Number of Target Synapses")
# plt.ylim(0, np.max(host_mem) * 1.2)
y = list(num_target_synapse.values())
x = np.arange(len(y))
width = 0.4
# plt.bar(x - width / 2, host_mem, width=width)
plt.bar(x, y)
# plt.legend(ncol=2)
plt.savefig("synapse_target.png", bbox_inches="tight", pad_inches=0.2)

plt.figure()
plt.grid()
plt.xlabel("MPI Process")
plt.ylabel("Number of Source Synapses")
# plt.ylim(0, np.max(host_mem) * 1.2)
y = list(num_source_synapse.values())
x = np.arange(len(y))
width = 0.4
# plt.bar(x - width / 2, host_mem, width=width)
plt.bar(x, y)
# plt.legend(ncol=2)
plt.savefig("synapse_source.png", bbox_inches="tight", pad_inches=0.2)
