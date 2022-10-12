import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

synapse_json = "../../multi-area-model-ngpu/synapse_full.json"

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
                num_source_synapse[source] += num

# sum_synapse = np.sum(y)
# print("sum_synapse:", sum_synapse) # 37 x 10^9

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
