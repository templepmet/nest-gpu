import matplotlib.pyplot as plt
import re
import numpy as np
from collections import defaultdict

nodes_list = [1, 2, 4, 8, 12, 16]

label_time = [
	'SpikeBufferUpdate_time',
	'poisson_generator_time',
	'neuron_Update_time',
	'copy_ext_spike_time',
	'SendExternalSpike_time',
	'SendSpikeToRemote_time',
	'RecvSpikeFromRemote_time',
	'NestedLoop_time',
	'GetSpike_time',
	'SpikeReset_time',
	'ExternalSpikeReset_time',
	'SendSpikeToRemote_MPI_time',
	'RecvSpikeFromRemote_MPI_time',
	'SendSpikeToRemote_CUDAcp_time',
	'RecvSpikeFromRemote_CUDAcp_time',
	'JoinSpike_time'
]

raw = defaultdict(lambda: {})

for nodes in nodes_list:
	with open(f'../scaling/1e6neuron/out_{nodes}.txt') as f:
		text = f.read()
		for label in label_time:
			sumt = 0
			for rank in range(nodes):
				t = float(re.findall(f'MPI rank {rank} :   {label}: (.*)\n', text)[0])
				sumt += t
			raw[nodes][label] = sumt

yy = []
for nodes in nodes_list:
	yy.append(list(raw[nodes].values()))

plt.figure()
plt.xticks(nodes_list)
plt.xlim(nodes_list[0], nodes_list[-1])
plt.stackplot(nodes_list, np.array(yy).T, labels=label_time)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.xlabel('Compute Nodes')
plt.ylabel('Execution Time [s]')
plt.savefig('time_psum.png', bbox_inches='tight', pad_inches=0.2)
