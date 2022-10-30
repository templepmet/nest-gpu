import matplotlib.pyplot as plt
import re
import numpy as np
from collections import defaultdict

nodes_list = [1, 2, 4, 8, 12, 16]

calc_label = [
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
	'ExternalSpikeReset_time'
]
comm_label = [
	'SendSpikeToRemote_MPI_time',
	'RecvSpikeFromRemote_MPI_time',
	'SendSpikeToRemote_CUDAcp_time',
	'RecvSpikeFromRemote_CUDAcp_time',
	'JoinSpike_time'
]

calc_time = {}
comm_time = {}
raw = defaultdict(lambda: {})

for nodes in nodes_list:
	with open(f'../scaling/1e6neuron/out_{nodes}.txt') as f:
		text = f.read()
		sumt = 0
		for label in calc_label:
			t = float(re.findall(f'MPI rank 0 :   {label}: (.*)\n', text)[0])
			raw[nodes][label] = t
			sumt += t
		calc_time[nodes] = sumt

		sumt = 0
		for label in comm_label:
			t = float(re.findall(f'MPI rank 0 :   {label}: (.*)\n', text)[0])
			raw[nodes][label] = t
			sumt += t
		comm_time[nodes] = sumt

# print(calc_time)
# print(comm_time)
# print(raw)

plt.figure()
plt.xticks(nodes_list)
plt.stackplot(nodes_list, [list(calc_time.values()), list(comm_time.values())], labels=['calc', 'comm'])
plt.legend(loc='upper left')
plt.xlabel('Compute Nodes')
plt.ylabel('Execution Time [s]')
plt.savefig('time.png', bbox_inches='tight', pad_inches=0.2)
