import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

sim_label="3da585141ae2447cd8e0bfff51efc918"
result_file = f"../../multi-area-model-ngpu/{sim_label}/result.txt"

time_label = [
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
	# 'SendSpikeToRemote_MPI_time',
	# 'RecvSpikeFromRemote_MPI_time',
	# 'SendSpikeToRemote_CUDAcp_time',
	'RecvSpikeFromRemote_CUDAcp_time', # not include, it is true.
	# 'JoinSpike_time'
]

# only last time = sum time

procs=32
data=[None]*procs
for i in range(procs):
	data[i] = {}
build_time=[0]*procs
sim_time=[0]*procs
sum_time=[0]*procs
host_mem=[0]*procs
gpu_mem=[0]*procs

with open(result_file) as f:
	text = f.read()
for p in range(procs):
	for lab in time_label:
		data[p][lab] = float(re.findall(f"MPI rank {p} :   {lab}: (.*)\n", text)[-1])
		build_time[p] = float(re.findall(f"MPI rank {p} : Building time: (.*)\n", text)[-1])
		sim_time[p] = float(re.findall(f"MPI rank {p} : Simulation time: (.*)\n", text)[-1])
		sum_time[p] += data[p][lab]

		host_mem[p] = int(re.findall(f"MPI Rank {p} : Host Memory :.*'VmPeak': (.+?),.*\n", text)[0])
		gpu_mem[p] = int(re.findall(f"MPI Rank {p} : GPU Memory :.*used: (.*) B\)\n", text)[0])

host_mem = np.array(host_mem) / 1e6
gpu_mem = np.array(gpu_mem) / 1e9

# print(sim_time)
# print(sum_time)
# print(np.array(sim_time) - np.array(sum_time))
# reason for differ
# building time is constant: build_real_time_ - start_real_time_
# simulation time is variable: end_real_time_ - build_real_time_
#  start_real_time_: create constructa real time
#  build_real_time_: first simulation real time
#  end_real_time_: each finish simulation real time
# -> so, end_real_time_ include else processing time not include simulatiobn
#   ex.) transfer GPU->CPU spike record data, python overhead...
# not ignore transfer GPU->CPU GPU->CPU spike record data time !!!!!!!!!

# print(host_mem)
# print(gpu_mem)

# time
# convert for plot
y = {}
for lab in time_label:
	yt = [0.0] * procs
	for p in range(procs):
		yt[p] = data[p][lab]
	y[lab] = yt

plt.rcParams['axes.axisbelow'] = True
plt.rcParams['font.size'] = 12
plt.figure()
plt.grid()
plt.xlabel('MPI Process')
plt.ylabel('Processing Time [s]')
bottom = np.array([0.0] * procs)
x = [i for i in range(procs)]
for lab in time_label:
	plt.bar(x, y[lab], bottom=bottom, align='center', label=lab)
	bottom += np.array(y[lab])

ax=plt.gca()
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 1), loc='upper left')
plt.savefig('time.png', bbox_inches='tight', pad_inches=0.2)

# memory
plt.figure()
plt.grid()
plt.xlabel('MPI Process')
plt.ylabel('Memory Size [GB]')
plt.ylim(0, np.max(host_mem) * 1.2)
x = np.arange(32)
width=0.4
plt.bar(x - width / 2, host_mem, width=width, label='Host VmPeak')
plt.bar(x + width / 2, gpu_mem, width=width, label='GPU Used')
plt.legend(ncol=2)
plt.savefig('memory.png', bbox_inches='tight', pad_inches=0.2)
