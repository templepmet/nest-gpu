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
comm_dir = os.path.join(sim_dir, "comm")
procs=32

iterations = 0
dist = [np.zeros(procs) for _ in range(procs)]
for p in range(procs):
	with open(os.path.join(comm_dir, f"send_{p}.txt")) as f:
		for line in f:
			row = line.split(",")
			cnt = [int(c) for c in row]
			dist[p] += np.array(cnt)
			if p == 0:
				iterations += 1

x = []
y = []
w=[]
for i in range(procs):
	for j in range(procs):
		x.append(i)
		y.append(j)
		w.append(dist[i][j] / iterations)

r = np.arange(32)

plt.rcParams["font.size"] = 12
plt.figure()
plt.axes().set_aspect("equal")
plt.hist2d(x, y, bins=r, weights=w)
plt.xlabel("Send Process")
plt.ylabel("Recv Process")
cbar = plt.colorbar()
cbar.set_label("Num of Spikes")
plt.grid()
plt.savefig(os.path.join(sim_dir, "comm_spike.png"), bbox_inches="tight", pad_inches=0.2)
