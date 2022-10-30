#!/bin/bash

# nodes_list=(1 2)
nodes_list=(1 2 4 8 12 16)

for nodes in ${nodes_list[@]}; do
 	echo $nodes
 	# mkdir -p mp$proc
	# sbatch -N $nodes -n $nodes -o ./log/job_%j_out.txt -e ./log/job_%j_err.txt parallel-sjob.sh
	sbatch -N $nodes -n $nodes -o ./scaling/1e6neuron/out_${nodes}.txt -e ./scaling/1e6neuron/err_${nodes}.txt parallel-sjob.sh
done
