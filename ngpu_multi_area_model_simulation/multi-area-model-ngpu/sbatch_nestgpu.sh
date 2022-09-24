#!/bin/bash -eu

if [ -f log/*.txt ]; then
	mv log/*.txt log/old/
fi

SCALING=0.1
sbatch --export=SCALING=$SCALING sjob_theory.sh
sbatch -d singleton sjob_simulation.sh
