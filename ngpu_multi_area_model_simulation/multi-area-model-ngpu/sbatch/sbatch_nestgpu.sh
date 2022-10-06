#!/bin/bash


if ls log/*.txt >/dev/null 2>&1
then
	mv log/*.txt log/old/
fi

SCALING=0.1
sbatch --export=SCALING=$SCALING sjob_theory.sh
sbatch -d singleton sjob_simulation.sh
sbatch -d singleton sjob_finish.sh
