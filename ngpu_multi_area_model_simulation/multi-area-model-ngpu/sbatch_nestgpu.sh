#!/bin/bash -eu

SCALING=0.1
sbatch --export=SCALING=$SCALING sjob_theory.sh
sbatch -d singleton sjob_simulation.sh
