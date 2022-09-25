#!/bin/bash


if ls log/*.txt >/dev/null 2>&1
then
	mv log/*.txt log/old/
fi

sbatch sjob_firing_rate.sh
sbatch sjob_cv_isi.sh
sbatch sjob_correl.sh
