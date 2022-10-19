# ngpu_multi_area_model_simulation
Material and data analysis for the preprint "Fast simulation of a multi-area spiking network model of macaque cortex on an MPI-GPU cluster"

--------------------------------------------------------------------------------

## Authors
Gianmarco Tiddia, Bruno Golosio, Jasper Albers, Johanna Senk, Francesco Simula, Jari Pronold, Viviana Fanti, Elena Pastorelli, Pier Stanislao Paolucci and Sacha J. van Albada

--------------------------------------------------------------------------------

## Outline
The directory ``multi-area-model-ngpu`` contains the [NEST GPU](https://github.com/golosio/NeuronGPU) implementation of the [Multi-area model](https://github.com/INM-6/multi-area-model). 

The directory ``analysis`` contains the scripts that extract the distributions and perform all the plots depicted in the manuscript. 


# NEST GPU implementation of the Multi-Area Model

In ``config.py`` is contained a template to submit jobs on a cluster with Slurm. To run the model in a local machine could be used OpenMP and MPI.

Running
```
run.sh
```
launches 10 simulations (with spike recording) with different seeds for random number generations, whereas
```
run_eval_time.sh
```
launches 10 simulations without spike recording. In particular those scripts use the files ``run_simulation.templ`` and ``run_eval_time.templ`` to generate the homonymous Python scripts. The simulation parameters could be modified by editing the .templ files.

Running
```
create_symbolic_links.sh
```
in the folder in which the simulations spike times are stored create the folders data0 - data9. Those folders contains in the subfolder ``spikes_pop_idx`` the spike times of each of the 254 populations of the model stored in spike_times_i.dat, where i goes from 0 to 253.
