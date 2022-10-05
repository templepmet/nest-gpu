import sys
import json
import numpy as np

from multiarea_model import MultiAreaModel
from multiarea_model.default_params import sim_params

scaling = 1.0
if len(sys.argv) > 1:
    scaling = float(sys.argv[1])

d = {}
conn_params = {'g': -11.,
               'K_stable': 'K_stable.npy',
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.41666667,
               'av_indegree_V1': 3950.,
	           'cc_weights_factor': 1.9,
	           'cc_weights_I_factor': 2.0}
input_params = {'rate_ext': 10.}
neuron_params = {'V0_mean': -150.,
                 'V0_sd': 50.}
network_params = {'N_scaling': scaling,
                  'K_scaling': scaling,
                  'fullscale_rates': 'tests/fullscale_rates.json',
                  'connection_params': conn_params,
                  'input_params': input_params,
                  'neuron_params': neuron_params}

sim_params = {'t_sim': 10000.0,
              't_presim': 500.0,
              'num_processes': 1,
              'local_num_threads': 1,
              'recording_dict': {'record_vm': False}}

theory_params = {'dt': 0.1}

M = MultiAreaModel(network_params, simulation=True,
                   sim_spec=sim_params,
                   theory=True,
                   theory_spec=theory_params)
p, r = M.theory.integrate_siegert()
print("Mean-field theory predicts an average "
      "rate of {0:.3f} spikes/s across all populations.".format(np.mean(r[:, -1])))

label_info = {'theory_label': M.simulation.label}
with open('label_info.json', 'w') as f:
    json.dump(label_info, f)
