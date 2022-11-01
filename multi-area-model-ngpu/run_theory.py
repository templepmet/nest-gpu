import os
import json

import numpy as np

from multiarea_model import MultiAreaModel
from multiarea_model.default_params import sim_params


def make_params():
    with open("sim_info.json") as f:
        sim_info = json.load(f)
    label = sim_info["label"]
    N_scale = float(sim_info["N_scale"])
    K_scale = float(sim_info["K_scale"])
    T_scale = float(sim_info["T_scale"])

    conn_params = {
        "g": -11.0,
        "K_stable": "K_stable.npy",
        "fac_nu_ext_TH": 1.2,
        "fac_nu_ext_5E": 1.125,
        "fac_nu_ext_6E": 1.41666667,
        "av_indegree_V1": 3950.0,
        "cc_weights_factor": 1.9,
        "cc_weights_I_factor": 2.0,
    }
    input_params = {"rate_ext": 10.0}
    neuron_params = {"V0_mean": -150.0, "V0_sd": 50.0}
    network_params = {
        "N_scaling": N_scale,
        "K_scaling": K_scale,
        "fullscale_rates": "tests/fullscale_rates.json",
        "connection_params": conn_params,
        "input_params": input_params,
        "neuron_params": neuron_params,
    }

    num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))

    sim_params = {
        "t_sim": 10000.0 * T_scale,
        "t_presim": 500.0 * T_scale,
        "num_processes": 32,
        "local_num_threads": num_threads,
        "recording_dict": {"record_vm": False},
    }
    return label, network_params, sim_params


def theory(label, network_params, sim_params):
    M = MultiAreaModel(
        label=label,
        network_spec=network_params,
        simulation=True,
        sim_spec=sim_params,
        theory=True,
        theory_spec={"dt": 0.1},
    )
    p, r = M.theory.integrate_siegert()
    print(
        "Mean-field theory predicts an average "
        "rate of {0:.3f} spikes/s across all populations.".format(np.mean(r[:, -1]))
    )


def main():
    params = make_params()
    theory(*params)


if __name__ == "__main__":
    main()
