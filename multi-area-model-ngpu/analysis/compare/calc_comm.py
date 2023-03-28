import os
import re
import sys
from collections import defaultdict

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np

import matplotlib; matplotlib.use('agg')

sim_label_base = "8nodes_4gpus_N1.00_K1.00_T1.00_0:546440.sqd"
sim_label_prop = "8nodes_4gpus_N1.00_K1.00_T1.00_0:500998.sqd"

time_label = [
    "SpikeBufferUpdate_time",
    "poisson_generator_time",
    "neuron_Update_time",
    "copy_ext_spike_time",
    "SendExternalSpike_time",
    "PackSendSpike_time",
    "SendRecvSpikeRemote_immed_time",
    "SendRecvSpikeRemote_delay_time",
    "UnpackRecvSpike_time",
    "CopySpikeFromRemote_time",
    "MpiBarrier_time",
    "copy_spike_time",
    "ClearGetSpikeArrays_time",
    "NestedLoop_time",
    "GetSpike_time",
    "SpikeReset_time",
    "ExternalSpikeReset_time",
    "RevSpikeBufferUpdate_time",
    "BufferRecSpikeTimes_time",
    "Other_time"
]

comm_immed_label = [
    "PackSendSpike_time",
    "SendRecvSpikeRemote_immed_time",
    "UnpackRecvSpike_time",
    "CopySpikeFromRemote_time",
    "MpiBarrier_time"
]

comm_delay_label = [
    "SendRecvSpikeRemote_delay_time",
]

PROCS = 32

def get_time(sim_label):
    result_file = os.path.join("..", "..", "simulation_result", sim_label, "result.txt")
    sum_time = defaultdict(lambda: 0)
    with open(result_file) as f:
        text = f.read()
        for p in range(PROCS):
            for lab in time_label:
                times = re.findall(f"MPI rank {p} :   {lab}: (.*)\(def\), (.*)\(host\), (.*)\(device\)\n", text)[-1]
                t = float(times[0])
                sum_time[lab] += t
    time_ret = defaultdict(lambda: 0)
    for lab in time_label:
        if lab in comm_immed_label:
            time_ret["comm_immed"] += sum_time[lab]
        elif lab in comm_delay_label:
            time_ret["comm_delay"] += sum_time[lab]
        else:
            time_ret["calc"] += sum_time[lab]
    for lab in time_ret:
        time_ret[lab] /= PROCS

    return time_ret

time_base = get_time(sim_label_base)
time_prop = get_time(sim_label_prop)

time_y = {}
for lab in time_base:
    time_y[lab] = [time_base[lab], time_prop[lab]]

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.size"] = 14
plt.figure()
plt.grid()
plt.ylabel("処理時間 [s]")

lab_table = {"calc": "計算", "comm_immed": "通信（即時）", "comm_delay": "通信（オーバラップ）"}

bottom = [0, 0]
for lab in time_base:
    plt.bar(
        ["既存手法", "提案手法"],
        time_y[lab],
        width=0.5,
        bottom=bottom,
        align="center",
        label=lab_table[lab],
    )
    bottom += np.array(time_y[lab])
plt.ylim(0, 330)

plt.legend(ncol=3, bbox_to_anchor=(1, 1), loc="lower right")
plt.savefig("time.png", bbox_inches="tight", pad_inches=0.2)
