import os

label="1nodes_8gpus_N0.01_K0.01_T0.01_0:355272.sqd"
label_ref="1nodes_8gpus_0.01scale_ref"

filenames = [f.name for f in os.scandir(os.path.join("simulation_result", label, "recordings"))]
filenames.remove("network_gids.txt")
for filename in filenames:
    sender = []
    sender_ref = []
    with open(os.path.join("simulation_result", label, "recordings", filename)) as f:
        for line in f.readlines()[1:]:
            row = line.split("\t")
            sender.append((float(row[1]), int(row[0])))
    with open(os.path.join("simulation_result", label_ref, "recordings", filename)) as f:
        for line in f.readlines()[1:]:
            row = line.split("\t")
            sender_ref.append((float(row[1]), int(row[0])))
    for i in range(min(len(sender), len(sender_ref))):
        if sender[i] != sender_ref[i]:
            print(filename, sender[i], sender_ref[i])
            break
