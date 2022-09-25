import numpy as np
import elephant
import matplotlib.pyplot as plt
from elephant.statistics import isi, cv
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef, covariance
from neo.core import SpikeTrain
from quantities import s
from eval_functions import __load_spike_times, __plot_hist, __smooth_hist
import os
import sys

recording_path = sys.argv[1]
write_path = recording_path + 'cv_isi/'
if not os.path.isdir(write_path):
    os.makedirs(write_path)

nrun = 10
name = 'spike_times_'
begin = 500.0
end = 10500.0
npop = 254
xmin = 0.0
xmax = 5.0
nx = 300

matrix_size = 200
spike_time_bin = 0.002

for i_run in range(nrun):
    dum = []
    print ('Processing dataset '+ str(i_run+1) + '/' + str(nrun), flush=True)
    
    for i in range(npop):
        if(os.path.isfile(write_path+'cv_isi_'+str(i)+'.dat') == False):
            dum.append(i)
    
    if(len(dum)==0):
        print("The dataset " + str(i_run+1) + " is complete!", flush=True)
        continue
    else:
        print("Calculating distributions for population:", dum, flush=True)
        spike_times_list = __load_spike_times(recording_path, name, begin, end, npop)
        for ipop in dum:
            print ("run ", i_run, "/", nrun, "  pop ", ipop, "/", npop, flush=True)
            spike_times = spike_times_list[ipop]
            cv_isi = []
            for st_row in spike_times:
                if (len(st_row) > 1):
                    cv_isi.append(cv(isi(np.array(st_row))))
        
            if len(cv_isi)>0:
                x, hist1 =  __smooth_hist(cv_isi, xmin, xmax, nx, bw_min=0.01)
                arr = np.column_stack((x, hist1))
                np.savetxt(write_path+'cv_isi_'+str(ipop)+'.dat', arr)
