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
write_path = recording_path + 'correl/'
if not os.path.isdir(write_path):
    os.makedirs(write_path)

nrun = 10
name = 'spike_times_'
begin = 500.0
end = 10500.0
npop = 254
xmin = -0.05
xmax = 0.15
nx = 400

matrix_size = 200
spike_time_bin = 0.002

for i_run in range(nrun):
    dum = []
    print ('Processing dataset '+ str(i_run+1) + '/' + str(nrun), flush=True)
    
    for i in range(npop):
        if(os.path.isfile(write_path+'correl2_'+str(i)+'.dat')==False):
            dum.append(i)
    
    if(len(dum)==0):
        print("The dataset "+str(i_run+1)+" is complete!", flush=True)
        continue
    else:
        print("Calculating distributions for populations:", dum, flush=True)
        spike_times_list = __load_spike_times(recording_path, name, begin, end, npop)
        for ipop in dum:
            print ("run ", i_run, "/", nrun, "  pop ", ipop, "/", npop, flush=True)
            spike_times = spike_times_list[ipop]
            st_list = []
            for j in range(matrix_size):
                spike_train = SpikeTrain(np.array(spike_times[j])*s,
                                         t_stop = (end/1000.0)*s)
                st_list.append(spike_train)
                                     
            binned_st = BinnedSpikeTrain(st_list, spike_time_bin*s, None,
                                        (begin/1000.0)*s, (end/1000.0)*s)
            #print (binned_st)
            cc_matrix = corrcoef(binned_st)
            correl = []
            for j in range(matrix_size):
                for k in range(matrix_size):
                    #print(j, k, cc_matrix[j][k])
                    if (j != k and cc_matrix[j][k]<xmax and cc_matrix[j][k]>xmin):
                        correl.append(cc_matrix[j][k])

            if len(correl)>0:
                x, hist1 =  __smooth_hist(correl, xmin, xmax, nx, bw_min=5e-5)
                arr = np.column_stack((x, hist1))
                np.savetxt(write_path+'corr_'+str(ipop)+'.dat', arr)
