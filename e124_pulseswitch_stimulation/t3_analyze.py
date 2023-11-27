'''

Title: vep signal inspection
Function: takes a single file, and averages the veps to see them better. 

Author: Jean Rintoul
Date: 23.10.2022

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
from scipy.stats import ttest_ind
import scipy.stats
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16


filename = 'fswitch_selected.npz'
# filename = 'fswitch_anesthesia_data.npz'

Fs                  = 1e5
prf                 = 1  
periods_of_interest = 1
pulse_length        = 0.01
#
data = np.load(filename,allow_pickle=True)
nfiltered_segmented_array = data['selected_array']
# nfiltered_segmented_array = data['nfiltered_segmented_array']
start    = 0.4
end      = 0.6

#
n_events,b = nfiltered_segmented_array.shape
time_segment = np.linspace(-start,end,num=b)



average_lfp  = np.mean(nfiltered_segmented_array,axis=0)
std_lfp      = np.std(nfiltered_segmented_array,axis=0)
sem_lfp      = np.std(nfiltered_segmented_array,axis=0)/np.sqrt(n_events)

# confidence interval. 
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m,h,se

m,ci,se = mean_confidence_interval(nfiltered_segmented_array,confidence=0.95)
print ('ci',len(ci))


print ('n_events: ', n_events)
# lfp_height   = np.max(average_lfp)- np.min(average_lfp)
# print ('LFP size', lfp_height)
error_lb     = ci
error_ub     = ci
# 
start_time = 0
start_times = []
for i in range(periods_of_interest):
    start_times.append((i)*(1.0/prf))
# 
end_times = []
for i in range(periods_of_interest):
    end_times.append((i)*(1.0/prf)+pulse_length)
# 

lfp_height = np.max(average_lfp) - np.min(average_lfp)
print ('lfp height:',lfp_height)
print (b)
# for j in range(b-1):
#     fig = plt.figure(figsize=(10,6))
#     ax  = fig.add_subplot(111)
#     plt.plot(time_segment,nfiltered_segmented_array[j,:])
#     for i in range(len(start_times) ):
#         plt.axvline(x=start_times[i],color ='k')
#     ax.set_title('filenumber '+str(j))
#     plt.show()

fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(time_segment,average_lfp,color='r')
plt.fill_between(time_segment, average_lfp - error_lb, average_lfp + error_ub,
                  color='gray', alpha=0.2)
# plot the start and end times. 
for i in range(len(start_times) ):
    plt.axvline(x=start_times[i],color ='k')
for i in range(len(end_times) ):
    plt.axvline(x=end_times[i],color ='k')  

ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Time(s)')
plt.autoscale(enable=True, axis='x', tight=True)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plot_filename = 't3_fswitch_'+str(n_events)+'.png'
plt.savefig(plot_filename, transparent=True)
plt.show()

