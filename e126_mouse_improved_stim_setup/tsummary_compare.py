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


# filename = 'fswitch_anesthesia_data.npz'

Fs                  = 1e5
prf                 = 1  
periods_of_interest = 1
pulse_length        = 0.01
start               = 0.4
end                 = 0.6
#
# 
filename                    = 't1-fswitch_data.npz'
data                        = np.load(filename,allow_pickle=True)
nfiltered_segmented_array   = data['nfiltered_segmented_array']
# a_array                      = fdata['nfiltered_segmented_array']
n_events,b                  = nfiltered_segmented_array.shape
time_segment                = np.linspace(-start,end,num=b)
# 
# 
ffilename                    = 't4-fswitch_data.npz'
fdata                        = np.load(ffilename,allow_pickle=True)
a_array                      = fdata['nfiltered_segmented_array']
# a_array                      = fdata['selected_array']
# 
# 
average_lfp  = np.mean(nfiltered_segmented_array,axis=0)
std_lfp      = np.std(nfiltered_segmented_array,axis=0)
sem_lfp      = np.std(nfiltered_segmented_array,axis=0)/np.sqrt(n_events)
# 
# 
a_average_lfp  = np.mean(a_array,axis=0)
a_std_lfp      = np.std(a_array,axis=0)
a_sem_lfp      = np.std(a_array,axis=0)/np.sqrt(n_events)
# 
timestep = 1/Fs
N = len(average_lfp)
fft_d = fft(average_lfp)
fft_d = np.abs(2.0/(N) * (fft_d))[1:(N)//2]
xf = np.fft.fftfreq( (N), d=timestep)[:(N)//2]
frequencies = xf[1:(N)//2]            
# 
fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_d,'k')
ax.set_xlim([0,100])
plt.show()

# 
# confidence interval. 
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m,h,se

m,ci,se = mean_confidence_interval(nfiltered_segmented_array,confidence=0.95)
print ('ci',len(ci))

am,aci,ase = mean_confidence_interval(a_array,confidence=0.95)
print ('ci',len(aci))

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
# 
lfp_height = np.max(average_lfp) - np.min(average_lfp)
print ('lfp height/std:',lfp_height)
# 
a_lfp_height = np.max(a_average_lfp) - np.min(a_average_lfp)
print ('a lfp height/std:',a_lfp_height)
# 
e1 = se
e2 = ase

fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(time_segment,average_lfp,color='k')
plt.plot(time_segment,a_average_lfp,color='r')
plt.fill_between(time_segment, average_lfp - e1, average_lfp + e1,
                  color='gray', alpha=0.2)
plt.fill_between(time_segment, a_average_lfp - e2, a_average_lfp + e2,
                  color='r', alpha=0.2)
# 
# plot the start and end times. 
for i in range(len(start_times) ):
    plt.axvline(x=start_times[i],color ='k')
for i in range(len(end_times) ):
    plt.axvline(x=end_times[i],color ='k')  
# 
ax.set_ylabel('Volts ($\mu$V)',fontsize=16)
ax.set_xlabel('Time(s)',fontsize=16)
plt.autoscale(enable=True, axis='x', tight=True)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# 
plt.legend(['0.5 Isoflurane','Phantom'],loc='upper right',framealpha=0.0,fontsize=16)
# 
plot_filename = 't1_comparison.png'
plt.savefig(plot_filename, transparent=True)
plt.show()

