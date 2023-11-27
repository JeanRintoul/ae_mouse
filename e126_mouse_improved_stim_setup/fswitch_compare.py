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

Fs                  = 1e5
prf                 = 1  
periods_of_interest = 1
pulse_length        = 0.01
start               = 0.2
end                 = 0.8
# 
# 
# erp 
# filename                    = 't1-fswitch_data_filter.npz'
filename                    = 'e124t1-fswitch_data_filter.npz'
data                        = np.load(filename,allow_pickle=True)
erp_array                   = data['nfiltered_segmented_array']
erp_events,b                = erp_array.shape
time_segment                = np.linspace(-start,end,num=b)
# 
# anesthesia
afilename                    = 't1-fswitch_data_anesthesia_filter.npz'

adata                        = np.load(afilename,allow_pickle=True)
a_array                      = adata['nfiltered_segmented_array']
a_events,b                   = a_array.shape
# 
# phantom
phantom_filename             = 't2-pulse_data_tight_filter.npz'
pdata                        = np.load(phantom_filename,allow_pickle=True)
p_array                      = pdata['nfiltered_segmented_array']
p_events,b                   = p_array.shape
# 
# 
average_lfp  = np.mean(-erp_array,axis=0)
std_lfp      = np.std(-erp_array,axis=0)
sem_lfp      = np.std(-erp_array,axis=0)/np.sqrt(erp_events)
# 
# 
a_average_lfp  = np.mean(-a_array,axis=0)
a_std_lfp      = np.std(-a_array,axis=0)
a_sem_lfp      = np.std(-a_array,axis=0)/np.sqrt(a_events)
# 
# phantom
p_average_lfp  = np.mean(-p_array,axis=0)
p_std_lfp      = np.std(-p_array,axis=0)
p_sem_lfp      = np.std(-p_array,axis=0)/np.sqrt(p_events)



# 
# timestep = 1/Fs
# N = len(average_lfp)
# fft_d = fft(average_lfp)
# fft_d = np.abs(2.0/(N) * (fft_d))[1:(N)//2]
# xf = np.fft.fftfreq( (N), d=timestep)[:(N)//2]
# frequencies = xf[1:(N)//2]            
# # 
# fig = plt.figure(figsize=(5,5))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_d,'k')
# ax.set_xlim([0,100])
# plt.show()

# 
# confidence interval. 
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m,h,se

m,ci,se = mean_confidence_interval(erp_array,confidence=0.95)
print ('ci',len(ci))

am,aci,ase = mean_confidence_interval(a_array,confidence=0.95)
print ('ci',len(aci))
#
pm,pci,pse = mean_confidence_interval(p_array,confidence=0.95)
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
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx

zero_idx = find_nearest(time_segment,0) +int(2000)
end_erp_idx = find_nearest(time_segment,0.1) +int(2000)

total_heights=[]
for i in range(erp_events):
    event = erp_array[i,:]
    lfp_height = np.max(event[zero_idx:end_erp_idx]) - np.min(event[zero_idx:end_erp_idx])
    total_heights.append(lfp_height)
print ('error erp height:',mean_confidence_interval(total_heights,confidence=0.95)[1] )


a_total_heights=[]
for i in range(a_events):
    event = a_array[i,:]
    lfp_height = np.max(event[zero_idx:end_erp_idx]) - np.min(event[zero_idx:end_erp_idx])
    a_total_heights.append(lfp_height)
# print ('a std erp height:',np.std(a_total_heights))
print ('a erp height:',mean_confidence_interval(a_total_heights,confidence=0.95)[1] )


p_total_heights=[]
for i in range(p_events):
    event = p_array[i,:]
    lfp_height = np.max(event[zero_idx:end_erp_idx]) - np.min(event[zero_idx:end_erp_idx])
    p_total_heights.append(lfp_height)
# print ('p std erp height:',np.std(p_total_heights))
print ('p erp height:',mean_confidence_interval(p_total_heights,confidence=0.95)[1] )

# peak to peak height should only be calculated after the stimulus
# 
lfp_height = np.max(average_lfp[zero_idx:]) - np.min(average_lfp[zero_idx:])
print ('ERP lfp height/std:',lfp_height)
# 
a_lfp_height = np.max(a_average_lfp[zero_idx:]) - np.min(a_average_lfp[zero_idx:])
print ('a lfp height/std:',a_lfp_height)
# 
p_lfp_height = np.max(p_average_lfp[zero_idx:]) - np.min(p_average_lfp[zero_idx:])
print ('p lfp height/std:',p_lfp_height)


# e = se
# a = ase
# p = pse

e = ci
a = aci
p = pci

fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(time_segment,average_lfp,color='k')
plt.plot(time_segment,a_average_lfp,color='r')
plt.plot(time_segment,p_average_lfp,color='b')
plt.fill_between(time_segment, average_lfp - e, average_lfp + e,
                  color='gray', alpha=0.2)
plt.fill_between(time_segment, a_average_lfp - a, a_average_lfp + a,
                  color='r', alpha=0.2)
plt.fill_between(time_segment, p_average_lfp - p, p_average_lfp + p,
                  color='b', alpha=0.2)
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
ax.set_xlim([-0.2,0.6])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# 
# plt.legend(['ERP iso=0.5,n='+str(erp_events),'Phantom,n='+str(p_events)],loc='lower right',framealpha=0.0,fontsize=16)
plt.legend(['ERP iso=0.5,n='+str(erp_events),'Anesthesia iso=1.5,n='+str(a_events),'Phantom,n='+str(p_events)],loc='upper right',framealpha=0.0,fontsize=16)
# 
plt.title('Frequency Switching Modulation Neural Response',fontsize=16)
plot_filename = 'fswitch_comparison.png'
plt.savefig(plot_filename)
plt.show()

