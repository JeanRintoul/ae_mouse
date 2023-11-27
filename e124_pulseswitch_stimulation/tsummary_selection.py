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
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx

# filename = 't1-fswitch_data.npz'
filename = 't4-fswitch_data.npz'

outfile = 't4-fswitch_phantom_data.npz'

Fs                  = 1e5
prf                 = 1  
periods_of_interest = 1
pulse_length        = 0.01
#
data = np.load(filename,allow_pickle=True)
nfiltered_segmented_array = data['nfiltered_segmented_array']
start    = data['start']
end      = data['end']
n_events,b = nfiltered_segmented_array.shape
time_segment = np.linspace(-start,end,num=b)
start_idx = find_nearest(time_segment,0)
end_idx   = start_idx + int(0.4*Fs)

selected_array = np.zeros((0,b))
print ('array_len',b)
print ('start and end:',start,end)

# I could now do some selection. 
for j in range(n_events-1):
    # 
    print ('file number',j)
    portion_of_interest = nfiltered_segmented_array[j,:]

    fig = plt.figure(figsize=(10,6))
    ax  = fig.add_subplot(111)
    plt.plot(time_segment,nfiltered_segmented_array[j,:])
    plt.axvline(x=0,color ='k')
    ax.set_title('filenumber '+str(j))
    plt.show()

    # 
    bit = portion_of_interest[:,None].T
    print ('len bit: ',bit.shape,selected_array.shape )

    # selected_array=np.concatenate((selected_array, bit ), axis=0)
    # selected_array= np.column_stack((selected_array,bit))
    n = int(input("Enter 1 if you want to remove, 0 to keep: "))
    if n == 0:
        selected_array=np.concatenate((selected_array, bit), axis=0)
    else: 
        print ('data is discarded')

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,selected_array=selected_array,time_segment=time_segment)
print ('saved out a data file!')


print ('now about to plot the new selected array - does it look any better?')
nfiltered_segmented_array = selected_array

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
error_lb     = se
error_ub     = se
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

fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(211)
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
ax2 = fig.add_subplot(212)
plt.plot(time_segment,nfiltered_segmented_array.T)

# plot the start and end times. 
for i in range(len(start_times) ):
    plt.axvline(x=start_times[i],color ='k')
for i in range(len(end_times) ):
    plt.axvline(x=end_times[i],color ='k')  

# ax2.text(0.5, 100, 'individual trials', color='black',fontsize = 6, ha='center')
plt.autoscale(enable=True, axis='x', tight=True)
ax2.set_ylabel('Individual trials ($\mu$V)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plot_filename = 't4_fswitch_prf_'+str(prf)+'events_'+str(n_events)+'.png'
plt.savefig(plot_filename, transparent=True)
plt.show()

