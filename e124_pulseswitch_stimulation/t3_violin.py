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

filename  = 't2-fswitch_selected.npz'
filename2 = 't2-fswitch_anesthesia_selected.npz'

Fs                  = 1e5
prf                 = 1  
periods_of_interest = 1
pulse_length        = 0.01
start    = 0.4
end      = 0.6
b = 100000
time_segment                = np.linspace(-start,end,num=b)
start_idx = find_nearest(time_segment,0)
end_idx = find_nearest(time_segment,0.3)
print ('start and end',start_idx,end_idx)
#
data = np.load(filename,allow_pickle=True)
d_array = data['selected_array']
d_events,d = d_array.shape


adata = np.load(filename2,allow_pickle=True)
a_array = adata['selected_array']
a_events,a = a_array.shape


# Now extract the lfp heights. 
a_heights = []
d_heights = []
for i in range(a_events): 
    x = a_array[i,start_idx:end_idx]
    a_heights.append(np.max(x)-np.min(x))
for i in range(d_events): 
    x = d_array[i,start_idx:end_idx]
    d_heights.append(np.max(x)-np.min(x))


# T-test. 
sample1 = a_heights
sample2 = d_heights
t_stat, p_value = ttest_ind(sample1, sample2) 
# print('T-statistic value: ', t_stat) 
print('P-Value: ', p_value)


names = ['high anesthesia','low anesthesia']

# Do a carrier amplitude plot. 
carrier_toviolin = [a_heights,d_heights]
fig = plt.figure(figsize=(5,3))
ax  = fig.add_subplot(111)
violin = ax.violinplot(carrier_toviolin,showmeans=True,showextrema=True)
for pc in violin["bodies"]:
    pc.set_facecolor("grey")
    # pc.set_edgecolor("black")
    pc.set_linewidth(1) 
    pc.set_alpha(0.5)
for partname in ('cbars','cmins','cmaxes','cmeans'):
        vp = violin[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1.6)
        vp.set_alpha(1) 
ax.set_xticks([1,2])  
ax.set_xticklabels(names)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.legend(['Carrier Amplitudes:'],str(np.round(p_value_carrier,2)),loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.set_ylim([-70,120])
plt.tight_layout()
plt.savefig('t2_anesthesia_comparison_violin.png', bbox_inches='tight')
plt.show()





# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(211)
# plt.plot(time_segment,average_lfp,color='r')
# plt.fill_between(time_segment, average_lfp - error_lb, average_lfp + error_ub,
#                   color='gray', alpha=0.2)
# # plot the start and end times. 
# for i in range(len(start_times) ):
#     plt.axvline(x=start_times[i],color ='k')
# for i in range(len(end_times) ):
#     plt.axvline(x=end_times[i],color ='k')  

# ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xlabel('Time(s)')
# plt.autoscale(enable=True, axis='x', tight=True)
# ax2 = fig.add_subplot(212)
# plt.plot(time_segment,nfiltered_segmented_array.T)

# # plot the start and end times. 
# for i in range(len(start_times) ):
#     plt.axvline(x=start_times[i],color ='k')
# for i in range(len(end_times) ):
#     plt.axvline(x=end_times[i],color ='k')  

# # ax2.text(0.5, 100, 'individual trials', color='black',fontsize = 6, ha='center')
# plt.autoscale(enable=True, axis='x', tight=True)
# ax2.set_ylabel('Individual trials ($\mu$V)')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = 't6_fswitch_prf_'+str(prf)+'events_'+str(n_events)+'.png'
# plt.savefig(plot_filename, transparent=True)
# plt.show()

