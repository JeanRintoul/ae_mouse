'''

Title: make the final violin plot for acoustically connected and not acoustically connected. 

Author: Jean Rintoul
Date: 19.06.23

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel


# Load in the acoustically connected VEP heights. 
filename          	= 'mouse_eeg.npz'
data              	= np.load(filename)
ac_height_list      = np.array(data['height_list'])
ac_veps         	= np.array(data['veps'])
ac_time_segment     = data['time_segment']
# 
ac_average_vep = np.median(ac_veps,axis=0)
ac_std_vep     = np.std(ac_veps,axis=0)
# 
# Load in the not acoustically connected VEP heights. 
filename          	= 'nac_veps.npz'
data              	= np.load(filename)
nac_height_list     = np.array(data['height_list'])
nac_veps            = np.array(data['veps'])
nac_time_segment    = data['time_segment']
# 
nac_average_vep = np.median(nac_veps,axis=0)
nac_std_vep     = np.std(nac_veps,axis=0)
# 
# 
# Data for violin plot. 
X 		 = [ac_height_list,nac_height_list]
x_labels = ['acoustically connected','not acoustically connected']
led_frequency 	= 4.0 
led_duration 	= 1/(2*led_frequency) # when the led turned on and off. 
start_time 		= 0.0
end_time   		= start_time+led_duration
# 
# t-test stats. 
t_stat, p_value = ttest_ind(ac_height_list,nac_height_list)
print("T-statistic value: ", t_stat)  
print("P-Value: ", p_value)

print ('number of repeats ac,nac:',len(nac_height_list),len(ac_height_list)    )
# 
# Plot
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
# ax.violinplot(x,showextrema = False)
# ax.violinplot(x, showmeans = True, showmedians = True)
# ax.violinplot(X, showmedians = True,positions=[1,2])
# ax.set_title('VEP height with and without US')

# Create the violin plot
plots = ax.violinplot(X, showmeans=True, showextrema=False, widths=0.8)

colors = ['Red', 'Black']
# Set the color of the violin patches
for pc, color in zip(plots['bodies'], colors):
    pc.set_facecolor(color)

ax.set_ylabel('VEP height ($\mu$V)',fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks([1,2])
ax.set_xticklabels(x_labels,fontsize=14)
plot_filename = 'violin_plot.png'
plt.savefig(plot_filename)
plt.show()

# plt.suptitle('AE location calibration')

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)

plt.plot(ac_time_segment,ac_average_vep,'r')
# plt.plot(nac_time_segment,nac_average_vep,'k')
# plt.legend(['mouse_eeg','not acoustically connected'],loc='upper right')
plt.fill_between(ac_time_segment, ac_average_vep - ac_std_vep, ac_average_vep + ac_std_vep,
                  color='red', alpha=0.2)
# plt.plot(nac_time_segment,nac_average_vep,'k')
# plt.fill_between(nac_time_segment, nac_average_vep - nac_std_vep, nac_average_vep + nac_std_vep,
#                   color='gray', alpha=0.2)
ax.set_xlim([np.min(nac_time_segment),np.max(nac_time_segment)])
plt.axvline(x=start_time,color ='k')
plt.axvline(x=end_time,color ='k')
ax.set_xlabel('time(s)',fontsize=14)
ax.set_ylabel('Volts ($\mu$V)',fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = 'veps_comparison.png'
plt.savefig(plot_filename)
plt.show()
