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
from scipy.signal import hilbert
from scipy.signal import find_peaks
import scipy.stats
from scipy.stats import ttest_ind
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16
# 
saveprefix          = './/images//'
# 
nognd_savepath  = 'D:\\ae_mouse\\e137_ae_neural_decoding\\t5_mouse_500kHz_noise\\NO_GND\\'
gnd_savepath    = 'D:\\ae_mouse\\e137_ae_neural_decoding\\t5_mouse_500kHz_noise\\GND\\'
# 
gain                        = 500
Fs                          = 5e6
carrier                     = 500000
measurement_channel         = 0
duration                    = 12
timestep                    = 1.0/Fs
m_channel                   = 0 
rf_channel                  = 4
marker_channel              = 7 
carrier                     = 500000
N                           = int(duration*Fs)
t                           = np.linspace(0, duration, N, endpoint=False)
# 

template_savepath      = 'D:\\ae_mouse\\e137_ae_neural_decoding\\'
template_filename      = 'noise_analysis.npz'
print ('filename: ', template_filename)
data                   = np.load(template_savepath+template_filename)
nognd                  = data['nognd']
gnd                    = data['gnd']
frequencies            = data['frequencies']
# 
# T-test. 
# sample1 = nognd
# sample2 = gnd
# t_stat, p_value = ttest_ind(sample1, sample2) 
# print('T-statistic value: ', t_stat) 
# print('P-Value: ', p_value)
# 
print ('nognd shape:',nognd.shape)
#
mean_gnd    = np.mean(gnd,axis=0)
mean_nognd  = np.mean(nognd,axis=0)
std_gnd     = np.std(gnd,axis=0)
std_nognd   = np.std(nognd,axis=0)
#
print ('len frequencies: ',len(frequencies))
#
# I need to remove the US onset and offset from the data. 
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   

# confidence interval. 
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m,h,se


carrier_idx = find_nearest(frequencies,carrier)
nognd_carrier = mean_nognd[carrier_idx]
gnd_carrier = mean_gnd[carrier_idx]
std_gnd_carrier     = std_gnd[carrier_idx]
std_nognd_carrier   = std_nognd[carrier_idx]
print ('mean carrier values nognd/gnd',nognd_carrier,gnd_carrier)
print ('std carrier values nognd/gnd',std_nognd_carrier,std_gnd_carrier)
# 
# Create a bar plot with and without GND.
# With statistical information in it. 
# Also calculate the noise around the carrier +- 10Hz. 
# start_idx   = find_nearest(frequencies,carrier + int(1) )   # 1-5Hz near the carrier 
start_idx   = find_nearest(frequencies,carrier - int(10) )   # 1-5Hz near the carrier 
end_idx     = find_nearest(frequencies,carrier + int(-1) )
nognd_noise_mean  = np.mean(mean_nognd[start_idx:end_idx])
nognd_noise_std   = np.std(mean_nognd[start_idx:end_idx])
print ('nognd noise mean/std: ',np.round(nognd_noise_mean,2),np.round(nognd_noise_std,2))
# 
# start_idx   = find_nearest(frequencies,carrier + int(1) )   # 1-5Hz near the carrier 
# end_idx     = find_nearest(frequencies,carrier + int(20) )
gnd_noise_mean  = np.mean(mean_gnd[start_idx:end_idx])
gnd_noise_std   = np.std(mean_gnd[start_idx:end_idx])
print ('gnd noise mean/std: ',np.round(gnd_noise_mean,2),np.round(gnd_noise_std,2))
# 
materials = ['GND noise floor', 'NO GND noise floor']
x_pos = np.arange(len(materials))
CTEs = [gnd_noise_mean,nognd_noise_mean]
error = [gnd_noise_std,nognd_noise_std]

print ('means:',CTEs)
print ('stds:',error)

# Build the plot
fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(111)
# ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,color='grey', ecolor='black', capsize=10)
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,color='grey', capsize=10)

ax.set_xticks(x_pos)
ax.set_xticklabels([])
# ax.yaxis.grid(True)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Save the figure and show
plt.tight_layout()
plt.savefig(saveprefix+'noisefloor_bar_plot_with_error_bars.png')
plt.show()



# Now calculate the SNR of GND, VS NO GND.  
# Also calculate the mean and std of the carrier signal. 
# 
# The noise is actually pretty similar around the carrier, the difference is the height of the carrier. 
# Altogether I think it is better to GND, if only to decrease the size of the carrier and minimize the DC. 
# Also, the DC component is much larger with not GNDed...
# 
# 
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(frequencies,mean_nognd,color='k')
plt.plot(frequencies,mean_gnd,color='r')
ax.set_xlim([0,1e6])
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Frequency(Hz)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(['nognd','gnd'],loc='upper right')
# plot_filename = t_series+'_vep_events_'+str(n_events)+'.png'
# plt.savefig(plot_filename, transparent=True)
plt.show()
# 
# 
# Create lists for the plot
materials = ['GND', 'NO GND']
x_pos = np.arange(len(materials))
CTEs = [gnd_carrier,nognd_carrier]
error = [std_gnd_carrier,std_nognd_carrier]

print ('means:',CTEs)
print ('stds:',error)

# Build the plot
fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(111)
# ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,color='grey', ecolor='black', capsize=10)
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,color='grey', capsize=10)

ax.set_xticks(x_pos)
ax.set_xticklabels([])
# ax.yaxis.grid(True)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Save the figure and show
plt.tight_layout()
plt.savefig(saveprefix+'carrier_bar_plot_with_error_bars.png')
plt.show()

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
# plot_filename = t_series+'_vep_events_'+str(n_events)+'.png'
# plt.savefig(plot_filename, transparent=True)
# plt.show()



