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
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16

saveprefix          = './/images//'
# 
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
# 
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
t  = np.linspace(0, duration, N, endpoint=False)
# 
template_savepath            = 'D:\\ae_mouse\\e137_ae_neural_decoding\\'
template_filename            = 'vertical_noise_analysis.npz'
print ('filename: ', template_filename)
data                         = np.load(template_savepath+template_filename)
vertical                     = data['vertical']
frequencies                  = data['frequencies']
# 
DV_list                      = [2,1,0,-1,-2,-3]
carrier_idx = find_nearest(frequencies,carrier)
start_idx   = find_nearest(frequencies,carrier - int(10) )   # 1-5Hz near the carrier 
end_idx     = find_nearest(frequencies,carrier + int(-1) )

start_idx2   = find_nearest(frequencies,carrier + int(1) )   # 1-5Hz near the carrier 
end_idx2     = find_nearest(frequencies,carrier + int(10) )

vertical_carriers = []
mean_at_heights   = []
std_at_heights    = []
for i in range(len(DV_list)):
    fft_dv = vertical[i]
    vertical_carrier = fft_dv[carrier_idx]
    vertical_carriers.append(vertical_carrier)
    #
    mean_at_height = np.mean(fft_dv[start_idx:end_idx])
    std_at_height  = np.std(fft_dv[start_idx:end_idx])

    mean_at_height2 = np.mean(fft_dv[start_idx2:end_idx2])
    std_at_height2  = np.std(fft_dv[start_idx2:end_idx2])   
    #
    mean_at_heights.append(mean_at_height+mean_at_height2)
    std_at_heights.append( (std_at_height+std_at_height2)/2  )
# 
ss = np.array(std_at_heights)

# acoustoelectric_magnitudes = [11.2,16.5,11.7,9.55,9.3,8.2]

# fig = plt.figure(figsize=(5,5))
# ax  = fig.add_subplot(111)
# plt.plot(DV_list,acoustoelectric_magnitudes,color='r')
# plt.plot(DV_list,acoustoelectric_magnitudes,'.r')
# ax.set_ylabel('Volts ($\mu$V)',fontsize=16)
# ax.set_xlabel('DV Height (mm)',fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.tight_layout()
# # plt.legend(['nognd','gnd'],loc='upper right')
# plot_filename = saveprefix+'vertical_aemags.png'
# plt.savefig(plot_filename, transparent=True)
# plt.show()
# 
fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111)
plt.plot(DV_list,ss,color='k')
plt.plot(DV_list,ss,'.k')

# plt.plot(frequencies,mean_gnd,color='r')
# ax.set_xlim([0,1e6])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.set_ylabel('Volts ($\mu$V)',fontsize=16)
ax.set_xlabel('DV Height (mm)',fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
# plt.legend(['nognd','gnd'],loc='upper right')
plot_filename = saveprefix+'vertical_noisefloor.png'
plt.savefig(plot_filename, transparent=True)
plt.show()
# 
# 


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



