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
# 
# start                       = 1
# stop                        = 7
# step                        = 1 
# file_list                   = range(start,stop,step)
# print ('file list',file_list)

file_list = [6,1,2,3,4,5]
# 
# nognd_savepath  = 'D:\\ae_mouse\\e137_ae_neural_decoding\\t5_mouse_500kHz_noise\\NO_GND\\'
vertical_savepath    = 'D:\\ae_mouse\\e137_ae_neural_decoding\\t5_mouse_500kHz_noise\\vertical_traversal_noise_analysis\\'
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
# Go from 2-10seconds. 
start_pause = int(2*Fs)
end_pause   = int(10*Fs)
# 
xf          = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
beta        = 20
window      = np.kaiser( (end_pause-start_pause), beta )
print('window len:',len(window))
# 
# 
vertical_list   = []
for n in range(len(file_list)):
    file_number = file_list[n] 
    print ('file_number',file_number)
    # vertical_savepath
    vertical_filename = vertical_savepath + 't'+str(file_number)+'_stream.npy'
    vertical_data     = np.load(vertical_filename)
    vertical_fsignal  = (1e6*vertical_data[m_channel]/gain)
    # 
    fft_vertical      = fft(vertical_fsignal[start_pause:end_pause]*window)
    fft_vertical      = np.abs(2.0/(end_pause-start_pause) * (fft_vertical))[1:(end_pause-start_pause)//2]
    # append the fft information into the list. 
    vertical_list.append(fft_vertical)
# 
# 
outfile = 'vertical_noise_analysis.npz'
# save out the information of interest. 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,vertical = np.array(vertical_list),frequencies=frequencies)
print ('saved out a data file!')
# 
# Now calculate the SNR of GND, VS NO GND.  
# Also calculate the mean and std of the carrier signal. 
# 
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(frequencies,vertical_list)
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Frequency(Hz)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plot_filename = t_series+'_vep_events_'+str(n_events)+'.png'
# plt.savefig(plot_filename, transparent=True)
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



