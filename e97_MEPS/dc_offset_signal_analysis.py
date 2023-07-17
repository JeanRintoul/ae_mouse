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
# 
# 
nac_file_number = 64
ac_file_number  = 34
# 
gain            = 500 
led_frequency   = 4   # in Hz. The number of times the LED is on per second. 
factor          = 1   
savepath        = 'D:\\mouse_aeti\\e97_MEPS\\t11_mouse\\'
Fs              = 5e6
duration        = 8.0
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 2 
# 
led_duration = 1/(2*led_frequency) # when the led turned on and off. 
start_time   = 0.0
end_time     = start_time+led_duration
start        = factor*led_duration     # this is where the trial starts seconds before marker. 
end          = factor*led_duration*2   # this is where the trial ends
# 
nac_filename = savepath + 't'+str(nac_file_number)+'_stream.npy'
nac_data     = np.load(nac_filename)
a,b          = nac_data.shape
# 
ac_filename    = savepath + 't'+str(ac_file_number)+'_stream.npy'
ac_data = np.load(ac_filename)
# 
t = np.linspace(0, duration, N, endpoint=False)
# convert it to microvolts by taking the gain into account. 
nac_signal = 1e6*nac_data[m_channel]/gain
ac_signal  = 1e6*ac_data[m_channel]/gain
# 
ac_rf_signal   = 10*ac_data[rf_channel]
nac_rf_signal  = 10*nac_data[rf_channel]

# create the marker channel. 
markerdata        = np.array(ac_data[marker_channel])
marker            = markerdata/np.max(markerdata)
diffs             = np.diff( markerdata )
zarray            = t*[0]
indexes           = np.argwhere(diffs > 0.2)[:,0] # leading edge
marker_up_length  = int(led_duration*Fs)
for i in range(len(indexes)):
    if i > 0:
        zarray[indexes[i]:(indexes[i]+marker_up_length) ] = 1
markerdata = zarray
# 
# mains_harmonics = [50,100,150,200,250,300]
# for i in range(len(mains_harmonics)):
#     mains_low  = mains_harmonics[i] -2
#     mains_high = mains_harmonics[i] +2
#     mains_sos = iirfilter(17, [mains_low,mains_high], rs=60, btype='bandstop',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
#     nac_fsignal = sosfiltfilt(mains_sos, nac_fsignal)
# 
low  = 0.5
high = 1000 
high = 50 
sos_low = iirfilter(17, [high], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
sos_sum = iirfilter(17, [990000,1010000], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
nac_fsignal  = sosfiltfilt(sos_low, nac_signal)
ac_fsignal   = sosfiltfilt(sos_low, ac_signal)
# 
nac_sum_fsignal  = sosfiltfilt(sos_sum, nac_signal)
ac_sum_fsignal   = sosfiltfilt(sos_sum, ac_signal)
# 
# rf signal. 
ac_dc_rf_signal    = sosfiltfilt(sos_low, ac_rf_signal)
nac_dc_rf_signal   = sosfiltfilt(sos_low, nac_rf_signal)
# 
ac_rf_sum_signal       = sosfiltfilt(sos_sum, ac_rf_signal)
nac_rf__sum_signal      = sosfiltfilt(sos_sum, nac_rf_signal)
# 
# 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
plt.plot(t,ac_fsignal,'r')
plt.plot(t,nac_fsignal,'k')
plt.plot(t,100*(markerdata)-200,'g')
ax.set_xlim([0,8])
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Time(s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(['acoustically connected(<50Hz i.e DC)','not acoustically connected(<50Hz i.e DC)','marker'],loc='upper right')

ax2 = fig.add_subplot(312)
plt.plot(t,ac_sum_fsignal,'r')
plt.plot(t,nac_sum_fsignal,'k')
plt.legend(['acoustically connected(1MHz filter)','not acoustically connected((1MHz filter))'],loc='upper right')
ax2.set_xlim([0,8])
ax2.set_ylabel('Volts ($\mu$V)')
ax2.set_xlabel('Time(s)')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax3 = fig.add_subplot(313)
plt.plot(t,ac_data[m_channel],'r')
plt.plot(t,nac_data[m_channel],'k')


plt.legend(['acoustically connected(raw)','not acoustically connected(raw)'],loc='upper right')
ax3.set_xlim([0,8])
ax3.set_ylabel('Volts ($\mu$V)')
ax3.set_xlabel('Time(s)')
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

plot_filename = 'dc_offset_analysis.png'
plt.savefig(plot_filename)
plt.show()
# 
# Looking at the RF signal. 
# 
# 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
plt.plot(t,ac_dc_rf_signal ,'r')
plt.plot(t,nac_dc_rf_signal,'k')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax2 = fig.add_subplot(212)
plt.plot(t,ac_rf_sum_signal,'r')
plt.plot(t,nac_rf__sum_signal,'k')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

plot_filename = 'rf_offset_analysis.png'
plt.savefig(plot_filename)
plt.show()
# 

# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_data,'k')
# ax.set_xlim([0,high])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = 'VEP_FFT.png'
# plt.savefig(plot_filename)
# plt.show()
# # 
# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(time_segment,average_lfp,color='r')
# plt.fill_between(time_segment, average_lfp - std_lfp, average_lfp + std_lfp,
#                   color='gray', alpha=0.2)
# plt.axvline(x=start_time,color ='k')
# plt.axvline(x=end_time,color ='k')
# ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xlabel('Time(s)')
# plt.autoscale(enable=True, axis='x', tight=True)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.title('VEP')
# plot_filename = 'VEP.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(t,marker,'r')
# plt.plot(t,fsignal/np.max(fsignal),'g')

# ax.set_xlim([0.5,7.5])
# ax.set_ylim([-2,2])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# ax2 = fig.add_subplot(312)
# plt.plot(time_segment,filtered_segmented_array.T)
# plt.axvline(x=start_time,color ='k')
# plt.axvline(x=end_time,color ='k')
# # ax2.text(0.5, 100, 'individual trials', color='black',fontsize = 6, ha='center')
# plt.autoscale(enable=True, axis='x', tight=True)
# ax2.set_ylabel('Individual trials ($\mu$V)')

# ax3 = fig.add_subplot(313)
# plt.plot(time_segment,average_lfp,color='r')
# plt.fill_between(time_segment, average_lfp - std_lfp, average_lfp + std_lfp,
#                   color='gray', alpha=0.2)
# plt.axvline(x=start_time,color ='k')
# plt.axvline(x=end_time,color ='k')
# ax3.set_xlabel('time(s)')
# ax3.set_ylabel('Volts ($\mu$V)')
# plt.autoscale(enable=True, axis='x', tight=True)

# # plot_filename = 'raw_VEP_data.png'
# # plt.savefig(plot_filename)
# plt.show()
