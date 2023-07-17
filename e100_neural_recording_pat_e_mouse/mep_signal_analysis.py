'''

Title: mep signal analysis. 
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
file_number     = 30
stim_df_frequency   = 2
gain            = 100
savepath        = 'D:\\mouse_aeti\\e100_neural_recording_pat_e_mouse\\t4_delta_wave_PRF1020\\'
Fs              = 5e6
duration        = 8.0
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 4 


low                 = 0.5
high                = 40
sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')


filename    = savepath + 't'+str(file_number)+'_stream.npy'
data = np.load(filename)
a,b = data.shape
t = np.linspace(0, duration, N, endpoint=False)
# convert it to microvolts by taking the gain into account. 
fsignal = 1e6*data[m_channel]/gain
rf_data = 10*data[rf_channel]
filtered_signal = sosfiltfilt(sos_low, fsignal)
#
#
# Lets look at why the RF monitor data goes weird... 
carrier_f     = 500000
carrier_f2    = carrier_f - stim_df_frequency
offset        = 1.0
# 
# 
l = carrier_f - offset
h = carrier_f + offset
sos_carrier_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
l2 = carrier_f2 - offset
h2 = carrier_f2 + offset
sos_carrier2_band = iirfilter(17, [l2,h2], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

df_l = 0.5
df_h = 5
sos_df_band = iirfilter(17, [df_l,df_h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
#
#
rf_data1    = sosfiltfilt(sos_carrier_band, rf_data)
rf_data2    = sosfiltfilt(sos_carrier2_band, rf_data)

fsignal1    = sosfiltfilt(sos_carrier_band, fsignal)
fsignal2    = sosfiltfilt(sos_carrier2_band, fsignal)
dfsignal    = sosfiltfilt(sos_df_band, fsignal)
# 
# 
# # find the fft of the data. 
# # TODO: this should be from the start to the end of the led flashing, and not including the start and end regions. 
start_pause     = int(2*Fs)
end_pause       = int(4*Fs)
window          = np.hanning(end_pause-start_pause)

fft_unfiltered_data = fft(fsignal[start_pause:end_pause])
fft_unfiltered_data = np.abs(2.0/(end_pause-start_pause) * (fft_unfiltered_data))[1:(end_pause-start_pause)//2]
xf                  = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies         = xf[1:(end_pause-start_pause)//2]
#
#
#
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(311)
plt.plot(t,rf_data1,'k')
ax = fig.add_subplot(312)
plt.plot(t,rf_data2,'k')
ax = fig.add_subplot(313)
plt.plot(t,rf_data,'k')
plt.show()
#
#
#
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
plt.plot(t,filtered_signal/np.max(filtered_signal),'k')
ax = fig.add_subplot(212)

plt.plot(t,dfsignal,'k')
plt.show()
#
#
#
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
plt.plot(t,rf_data/np.max(rf_data),'grey')
plt.plot(t,filtered_signal/np.max(filtered_signal),'k')

ax2 = fig.add_subplot(212)
plt.plot(frequencies,fft_unfiltered_data,'k')
ax2.set_xlim([0,10])
# ax2.set_ylim([0,140])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plot_filename = 'MEP_timeseries.png'
plt.savefig(plot_filename)
plt.show()
#
#
# fft_data        = fft(fsignal[start_pause:end_pause]*window)
# fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]

# fft_demod        = fft(demodulated[start_pause:end_pause]*window)
# fft_demod        = np.abs(2.0/(end_pause-start_pause) * (fft_demod))[1:(end_pause-start_pause)//2]
# print ('fft bin size ', frequencies[2]-frequencies[1])

# rf_data = rf_data/np.max(rf_data)
# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_data,'r')
# plt.plot(frequencies,fft_demod,'b')
# plt.plot(frequencies,fft_unfiltered_data,'k')
# ax.set_xlim([0,1e6])
# plot_filename = 'FFT.png'
# plt.savefig(plot_filename)
# plt.show()
# # 
# # 
# lim1 = int(0.5*Fs)
# lim2 = int(7.5*Fs)
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# # plt.plot(t[lim1:lim2],0.5*marker[lim1:lim2],'r')
# plt.plot(t[lim1:lim2],rf_data[lim1:lim2],'grey')
# plt.plot(t[lim1:lim2],fsignal[lim1:lim2]/np.max(fsignal[lim1:lim2]),'r')
# plt.plot(t[lim1:lim2],demodulated[lim1:lim2]/np.max(demodulated[lim1:lim2]),'k')
# # ax.set_xlim([1.8])
# # ax.set_ylim([-2,2])
# plt.legend(['rf data','lp lfp','demod'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.autoscale(enable=True, axis='x', tight=True)
# plot_filename = 'raw_and_demod_data.png'
# plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# # plt.plot(t[lim1:lim2],0.5*marker[lim1:lim2],'r')
# plt.plot(t[lim1:lim2],100*rf_data[lim1:lim2],'grey')
# plt.plot(t[lim1:lim2],fsignal[lim1:lim2],'r')
# plt.plot(t[lim1:lim2],demodulated[lim1:lim2],'k')
# # ax.set_xlim([lim1,lim2])
# # ax.set_ylim([-2,2])
# plt.legend(['rf data','lp lfp','demod'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.autoscale(enable=True, axis='x', tight=True)
# plot_filename = 'raw_and_demod_data_unscaled.png'
# plt.savefig(plot_filename)
# plt.show()