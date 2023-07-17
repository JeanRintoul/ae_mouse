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
file_number         = 17
stim_df_frequency   = 0

gain            = 1000
savepath        = 'D:\\mouse_aeti\\e100_neural_recording_pat_e_mouse\\t6_mep_delta_wave_motorcortex\\'
Fs              = 5e6
duration        = 8.0
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 4 


low             = 0.1
high            = 10
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
emg_data = 1e6*data[v_channel]
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

df_l = 0.1
df_h = stim_df_frequency +3
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
rf_atdf   = sosfiltfilt(sos_df_band, rf_data)
emg_l   = 0.5
emg_h   = stim_df_frequency +3
sos_emg_band = iirfilter(17, [emg_l,emg_h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
#
emgdfsignal = sosfiltfilt(sos_emg_band,emg_data)
# 
# 
# # find the fft of the data. 
# # TODO: this should be from the start to the end of the led flashing, and not including the start and end regions. 
start_pause     = int(1*Fs)
end_pause       = int(3*Fs)
# window          = np.hanning(end_pause-start_pause)

fft_unfiltered_data = fft(fsignal[start_pause:end_pause])
fft_unfiltered_data = np.abs(2.0/(end_pause-start_pause) * (fft_unfiltered_data))[1:(end_pause-start_pause)//2]

fft_emg_data = fft(emg_data[start_pause:end_pause])
fft_emg_data = np.abs(2.0/(end_pause-start_pause) * (fft_emg_data))[1:(end_pause-start_pause)//2]

fft_rf_data = fft(rf_data[start_pause:end_pause])
fft_rf_data = np.abs(2.0/(end_pause-start_pause) * (fft_rf_data))[1:(end_pause-start_pause)//2]



xf                  = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies         = xf[1:(end_pause-start_pause)//2]
#
#
#
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(t,rf_data1,'k')
# ax = fig.add_subplot(312)
# plt.plot(t,rf_data2,'k')
# ax = fig.add_subplot(313)
# plt.plot(t,rf_data,'k')
# plt.show()
#
#
#
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(t,1e6*data[m_channel]/gain,'k')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = 'raw_timeseries.png'
plt.savefig(plot_filename)
plt.show()
#
#
#
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(311)
plt.plot(t,rf_data/np.max(rf_data),'grey')
plt.plot(t,filtered_signal/np.max(filtered_signal),'k')
# plt.plot(t,emgdfsignal/np.max(emgdfsignal),'r')
# plt.plot(t,rf_atdf/np.max(rf_atdf),'m')
# plt.legend(['rf data','neural data','emg data','rf at df'], loc='upper right')
plt.legend(['rf data','neural data'], loc='upper right')
ax2 = fig.add_subplot(312)
plt.plot(frequencies,fft_unfiltered_data,'k')
# plt.plot(frequencies,fft_emg_data,'r')
ax2.set_ylim([0,300])
ax2.set_xlim([0,10])
plt.legend(['neural data'], loc='upper right')
ax3 = fig.add_subplot(313)
plt.plot(frequencies,fft_rf_data,'r')
ax3.set_ylim([0,0.005])
ax3.set_xlim([0,10])
ax3.set_xlabel('frequencies(Hz)')
plt.legend(['rf data'], loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
plot_filename = 'MEP_timeseries.png'
plt.savefig(plot_filename)
plt.show()
#
#

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(t,rf_data/np.max(rf_data),'grey')
plt.plot(t,filtered_signal/np.max(filtered_signal),'k')
plt.legend(['rf data','neural data'], loc='upper right')
ax.set_xlim([0,duration])
ax.set_xlabel('time(s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = 'MEP_RFNEURAL_data.png'
plt.savefig(plot_filename)
plt.show()
#
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(121)
plt.plot(frequencies,fft_unfiltered_data,'k')
ax.set_xlabel('frequencies(Hz)')
ax.set_ylim([0,300])
ax.set_xlim([0,10])
plt.legend(['neural data'], loc='upper right')
ax.set_ylabel('Volts ($\mu$V)')
ax2 = fig.add_subplot(122)
plt.plot(frequencies,1e6*fft_rf_data,'r')
ax2.set_ylim([0,4000])
ax2.set_xlim([0,10])
# ax2.set_ylabel('Volts (V)')
plt.legend(['rf data'], loc='upper right')
ax2.set_xlabel('frequencies(Hz)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plot_filename = 'MEP_RFNEURAL_fft.png'
plt.savefig(plot_filename)
plt.show()