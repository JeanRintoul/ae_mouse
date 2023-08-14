#!/usr/bin/python
'''
 Author: Jean Rintoul Date: 18/01/2023

'''
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib
from scipy import interpolate
import serial, time
from subprocess import check_output
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.fft import fft, fftfreq
import mouse_library as m
import pandas as pd
from datetime import datetime
from scipy.signal import iirfilter,sosfiltfilt
# 
# Increment this for each test. 
# 
test_no = 1
gain    = 100 # gain of EMG electrode? 
# Change this difference frequency. 
dfx = 2 
# 
aeti_variables = {
'type':'demodulation',        # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 5e6,                    # 
# 'Fs': Fs,                   # 
# 'Fs': 1e6,                  # this will change just the sample rate, and not the f generated rate if those two lines are left uncommented. 
# 'USMEP': 1,                 # when usmep == 1, the current and pressure Fs can be different to the recorded Fs. In this case the US is at 5Mhz, but the recording frequency is 100kHz. This decreases the amount of data that needs to be dumped to disk. 
'duration': 8.0, 
'position': test_no,
'pressure_amplitude': 0.0,      # how much is lost through skull??? 400kPa, 0.08 is about 200kPz. 0.15 is about 400kPa. 
'pressure_frequency': 500000.0,
# 'pressure_prf':1020,          # pulse repetition frequency for the sine wave. Hz. 
# 'pressure_ISI':0,             # inter trial interval in seconds. 
# 'pressure_frequency': 0.0,
'current_amplitude': 0.1,       # its actually a voltage .. Volts. 
'current_frequency': 500000.0,  # 
'ae_channel': 0,                # the channel of the measurement probe. 
'rf_monitor_channel': 4,        # this output of the rf amplifier.  
'e_channel': 6,                 # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5,   # this is the current measurement channel of the transformer. 
'marker_channel':7,
'end_null': 0.1,                # start of end null. 
'end_pause': 0.89,              # start of end ramp
'start_null': 0.1,              # percent of file set to zero at the beginning. 
'start_pause': 0.88,            # percent of file in ramp mode or null at start.
'no_ramp':0.0,                  # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                    # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':1,             # the current and voltage monitor both have attenuators on them 
'command_c':'code\\mouse_stream',
'save_folder_path':'D:\\ae_mouse\\e104_mep_us_frequency_proofpoint',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
#
# If I need to do a ramp then just use one US frequency. 
#  
#  
result, data_out            = m.aeti_recording(**aeti_variables)
data                        = m.copy_to_folder_and_return_data(**aeti_variables)
rf_channel      = aeti_variables['rf_monitor_channel']
marker_channel  = aeti_variables['marker_channel']
m_channel       = aeti_variables['ae_channel'] 
v_channel       = aeti_variables['e_channel'] 
i_channel       = aeti_variables['current_monitor_channel'] 
current_signal_frequency = aeti_variables['current_frequency'] 
acoustic_frequency = aeti_variables['pressure_frequency']
Fs              = aeti_variables['Fs'] 
duration        = aeti_variables['duration'] 
savepath        = aeti_variables['save_folder_path']
timestep        = 1.0/Fs
N               = int(Fs*duration)
print("expected no. samples:",N)
t               = np.linspace(0, duration, N, endpoint=False)
start_pause     = int(aeti_variables['start_pause'] * N+1)
end_pause       = int(aeti_variables['end_pause'] *N-1)
resistor_current_mon    = 49.9  # 49.9 Ohms for current monitor, 
# 
fft_m = 1e6*fft(data[m_channel][start_pause:end_pause])/gain
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
fft_us = fft(data[rf_channel][start_pause:end_pause])
fft_us = np.abs(2.0/(end_pause-start_pause) * (fft_us))[1:(end_pause-start_pause)//2]
fft_v = fft(data[v_channel][start_pause:end_pause])
fft_v = np.abs(2.0/(end_pause-start_pause) * (fft_v))[1:(end_pause-start_pause)//2]
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 
'''
Here we specify which channel has gain and neural data, and which channel measures EMG data. 
'''
# this is only the case when I have it in practice mode(i.e. looking at the v chan)
rf_channel = v_channel

# EMG data is on the differential voltage input channel this time? 
emg_data    = 1e6*data[v_channel]
neural_data = 1e6*data[m_channel]/gain
rf_data     = 10*data[rf_channel]
# 
# Filter for the difference frequency. 
# 
offset = 0.9
df_l = dfx - offset
df_h = dfx + offset
sos_df_band = iirfilter(17, [df_l,df_h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
emg_df_data     = sosfiltfilt(sos_df_band, emg_data)
neural_df_data  = sosfiltfilt(sos_df_band, neural_data)
rf_df_data      = sosfiltfilt(sos_df_band, rf_data)
# 
# RF filter. fairly wide filter around the frequencies of the US to show the mixing peaks. 
#  
a_offset = 10
l = acoustic_frequency - a_offset
h = acoustic_frequency + a_offset
sos_carriers_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
emg_carriers_data    = sosfiltfilt(sos_carriers_band, emg_data)
neural_carriers_data = sosfiltfilt(sos_carriers_band, neural_data)
rf_carriers_data     = sosfiltfilt(sos_carriers_band, rf_data)
# 
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(t,data[6],'k')
plt.show()
# 
# difference frequencies plot. 
# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(311)
# plt.plot(t,emg_df_data,'r')
# ax.set_xlim([0,duration])
# ax.set_ylabel('Volts ($\mu$V)')
# plt.legend(['emg df data'],loc='upper right')
# ax2  = fig.add_subplot(312)
# plt.plot(t,neural_df_data,'grey')
# ax2.set_ylabel('Volts ($\mu$V)')
# plt.legend(['neural df data'],loc='upper right')
# ax2.set_xlim([0,duration])
# ax3  = fig.add_subplot(313)
# plt.plot(t,rf_df_data,'k')
# ax3.set_xlim([0,duration])
# ax3.set_ylabel('Volts (V)')
# plt.legend(['rf df data'],loc='upper right')
# ax3.set_xlabel('time(s)')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plot_filename = savepath + '\\t'+str(test_no)+'_df_amplitudes.png'
# plt.savefig(plot_filename)
# plt.show()
# #
# Carrier amplitudes plot. 
# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(311)
# plt.plot(t,emg_carriers_data,'r')
# ax.set_xlim([0,duration])
# ax.set_ylabel('Volts ($\mu$V)')
# plt.legend(['emg carrier data'],loc='upper right')
# ax2  = fig.add_subplot(312)
# plt.plot(t,neural_carriers_data,'grey')
# ax2.set_xlim([0,duration])
# ax2.set_ylabel('Volts ($\mu$V)')
# plt.legend(['neural carrier data'],loc='upper right')
# ax3  = fig.add_subplot(313)
# plt.plot(t,rf_carriers_data,'k')
# plt.legend(['rf carrier data'],loc='upper right')
# ax3.set_xlim([0,duration])
# ax3.set_xlabel('time(s)')
# ax3.set_ylabel('Volts (V)')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plot_filename = savepath + '\\t'+str(test_no)+'_carrier_amplitudes.png'
# plt.savefig(plot_filename)
# plt.show()
# 
# Further analysis can be done offline. 
