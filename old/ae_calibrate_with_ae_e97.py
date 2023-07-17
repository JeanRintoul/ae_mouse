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
# increment this for each test. 
# 
test_no = 1
gain    = 1
# 
aeti_variables = {
'type':'demodulation',        # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 5e6,                    # 
# 'Fs': 1e4,                  # this will change just the sample rate, and not the f generated rate if those two lines are left uncommented. 
# 'USMEP': 1,                 # when usmep == 1, the current and pressure Fs can be different to the recorded Fs. In this case the US is at 5Mhz, but the recording frequency is 100kHz. This decreases the amount of data that needs to be dumped to disk. 
'duration': 4.0, 
'position': test_no,
'pressure_amplitude': 0.1,      # how much is lost through skull??? 400kPa, 0.08 is about 200kPz. 0.15 is about 400kPa. 
'pressure_frequency': 500000.0,
# 'pressure_frequency': 0.0,
'current_amplitude': 0.0,       # its actually a voltage .. Volts. 
# 'current_frequency': 8,       # 
'current_frequency': 8000,      # 
#'current_frequency': 508000,   # 
#'current_frequency': 500008,   # 
#'current_frequency': 500002,   # 
# 'current_frequency': 4,       # 
# 'current_amplitude': 6.0,     # 
# 'ti_frequency': 0,            # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 0,                # the channel of the measurement probe. 
'rf_monitor_channel': 2,        # this output of the rf amplifier.  
'e_channel': 6,                 # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5,   # this is the current measurement channel of the transformer. 
'marker_channel':7,
'end_null': 0.02,               # start of end null. 
'end_pause': 0.875,             # start of end ramp
# 'start_null': 0.36,           # percent of file set to zero at the beginning. 
# 'start_pause': 0.4,           # percent of file in ramp mode or null at start.
'start_null': 0.125,            # percent of file set to zero at the beginning. 
'start_pause': 0.25,            # percent of file in ramp mode or null at start.
'no_ramp':0.0,                  # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                    # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':1,             # the current and voltage monitor both have attenuators on them 
'command_c':'code\\mouse_stream',
'save_folder_path':'D:\\mouse_aeti\\e97_MEPS',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
#  
result, data_out            = m.aeti_recording(**aeti_variables)
# print ('impedance:',data_out[0])
data,idx_lag,original_data  = m.align_data_to_marker_channel(**aeti_variables)
# print ('impedance:',data_out[0])
# data,idx_lag,original_data  = m.align_data_to_marker_channel(**aeti_variables)
# #  
rf_channel = aeti_variables['rf_monitor_channel']
marker_channel = aeti_variables['marker_channel']
m_channel = aeti_variables['ae_channel'] 
v_channel = aeti_variables['e_channel'] 
i_channel = aeti_variables['current_monitor_channel'] 
current_signal_frequency = aeti_variables['current_frequency'] 
acoustic_frequency = aeti_variables['pressure_frequency']
Fs       = aeti_variables['Fs'] 
duration = aeti_variables['duration'] 
savepath = aeti_variables['save_folder_path']
#  
timestep = 1.0/Fs
N = int(Fs*duration)
# print("expected no. samples:",N)
t               = np.linspace(0, duration, N, endpoint=False)
start_pause     = int(aeti_variables['start_pause'] * N+1)
end_pause       = int(aeti_variables['end_pause'] *N-1)
# print ('start and end:',start_pause,end_pause)
resistor_current_mon    = 49.9  # 49.9 Ohms for current monitor, 
# 
fft_m = 1e6*fft(data[m_channel][start_pause:end_pause])/gain
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
# 
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 
fft_us = fft(data[rf_channel][start_pause:end_pause])
fft_us = np.abs(2.0/(end_pause-start_pause) * (fft_us))[1:(end_pause-start_pause)//2]
# 
# 
fft_v = fft(data[v_channel][start_pause:end_pause])
fft_v = np.abs(2.0/(end_pause-start_pause) * (fft_v))[1:(end_pause-start_pause)//2]
# 
# 
fft_i = fft(data[i_channel][start_pause:end_pause])
fft_i = np.abs(2.0/(end_pause-start_pause) * (fft_i))[1:(end_pause-start_pause)//2]
# 
# 
df = int(abs(acoustic_frequency - current_signal_frequency))
sf = int(abs(acoustic_frequency + current_signal_frequency))
# print ('frequencies of interest')
# 
#
carrier_idx = m.find_nearest(frequencies,acoustic_frequency)
df_idx = m.find_nearest(frequencies,df)
sf_idx = m.find_nearest(frequencies,sf)
v_idx = m.find_nearest(frequencies,current_signal_frequency) 

i = fft_i[v_idx]/resistor_current_mon
z = fft_v[v_idx]/i
print ('i (ma) :',np.round(i*1000,3)  )
print ('z (ohms) :',np.round(z,3))
# 
# 
# For this calibration, we are measuring the ae signal on the measurement electrode. 
# i.e. on 
print ('Amplitude at df and sf:',2*fft_v[df_idx],2*fft_v[sf_idx])


now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
print ('df (\u03BCV):',1e6*fft_v[df_idx])
print ('carrier (\u03BCV):',1e6*fft_v[carrier_idx])
print ('sf (\u03BCV):',1e6*fft_v[sf_idx])
# 
# First plot is the raw signal, 
# Second plot is the 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
# plt.plot(t,10*data[rf_channel],color='b')
# plt.plot(t,10*data[v_channel],color='r')
plt.plot(frequencies,fft_us,'b')
plt.plot(frequencies,fft_v,'r')
# plt.plot(frequencies,fft_m,'k')
plt.legend(['fft us','fft v'])
ax.set_xlim([0,int(np.max(frequencies))/2])

ax2 = fig.add_subplot(212)
plt.axvline(df,color='k')
plt.axvline(sf,color='k')
# plt.plot(frequencies,fft_us,'b')
# plt.plot(frequencies,fft_m,'c')
plt.plot(frequencies,1e6*fft_v,'r')
# plt.plot(frequencies,fft_i,'orange')
ax2.set_ylim([0,500])
ax2.set_xlim([acoustic_frequency - 2*current_signal_frequency,acoustic_frequency + 2*current_signal_frequency])
# ax2.set_xlim([0,40])
# ax2.set_ylim([0,200])
plt.legend(['fft us','fft v'])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.suptitle('AE location calibration')
plot_filename = savepath + '\\t'+str(test_no)+'_datacheck.png'
plt.savefig(plot_filename)
plt.show()
# 
# 
# print ('Second plot')
# low  = abs(current_signal_frequency - acoustic_frequency) - 1.8
# high = abs(current_signal_frequency - acoustic_frequency) + 2
# print ('low and high',low,high)
# sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# fsignal  = sosfiltfilt(sos_low,data[m_channel])
# print (len(fsignal),len(t),len(data[v_channel]))
# print ('df measurement idx',fft_m[df_idx])
# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(frequencies,fft_m,'k')
# ax.set_xlim([0,40])
# ax.set_ylim([0,2000])
# ax2 = fig.add_subplot(212)
# plt.plot(t,data[m_channel],'k')
# plt.plot(t,fsignal,'r')
# plt.show()
# # 
# # 
# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(t,data[v_channel],color='r')
# ax2 = fig.add_subplot(212)
# plt.plot(fsignal,color='k')
# plt.show()
# # 
# 
# 