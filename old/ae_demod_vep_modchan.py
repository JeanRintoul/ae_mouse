#!/usr/bin/python
'''
 Author: Jean Rintoul Date: 13/08/2021

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
from scipy.signal import iirfilter,sosfiltfilt
# 
# increment this for each test. 
# 
# Need at least
# 
test_no = 1
gain    = 1000
# 
# double chEnables[8]                 = {1,1,1,0,0,0,0,1};
# for demodulation, I'd like the rf monitor channel, and the measurement channel only. 
# ae channel is ch 0.
# rf chan is ch 7.
# marker channel is ch 2.
# 
aeti_variables = {
'type':'demodulation',      # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
# 'Fs': 5e6,                # 
'Fs': 1e7,                  # 10M
'duration': 3.0,            # overflows at 4seconds, 3.5 seconds works. 
'position': test_no,
'pressure_amplitude': 0.5, # 0.15 is about 400kPa. 
'pressure_frequency': 672800.0,
'current_amplitude': 0.0,   # its actually a voltage .. Volts. 
'current_frequency': 1000,  # 
# 'ti_frequency': 0,        # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 0,            # the channel of the measurement probe. 
'e_channel': 4,             # this is the voltage measured between the stimulator probes. 
'rf_monitor_channel': 1,    # this output of the rf amplifier. 
'current_monitor_channel': 5, # this is the current measurement channel of the transformer. 
'start_null': 0.1,          # percent of file set to zero at the beginning. 
'start_pause': 0.3,         # percent of file in ramp mode. 
'end_pause': 0.9,           # end ramp. 
'no_ramp':0.0,              # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':10,        # the current and voltage monitor both have attenuators on them 
'marker_channel':2,
'command_c':'mouse_stream',
'save_folder_path':'D:\\mouse_aeti\\e82_ae_demod',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
#  
result, data_out            = m.aeti_recording(**aeti_variables)
# print ('impedance:',data_out[0])
data,idx_lag,original_data  = m.align_data_to_marker_channel(**aeti_variables)
#  
print ('data shape: ',data.shape)
rf_channel = aeti_variables['rf_monitor_channel']
marker_channel = aeti_variables['marker_channel']
m_channel = aeti_variables['ae_channel'] 
v_channel = aeti_variables['e_channel'] 
i_channel = aeti_variables['current_monitor_channel'] 
current_signal_frequency = aeti_variables['current_frequency'] 
Fs       = aeti_variables['Fs'] 
duration = aeti_variables['duration'] 
savepath = aeti_variables['save_folder_path']

timestep = 1.0/Fs
N = int(Fs*duration)
# print("expected no. samples:",N)
t               = np.linspace(0, duration, N, endpoint=False)
start_pause     = int(aeti_variables['start_pause'] * N+1)
end_pause       = int(aeti_variables['end_pause'] *N-1)
# print ('start and end:',start_pause,end_pause)
resistor_current_mon    = 49.9  # 49.9 Ohms for current monitor, 
# 
# 
fft_m = 1e6*fft(data[m_channel][start_pause:end_pause])/gain
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
# 
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 

low  = 1
high = 300
sos_band = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
markerdata = data[marker_channel]
marker     = markerdata/np.max(markerdata)
rawdata    = 1e6*data[m_channel]/gain
# basedata   = 1e6*baseline_data[m_channel]/gain
rfdata     = 10*data[rf_channel]

filtered_rawdata   = rawdata
mains_harmonics = [50,100,150]
for i in range(len(mains_harmonics)):
    mains_low  = mains_harmonics[i] -3
    mains_high = mains_harmonics[i] +3
    mains_sos = iirfilter(17, [mains_low,mains_high], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
    filtered_rawdata  = sosfiltfilt(mains_sos, filtered_rawdata)
# 
filtered_rawdata    = sosfiltfilt(sos_band, filtered_rawdata)

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(t,filtered_rawdata,'r')
# plt.plot(t,marker,'r')
# plt.show()

# Plot the fft of the measurement data, to see if I can see the 
# VEP.
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(frequencies,fft_m,'r')
# ax.set_xlim([0,40])
# ax.set_ylim([0,20])
# ax = fig.add_subplot(212)
# plt.plot(frequencies,fft_m,'b')
# ax.set_xlim([600000,700000])
# plot_filename = savepath + '\\t'+str(test_no)+'_datacheck.png'
# plt.savefig(plot_filename)
# plt.show()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(311)
plt.plot(t,filtered_rawdata,color='r')
plt.plot(t,50*data[marker_channel],color='g')
ax.set_xlim(0,np.max(t))
ax2 = fig.add_subplot(312)
plt.plot(t,data[rf_channel],color='k')
# plt.plot(t,data[m_channel],color='r')
ax2.set_xlim(0,np.max(t))
ax3 = fig.add_subplot(313)
plt.plot(frequencies,fft_m,'r')
# ax3.set_ylim([0,5])
ax3.set_ylim([0,50])
# ax3.set_xlim([0,800000])
ax3.set_xlim([0,100])
# ax3.set_xlim([672000,673000])
plot_filename = savepath + '\\t'+str(test_no)+'_datacheck.png'
plt.savefig(plot_filename)
plt.show()

