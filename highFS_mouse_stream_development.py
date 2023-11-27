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
# ch5 has the differential measurement of the function generator. 
# 
test_no = 1
gain    = 1
# 
pressure_prf        = 0.5
current_prf         = 0.5
# Fs = current_frequency*10 
# Fs      = 1e7  # this is the max frequency I can have on the recording. The function gneerators can go higher though. 
# there appears to be a sweet spot when Fs is x10 the carrier. x20 is NOT better... weirdly.  
carrier = 5e5
Fs      = carrier*10
# carrier = 1e6
# Fs      = carrier*10
dfx 	= 1000
# 
# the other option is to just have a single channel come in. 
# 
measurement_channel = 0
# 
# the problem is there is so much spectral spreading - this may be because of how I was creating the sine wave?
# because the output frequency 
# 
aeti_variables = {
'type':'demodulation',        # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': Fs,                    # 
'USMEP': 0,                   # when usmep == 1, the current and pressure Fs can be different to the recorded Fs. In this case the US is at 5Mhz, but the recording frequency is 100kHz. This decreases the amount of data that needs to be dumped to disk. 
'duration': 4.0, 
'position': test_no,
'pressure_amplitude': 2,    # how much is lost through skull??? 400kPa, 0.08 is about 200kPz. 0.15 is about 400kPa. 
'pressure_frequency': carrier,
# 'pressure_fswitching': carrier+dfx,
# 'pressure_ISI':0,
# 'pressure_prf':pressure_prf,            # pulse repetition frequency for the sine wave. Hz. 
# 'pressure_burst_length':0.25,  # in seconds(maxes out at 50% duty cycle). pressure burst length is calculated as: prf_counter/gen_pressure_sample_frequency
'current_amplitude': 0,       # its actually a voltage .. Volts. 
'current_frequency': 50,  # 
'current_fswitching': 100,
'current_ISI':0,
'current_burst_length':0.25, # Duration should be? 
'current_prf':current_prf, # twice per second. 
# 'ti_frequency': carrier + dfx,  # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 0,                  # the channel of the measurement probe. 
'rf_monitor_channel': 4,          # this output of the rf amplifier.  
# 'e_channel': 6,                 # this is the voltage measured between the stimulator probes. 
# 'current_monitor_channel': 5,   # this is the current measurement channel of the transformer. 
'e_channel': 6,                   # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5,     # this is the current measurement channel of the transformer. 
'marker_channel':7,
'end_null': 0.1,                # start of end null. 
'end_pause': 0.8,               # start of end ramp
'start_null': 0.2,              # percent of file set to zero at the beginning. 
'start_pause': 0.3,             # percent of file in ramp mode or null at start.
'no_ramp':0.0,                  # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                    # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':1,             # the current and voltage monitor both have attenuators on them 
'command_c':'code\\mouse_stream',
'save_folder_path':'D:\\ae_mouse\\e120_highFs_mouse_stream',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
#  Do a recording and copy it into the experiment folder. 
result, data_out            = m.aeti_recording(**aeti_variables)
data                        = m.copy_to_folder_and_return_data(**aeti_variables)

print ('data shape',data.shape)

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
print("expected no. samples:",N)
t               = np.linspace(0, duration, N, endpoint=False)
start_pause     = int(aeti_variables['start_pause'] * N+1)
end_pause       = int(aeti_variables['end_pause'] *N-1)
# print ('start and end:',start_pause,end_pause)

fsignal = data[0]

fft_m = fft(fsignal[start_pause:end_pause])
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 

plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2

start = 0
stop  = duration

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(411)
plt.plot(t,fsignal,'k')
ax2 = fig.add_subplot(412)
plt.plot(t,data[1],'k')
ax3 = fig.add_subplot(413)
plt.plot(t,data[2],'k')
ax4 = fig.add_subplot(414)
plt.plot(t,data[3],'k')
plt.show()
# 
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(311)
plt.plot(t,fsignal,'k')
ax2 = fig.add_subplot(312)
plt.plot(frequencies,fft_m,'k')
ax2.set_xlim([0,2e6])
ax3 = fig.add_subplot(313)
plt.plot(t,fsignal,'k')
ax3.set_xlim([1.5,1.5+0.00001])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
plt.show()
# 
# 