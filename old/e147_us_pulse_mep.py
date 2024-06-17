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
bgain               = 10 
gain                = 500
Fs                  = 5e6
carrier             = 500000
measurement_channel = 0
# this is the variable to change. 
# compute the start and end points to be accurate. 
duration = 6
test_no  = 1
# 
# 
dfx 			= 0
name_prefix     = str(dfx)
cut             = 40
sos_low_band    = iirfilter(17, [cut], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# EMG filter. 
sos_emg_band    = iirfilter(17, [52,1000], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
current_signal_frequency = carrier - dfx
print ('current_signal_frequency: ', current_signal_frequency)
start_time = np.round(0.8/duration ,2)
end_time   = np.round((duration - 0.8)/duration,2)

start_time  = 0.15
end_time    = 1.0
print ('start and end',start_time,end_time)
# 
current_amplitude   = 0.0
# pressure_amplitude  = 0.35*1.2  # +20% 
# pressure_amplitude  = 0.3*1.2
pressure_amplitude  = 0.1

state               = '_acdc'
rep                 = 3
# 
# 
aeti_variables = {
'type':'demodulation',        # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': Fs,                     # 
'USMEP': 1,                   # when usmep == 1, the current and pressure Fs can be different to the recorded Fs. In this case the US is at 5Mhz, but the recording frequency is 100kHz. This decreases the amount of data that needs to be dumped to disk. 
'duration': duration, 
# 'position': test_no,
'fileprefix': name_prefix,
'pressure_amplitude': pressure_amplitude,      	# how much is lost through skull??? 400kPa, 0.08 is about 200kPz. 0.15 is about 400kPa. 
'pressure_frequency': carrier,
## 'pressure_prf':prf,             	# pulse repetition frequency for the sine wave. Hz.
'pressure_ISI':2.0,             	# inter trial interval in seconds. 
'pressure_burst_length': 0.5,  	    # burst length in seconds. 
# 'current_ISI':0.0,              # time in seconds between repeats. default length of time for pulse is 0.05s. see c code. 
# 'current_burst_length':0.5,     # 50ms 
# 'current_prf':current_prf,      # twice per second. 
'current_amplitude':current_amplitude,       
'current_frequency':current_signal_frequency,   # 
'current_ISI':2.0,  			# time in seconds between repeats. default length of time for pulse is 0.05s. see c code. 
'current_burst_length':0.5,    	# 50ms 
'ae_channel': 0,                # the channel of the measurement probe. 
'rf_monitor_channel': 4,        # this output of the rf amplifier.  
'marker_channel':7,
'e_channel': 6,                 # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5,   # this is the current measurement channel of the transformer. 
'end_null': 0.0,                # start of end null. 
'end_pause': end_time,          # start of end ramp
'start_null': 0.05,             # percent of file set to zero at the beginning. 
'start_pause': start_time,      # percent of file in ramp mode or null at start.
'no_ramp':1,                    # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                    # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':1,             # the current and voltage monitor both have attenuators on them 
'command_c':'code\\mouse_stream',
'save_folder_path':'D:\\ae_mouse\\e147_meps',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
# 

# Leave one out test. 
# Test 1: ae (p and v) 
# aeti_variables['pressure_amplitude'] = pressure_amplitude
# aeti_variables['current_amplitude'] = current_amplitude
# aeti_variables['fileprefix'] = 'ae_g'+str(gain)
# result, data_out            = m.aeti_recording(**aeti_variables)
# data                        = m.copy_to_folder_and_return_data(**aeti_variables)
# Test 2: p only
aeti_variables['pressure_amplitude'] = pressure_amplitude
aeti_variables['current_amplitude'] = 0
aeti_variables['fileprefix'] = 'p_p_' +str(pressure_amplitude)+state+'_rep'+str(rep)
result, data_out            = m.aeti_recording(**aeti_variables)
data                        = m.copy_to_folder_and_return_data(**aeti_variables)
# # Test 3: v only
# aeti_variables['pressure_amplitude'] = 0
# aeti_variables['current_amplitude'] = current_amplitude
# aeti_variables['fileprefix'] = 'v_v' +str(current_amplitude)+'_g'+str(gain)
# result, data_out            = m.aeti_recording(**aeti_variables)
# data                        = m.copy_to_folder_and_return_data(**aeti_variables)

# x = range(1,2)
# print ('x',x )
# for i in x:
# 	print (i)
# 	aeti_variables['position'] = i
#  Do a recording and copy it into the experiment folder. 
# result, data_out            = m.aeti_recording(**aeti_variables)
# data                        = m.copy_to_folder_and_return_data(**aeti_variables)
# 
# # #  
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
# this is if I use the preamp. 
fsignal 		      = 1e6*data[m_channel]/bgain
rfsignal 		      = 10*data[rf_channel]
emg_channel           = 2 
emgsignal             = 1e6*data[emg_channel]/gain

# Calculate the current output, and the impedance between the stimulation electrodes. 
# i_signal = -5*data[i_channel]/49.9 
# v_signal = 10*data[v_channel]
# V_pp = np.max(v_signal) - np.min(v_signal)
# I_pp = np.max(i_signal) - np.min(i_signal)
# Z = np.abs(V_pp /I_pp)
# print ('max I(mA),V(V),Z(Ohms)', 1000*I_pp,V_pp,Z)
# # 
low_signal    = sosfiltfilt(sos_low_band, fsignal)
emg_signal    = sosfiltfilt(sos_emg_band, emgsignal)
# 
fft_m = fft(fsignal[start_pause:end_pause])
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
# 
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 
df_idx = m.find_nearest(frequencies,0.5)
# 
st = int(0.0*Fs)
en = int(duration*Fs)
# 
print ('iso freq:',np.round(frequencies[df_idx],2 ))
print ('df:',2*fft_m[df_idx])
print ('pp height: ', np.max(low_signal[st:en])-np.min(low_signal[st:en]))

# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 
f = 18 
# 
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(411)
plt.plot(t,rfsignal,'k')
ax.set_xlim([0,duration])
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
plt.legend(['RF signal'],loc='upper right')
ax2 = fig.add_subplot(412)
plt.plot(frequencies,fft_m,'k')
plt.axvline(x=frequencies[df_idx],color='r')
ax2.set_xlim([0,cut])
plt.legend(['FFT measurement signal'],loc='upper right')
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax3 = fig.add_subplot(413)
plt.plot(t,low_signal,'r')
ax3.set_xlim([0,duration])
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
plt.legend(['low f signal'],loc='upper right')
ax4 = fig.add_subplot(414)
plt.plot(t,emg_signal,'k')
plt.legend(['emg'],loc='upper right')
ax4.set_xlim([0,duration])
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
plt.tight_layout()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
plot_filename = savepath + '\\tp_p_'+str(pressure_amplitude)+state+'rep'+str(rep)+'_US_MEP.png'
plt.savefig(plot_filename)
plt.show()
# 
# 