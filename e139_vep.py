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
# 
multifile_start             = 1  
multifile_end               = 11
test_no = 1
gain    = 500
Fs      = 1e5
measurement_channel = 0
# this is the variable to change. 
# dfx     = 1
# compute the start and end points to be accurate. 
duration = 12
# 5 seconds of data. 
# 0.5 seconds of nothing at start and end. 
# 
start_time = np.round(0.8/duration ,2)
end_time   = np.round((duration - 1.0)/duration,2)
print ('start and end',start_time,end_time)
# 
# 
aeti_variables = {
'type':'demodulation',        # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': Fs,                     # 
'USMEP': 1,                   # when usmep == 1, the current and pressure Fs can be different to the recorded Fs. In this case the US is at 5Mhz, but the recording frequency is 100kHz. This decreases the amount of data that needs to be dumped to disk. 
'duration': duration, 
'position': test_no,
'pressure_amplitude': 0.0,        # how much is lost through skull??? 400kPa, 0.08 is about 200kPz. 0.15 is about 400kPa. 
'pressure_frequency': 0.0,
# 'pressure_prf':0,                # pulse repetition frequency for the sine wave. Hz.
# 'pressure_ISI':1.0,              # inter trial interval in seconds. 
# 'pressure_burst_length': 0.05,   # burst length in seconds. 
'current_amplitude':0.0,       
'current_frequency':0.0,      # 
# 'current_ISI':1.0,  # time in seconds between repeats. default length of time for pulse is 0.05s. see c code. 
# 'current_burst_length':0.05,    # 50ms 
# 'current_prf':0,
# # 'ti_frequency': carrier + dfx,  # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 0,                # the channel of the measurement probe. 
'rf_monitor_channel': 4,        # this output of the rf amplifier.  
'marker_channel':7,
# channel assignments for = {1,1,0,0,1,0,0,1};
# 'ae_channel': 0,                # the channel of the measurement probe. 
# 'rf_monitor_channel': 2,        # this output of the rf amplifier.  
# 'marker_channel':3,
'e_channel': 6,                 # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5,   # this is the current measurement channel of the transformer. 
'end_null': 0.0,                  # start of end null. 
'end_pause': end_time,            # start of end ramp
'start_null': 0.05,               # percent of file set to zero at the beginning. 
'start_pause': start_time,        # percent of file in ramp mode or null at start.
'no_ramp':0.0,                  # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                    # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':1,             # the current and voltage monitor both have attenuators on them 
'command_c':'code\\mouse_stream',
'save_folder_path':'D:\\ae_mouse\\e139_ae_neural_decoding',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
# 
rf_channel = aeti_variables['rf_monitor_channel']
marker_channel = aeti_variables['marker_channel']
m_channel = aeti_variables['ae_channel'] 
# v_channel = aeti_variables['e_channel'] 
# i_channel = aeti_variables['current_monitor_channel'] 
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
iteration_number        = 0
summed_total            = [0]*t
average_signal          = [0]*t
m_channel = aeti_variables['ae_channel'] 


x = range(multifile_start,multifile_end)

print ('x',x )
for i in x:
    print (i)
    # Divide by this number to get the average.  
    iteration_number = iteration_number + 1
    aeti_variables['position'] = i
    # Do a recording and copy it into the experiment folder. 
    result, data_out            = m.aeti_recording(**aeti_variables)
    data                        = m.copy_to_folder_and_return_data(**aeti_variables)
    # 
    if i ==0:
        summed_total = 1e6*data[m_channel]/gain
    else: 
        summed_total = summed_total + 1e6*data[m_channel]/gain

average_signal = summed_total/(iteration_number)
# 
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# plt.plot(data[1],'k')
# # ax.set_xlim([start,stop])
# plt.show()
#
#  
start_pause     = int(aeti_variables['start_pause'] * N+1)
end_pause       = int(aeti_variables['end_pause'] *N-1)
# print ('start and end:',start_pause,end_pause)

# lfp band filter. 
df_l = 0.3
df_h = 300
sos_lfp_band = iirfilter(17, [df_l, df_h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

# spike band filter. 
df_l = 300
df_h = 1500
sos_spike_band = iirfilter(17, [df_l, df_h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
# this is if I use the preamp. 
# fsignal 		      = 1e6*data[m_channel]/gain
# do averaging to see the signal. 
fsignal               = average_signal
# 
fft_m = fft(fsignal[start_pause:end_pause])
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
# 
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 
# df = int(abs(dfx))
# sf = int(abs(carrier*2 + dfx))
# df = dfx 
# # carrier_idx = m.find_nearest(frequencies,acoustic_frequency)
# df_idx = m.find_nearest(frequencies,dfx)
# sf_idx = m.find_nearest(frequencies,sf)
# 
# # print ('Amplitude at df and sf:',2*fft_v[df_idx],2*fft_v[sf_idx])
# print ('df and sf:',2*fft_m[df_idx],2*fft_m[sf_idx])
# 

# vdata = data[v_channel]
# idata = 0.1*data[i_channel] # 100mA per V i.e. 100*data[i_channel] gives i in mA. 
# impedance = np.max(vdata)/np.max(idata)

# print ('v,i(ma),z:',np.max(vdata),np.max(1000*idata),impedance)

# plt.rc('font', family='serif')
# plt.rc('font', serif='Arial')
# plt.rcParams['axes.linewidth'] = 2

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(311)
plt.plot(t,fsignal,'k')
ax2 = fig.add_subplot(312)
plt.plot(frequencies,fft_m,'k')
ax2.set_xlim([0,40])
ax2.set_ylim([0,100])
ax3 = fig.add_subplot(313)
# plt.plot(t,data[7],'k')
# ax2.set_ylim([0,500])
plt.show()
