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
from math import floor
# 

gain                = 1000
# 
Fs                  = 2e6
carrier             = 500000
measurement_channel = 0
# 
# compute the start and end points to be accurate. 
duration = 12
test_no  = 4
# 
# prf     = 1020
# 
current_signal_frequency = 4
print ('current_signal_frequency: ', current_signal_frequency)
start_time = np.round(0.8/duration ,2)
end_time   = np.round((duration - 0.4)/duration,2)
print ('start and end',start_time,end_time)
# 
# 
prf = 120 
# 
aeti_variables = {
'type':'demodulation',        # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': Fs,                     # 
'USMEP': 1,                   # when usmep == 1, the current and pressure Fs can be different to the recorded Fs. In this case the US is at 5Mhz, but the recording frequency is 100kHz. This decreases the amount of data that needs to be dumped to disk. 
'duration': duration, 
'position': test_no,
'pressure_amplitude': 0.1,        # how much is lost through skull??? 400kPa, 0.08 is about 200kPz. 0.15 is about 400kPa. 
'pressure_frequency': carrier,
# 'pressure_prf':prf,             # pulse repetition frequency for the sine wave. Hz.
# 'pressure_ISI':0.0,             # inter trial interval in seconds. 
# 'pressure_burst_length': 0.1,   # burst length in seconds. 
'current_amplitude':0.0,       
'current_frequency':current_signal_frequency,   # 
# 'current_ISI':0.0,  			# time in seconds between repeats. default length of time for pulse is 0.05s. see c code. 
# 'current_burst_length':0.05,  # 50ms 
# 'current_prf':0,
# 'ti_frequency': carrier + dfx,  # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 0,                # the channel of the measurement probe. 
'rf_monitor_channel': 4,        # this output of the rf amplifier.  
'e_channel': 6,                 # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5,   # this is the current measurement channel of the transformer. 
'marker_channel':7,
'end_null': 0.0,                # start of end null. 
'end_pause': end_time,          # start of end ramp
'start_null': 0.05,             # percent of file set to zero at the beginning. 
'start_pause': start_time,      # percent of file in ramp mode or null at start.
'no_ramp':0.0,                  # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                    # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':1,             # the current and voltage monitor both have attenuators on them 
'command_c':'code\\mouse_stream',
'save_folder_path':'D:\\ae_mouse\\e135_ae_neural_decoding',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
# 
x = range(1,21)
print ('x',x )
for i in x:
    print (i)
    aeti_variables['position'] = i
    # Do a recording and copy it into the experiment folder. 
    result, data_out            = m.aeti_recording(**aeti_variables)
    data                        = m.copy_to_folder_and_return_data(**aeti_variables)
# 
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# plt.plot(data[1],'k')
# # ax.set_xlim([start,stop])
# plt.show()
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

marker_signal = data[marker_channel]

# m_channel  = 0 
# v_channel  = 3 
# rf_channel = 2 
# 
# this is if I use the preamp. 
fsignal 		      = 1e6*data[m_channel]/gain

lfp_filter = iirfilter(17, [0.5,300], rs=60, btype='bandpass',
                   analog=False, ftype='cheby2', fs=Fs,
                   output='sos')
lfp_signal  = sosfiltfilt(lfp_filter, fsignal)

# Play with windowing to decrese the spectral leakage of the big signal next to the small signal. 
print ('window length:', end_pause - start_pause)
multiple = floor((end_pause - start_pause)/carrier)
print ('sub-multiple:', floor((end_pause - start_pause)/carrier) )
print ('start time:', start_pause*timestep)
print ('end time:', end_pause*timestep)
# this makes it a perfect multiple, though it doesnt start on a zero point. 
# start_pause  = start_pause + carrier
# end_pause 	 = start_pause + int( multiple-1 )*carrier
# 
# https://numpy.org/doc/stable/reference/generated/numpy.kaiser.html#numpy.kaiser
# beta = 14 is a very thin window. 
# beta = 0 rectangular
# beta = 6 similar to hanning
# beta = 8.6 similar to blackman - I can see 10Hz here. 200 microvolts signal. 
# 
# averaging improves signal to noise. windowing removes the spectral apodization.  
# 
window 		 = np.kaiser( (end_pause - start_pause), 20.0 )
# 
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# plt.plot(fsignal[start_pause:end_pause],'k')
# plt.show()
# 
fft_m = fft(fsignal[start_pause:end_pause] * window)
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
# 
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 
# 
df = int(abs(carrier - current_signal_frequency))
print ('df is',df)
sf = int(abs(carrier + current_signal_frequency))

carrier_idx = m.find_nearest(frequencies,acoustic_frequency)
df_idx = m.find_nearest(frequencies,df)
sf_idx = m.find_nearest(frequencies,sf)
# 
# prf_idx = m.find_nearest(frequencies,prf)
sig_idx = m.find_nearest(frequencies,current_signal_frequency)
# prf_df_idx = m.find_nearest(frequencies,(current_signal_frequency+prf) )
# # 
print ('df/sf/carrier/sig:',frequencies[df_idx],frequencies[sf_idx],frequencies[carrier_idx],frequencies[sig_idx])
print ('df and sf/carrier/sig:',2*fft_m[df_idx],2*fft_m[sf_idx],2*fft_m[carrier_idx],2*fft_m[sig_idx])
# 
# print ('df and sf:',2*fft_m[prf_idx],2*fft_m[sig_idx],2*(fft_m[prf_df_idx] ))
# print ('ratio orig/df:',fft_m[sig_idx]/(fft_m[prf_df_idx] ))
# 
# 
# vdata = data[v_channel]
# idata = 0.1*data[i_channel] # 100mA per V i.e. 100*data[i_channel] gives i in mA. 
# impedance = np.max(vdata)/np.max(idata)
# print ('v,i(ma),z:',np.max(vdata),np.max(1000*idata),impedance)

plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2

band = 40
yheight = 1.0
# 
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(311)
plt.plot(t,data[m_channel],'r')
plt.plot(t,lfp_signal/np.max(lfp_signal),'k')
plt.plot(t,-1+marker_signal/np.max(marker_signal),'grey')
ax2 = fig.add_subplot(312)
plt.axvline(x=frequencies[df_idx],color='r')
plt.axvline(x=frequencies[sf_idx],color='r')
plt.plot(frequencies,fft_m,'k')
ax2.set_xlim([carrier - current_signal_frequency - band,carrier + current_signal_frequency + band])
# ax2.set_xlim([carrier - current_signal_frequency - 100,carrier - current_signal_frequency + 100])
# ax2.set_xlim([0,1040])
ax2.set_ylim([0,yheight])
# ax2.set_xlim([0,1e6])
ax3 = fig.add_subplot(313)
# plt.plot(t,10*data[rf_channel],'k')
plt.plot(frequencies,fft_m,'k')
ax3.set_xlim([0,40])
ax3.set_ylim([0,40])
plt.show()

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(211)
plt.axvline(x=frequencies[df_idx],color='r')
plt.axvline(x=frequencies[sf_idx],color='r')
plt.plot(frequencies,fft_m,'k')
ax.set_xlim([carrier - current_signal_frequency - band,carrier + current_signal_frequency + band])
ax.set_ylim([0,yheight])
ax2 = fig.add_subplot(212)
plt.plot(t,lfp_signal,'k')
plt.plot(t,-10+marker_signal/np.max(marker_signal),'grey')
plt.tight_layout()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = savepath + '\\'+'duration.png'
plt.savefig(plot_filename)
plt.show()

