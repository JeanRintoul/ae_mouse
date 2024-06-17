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
brain_gain          = 10
emg_gain            = 500 

Fs                  = 5e6
carrier             = 500000
measurement_channel = 0
# this is the variable to change. 
# compute the start and end points to be accurate. 
duration = 6
test_no  = 1

name_prefix 	= '111'
dfx 			= 0.5
cut 			= 1000
sos_low_band    = iirfilter(17, [cut], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

sos_emg_band    = iirfilter(17, [40,1000], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

sos_mains_stop    = iirfilter(17, [50-2,50+2], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

current_signal_frequency = carrier - dfx
print ('current_signal_frequencyr', current_signal_frequency)
start_time = np.round(0.8/duration ,2)
end_time   = np.round((duration - 0.2)/duration,2)
# 
start_time  = 0.2
end_time    = 0.8 
print ('start and end',start_time,end_time)
# 
# current_amplitude  = 12.0
# pressure_amplitude = 0.35
# 
# NOTE: using this RF amplifier, 116V is at vout = 0.4, vout = 0.35 is Vpp = 103. 
current_amplitude  = 10
pressure_amplitude = 0.15
# 
# current_amplitude  = 0.07
# pressure_amplitude = 0.25
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
# 'pressure_prf':prf,             	# pulse repetition frequency for the sine wave. Hz.
# 'pressure_ISI':0.0,             	# inter trial interval in seconds. 
# 'pressure_burst_length': 0.1,  	# burst length in seconds. 
'current_amplitude':current_amplitude,       
'current_frequency':current_signal_frequency,   # 
# 'current_ISI':0.0,  				# time in seconds between repeats. default length of time for pulse is 0.05s. see c code. 
# 'current_burst_length':0.05,    	# 50ms 
# 'current_prf':0, 
# 'marker_channel':4,
'ae_channel': 0,                # the channel of the measurement probe. 
'rf_monitor_channel': 4,        # this output of the rf amplifier.  
'marker_channel':7,
'e_channel': 6,                 # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5,   # this is the current measurement channel of the transformer. 
'end_null': 0.0,                # start of end null. 
'end_pause': end_time,          # start of end ramp
'start_null': 0.05,             # percent of file set to zero at the beginning. 
'start_pause': start_time,      # percent of file in ramp mode or null at start.
'no_ramp':0.0,                  # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':brain_gain,                    # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':1,             # the current and voltage monitor both have attenuators on them 
'command_c':'code\\mouse_stream',
'save_folder_path':'D:\\ae_mouse\\e149_meps\\17\\',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
# 
order = [1,2,3]
import random
random_order = random.sample(order, len(order))
print ('randomized loo test ordering: ',random_order)
# 
# random_order = [2,1,3]
# randomized order leave one out test. 
for i in range(len(order)):  
    num = random_order[i]
    print ('num:',num)
    if num == 1: # ae test
        aeti_variables['pressure_amplitude'] = pressure_amplitude
        aeti_variables['current_amplitude'] = current_amplitude
        aeti_variables['fileprefix'] = 'ae_g'+str(emg_gain)
    if num == 2: # p test
        aeti_variables['pressure_amplitude'] = pressure_amplitude
        aeti_variables['current_amplitude'] = 0
        aeti_variables['fileprefix'] = 'p_p' +str(pressure_amplitude)+'_g'+str(emg_gain)
    if num == 3: # v test 
        aeti_variables['pressure_amplitude'] = 0
        aeti_variables['current_amplitude'] = current_amplitude
        aeti_variables['fileprefix'] = 'v_v' +str(current_amplitude)+'_g'+str(emg_gain)
    result, data_out            = m.aeti_recording(**aeti_variables)
    data                        = m.copy_to_folder_and_return_data(**aeti_variables)
# 
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
emg_channel = 2 
# #  
timestep = 1.0/Fs
N = int(Fs*duration)
print("expected no. samples:",N)
t               = np.linspace(0, duration, N, endpoint=False)
start_pause     = int(aeti_variables['start_pause'] * N+1)
end_pause       = int(aeti_variables['end_pause'] *N-1)
# print ('start and end:',start_pause,end_pause)

# # this is if I use the preamp. 
fsignal 		      = 1e6*data[m_channel]/brain_gain
rfsignal              = 10*data[rf_channel]
emgsignal              = 1e6*data[emg_channel]/emg_gain
# # Calculate the current output, and the impedance between the stimulation electrodes. 
i_signal = -5*data[i_channel]/49.9 
v_signal = 10*data[v_channel]
I_pp = np.max(i_signal) - np.min(i_signal)
V_pp = np.max(v_signal) - np.min(v_signal)
RF_pp = np.max(rfsignal) - np.min(rfsignal)
print ('Vpp/RFpp: ', V_pp,RF_pp)
print ('Ipp: ', I_pp)
#
# 
# low_signal    = sosfiltfilt(sos_low_band, fsignal)
# emg_signal    = sosfiltfilt(sos_emg_band, emgsignal)
# emg_signal    = sosfiltfilt(sos_mains_stop, emg_signal)
# 
# 
# 
# fft_m = fft(fsignal[start_pause:end_pause])
# fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
# # 
# # fft_v = fft(v_signal[start_pause:end_pause])
# # fft_v = np.abs(2.0/(end_pause-start_pause) * (fft_v))[1:(end_pause-start_pause)//2]
# # 
# xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# frequencies = xf[1:(end_pause-start_pause)//2]
# df_idx = m.find_nearest(frequencies,dfx)
# #  
# st = int((start_time+1)*Fs)
# en = int((end_time-1)*Fs)
# # # 
# print ('df pp:',2*fft_m[df_idx])
# print ('pp height: ', np.max(low_signal[st:en])-np.min(low_signal[st:en]))
# # 
# fig = plt.figure(figsize=(8,7))
# ax = fig.add_subplot(211)
# plt.plot(t,v_signal,'k')
# ax2 = fig.add_subplot(212)
# plt.plot(frequencies,fft_v,'k')
# ax2.set_xlim([0,1e6])
# plt.show()


# plt.rc('font', family='serif')
# plt.rc('font', serif='Arial')
# plt.rcParams['axes.linewidth'] = 2
# # 
# # 
# # 
# fig = plt.figure(figsize=(8,7))
# ax = fig.add_subplot(511)
# plt.plot(t,v_signal,'k')
# plt.legend(['v signal'],loc='upper right', framealpha=0,labelcolor='white')
# ax.set_xlim([0,duration])
# ax2 = fig.add_subplot(512)
# plt.axvline(x=frequencies[df_idx],color='r')
# plt.plot(t,rfsignal,'r')
# plt.legend(['rf signal'],loc='upper right', framealpha=0)
# ax2.set_xlim([0,duration])

# ax3 = fig.add_subplot(513)
# plt.axvline(x=dfx,color='r')
# plt.plot(frequencies,fft_m,'k')
# plt.legend(['fft m'],loc='upper right', framealpha=0)
# ax3.set_xlim([0,55])

# ax4 = fig.add_subplot(514)
# plt.plot(t,low_signal,'r')
# plt.legend(['low f signal'],loc='upper right', framealpha=0)
# ax4.set_xlim([0,duration])

# ax5 = fig.add_subplot(515)
# plt.plot(t,emg_signal,'k')
# plt.legend(['emg'],loc='upper right', framealpha=0)
# ax5.set_xlim([0,duration])
# # plt.tight_layout()
# # plot_filename = savepath + '\\t'+str(test_no)+'_output_wave.png'
# # plt.savefig(plot_filename)
# plt.show()
# # # 