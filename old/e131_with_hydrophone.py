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
from scipy.signal import iirfilter,sosfiltfilt
# 
# increment this for each test. 
# 
sensitivity = 0.033    #  hydrophone sensitivity is 33mV/MPa at 500kHz


test_no = 1
gain    = 100
# 
dfx 	= 1000
carrier = 500000 
hydrophone_channel = 1

aeti_variables = {
'type':'demodulation',      # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 5e6,                  # 
'duration': 4.0, 
'position': test_no,
'pressure_amplitude': 0.1,  # how much is lost through skull??? 400kPa, 0.08 is about 200kPz. 0.15 is about 400kPa. 
'pressure_frequency': carrier,
# 'pi_frequency': carrier + dfx,
# 'pressure_prf':1020,      # pulse repetition frequency for the sine wave. Hz.
# 'pressure_ISI':0.0,       # inter trial interval in seconds. 
# 'pressure_burst_length':0.05,  # burst length in seconds. 
'current_amplitude': 0,     # its actually a voltage .. Volts. 
'current_frequency': carrier,  # 
'current_ISI':1.0,  # time in seconds between repeats. default length of time for pulse is 0.05s. see c code. 
'current_burst_length':0.05,    # 50ms 
'current_prf':0,
#'ti_frequency': 0,         # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 0,            # the channel of the measurement probe. 
'rf_monitor_channel': 4,    # this output of the rf amplifier.  
'e_channel': 6,             # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5, # this is the current measurement channel of the transformer. 
'marker_channel':7,
'end_null': 0.02,           # start of end null. 
'end_pause': 0.875,         # start of end ramp
'start_null': 0.125,        # percent of file set to zero at the beginning. 
'start_pause': 0.25,        # percent of file in ramp mode or null at start. 
'no_ramp':0.0,              # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':10,        # the current and voltage monitor both have attenuators on them 
'command_c':'code\\mouse_stream',
# 'command_c':'code\\mouse_4stream',
'save_folder_path':'D:\\ae_mouse\\e131_postDAQdisaster',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
#  
result, data_out            = m.aeti_recording(**aeti_variables)
data                        = m.copy_to_folder_and_return_data(**aeti_variables)

print ('data shape', data.shape)
#  
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
# prf      = aeti_variables['pressure_prf']
prf = 1020
#  
timestep = 1.0/Fs
N = int(Fs*duration)
# print("expected no. samples:",N)
t               = np.linspace(0, duration, N, endpoint=False)
start_pause     = int(aeti_variables['start_pause'] * N+1)
end_pause       = int(aeti_variables['end_pause'] *N-1)

# start_pause     = int(1*Fs)
# end_pause       = int(3*Fs)

# print ('start and end:',start_pause,end_pause)
resistor_current_mon    = 49.9  # 49.9 Ohms for current monitor, 


pressure        = np.array(data[hydrophone_channel])*1000/sensitivity  # KPa
pressure_pp     = np.max(pressure)
print ('pressure(kPa)',pressure_pp*2)

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(t,data[hydrophone_channel],'k')
# plt.legend(['raw hydrophone input'],loc='upper right')
# ax2 = fig.add_subplot(312)
# plt.plot(t,pressure,'k')
# plt.legend(['pressure(kPa)'],loc='upper right')
# ax3 = fig.add_subplot(313)
# plt.plot(t,data[rf_channel],'r')
# plt.legend(['RF chan'],loc='upper right')
# plot_filename = savepath + '\\t'+str(test_no)+'_hydrophone.png'
# plt.savefig(plot_filename)
# plt.show()

fft_p = fft(data[hydrophone_channel][start_pause:end_pause])
fft_p = np.abs(2.0/(end_pause-start_pause) * (fft_p))[1:(end_pause-start_pause)//2]
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
# fft_v = fft(10*data[v_channel][start_pause:end_pause])
# fft_v = np.abs(2.0/(end_pause-start_pause) * (fft_v))[1:(end_pause-start_pause)//2]
# 

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
plt.plot(t,data[7],'k')
plt.legend(['raw hydrophone input'],loc='upper right')
ax2 = fig.add_subplot(212)
plt.plot(frequencies,fft_p,'k')
plt.legend(['fft pressure'],loc='upper right')
plot_filename = savepath + '\\t'+str(test_no)+'_ffthydrophone.png'
plt.savefig(plot_filename)
plt.show()




# 
# 
# df = int(acoustic_frequency - current_signal_frequency)
# sf = int(acoustic_frequency + current_signal_frequency)
# print ('frequencies of interest')
# # 
# #
# df_idx = m.find_nearest(frequencies,df)
# sf_idx = m.find_nearest(frequencies,sf)
# # 
# # 
# # For this calibration, we are measuring the ae signal on the measurement electrode. 
# # i.e. on 
# # print ('Amplitude at df and sf:',2*fft_v[df_idx],2*fft_v[sf_idx])
# # For this calibration, report the amplitude at the PRF. 
# prf_idx = m.find_nearest(frequencies,prf)
# print ('size of signal at PRF:(microvolts)',2*fft_m[prf_idx])

# carrier_idx = m.find_nearest(frequencies,acoustic_frequency)
# print ('size of signal at carrier:(microvolts)',2*fft_m[carrier_idx])

# prf_sum_idx = m.find_nearest(frequencies,prf+acoustic_frequency*2)
# print ('size of signal at sum PRF:(microvolts)',2*fft_m[prf_sum_idx])
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(t,data[marker_channel],'b')
# plt.show()
# 
# 
# fft_usm = fft(data[1][start_pause:end_pause])
# fft_usm = np.abs(2.0/(end_pause-start_pause) * (fft_usm))[1:(end_pause-start_pause)//2]
# Plot the FFT of the measurement signal, clearly showing the 1000 PRF. 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# # plt.plot(t,data[1],color='b')
# plt.plot(t,data[rf_channel],color='b')
# # check the PRF in this data by putting two vertical lines to show a period. 
# vline_1 = int(Fs*duration/2)
# vline_2 = vline_1 + int(Fs/prf)
# print ('vline 1 and 2:', t[vline_1],t[vline_2] ) 
# plt.axvline(t[vline_1],color='k')
# plt.axvline(t[vline_2],color='k')
# # 
# # plt.legend(['time series US'])
# ax2 = fig.add_subplot(312)
# # plt.plot(frequencies,fft_usm,color='b')
# plt.plot(frequencies, fft_us,color='b')
# plt.legend(['fft US'])
# ax2.set_xlim(0,800000)
# ax3 = fig.add_subplot(313)
# plt.plot(frequencies,fft_m,color='r')
# # ax3.set_xlim(0,2000)
# # ax3.set_ylim(0,20)
# plt.legend(['measurement data'])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# # plt.suptitle('AE location calibration with PRF')
# plot_filename = savepath + '\\t'+str(test_no)+'_datacheck.png'
# plt.savefig(plot_filename)
# plt.show()
# 
# Plot the FFT of the measurement signal, clearly showing the 1000 PRF. 
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_m,'b')
# ax.set_xlim(0,2000)
# plt.show()
# 
# 
# First plot is the raw signal, 
# Second plot is the 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# # plt.plot(t,10*data[rf_channel],color='b')
# # plt.plot(t,10*data[v_channel],color='r')
# plt.plot(frequencies,fft_us,'b')
# plt.plot(frequencies,fft_v,'r')
# ax.set_xlim([0,800000])

# ax2 = fig.add_subplot(212)
# plt.plot(frequencies,fft_us,'b')
# plt.plot(frequencies,fft_v,'r')
# plt.axvline(df,color='k')
# plt.axvline(sf,color='k')
# ax2.set_xlim([acoustic_frequency - 2*current_signal_frequency,acoustic_frequency + 2*current_signal_frequency])

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plt.suptitle('AE location calibration')
# plot_filename = savepath + '\\t'+str(test_no)+'_datacheck.png'
# plt.savefig(plot_filename)
# plt.show()



