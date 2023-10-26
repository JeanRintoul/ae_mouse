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
test_no = 18
gain    = 1000
# 
# Fs = current_frequency*10 
# Fs = 5e6
# carrier = 1e6*2
carrier = 5e5 # *2
dfx 	= 10
# 
measurement_channel = 0
prf                 = 1020
# 
aeti_variables = {
'type':'demodulation',        # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
# 'Fs': 5e6,                    # 
'Fs': 5e6,
# 'Fs': Fs,                   # 
# 'Fs': 1e6,                  # this will chang e just the sample rate, and not the f generated rate if those two lines are left uncommented. 
'USMEP': 1,                   # when usmep == 1, the current and pressure Fs can be different to the recorded Fs. In this case the US is at 5Mhz, but the recording frequency is 100kHz. This decreases the amount of data that needs to be dumped to disk. 
'duration': 6.0, 
'position': test_no,
'pressure_amplitude': 0.6,      # how much is lost through skull??? 400kPa, 0.08 is about 200kPz. 0.15 is about 400kPa. 
'pressure_frequency': carrier,
'pi_frequency': carrier + dfx,
'pressure_prf':0,            # pulse repetition frequency for the sine wave. Hz. 
# 'pressure_burst_length':0.3, 
# 'pressure_ISI':0,
# 
# Test 1: pulsed e stim. 
# 'pressure_ISI':0,
# 'pressure_prf':prf,            # pulse repetition frequency for the sine wave. Hz. 
# # 'pressure_burst_length':0.03,  # in seconds(maxes out at 50% duty cycle). pressure burst length is calculated as: prf_counter/gen_pressure_sample_frequency
# 'pressure_burst_length':0.05,  # in seconds(maxes out at 50% duty cycle). pressure burst length is calculated as: prf_counter/gen_pressure_sample_frequency
# i.e. x/1e7 = 0.003s . 3 cycles so 0.003 seconds... 3 milliseconds
# 
# Test 2 and 3: sine wave. Change carrier to 3Hz, Use preamp, and high pass filter to try to see spikes. 
# 'pressure_frequency': carrier,
# 
# Test 4: RF TI at 2Mhz. Filter on preamp. Consider using RF Amplifier? 
# 'pi_frequency':carrier + dfx,
# 
'current_amplitude': 0,       # its actually a voltage .. Volts. 
# 'current_frequency': 8000,     # 
# 'current_frequency': current_frequency, # 
'current_frequency': carrier,  # 
# 'current_ISI':0,
# 'current_burst_length':0.0004,
# 'current_prf':1020,
# 'ti_frequency': carrier + dfx,  # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 0,                # the channel of the measurement probe. 
'rf_monitor_channel': 4,        # this output of the rf amplifier.  
# 'e_channel': 6,                 # this is the voltage measured between the stimulator probes. 
# 'current_monitor_channel': 5,   # this is the current measurement channel of the transformer. 
'e_channel': 6,                 # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5,   # this is the current measurement channel of the transformer. 
'marker_channel':7,
'end_null': 0.1,                # start of end null. 
'end_pause': 0.8,               # start of end ramp
'start_null': 0.2,              # percent of file set to zero at the beginning. 
'start_pause': 0.3,             # percent of file in ramp mode or null at start.
'no_ramp':0.0,                  # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                    # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':1,             # the current and voltage monitor both have attenuators on them 
'command_c':'code\\mouse_stream',
'save_folder_path':'D:\\ae_mouse\\e118_ionic_mixing_test',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
#  Do a recording and copy it into the experiment folder. 
result, data_out            = m.aeti_recording(**aeti_variables)
data                        = m.copy_to_folder_and_return_data(**aeti_variables)

# #  
pressure_amplitude = aeti_variables['pressure_amplitude']
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

# this is if I use the preamp. 
fsignal 		      = 1e6*data[m_channel]/gain
# this is if I use the daq differntial input. 
# fsignal               = data[measurement_channel]
rfsignal                = 10*data[rf_channel]

lfp_data     	      = sosfiltfilt(sos_lfp_band, fsignal)
spike_data            = sosfiltfilt(sos_spike_band, fsignal)
# 
fft_m = fft(fsignal[start_pause:end_pause])
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
# 
fft_rf = fft(rfsignal[start_pause:end_pause])
fft_rf = np.abs(2.0/(end_pause-start_pause) * (fft_rf))[1:(end_pause-start_pause)//2]

xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 
df = int(abs(carrier - dfx))
sf = int(abs(carrier + dfx))
carrier_idx = m.find_nearest(frequencies,acoustic_frequency)
df_idx = m.find_nearest(frequencies,10)
sf_idx = m.find_nearest(frequencies,sf)
# 
# print ('Amplitude at df and sf:',2*fft_v[df_idx],2*fft_v[sf_idx])
# print ('df and sf:',2*fft_m[df_idx],2*fft_m[sf_idx])
print ('pressure vin:',pressure_amplitude)
print ('df p-p:',2*fft_m[df_idx])
print ('rf p-p:',2*fft_rf[carrier_idx], np.max(rfsignal[start_pause:end_pause]) )
# 
# 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(411)
plt.plot(t,rfsignal,'r')
plt.legend(['rf monitor signal'],loc='upper right')
ax2 = fig.add_subplot(412)
plt.plot(frequencies,fft_rf,'r')
plt.legend(['rf fft'],loc='upper right')
ax3 = fig.add_subplot(413)
plt.plot(t,data[measurement_channel],'k')
plt.legend(['measurement chan'],loc='upper right')
ax4 = fig.add_subplot(414)
plt.plot(frequencies,fft_m,'k')
ax4.set_xlim([0,110])
plt.legend(['measurement fft'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
plot_filename = savepath + '\\t'+str(test_no)+'_rfti_rfamplifier.png'
plt.savefig(plot_filename)
plt.show()
# 
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_rf,'r')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.show()
# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(t,data[measurement_channel],'k')
# ax2 = fig.add_subplot(312)
# plt.plot(t,lfp_data,'k')
# plt.plot(t,10*data[measurement_channel]/np.max(data[measurement_channel]),'r')
# plt.legend(['lfp'],loc='upper right')
# ax3 = fig.add_subplot(313)
# plt.plot(t,spike_data,'k')
# plt.plot(t,10*data[measurement_channel]/np.max(data[measurement_channel]),'r')
# plt.legend(['spike'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plot_filename = savepath + '\\t'+str(test_no)+'_estim.png'
# plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(411)
# plt.plot(t,data[m_channel],'k')
# ax2 = fig.add_subplot(412)
# plt.plot(t,data[v_channel],'k')
# ax3 = fig.add_subplot(413)
# plt.plot(t,data[i_channel],'k')
# ax4 = fig.add_subplot(414)
# # plt.plot(t,data[rf_channel],'k')
# # plt.plot(frequencies,fft_us,'k')
# plt.show()
# # 
# print ('carrier:',2*fft_m[carrier_idx])
# print ('rf antenna f:',2*fft_m[v_idx])
# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(411)
# plt.plot(frequencies,fft_v,'r')
# ax.set_ylabel('Volts(V)')
# plt.legend(['voltage output monitor'],loc='upper right')

# ax2 = fig.add_subplot(412)
# plt.plot(frequencies,fft_m,'k')
# ax2.set_ylabel('Volts ($\mu$V)')
# ax2.set_xlim([carrier-20000,carrier+20000])
# ax2.set_ylim([0,200])
# # ax2.set_xlim([400000,600000])
# ax2.set_xlabel('Frequencies(Hz)')
# plt.legend(['measurement in electrolyte'],loc='upper right')

# ax3 = fig.add_subplot(413)
# plt.plot(frequencies,fft_m,'k')
# ax3.set_xlim([0,dfx*1.5])
# # ax3.set_ylim([0,200])
# plt.legend(['dfx close up'],loc='upper right')

# ax4 = fig.add_subplot(414)
# plt.plot(frequencies,fft_m,'k')
# ax4.set_xlim([2*carrier,2*carrier+dfx*2])
# # ax4.set_ylim([0,200])
# plt.legend(['sfx close up'],loc='upper right')

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# ax4.spines['right'].set_visible(False)
# ax4.spines['top'].set_visible(False)

# plt.suptitle('RF transmitter/receiver test')
# # plot_filename = savepath + '\\rf_receiver_test.png'
# plot_filename = savepath + '\\t'+str(test_no)+'_rf.png'
# plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(t,data[v_channel],'r')

# ax2 = fig.add_subplot(212)
# plt.plot(t,lfp_data,'k')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = savepath + '\\lfp_filtered.png'
# plt.savefig(plot_filename)
# plt.show()

# First plot is the raw signal, 
# Second plot is the 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# # plt.plot(t,10*data[rf_channel],color='b')
# # plt.plot(t,10*data[v_channel],color='r')
# plt.plot(frequencies,fft_us,'b')
# plt.plot(frequencies,fft_v,'r')
# # plt.plot(frequencies,fft_m,'k')
# plt.legend(['fft us','fft v'],loc='upper right')
# ax.set_xlim([0,int(np.max(frequencies))/2])

# ax2 = fig.add_subplot(212)
# plt.axvline(df,color='k')
# plt.axvline(sf,color='k')
# # plt.plot(frequencies,fft_us,'b')

# plt.plot(frequencies,1e6*fft_v,'r')
# plt.plot(frequencies,fft_m,'c')
# # plt.plot(frequencies,fft_i,'orange')
# # ax2.set_ylim([0,500])
# ax2.set_xlim([acoustic_frequency - 2*current_signal_frequency,acoustic_frequency + 2*current_signal_frequency])
# # ax2.set_xlim([0,40])
# ax2.set_ylim([0,2000])
# plt.legend(['fft us','fft v'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plt.suptitle('AE location calibration')
# plot_filename = savepath + '\\t'+str(test_no)+'_datacheck.png'
# plt.savefig(plot_filename)
# plt.show()
# 
# 
# At low frequencies, I don't have enough repeats, and I have too much Johnson noise
# to see the difference frequency. I definitely cannot see it in the 
df = abs(current_signal_frequency - acoustic_frequency)
print ('df:', df)

# low  = df - 4
# if low < 0:
#     low = 0.1
# high = df+4
# sos_lowdf = iirfilter(17, [low,high], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# fsignal  = sosfiltfilt(sos_lowdf, 1e6*data[m_channel]/gain )
# fvsignal  = sosfiltfilt(sos_lowdf, data[v_channel] )
# # 
# subsection = fsignal[start_pause:end_pause]
# df_pp_height = np.max(subsection) -np.min(subsection)
# print ('df pp height:', df_pp_height)
# 
# plot of what comes through the preamp
# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(311)
# # plt.plot(t,10*data[rf_channel],'r')
# # plt.plot(t,data[v_channel],'k')
# plt.plot(10*data[rf_channel],'r')
# plt.plot(data[v_channel],'k')
# plt.legend(['rf chan','v chan'],loc='upper right')
# ax2  = fig.add_subplot(312)
# # plt.plot(t,fsignal,'k')
# plt.plot(fsignal,'k')
# plt.legend(['tight filtered df'],loc='upper right')
# ax3  = fig.add_subplot(313)
# plt.plot(frequencies,fft_m,'k')
# ax3.set_xlim([0,100])
# ax3.set_ylim([0,1000])
# plt.legend(['measurement channel fft'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plot_filename = savepath + '\\t'+str(test_no)+'low_frequency_df.png'
# plt.savefig(plot_filename)
# plt.show()


# 
# plot of the output currents and voltages. 
# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(311)
# plt.plot(t,data[v_channel],'r')
# plt.legend(['v monitor raw'],loc='upper right')
# ax2  = fig.add_subplot(312)
# plt.plot(t,i_data,'k')
# plt.legend(['i monitor'],loc='upper right')
# ax3  = fig.add_subplot(313)
# plt.plot(frequencies,fft_v,'r')
# plt.plot(frequencies,fft_i,'k')
# ax3.set_xlim([0,int(np.max(frequencies))/2])
# plt.legend(['voltage out fft(V)','current fft'],loc='upper right')
# # plot_filename = savepath + '\\t'+str(test_no)+'_v_i_vhannel.png'
# # plt.savefig(plot_filename)
# plt.show()




