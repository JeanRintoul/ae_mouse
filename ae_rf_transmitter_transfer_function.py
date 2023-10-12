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
test_no = 32
gain    = 100
# 
# This control of the sample rate is to avoid aliasing in the recorded electric signal. 
# The sample rate should be an integer sub-multiple of the 
# current_frequency = 499990
# 
# Fs = current_frequency*10 
# Fs = 5e6
dfx = 4 
# 
aeti_variables = {
'type':'demodulation',        # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 5e6,                    # 
# 'Fs': Fs,                   # 
# 'Fs': 1e6,                  # this will change just the sample rate, and not the f generated rate if those two lines are left uncommented. 
# 'USMEP': 1,                 # when usmep == 1, the current and pressure Fs can be different to the recorded Fs. In this case the US is at 5Mhz, but the recording frequency is 100kHz. This decreases the amount of data that needs to be dumped to disk. 
'duration': 4.0, 
'position': test_no,
'pressure_amplitude': 0.0,      # how much is lost through skull??? 400kPa, 0.08 is about 200kPz. 0.15 is about 400kPa. 
'pressure_frequency': 500000.0,
# 'pi_frequency': 500000 +dfx,
'current_amplitude': 10.0,      # its actually a voltage .. Volts. 
'current_frequency': 8000,     # 
# 'current_frequency': current_frequency, # 
# 'current_frequency': 499996,  # 
#'current_frequency': 500000,   # 
#'current_frequency': 500200,   # 
# 'current_frequency': 4,       # 
# 'current_amplitude': 6.0,     # 
# 'ti_frequency': 0,            # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 0,                # the channel of the measurement probe. 
'rf_monitor_channel': 4,        # this output of the rf amplifier.  
# 'e_channel': 6,                 # this is the voltage measured between the stimulator probes. 
# 'current_monitor_channel': 5,   # this is the current measurement channel of the transformer. 
'e_channel': 6,                 # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5,   # this is the current measurement channel of the transformer. 
'marker_channel':7,
'end_null': 0.0,               # start of end null. 
'end_pause': 1.0,             # start of end ramp
# 'start_null': 0.36,           # percent of file set to zero at the beginning. 
# 'start_pause': 0.4,           # percent of file in ramp mode or null at start.
# 'end_null': 0.0,               # start of end null. 
# 'end_pause': 1.0,             # start of end ramp
# 'start_null': 0.0,           # percent of file set to zero at the beginning. 
# 'start_pause': 0.0,           # percent of file in ramp mode or null at start.
'start_null': 0.05,            # percent of file set to zero at the beginning. 
'start_pause': 0.1,            # percent of file in ramp mode or null at start.
'no_ramp':0.0,                  # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                    # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':1,             # the current and voltage monitor both have attenuators on them 
'command_c':'code\\mouse_stream',
'save_folder_path':'D:\\ae_mouse\\e112_RF_antenna_TI',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
# 
# 
# 
frequencies_list = list(np.linspace(30,int(1e6),num=30))
# frequencies_list = [100000,500000,1000000] # frequencies for dummy test. 
print ('frequencies', frequencies_list)
# 
Fs       = aeti_variables['Fs'] 
duration = aeti_variables['duration'] 
timestep = 1.0/Fs
N = int(Fs*duration)
t               = np.linspace(0, duration, N, endpoint=False)
start_pause     = int(aeti_variables['start_pause'] * N+1)
end_pause       = int(aeti_variables['end_pause'] *N-1)
savepath = aeti_variables['save_folder_path']
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]

# 
data_response       = []
for i in range(len(frequencies_list)): 
  # f =  round(frequencies[i],2)
  f = round(frequencies_list[i],2)
  print ('f',f)
  aeti_variables['current_frequency'] = f
  print('current_frequency(Hz): ',f)  
  aeti_variables['position']  = f   # this is to label the file differently. 
  result, data_out            = m.aeti_recording(**aeti_variables)
  data                        = m.copy_to_folder_and_return_data(**aeti_variables)
  # get the rf channel of the data and save off the fft amplitude at the pressure frequency. 
  e_channel = aeti_variables['e_channel']
  m_channel = aeti_variables['ae_channel']
  fft_e = fft(data[e_channel][start_pause:end_pause])
  fft_e = np.abs(2.0/(end_pause-start_pause) * (fft_e))[1:(end_pause-start_pause)//2]
  carrier_idx = m.find_nearest(frequencies,f)
  amplitude_rf = fft_e[carrier_idx]
  #
  fft_m = 1e6*fft(data[m_channel][start_pause:end_pause])/gain
  fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
  #
  amplitude_measured = fft_m[carrier_idx]
  #
  print ('f/amplitude_measured:',f,amplitude_measured)
  data_response.append([f,amplitude_rf,amplitude_measured])

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #
d = np.array(data_response)
print ('d shape',d.shape,d)
d_f = d[:,0]
d_rf = d[:,1]
d_mf = d[:,2]
print ('d_f',d_f)

print ('d_rf',d_rf)

print ('d_mf',d_mf)

outfile                      = savepath + '\\single_wave.npz'
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,d=d)
print ('saved out a data file!')

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
plt.plot(d_f,d_rf,'-k.')
ax.set_ylabel('Volts (V)')
plt.legend(['rf v transmitter'],loc='upper right')
ax.set_xlim([0,1000010])
ax2 = fig.add_subplot(212)
plt.plot(d_f,d_mf,'-r.')
plt.legend(['measured carrier amplitude@electrode'],loc='upper right')
ax2.set_ylabel('Volts ($\mu$V)')
ax2.set_xlabel('Frequency(Hz)')
ax2.set_xlim([0,1000010])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.suptitle('RF versus measured carrier response amplitude')
plot_filename = savepath + '\\t'+str(test_no)+'_rf_response.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(d_f,d_rf/np.max(d_rf),'-k.')
plt.plot(d_f,d_mf/np.max(d_mf),'-r.')
ax.set_xlim([0,1000010])
ax.set_ylabel('Volts (V)')
plt.legend(['rf monitor','measured carrier amplitude@electrode'],loc='upper right')
ax2.set_ylabel('normalized')
ax2.set_xlabel('Frequency(Hz)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.suptitle('RF versus measured carrier response shape')
plot_filename = savepath + '\\t'+str(test_no)+'_rf_response_normalized.png'
plt.savefig(plot_filename)
plt.show()




# #  
# result, data_out            = m.aeti_recording(**aeti_variables)
# # print ('impedance:',data_out[0])
# data                        = m.copy_to_folder_and_return_data(**aeti_variables)

# # #  
# rf_channel = aeti_variables['rf_monitor_channel']
# marker_channel = aeti_variables['marker_channel']
# m_channel = aeti_variables['ae_channel'] 
# v_channel = aeti_variables['e_channel'] 
# i_channel = aeti_variables['current_monitor_channel'] 
# current_signal_frequency = aeti_variables['current_frequency'] 
# acoustic_frequency = aeti_variables['pressure_frequency']
# Fs       = aeti_variables['Fs'] 
# duration = aeti_variables['duration'] 
# savepath = aeti_variables['save_folder_path']
# #  
# timestep = 1.0/Fs
# N = int(Fs*duration)
# print("expected no. samples:",N)
# t               = np.linspace(0, duration, N, endpoint=False)
# start_pause     = int(aeti_variables['start_pause'] * N+1)
# end_pause       = int(aeti_variables['end_pause'] *N-1)

# # fake start and end due to timing mismatch. 
# start_pause     = int(0.25*N)
# end_pause       = int(0.75*N)



# # print ('start and end:',start_pause,end_pause)
# resistor_current_mon    = 49.9  # 49.9 Ohms for current monitor, 
# # 
# fft_m = 1e6*fft(data[m_channel][start_pause:end_pause])/gain
# fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
# # 
# xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# frequencies = xf[1:(end_pause-start_pause)//2]
# # 
# fft_us = fft(data[rf_channel][start_pause:end_pause])
# fft_us = np.abs(2.0/(end_pause-start_pause) * (fft_us))[1:(end_pause-start_pause)//2]
# # 
# # 
# fft_v = fft(data[v_channel][start_pause:end_pause])
# fft_v = np.abs(2.0/(end_pause-start_pause) * (fft_v))[1:(end_pause-start_pause)//2]
# # 
# # Bono TX. 
# # i_data = 5 * data[i_channel]/resistor_current_mon
# # my voltage mon
# i_data = data[i_channel]/resistor_current_mon
# fft_i = fft(i_data[start_pause:end_pause])
# fft_i = np.abs(2.0/(end_pause-start_pause) * (fft_i))[1:(end_pause-start_pause)//2]
# # 
# # 
# fft_fg = fft(data[3][start_pause:end_pause])
# fft_fg = np.abs(2.0/(end_pause-start_pause) * (fft_fg))[1:(end_pause-start_pause)//2]
# # 
# # 
# df = int(abs(acoustic_frequency - current_signal_frequency))
# sf = int(abs(acoustic_frequency + current_signal_frequency))
# # print ('frequencies of interest')
# #     resistor_current_mon    = 49.9  # 49.9 Ohms for current monitor, 
# # Now calculate the resistance and the reactance.   
# # i_data         = -5*data[i_channel]/resistor_current_mon
# #
# carrier_idx = m.find_nearest(frequencies,acoustic_frequency)
# df_idx = m.find_nearest(frequencies,df)
# sf_idx = m.find_nearest(frequencies,sf)
# v_idx = m.find_nearest(frequencies,current_signal_frequency) 
# # Note: this is going to be the average pp height. 
# i = fft_i[v_idx]
# z = fft_v[v_idx]/i
# print (i,fft_v[v_idx])

# print ('v_pp (V) :',np.round(fft_v[v_idx]*2,3)  )
# print ('i (ma) :',np.round(i*1000,3)  )
# print ('z (ohms) :',np.round(z,3))
# # 1.25
# # 426550
# # For this calibration, we are measuring the ae signal on the measurement electrode. 
# # i.e. on 
# print ('Amplitude at df and sf:',2*fft_v[df_idx],2*fft_v[sf_idx])
# now = datetime.now()

# current_time = now.strftime("%H:%M:%S")
# print("Current Time =", current_time)
# print ('df (\u03BCV):',1e6*fft_v[df_idx])
# print ('carrier (\u03BCV):',1e6*fft_v[carrier_idx])
# print ('sf (\u03BCV):',1e6*fft_v[sf_idx])
# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(t,data[rf_channel],'k')
# plt.show()

# # First plot is the raw signal, 
# # Second plot is the 
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
# # 
# # 
# # At low frequencies, I don't have enough repeats, and I have too much Johnson noise
# # to see the difference frequency. I definitely cannot see it in the 
# df = abs(current_signal_frequency - acoustic_frequency)
# print ('df:', df)

# # low  = df - 4
# # if low < 0:
# #     low = 0.1
# # high = df+4
# # sos_lowdf = iirfilter(17, [low,high], rs=60, btype='bandpass',
# #                        analog=False, ftype='cheby2', fs=Fs,
# #                        output='sos')
# # fsignal  = sosfiltfilt(sos_lowdf, 1e6*data[m_channel]/gain )
# # fvsignal  = sosfiltfilt(sos_lowdf, data[v_channel] )
# # # 
# # subsection = fsignal[start_pause:end_pause]
# # df_pp_height = np.max(subsection) -np.min(subsection)
# # print ('df pp height:', df_pp_height)
# # 
# # plot of what comes through the preamp
# # fig = plt.figure(figsize=(10,6))
# # ax  = fig.add_subplot(311)
# # # plt.plot(t,10*data[rf_channel],'r')
# # # plt.plot(t,data[v_channel],'k')
# # plt.plot(10*data[rf_channel],'r')
# # plt.plot(data[v_channel],'k')
# # plt.legend(['rf chan','v chan'],loc='upper right')
# # ax2  = fig.add_subplot(312)
# # # plt.plot(t,fsignal,'k')
# # plt.plot(fsignal,'k')
# # plt.legend(['tight filtered df'],loc='upper right')
# # ax3  = fig.add_subplot(313)
# # plt.plot(frequencies,fft_m,'k')
# # ax3.set_xlim([0,100])
# # ax3.set_ylim([0,1000])
# # plt.legend(['measurement channel fft'],loc='upper right')
# # ax.spines['right'].set_visible(False)
# # ax.spines['top'].set_visible(False)
# # ax2.spines['right'].set_visible(False)
# # ax2.spines['top'].set_visible(False)
# # ax3.spines['right'].set_visible(False)
# # ax3.spines['top'].set_visible(False)
# # plot_filename = savepath + '\\t'+str(test_no)+'low_frequency_df.png'
# # plt.savefig(plot_filename)
# # plt.show()


# # 
# # plot of the output currents and voltages. 
# # fig = plt.figure(figsize=(10,6))
# # ax  = fig.add_subplot(311)
# # plt.plot(t,data[v_channel],'r')
# # plt.legend(['v monitor raw'],loc='upper right')
# # ax2  = fig.add_subplot(312)
# # plt.plot(t,i_data,'k')
# # plt.legend(['i monitor'],loc='upper right')
# # ax3  = fig.add_subplot(313)
# # plt.plot(frequencies,fft_v,'r')
# # plt.plot(frequencies,fft_i,'k')
# # ax3.set_xlim([0,int(np.max(frequencies))/2])
# # plt.legend(['voltage out fft(V)','current fft'],loc='upper right')
# # # plot_filename = savepath + '\\t'+str(test_no)+'_v_i_vhannel.png'
# # # plt.savefig(plot_filename)
# # plt.show()




