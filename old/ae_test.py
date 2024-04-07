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
# 
# 
test_no = 8
gain = 1000
# 
aeti_variables = {
'type':'demodulation',         # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 5e6,                  # 
'duration': 4.0, 
'position': test_no,
'pressure_amplitude': 0.1,  # how much is lost through skull??? 400kPa, 0.08 is about 200kPz. 0.15 is about 400kPa. 
'pressure_frequency': 500000.0,
'current_amplitude': 2.0,   #  its actually a voltage .. Volts. 
'current_frequency': 8000,  # 
# 'ti_frequency': 0,        # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 0,            # the channel of the measurement probe. 
'e_channel': 6,             # this is the voltage measured between the stimulator probes. 
'rf_monitor_channel': 4,    # this output of the rf amplifier. 
'current_monitor_channel': 5,  # this is the current measurement channel of the transformer. 
'start_null': 0.125,            # percent of file set to zero at the beginning. 
'start_pause': 0.25,            # percent of file in ramp mode or null at start.
'end_pause': 0.875,           # 
'end_null': 0.05,               # start of end null. 
'no_ramp':0.0,              # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':10,        # the current and voltage monitor both have attenuators on them 
'marker_channel':7,
'command_c':'code\\mouse_stream',
'save_folder_path':'D:\\mouse_aeti\\e110_F21_noise_reduction_characterization',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
#  
result, data_out            = m.aeti_recording(**aeti_variables)
# print ('impedance:',data_out[0])
data                        = m.copy_to_folder_and_return_data(**aeti_variables)
#  
rf_channel = aeti_variables['rf_monitor_channel']
marker_channel = aeti_variables['marker_channel']
m_channel = aeti_variables['ae_channel'] 
v_channel = aeti_variables['e_channel'] 
i_channel = aeti_variables['current_monitor_channel'] 
current_signal_frequency = aeti_variables['current_frequency'] 
Fs       = aeti_variables['Fs'] 
duration = aeti_variables['duration'] 
savepath = aeti_variables['save_folder_path']


p_f =   aeti_variables['pressure_frequency']


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
fft_us = fft(data[rf_channel][start_pause:end_pause])
fft_us = np.abs(2.0/(end_pause-start_pause) * (fft_us))[1:(end_pause-start_pause)//2]
# 
fft_v = fft(data[v_channel][start_pause:end_pause])
fft_v = np.abs(2.0/(end_pause-start_pause) * (fft_v))[1:(end_pause-start_pause)//2]
# 
fft_i = fft(data[i_channel][start_pause:end_pause])
fft_i = np.abs(2.0/(end_pause-start_pause) * (fft_i))[1:(end_pause-start_pause)//2]
# 

filter_cutoff           = 1000
lp_filter               = iirfilter(17, [filter_cutoff], rs=60, btype='lowpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')
filtered_d = sosfiltfilt(lp_filter, data[m_channel])

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(t,data[m_channel],color='orange')
plt.plot(t,filtered_d,color='k')
plot_filename = savepath + '\\t'+str(test_no)+'_datacheck.png'
plt.savefig(plot_filename)
plt.show()

# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# # plt.plot(t,10*data[rf_channel],color='b')
# plt.plot(t,data[m_channel],color='orange')
# ax2 = fig.add_subplot(312)
# # plt.plot(frequencies,fft_us,'b')
# # plt.plot(frequencies,fft_v,'r')
# # plt.plot(frequencies,fft_i,'b')
# plt.plot(frequencies,fft_m,'g')
# plt.axvline(x=p_f + current_signal_frequency)
# # ax2.set_xlim([0,800000])
# ax2.set_xlim([p_f - 2*current_signal_frequency,p_f + 2*current_signal_frequency])

# ax3 = fig.add_subplot(313)
# # plt.plot(frequencies,fft_i,'b')
# plt.plot(frequencies,10*fft_v,'r')
# plt.axvline(x=p_f + current_signal_frequency)
# plt.axvline(x=p_f)
# # ax3.set_xlim([0,800000])
# ax3.set_xlim([p_f - 2*current_signal_frequency,p_f + 2*current_signal_frequency])

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)

# plot_filename = savepath + '\\t'+str(test_no)+'_datacheck.png'
# plt.savefig(plot_filename)
# plt.show()



# 
# create the marker channel. 
# diffs = np.diff(data[marker_channel] )
# 
# zarray = t*[0]
# indexes = np.argwhere(diffs > 3.0)[:,0]
# dindexes = np.argwhere(diffs < -3.0)[:,0]
# # ind_list = dindexes - indexes
# print ('indexes:',indexes)
# if (len(dindexes) == len(indexes)):
#     ind_list = dindexes - indexes
# elif (len(dindexes)>len(indexes)):
#     ind_list = dindexes[0:len(indexes)]- indexes
#     print ('indexes length issue 2',dindexes, indexes)        
# else: 
#     ind_list = dindexes - indexes[0:len(dindexes)]
#     print ('indexes length issue',ind_list,dindexes,indexes)

# # print ('lengths list', ind_list,np.argwhere(ind_list < 1010)[0][0] )
# start_index = indexes[np.argwhere(ind_list < 1010)[0][0]]
# #  remove the start index so it is now just the led flashes. 
# indexes = np.delete(indexes, start_index)
# print ('len indexes', len(indexes),indexes)
# zarray[indexes] = 1.0


# # remove the mains noise. 
# fsignal = 1e6*data[m_channel]/gain
# mains_harmonics = [50,100,150,158,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950]
# for i in range(len(mains_harmonics)):
#     mains_low  = mains_harmonics[i] -2
#     mains_high = mains_harmonics[i] +2
#     mains_sos = iirfilter(17, [mains_low,mains_high], rs=60, btype='bandstop',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
#     fsignal = sosfiltfilt(mains_sos, fsignal)

# low  = 0.1
# high = 300
# sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# low_filtered_data  = sosfiltfilt(sos_low, fsignal)


# low  = 300
# high = 1000
# sos_high = iirfilter(17, [low,high], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# high_filtered_data  = sosfiltfilt(sos_high, fsignal)

# # This is the data to be included in the lfp averaging. 
# filtered_data  = low_filtered_data
# # segment range
# start = 0.1 
# end   = 0.7
# pre_event_idxcount  = int(start*Fs)
# post_event_idxcount = int(end*Fs)
# filtered_segmented_data        = []
# for i in range(len(indexes)):  # the last index may get chopped short. 
#   filtered_segmented_data.append(filtered_data[(indexes[i]-pre_event_idxcount):(indexes[i]+post_event_idxcount)    ])
# filtered_segmented_array = np.array(filtered_segmented_data)
# a,b = filtered_segmented_array.shape
# print ('shape', a,b)
# time_segment = np.linspace(-start,end,num=b)
# average_lfp = np.median(filtered_segmented_array,axis=0)
# std_lfp     = np.std(filtered_segmented_array,axis=0)
# print ('n_events: ', a)

# It would be useful to see the raw measured data, and it's fourier transform. 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(t, data[m_channel], color = 'purple')
# plt.plot(t,np.max(data[m_channel])*data[marker_channel]/np.max(data[marker_channel]),color='green')
# plt.plot(t, np.max(data[m_channel])*zarray, color = 'orange')
# plt.legend(['raw data','led on'],loc='upper left')
# ax.set_ylabel('Volts(V)')
# ax2 = fig.add_subplot(212)
# plt.plot(frequencies, fft_m, color = 'purple')
# ax2.set_xlim([0,100])
# ax2.set_xlabel('Frequency(Hz)')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = savepath + '\\t'+str(test_no)+'_raw_vep_data.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(time_segment,filtered_segmented_array.T)
# plt.axvline(x=0.0,color ='k')
# ax.text(0.5, 100, 'individual trials', color='black',fontsize = 6, ha='center')
# plt.autoscale(enable=True, axis='x', tight=True)
# ax.set_ylabel('Volts ($\mu$V)')
# # plot the average lfp. 
# ax2=fig.add_subplot(312)
# plt.plot(time_segment,average_lfp,color='r')
# plt.fill_between(time_segment, average_lfp - std_lfp, average_lfp + std_lfp,
#                   color='gray', alpha=0.2)
# plt.axvline(x=0.0,color ='k')
# ax2.text(0.5, 100, 'averaged LFP(0.1-300Hz)', fontsize = 6)
# ax2.set_xlabel('time(s)')
# ax2.set_ylabel('Volts ($\mu$V)')
# plt.autoscale(enable=True, axis='x', tight=True)

# ax3 = fig.add_subplot(313)
# plt.plot(frequencies,fft_m,color='b')
# plt.xlim([0,45])
# ax3.set_xlabel('frequency(Hz)')
# ax3.set_ylabel('Volts ($\mu$V)')

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)


# plt.suptitle('Visual Evoked Potential Inspection')
# plot_filename = savepath + '\\t'+str(test_no)+'_vep.png'
# plt.savefig(plot_filename)
# plt.show()


