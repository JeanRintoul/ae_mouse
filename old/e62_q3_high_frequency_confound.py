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

# increment this for each test. 
# test_no = '\\1'
test_no = 1
# 
# Warren Grill suggested 1mA per 50ms
# 
aeti_variables = {
'type':'impedance',         # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 5e6,                  # 
'duration': 8.0, 
'position': test_no,
'pressure_amplitude': 0.0,
'pressure_frequency': 10000.0,
'current_amplitude': 1.0,   #  its actually a voltage .. Volts. 
'current_frequency': 500000,
'ti_frequency': 0,          # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 7,            # the channel of the measurement probe. 
'e_channel': 5,             # this is the voltage measured between the stimulator probes. 
'rf_monitor_channel': 1,    # this output of the rf amplifier. 
'current_monitor_channel': 3,  # this is the current measurement channel of the transformer. 
'start_pause': 0.5,         # percent of file in ramp mode. 
'end_pause': 0.5,           # 
'no_ramp':0.0,              # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':1,
'marker_channel':0,
'command_c':'mouse_stream',
'save_folder_path':'D:\mouse_aeti\e62_mouse_1\q3\\hfc_w_dceliminator',
}
#  
result, data_out            = m.aeti_recording(**aeti_variables)
print ('impedance:',data_out[0])
data,idx_lag,original_data  = m.align_data(**aeti_variables)

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
# start_pause     = int(aeti_variables['start_pause'] * N+1)
# end_pause       = int(aeti_variables['end_pause'] *N-1)

# since it is a continuous ramp, do it this way. 
start_pause = 100 
end_pause   = N-1
# print ('start and end:',start_pause,end_pause)
resistor_current_mon    = 49.9  # 49.9 Ohms for current monitor, 
i_data         = -5*data[i_channel]/resistor_current_mon
i_data         = 1000*i_data # convert to mA. 
v_data         = -10*data[v_channel]
m_data         = data[m_channel]

fft_m = fft(data[m_channel][start_pause:end_pause])
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]

xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]

print('len freqs',len(frequencies),len(fft_m))
indexes = np.argwhere(np.diff(data[marker_channel]) > 1)[:,0]
# print ('indexes',indexes.shape)
# 300 ms post event, -100 ms pre-event? 
start = 0.05 
end   = 0.25
pre_event_idxcount  = int(0.05*Fs)
post_event_idxcount = int(0.25*Fs)

# print ('idx example',indexes)
# 
# Do a filtfilt low pass the data < 1000 Hz? 
# Why is my data all Nans? 
channel_of_interest = m_channel
low_cut  = 2  # where do papers normally put this cut-off? 
high_cut = 300

sos = signal.iirfilter(17, [low_cut, high_cut], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
filtered_data = signal.sosfiltfilt(sos, data[m_channel,:])
# print ('filtered data:',filtered_data)

# here is a problem - why is it zero?
spike_cut_low   = 300 
spike_cut_high  = 1000
spike_sos = signal.iirfilter(17, [spike_cut_low, spike_cut_high], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
spike_filtered_data = signal.sosfiltfilt(spike_sos, data[m_channel,:])
# print ('data shape',spike_filtered_data.shape)
# print ('data:',len(spike_filtered_data),len(filtered_data))
# print ('spike data:',spike_filtered_data)

spike_filtered_segmented_data  = []
filtered_segmented_data        = []
segmented_data                 = []
for i in range(len(indexes)):  # the last index may get chopped short. 
  segmented_data.append(data[m_channel,(indexes[i]-pre_event_idxcount):(indexes[i]+post_event_idxcount)    ])

  filtered_segmented_data.append(filtered_data[(indexes[i]-pre_event_idxcount):(indexes[i]+post_event_idxcount)    ])
  spike_filtered_segmented_data.append(spike_filtered_data[(indexes[i]-pre_event_idxcount):(indexes[i]+post_event_idxcount)    ])


spike_filtered_segmented_array = np.array(spike_filtered_segmented_data)
filtered_segmented_array = np.array(filtered_segmented_data)
segmented_array = np.array(segmented_data)

# print (segmented_array[0,:])
# print ('spike data:',spike_filtered_data.shape)
print ('seg array shape:',segmented_array.shape)
a,b = segmented_array.shape
print ('b',a,b)
time_segment = np.linspace(-start,end,num=b)
# print ('time segment length: ',len(time_segment))
# 
# Downsample the data so it is more reasonable to plot. 
# 
# f = signal.resample(segmented_array, int(b/100),axis=1 )
# tnew = np.linspace(-0.1,0.3, int(b/100), endpoint=False)
# print ('length of downsampled signal', len(f),len(t))
# 

# print ('filterewd segmented array:',filtered_segmented_array.shape)
# print ('spike filt seg:',spike_filtered_segmented_array.shape)
# print ('data sample',filtered_segmented_array[0,0:50])

average_lfp = np.mean(filtered_segmented_array,axis=0)
std_lfp     = np.std(filtered_segmented_array,axis=0)
# print ('average lfp shape: ',average_lfp.shape)
average_spike = np.mean(spike_filtered_segmented_array,axis=0)
std_spike     = np.std(spike_filtered_segmented_array,axis=0)

print ('n_events: ', a)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
plt.plot(t, i_data, color = 'g')
plt.plot(t, v_data, color = 'b',linewidth=2)
plt.legend(['i','v'])

ax2 = fig.add_subplot(212)
plt.plot(t, m_data, color = 'purple')
plt.xlabel('time(s)')
plt.legend(['measured data'])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plot_filename = savepath + '\\'+str(test_no)+'_vim.png'
plt.savefig(plot_filename)
plt.show()


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(221)
plt.plot(t, v_data, color = 'b',linewidth=2)
plt.plot(t, i_data, color = 'g')
plt.plot(t, data[m_channel], color = 'r')
plt.plot(t, data[marker_channel],color='purple')
# plt.plot(t[1:], np.diff(data[marker_channel]),color='orange')
plt.legend(['V','I(mA)','measure','marker'],loc='upper right', fontsize = 6)

ax2=fig.add_subplot(222)
# plt.plot(frequencies,fft_m)
# plt.xlim([0,20000])
# plt.plot(time_segment,segmented_array[0,:])
plt.plot(time_segment,filtered_segmented_array.T)
plt.axvline(x=0.0,color ='k')
# ax2.text(0.1, 0.01, 'individual trials', color='black',fontsize = 6, ha='center')

# plot the average lfp. 
ax3=fig.add_subplot(223)
plt.plot(time_segment,average_lfp,color='r')
plt.fill_between(time_segment, average_lfp - std_lfp, average_lfp + std_lfp,
                  color='gray', alpha=0.2)
plt.axvline(x=0.0,color ='k')
# ax3.text(0.1, 0.01, 'averaged LFP(2-300Hz)', fontsize = 6)

ax4=fig.add_subplot(224)
plt.plot(time_segment,average_spike,color='r')
plt.fill_between(time_segment, average_spike - std_spike, average_spike + std_spike,
                  color='gray', alpha=0.2)
plt.axvline(x=0.0,color ='k')
# ax4.text(0.1, 0.01, 'spike LFP(300-1000Hz)', color='black', fontsize=6, ha='center')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)

plt.xlabel('time(s)')
plot_filename = savepath + '\\'+str(test_no)+'_lfp.png'
plt.savefig(plot_filename)
plt.show()


# Classic LFP plot. 
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# plt.plot(time_segment,average_lfp,color='r')
# plt.fill_between(time_segment, average_lfp - std_lfp, average_lfp + std_lfp,
#                   color='gray', alpha=0.2)
# plt.axvline(x=0.0,color ='k')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # plt.xlabel('time(s)')
# plot_filename = savepath + '\\'+str(test_no)+'_lfp_plot.png'
# plt.savefig(plot_filename)
# plt.show()


# Frequency plot
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
plt.plot(frequencies,fft_m,color='k')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([490000,501000])
# plt.xlabel('time(s)')
plot_filename = savepath + '\\'+str(test_no)+'_ti_frequency.png'
plt.savefig(plot_filename)
plt.show()