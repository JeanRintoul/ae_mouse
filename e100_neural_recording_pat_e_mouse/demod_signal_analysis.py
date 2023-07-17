'''

Title: vep signal inspection
Function: takes a single file, and averages the veps to see them better. 

Author: Jean Rintoul
Date: 23.10.2022

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
# 
# Convincing ones. 
# the first part of file 2 is AWESOME in terms of demodulation accuracy. 
# file 5. 
# file 7 has a little bit of good, then it goes bad at acoustic cut point. 
# file 10 has a little good. little
# file 11 little good. 
# 
# file 15 has a clear phase inversion. 
# 
# file 16 is awesome. 
# file 18 has good bits. 
# 19 is pretty awesome. 
# file 21 has awesome parts. 

# 3 appears to have phase inversion. 
# Each time there is a break in the RF data in amplitude, something weird happens to the demodulated signal. 
# file 5 is pretty good. 
# 6 is bad. 
# 
# We need to use a PRF to avoid reflections dominating the modulated data? 
# Can we have a larger signal at 1kHz? 
# 
# 
file_number     = 16
gain            = 1000
savepath        = 'D:\\mouse_aeti\\e100_neural_recording_pat_e_mouse\\t3_delta_wave_demodulation\\'
Fs              = 5e6
duration        = 8.0
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 4 

def demodulate(measured_signal):
    offset              = np.min(measured_signal)
    offset_adjustment   = offset*np.cos(2 * np.pi * carrier_f * t)
    IQ                  = (measured_signal+offset_adjustment)*np.exp(1j*(2*np.pi*carrier_f*t )) 
    idown               = np.real(IQ)
    qdown               = np.imag(IQ)
    idown               = sosfiltfilt(lp_filter, idown)
    qdown               = sosfiltfilt(lp_filter, qdown)  
    rsignal             = -(idown + qdown)
    rsignal             = rsignal - np.mean(rsignal) 
    return rsignal

carrier_f               = 500000
filter_cutoff           = 200


low  = 1
# high = 1000 
high = 16
sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')


sos_center_cut = iirfilter(17, [carrier_f-2,carrier_f+2], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

# define the carrier frequency. 
fc = carrier_f
l = fc - high
h = fc + high
sos_carrier_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

lp_filter               = iirfilter(17, [filter_cutoff], rs=60, btype='lowpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')

# led_duration = 1/(2*led_frequency) # when the led turned on and off. 
# start_time = 0.0
# end_time   = start_time+led_duration
# start = factor*led_duration   # this is where the trial starts seconds before marker. 
# end   = factor*led_duration*2   # this is where the trial ends

filename    = savepath + 't'+str(file_number)+'_stream.npy'
data = np.load(filename)
a,b = data.shape
t = np.linspace(0, duration, N, endpoint=False)
# convert it to microvolts by taking the gain into account. 
fsignal = 1e6*data[m_channel]/gain
unfiltered_data = fsignal
rf_data = 10*data[rf_channel]
# do the demodulation
fraw           = sosfiltfilt(sos_carrier_band, fsignal) 
fraw           = sosfiltfilt(sos_center_cut, fraw) 

demod_data     = demodulate(fraw)
# 
demodulated     = sosfiltfilt(sos_low, demod_data)
fsignal         = sosfiltfilt(sos_low, fsignal)

# find the fft of the data. 
# TODO: this should be from the start to the end of the led flashing, and not including the start and end regions. 
start_pause     = int(0.3 * N+1)
end_pause       = int(0.8 * N-1)
window          = np.hanning(end_pause-start_pause)

fft_unfiltered_data = fft(unfiltered_data[start_pause:end_pause]*window)
fft_unfiltered_data = np.abs(2.0/(end_pause-start_pause) * (fft_unfiltered_data))[1:(end_pause-start_pause)//2]


fft_data        = fft(fsignal[start_pause:end_pause]*window)
fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]

fft_demod        = fft(demodulated[start_pause:end_pause]*window)
fft_demod        = np.abs(2.0/(end_pause-start_pause) * (fft_demod))[1:(end_pause-start_pause)//2]


xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies     = xf[1:(end_pause-start_pause)//2]

print ('fft bin size ', frequencies[2]-frequencies[1])

# create the marker channel. 
# markerdata        = np.array(data[marker_channel])
# marker            = markerdata/np.max(markerdata)
# diffs             = np.diff( markerdata )
# zarray            = t*[0]
# indexes           = np.argwhere(diffs > 0.2)[:,0] # leading edge
# marker_up_length  = int(led_duration*Fs)
# for i in range(len(indexes)):
#     if i > 0:
#         zarray[indexes[i]:(indexes[i]+marker_up_length) ] = 1
# markerdata = zarray
# 
# pre_event_idxcount              = int(start*Fs)
# post_event_idxcount             = int(end*Fs)
# data_to_segment                 = fsignal
# filtered_segmented_data         = []
# for i in range(len(indexes)-1):  # the last index may get chopped short. 
#   filtered_segmented_data.append(data_to_segment[(indexes[i+1]-pre_event_idxcount):(indexes[i+1]+post_event_idxcount)])
# # 
# filtered_segmented_array = np.array(filtered_segmented_data)
# a,b = filtered_segmented_array.shape
# print ('shape', a,b)
# time_segment = np.linspace(-start,end,num=b)
# average_lfp = np.median(filtered_segmented_array,axis=0)
# std_lfp     = np.std(filtered_segmented_array,axis=0)
# print ('n_events: ', a)
# # 
# # 
# lfp_height = np.max(average_lfp)- np.min(average_lfp)
# print ('LFP size', lfp_height)
# 
# 
rf_data = rf_data/np.max(rf_data)
# 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(frequencies,fft_data,'r')
plt.plot(frequencies,fft_demod,'b')
plt.plot(frequencies,fft_unfiltered_data,'k')
ax.set_xlim([0,1e6])
plot_filename = 'FFT.png'
plt.savefig(plot_filename)
plt.show()
# 
# 
lim1 = int(0.5*Fs)
lim2 = int(7.5*Fs)
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
# plt.plot(t[lim1:lim2],0.5*marker[lim1:lim2],'r')
plt.plot(t[lim1:lim2],rf_data[lim1:lim2],'grey')
plt.plot(t[lim1:lim2],fsignal[lim1:lim2]/np.max(fsignal[lim1:lim2]),'r')
plt.plot(t[lim1:lim2],demodulated[lim1:lim2]/np.max(demodulated[lim1:lim2]),'k')
# ax.set_xlim([1.8])
# ax.set_ylim([-2,2])
plt.legend(['rf data','lp lfp','demod'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.autoscale(enable=True, axis='x', tight=True)
plot_filename = 'raw_and_demod_data.png'
plt.savefig(plot_filename)
plt.show()


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
# plt.plot(t[lim1:lim2],0.5*marker[lim1:lim2],'r')
plt.plot(t[lim1:lim2],100*rf_data[lim1:lim2],'grey')
plt.plot(t[lim1:lim2],fsignal[lim1:lim2],'r')
plt.plot(t[lim1:lim2],demodulated[lim1:lim2],'k')
# ax.set_xlim([lim1,lim2])
# ax.set_ylim([-2,2])
plt.legend(['rf data','lp lfp','demod'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.autoscale(enable=True, axis='x', tight=True)
plot_filename = 'raw_and_demod_data_unscaled.png'
plt.savefig(plot_filename)
plt.show()