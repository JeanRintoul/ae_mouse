'''

Title: vep signal inspection
Function: takes a single file, and averages the veps to see them better. 

Author: Jean Rintoul
Date: 23.10.2022

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift,ifft
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
import sys
sys.path.append('D:\\ae_mouse')  #  so that we can import from the parent folder. 
import mouse_library as m

# the 30 hz I see is coming from 1050Hz. 
carrier_f             = 500000
# carrier_f             = 501000
carrier_f               = 1000
# carrier_f             = 120
# 
file_number     =10 # 6-9 is 1Mhz. 
gain            = 500
savepath  = 'D:\\ae_mouse\\e104_mep_us_frequency_proofpoint\\t1\\'
Fs              = 5e6
duration        = 4.0
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 4 

def demodulate2(measured_signal,craw):
    rsignal             = sosfiltfilt(lp_filter, -measured_signal*craw) 
    # rsignal             = rsignal - np.mean(rsignal) 
    return rsignal

# 
# Use the inverse FFT to shift the band. 
# 
# 
filename    = savepath + 't'+str(file_number)+'_stream.npy'
data = np.load(filename)
a,b = data.shape
t = np.linspace(0, duration, N, endpoint=False)
# convert it to microvolts by taking the gain into account. 
fsignal = 1e6*data[m_channel]/gain
unfiltered_data = fsignal
rf_data = 10*data[rf_channel]
# 
# 
fft_unfiltered_data = fft(unfiltered_data)

xf              = np.fft.fftfreq( (N), d=timestep)[:(N)//2]
frequencies     = xf[1:(N)//2]
print ('fft bin size ', frequencies[2]-frequencies[1])
# index of frequency 1000. 
dfx         = 1000 
max_limit   = 10
df_idx  = m.find_nearest(frequencies,dfx)
df2_idx = m.find_nearest(frequencies,dfx+max_limit)
new_fft = [0]*fft_unfiltered_data
# modded_fft_data = fft_unfiltered_data[df_idx:df2_idx]
new_fft[0:(df2_idx- df_idx) ] = fft_unfiltered_data[df_idx:df2_idx]
result_signal = ifft(new_fft)

original_filtered_fft =[0]*fft_unfiltered_data
original_filtered_fft[0:(df2_idx- df_idx) ] = fft_unfiltered_data[0:(df2_idx- df_idx) ]
origin_signal = ifft(original_filtered_fft)

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(311)
plt.plot(t,result_signal,'k')
ax2 = fig.add_subplot(312)
plt.plot(t,origin_signal,'r')
ax3 = fig.add_subplot(313)
plt.plot(t,(result_signal -np.min(result_signal) )/(np.max(result_signal)-np.min(result_signal)) ,'k')
plt.plot(t,(origin_signal -np.min(origin_signal))/(np.max(origin_signal) - np.min(origin_signal) ),'r')
plt.show()


# def demodulate(measured_signal):
#     offset              = np.min(measured_signal)
#     offset_adjustment   = offset*np.cos(2 * np.pi * carrier_f * t)
#     IQ                  = (measured_signal+offset_adjustment)*np.exp(1j*(2*np.pi*carrier_f*t )) 
#     idown               = np.real(IQ)
#     qdown               = np.imag(IQ)
#     idown               = sosfiltfilt(lp_filter, idown)
#     qdown               = sosfiltfilt(lp_filter, qdown)  
#     rsignal             = -(idown + qdown)
#     rsignal             = rsignal - np.mean(rsignal) 
#     return rsignal
# # 
# c_band = iirfilter(17, [carrier_f-1,carrier_f+1], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')

# # 
# low                     = 1
# high                    = 20
# center_cut_offset       = 1 
# # 
# sos_low = iirfilter(17, [high], rs=60, btype='lowpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# # 
# # center cut band pass to remove carrier frequency. 
# sos_center_cut = iirfilter(17, [carrier_f-center_cut_offset,carrier_f+center_cut_offset], rs=60, btype='bandstop',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# sos_center_cut_band = iirfilter(17, [carrier_f-center_cut_offset,carrier_f+center_cut_offset], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# # 
# # signal of interest around the carrier. 
# l = carrier_f - 800
# h = carrier_f + 100000
# sos_carrier_band = iirfilter(17, [700], rs=60, btype='highpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# # low pass filter. 
# lp_filter               = iirfilter(17, [high], rs=60, btype='lowpass',
#                             analog=False, ftype='cheby2', fs=Fs,
#                             output='sos')
# # 
# # 
# bottom_cut_band = iirfilter(17, [carrier_f-center_cut_offset,carrier_f+center_cut_offset], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# # 
# # 

# # do the demodulation
# fraw            = sosfiltfilt(sos_carrier_band, fsignal) 
# craw            = sosfiltfilt(c_band, fsignal) 
# demod_data      = demodulate(fraw)

# demod_data2     = demodulate2(fsignal,fraw)
# # final low pass filter. 
# demodulated     = demod_data
# fsignal         = sosfiltfilt(sos_low, fsignal)
# # first carrier 
# demodulated2    = demod_data2 # sosfiltfilt(sos_low, demod_data)
# # second carrier. 
# # demodulated3    = demodulate2(fraw,craw)

# # find the fft of the data. 
# # TODO: this should be from the start to the end of the led flashing, and not including the start and end regions. 
# start_pause     = int(0.25 * N+1)
# end_pause       = int(0.75 * N-1)
# window          = np.hanning(end_pause-start_pause)

# fft_unfiltered_data = fft(unfiltered_data[start_pause:end_pause])
# fft_unfiltered_data = np.abs(2.0/(end_pause-start_pause) * (fft_unfiltered_data))[1:(end_pause-start_pause)//2]

# fft_data        = fft(fsignal[start_pause:end_pause])
# fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]

# fft_demod        = fft(demodulated[start_pause:end_pause])
# fft_demod        = np.abs(2.0/(end_pause-start_pause) * (fft_demod))[1:(end_pause-start_pause)//2]

# fft_demod2        = fft(demodulated2[start_pause:end_pause])
# fft_demod2        = np.abs(2.0/(end_pause-start_pause) * (fft_demod2))[1:(end_pause-start_pause)//2]


# fft_rfdata        = fft(rf_data[start_pause:end_pause])
# fft_rfdata        = np.abs(2.0/(end_pause-start_pause) * (fft_rfdata))[1:(end_pause-start_pause)//2]


# xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# frequencies     = xf[1:(end_pause-start_pause)//2]
# print ('fft bin size ', frequencies[2]-frequencies[1])
# # 

# fig = plt.figure(figsize=(6,6))
# ax1 = fig.add_subplot(311)
# plt.plot(t,unfiltered_data ,'k')
# ax2 = fig.add_subplot(312)
# plt.plot(t,craw ,'r')
# plt.legend(['carrier band','r'],loc='upper right')
# ax3 = fig.add_subplot(313)
# plt.plot(frequencies,fft_unfiltered_data,'k')
# plt.plot(frequencies,fft_rfdata,'r')
# plt.show()
# # 
# # 
# fig = plt.figure(figsize=(10,6))

# ax1 = fig.add_subplot(311)
# plt.plot(frequencies,fft_data,'k')
# ax1.set_xlim([0,high])
# plt.legend(['raw'])
# # ax1.set_ylim([0,100])

# ax2 = fig.add_subplot(312)
# plt.plot(frequencies,fft_demod,'r')
# plt.plot(frequencies,fft_demod2,'k')
# ax2.set_xlim([0,high])
# plt.legend(['demod'],loc='upper right')

# ax3 = fig.add_subplot(313)
# plt.plot(frequencies,fft_unfiltered_data,'k')
# ax3.set_xlim([carrier_f,carrier_f + high])
# plt.legend(['at carrier'],loc='upper right')
# # ax3.set_ylim([0,1])
# # # 
# # ax4 = fig.add_subplot(414)
# # plt.plot(frequencies,fft_demod,'r')
# # # ax4.set_xlim([carrier_f - high,carrier_f + high ])
# # plt.legend(['demod'])
# # ax4.set_ylim([0,0.0001])
# # 
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# # ax4.spines['right'].set_visible(False)
# # ax4.spines['top'].set_visible(False)
# plot_filename = 'FFT.png'
# plt.savefig(plot_filename)
# plt.show()
# # 
# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(t,(rf_data -np.min(rf_data))/(np.max(rf_data)-np.min(rf_data)) ,'grey')
# plt.plot(t,(fsignal-np.min(fsignal))/(np.max(fsignal) - np.min(fsignal)) ,'r')
# # plt.plot(t,(demodulated-np.min(demodulated))/(np.max(demodulated) - np.min(demodulated)),'k')
# plt.plot(t,(demodulated2-np.min(demodulated2))/(np.max(demodulated2) - np.min(demodulated2)),'m')

# ax.set_xlim([0,duration])
# # ax.set_ylim([-2,2])
# plt.legend(['rf data','lp lfp','demod','demod2'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.autoscale(enable=True, axis='x', tight=True)
# plot_filename = 'raw_and_demod_data.png'
# plt.savefig(plot_filename)
# plt.show()

