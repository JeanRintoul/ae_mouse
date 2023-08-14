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
# carrier_f             = 500000
# carrier_f             = 501020
carrier_f               = 1020
# carrier_f             = 120
# 
file_number     = 6 # 6-9 is 1Mhz. 
gain            = 1000
savepath  = 'D:\\ae_mouse\\e105_rfae_meps\\t1\\'
Fs              = 5e6
duration        = 8.0
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 4 
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
start_pause                 = int(0 * N)
end_pause                   = int(4 * (N-1))
fft_unfiltered_data         = fft(unfiltered_data[start_pause:end_pause])
# fft_unfiltered_data       = np.abs(2.0/(end_pause-start_pause) * (fft_unfiltered_data))[1:(end_pause-start_pause)//2]
xf                          = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# frequencies                 = xf[1:(end_pause-start_pause)//2]
# frequencies                 = xf[1:(end_pause-start_pause)]
# 
xf              = np.fft.fftfreq( (N), d=timestep)[:(N)//2]
frequencies     = xf[1:(N)//2]


fft_seconds = np.flip(fft_unfiltered_data)

print ('fft bin size ', frequencies[2]-frequencies[1])
# index of frequency 1000. 
dfx         = carrier_f 
min_limit   = 5 # this is an index. 
max_limit   = 20 # this is a frequency
df_idx  = m.find_nearest(frequencies,dfx)
df2_idx = m.find_nearest(frequencies,dfx+max_limit)
df3_idx = m.find_nearest(frequencies,dfx-max_limit)

positive_fft = [0]*fft_unfiltered_data
positive_fft[min_limit:(df2_idx- df_idx) ] = fft_unfiltered_data[(df_idx+min_limit):df2_idx]
positive_result_signal = ifft(positive_fft)

negative_fft = [0]*fft_unfiltered_data
negative_fft[min_limit:(df2_idx - df_idx) ] = np.flip(fft_unfiltered_data[df3_idx:df_idx])[min_limit:]
negative_result_signal = ifft(negative_fft)

total_signal = positive_result_signal + negative_result_signal

positive_fft2 = [0]*fft_seconds
positive_fft2[min_limit:(df2_idx- df_idx) ] = fft_seconds[(df_idx+min_limit):df2_idx]
positive_result_signal2 = ifft(positive_fft2)

negative_fft2 = [0]*fft_seconds
negative_fft2[min_limit:(df2_idx - df_idx) ] = np.flip(fft_seconds[df3_idx:df_idx])[min_limit:]
negative_result_signal2 = ifft(negative_fft2)

total_signal2 = positive_result_signal2 + negative_result_signal2

original_filtered_fft =[0]*fft_unfiltered_data
original_filtered_fft[0:(df2_idx- df_idx) ] = fft_unfiltered_data[0:(df2_idx- df_idx) ]
origin_signal = ifft(original_filtered_fft)
# remove the DC part
original_filtered_fftx =[0]*fft_unfiltered_data
original_filtered_fftx[min_limit:(df2_idx- df_idx) ] = fft_unfiltered_data[min_limit:(df2_idx- df_idx) ]
origin_signalx = ifft(original_filtered_fftx)


print ('len frequencies,len fft data',len(frequencies),len(fft_unfiltered_data))
# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111)
# plt.plot(fft_unfiltered_data,'k')
# plt.show()

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(311)
plt.plot(t,-negative_result_signal,'k')
plt.plot(t,-positive_result_signal,'r')

ax2 = fig.add_subplot(312)
plt.plot(t,positive_result_signal2,'k')
plt.plot(t,negative_result_signal2,'r')
plt.plot(t,origin_signalx/100,'b') # filtered signal. 

ax2 = fig.add_subplot(313)
# plt.plot(t,origin_signal,'k')
# plt.plot(t,origin_signalx,'b') # filtered signal. 
plt.plot(t,-negative_result_signal+positive_result_signal2,'b') # filtered signal.
plt.plot(t,-positive_result_signal+negative_result_signal2,'g') # filtered signal.

# ax3 = fig.add_subplot(313)
# # plt.plot(t,total_signal,'m')
# # plt.plot(t,-total_signal2,'k')
# plt.plot(t,total_signal-total_signal2,'b')
# plt.plot(t,-(positive_result_signal -np.min(positive_result_signal) )/(np.max(positive_result_signal)-np.min(positive_result_signal)) ,'k')
# plt.plot(t,-1+(origin_signal -np.min(origin_signal))/(np.max(origin_signal) - np.min(origin_signal) ),'r')
plt.show()

