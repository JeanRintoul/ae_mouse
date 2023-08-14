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
# the 30 hz I see is coming from 1050Hz. 
carrier_f               = 1020
# 
file_number     = 4  # 6-9 is 1Mhz. 
gain            = 1000
savepath        = 'D:\\ae_mouse\\e105_rfae_meps\\t1\\'
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


filename    = savepath + 't'+str(file_number)+'_stream.npy'
data = np.load(filename)
a,b = data.shape
t = np.linspace(0, duration, N, endpoint=False)
# convert it to microvolts by taking the gain into account. 
fsignal = 1e6*data[m_channel]/gain
rf_data = 10*data[rf_channel]
v_data = data[v_channel]
# now filter the data to see the gamme band 30-100Hz,
low                     = 30
high                    = 100
# 
sos_gamma = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
gamma_lfp            = sosfiltfilt(sos_gamma, fsignal) 

sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
low  = 0.5
high = 20
sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
filtered_lfp            = sosfiltfilt(sos_low, fsignal) 


start_pause     = int(0.3 * N+1)
end_pause       = int(0.8 * N-1)
window          = np.hanning(end_pause-start_pause)

fft_unfiltered_data = fft(fsignal[start_pause:end_pause])
fft_unfiltered_data = np.abs(2.0/(end_pause-start_pause) * (fft_unfiltered_data))[1:(end_pause-start_pause)//2]
xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies     = xf[1:(end_pause-start_pause)//2]
print ('fft bin size ', frequencies[2]-frequencies[1])

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(311)
plt.plot(t,gamma_lfp ,'k')
ax2 = fig.add_subplot(312)
plt.plot(frequencies,fft_unfiltered_data,'k')
ax2.set_xlim([0,100])
ax2.set_ylim([0,5])
ax3 = fig.add_subplot(313)
plt.plot(t,filtered_lfp,'k')

plt.show()





# # do the demodulation
# # fraw            = sosfiltfilt(sos_carrier_band, fsignal) 
# # craw            = sosfiltfilt(c_band, fsignal) 
# # demod_data      = demodulate(fraw)
# # # final low pass filter. 
# # demodulated     = sosfiltfilt(sos_low, demod_data)
# # fsignal         = sosfiltfilt(sos_low, fsignal)






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
# c_band = iirfilter(17, [carrier_f-2,carrier_f+2], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')

# # 
# low                     = 1
# high                    = 10
# center_cut_offset       = 1 
# # 
# sos_low = iirfilter(17, [high], rs=60, btype='lowpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# # 
# # center cut band pass to remove carrier frequency. 
# # sos_center_cut = iirfilter(17, [carrier_f-center_cut_offset,carrier_f+center_cut_offset], rs=60, btype='bandstop',
# #                        analog=False, ftype='cheby2', fs=Fs,
# #                        output='sos')
# # 
# # signal of interest around the carrier. 
# l = carrier_f - high
# h = carrier_f + high
# sos_carrier_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# # low pass filter. 
# lp_filter               = iirfilter(17, [high], rs=60, btype='lowpass',
#                             analog=False, ftype='cheby2', fs=Fs,
#                             output='sos')


# # find the fft of the data. 
# # TODO: this should be from the start to the end of the led flashing, and not including the start and end regions. 
# start_pause     = int(0.3 * N+1)
# end_pause       = int(0.8 * N-1)
# window          = np.hanning(end_pause-start_pause)

# fft_unfiltered_data = fft(unfiltered_data[start_pause:end_pause])
# fft_unfiltered_data = np.abs(2.0/(end_pause-start_pause) * (fft_unfiltered_data))[1:(end_pause-start_pause)//2]

# fft_data        = fft(fsignal[start_pause:end_pause])
# fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]



# fft_rfdata        = fft(rf_data[start_pause:end_pause])
# fft_rfdata        = np.abs(2.0/(end_pause-start_pause) * (fft_rfdata))[1:(end_pause-start_pause)//2]


# xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# frequencies     = xf[1:(end_pause-start_pause)//2]
# # 
# print ('fft bin size ', frequencies[2]-frequencies[1])
# # 
# # 
# fig = plt.figure(figsize=(6,6))
# ax1 = fig.add_subplot(311)
# plt.plot(t,unfiltered_data ,'k')
# ax2 = fig.add_subplot(312)
# plt.plot(t,craw ,'r')
# plt.plot(t,5*(fsignal-np.min(fsignal))/(np.max(fsignal) - np.min(fsignal)) ,'grey')
# plt.plot(t,5*(1-(demodulated-np.min(demodulated))/(np.max(demodulated) - np.min(demodulated))),'k')
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
# ax2.set_xlim([0,high])
# plt.legend(['demod'])

# ax3 = fig.add_subplot(313)
# plt.plot(frequencies,fft_unfiltered_data,'k')
# ax3.set_xlim([carrier_f,carrier_f + high])
# plt.legend(['at carrier'])
# ax3.set_ylim([0,1])
# # 
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
# # plt.plot(t,(v_data -np.min(v_data))/(np.max(v_data)-np.min(v_data)) ,'pink')

# plt.plot(t,(fsignal-np.min(fsignal))/(np.max(fsignal) - np.min(fsignal)) ,'r')
# plt.plot(t,1-(demodulated-np.min(demodulated))/(np.max(demodulated) - np.min(demodulated)),'k')
# ax.set_xlim([0,duration])
# # ax.set_ylim([-2,2])
# plt.legend(['rf data','lp lfp','demod'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.autoscale(enable=True, axis='x', tight=True)
# plot_filename = 'raw_and_demod_data.png'
# plt.savefig(plot_filename)
# plt.show()

