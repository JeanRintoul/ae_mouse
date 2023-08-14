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
from scipy.signal import hilbert
# 
# 
# in the mouse yesterday. 

# savepath       = 'D:\\ae_mouse\\e105_rfae_meps\\t6_phantom_demodulation_amplitudes\\'
# file_number    = 4
# Fs             = 5e6
# duration       = 8.0	
# savepath       = 'D:\\ae_mouse\\e105_rfae_meps\\t2_phantom_detangling\\demodulation_challenge\\'
# file_number    = 5
# Fs              = 5e3
# duration        = 12.0	
# 
savepath        = 'D:\\ae_mouse\\e105_rfae_meps\\'
file_number     = 1
Fs              = 5e4
duration        = 10.0	

gain            = 1000
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
fsignal = fsignal-np.mean(fsignal)
# downsample the signal
# new_Fs                        = 1000 
# downsampling_factor           = int(Fs/new_Fs)
# high = new_Fs 
# low_filter = iirfilter(17, [high], rs=60, btype='lowpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# s     	= sosfiltfilt(low_filter, fsignal)
# dsignal = s[::downsampling_factor]
# t 		= t[::downsampling_factor]
# no downsampling
new_Fs 			= Fs
dsignal 		= fsignal

newN = int(len(t))
def demodulate(in_signal,carrier_f): 
    idown = in_signal*np.cos(2*np.pi*carrier_f*t)
    qdown = in_signal*np.sin(2*np.pi*carrier_f*t)   
    demodulated_signal = idown + 1j*qdown
    return np.abs(demodulated_signal)
# 
# 
# prf_list = [80,80*2,80*3]
# # prf_list = [80,160,240,320]
# prf_list = [180,180*2,500000*2+180,500000*2-180]
prf_list = [80]
# prf_list = [1030,1030*2,1e6+1030]
# prf_list = [1e6+1030]
# prf_list = [1030,1e6]
# prf_list = [1030]
# prf_list = [180,180*2,500000 - 180,500000 + 180,500000]
# prf_list = [180,180*2, 500000*2+180,500000*2+180*2  ]
# prf_list = [180,180*2, 500000*2+180,500000*2+180*2  ]

# prf_list = [180,180*2,180*3,180*4]
# 
# prf             	= prf_list[0]
# prf             	= 500000 - 180
# prf             	= 500000*2 + 180
# 
signal_of_interest 	= 26 
demod_outputs 		= []
f_outputs           = []
for i in range(len(prf_list)):
	print ('PRF:',prf_list[i])
	prf             	= prf_list[i]
	low  = prf -(signal_of_interest + 5)
	high = prf +(signal_of_interest + 5)
	modulation_filter = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=new_Fs,
                       output='sos')
	modulated_signal 	= sosfiltfilt(modulation_filter, dsignal)
	demodulated_signal  = demodulate(modulated_signal,prf)

	# Now finally filter both original and new signals? 
	low  = signal_of_interest-6 
	high = signal_of_interest+1
	signal_isolation_filter = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=new_Fs,
                       output='sos')
	filtered_demod_signal    = sosfiltfilt(signal_isolation_filter, demodulated_signal)

	# add all the prf results together. 
	if i > 0:
		outputs 	= outputs + demodulated_signal
		f_outputs 	= f_outputs + filtered_demod_signal
	else: 
		outputs 	= demodulated_signal
		f_outputs 	= filtered_demod_signal
	# print (outputs.shape)

filtered_real_signal     = sosfiltfilt(signal_isolation_filter, dsignal)

demodulated_signal 		= outputs
filtered_demod_signal 	= f_outputs
# 
# Calculate FFT spectrums. 
timestep        = 1.0/new_Fs
start_pause     = int(0.125 * newN) #  start 1 second in to exclude the settling of the filter. 
end_pause       = int( newN)
fft_data        = fft(dsignal[start_pause:end_pause])
fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]

fft_ddata       = fft(demodulated_signal[start_pause:end_pause])
fft_ddata       = np.abs(2.0/(end_pause-start_pause) * (fft_ddata))[1:(end_pause-start_pause)//2]
xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies     = xf[1:(end_pause-start_pause)//2]


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(411)
plt.plot(t[start_pause:end_pause],filtered_real_signal[start_pause:end_pause],'k')
ax2 = fig.add_subplot(412)
plt.plot(t[start_pause:end_pause],filtered_demod_signal[start_pause:end_pause],'r')

ax3 = fig.add_subplot(223)
plt.plot(frequencies,fft_data,'k')
ax3.set_xlim([0,signal_of_interest+10])
ax4 = fig.add_subplot(224)
plt.plot(frequencies,fft_ddata,'r')
# ax3.set_xlim([0,signal_of_interest+10])
ax4.set_xlim([0,signal_of_interest+10])
plot_filename = '_demod_result.png'
plt.savefig(plot_filename)
plt.show()


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(frequencies,fft_data,'k')
# # plt.plot(frequencies,fft_rfdata,'r')
# ax.set_xlim([1000000-200,1000000+200])
# ax2 = fig.add_subplot(212)
# plt.plot(frequencies,fft_data,'k')
# # plt.plot(frequencies,fft_rfdata,'r')
# ax2.set_xlim([500000-200,500000+200])
# # ax2.set_xlim([0,5])
# # ax2.set_ylim([0,20])
# plt.show()


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_data,'k')
# # ax.set_xlim([0,1000000])
# ax.set_xlim([0,5000])
# ax.set_ylim([0,2.5])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = 'dFFT.png'
# plt.savefig(plot_filename)
# plt.show()
# # 
# 

