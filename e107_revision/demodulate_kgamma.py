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
from scipy.signal import spectrogram
from scipy.signal import decimate
import scipy.signal
import pandas as pd
# 
signal_of_interest 	= 90.0
low 				= 0.1

savepath        = 'D:\\ae_mouse\\e107_revision\\t2_mouse_neural_decoding_only\\'
file_number     = 12
Fs              = 5e6
Fs              = 1e4
duration        = 12	
prf_list 		= [1020*1, 1020*3,500000*2] # 1020*2 no  # 1020*1, 1020*3
prf_list 		= [1020]
# prf_list 		= [750000] # no 500000, 500000+10201020*2 no  # 1020*1, 1020*3, 500000*2 + 1020, 500000*2-1020,500000*2-2*1020
# 
factor_1 		= 1
factor_2 		= 5
# factor_2 		= 2
gain            = 1000

# savepath        = 'D:\\ae_mouse\\e105_rfae_meps\\t4_mouse\\'
# file_number     = 11
# Fs              = 1e4
# duration        = 12.0	
# prf_list 		= [180]
# factor_1 		= 10
# factor_2 		= 2
# gain          = 1000

# the mouse from yesterday. 
# savepath        = 'D:\\ae_mouse\\e107_revision\\t2_mouse_neural_decoding_only\\'
# file_number     = 2
# Fs              = 5e6
# duration        = 8.0	
# prf_list 		= [2020]
# factor_1 		= 75
# factor_2 		= 50
# gain            = 1000
# # 
# # Phantom files 1-10 are all the same. 
# savepath        = 'D:\\ae_mouse\\e105_rfae_meps\\t2_phantom_detangling\\demodulation_challenge\\'
# file_number     = 5
# Fs              = 5e3
# duration        = 12.0	

# # file_number     = 11
# # Fs              = 5e6
# # duration        = 4.0	
# prf_list 		= [80]
# gain            = 2000
# factor_1 		= 1
# factor_2 		= 5
# 
# Why does it work so much better when there is a lower sampling rate? I had a 3k low pass in place, so I was effectively doing signal averaging to remove noise. 
# TODO: I should be able to recreate this in software? 
# Then I should copy this over and try it on the mouse signals I took yesterday with the high sample rate. 

# The IIR filter incorporates a modulation artefact when I have a high Fs. 
# # I should really lower the Fs before using it. 

# Noise rejection algorithm: 
# Use first, second and third harmonics of the PRF, to eliminate peaks that are above SNR, which are not repeated. 
# 

timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 4
# 
filename    = savepath + 't'+str(file_number)+'_stream.npy'
data 		= np.load(filename)
a,b 		= data.shape
t 			= np.linspace(0, duration, N, endpoint=False)
# convert it to microvolts by taking the gain into account. 
fsignal 	= 1e6*data[m_channel]/gain
fsignal 	= fsignal-np.mean(fsignal)
# 
# 
start_pause_in_seconds = 0.125/2
start_pause     = int((start_pause_in_seconds) *N) #  start 1 second in to exclude the settling of the filter. 
end_pause       = int(N)
fft_data        = fft(fsignal[start_pause:end_pause])
fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]
xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies     = xf[1:(end_pause-start_pause)//2]
# 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(311)
plt.plot(t,fsignal,'k')
plt.legend(['raw signal'],loc='upper right')
ax2 = fig.add_subplot(312)
plt.plot(frequencies,fft_data,'k')
plt.legend(['fft of raw signal'],loc='upper right')
ax3 = fig.add_subplot(313)
plt.plot(t,data[rf_channel],'r')
plt.legend(['rf signal'],loc='upper right')
plt.show()
# 
def demodulate(in_signal,carrier_f): 
    idown = in_signal*np.cos(2*np.pi*carrier_f*t)
    qdown = in_signal*np.sin(2*np.pi*carrier_f*t)   
    demodulated_signal = idown + 1j*qdown
    return np.abs(demodulated_signal)
# 
def demodulate2(in_signal,carrier_signal): 
    return in_signal*carrier_signal
# 
# 

high 				= signal_of_interest
# 
demod_outputs 		= []
f_outputs           = []
for i in range(len(prf_list)):
	print ('PRF:',prf_list[i])
	prf             	= prf_list[i]
	#  
	nFs = Fs
	# I have an issue with filter artefacts- perhaps if I downsample here I remove this problem? 
	# nFs 							= 5e3
	# newN 							= int(duration*nFs)
	# downsampling_factor           = int(Fs/nFs)
	# # 	# #   	# #   	# #
	# averaging_filter = iirfilter(17, [4000], rs=60, btype='lowpass',
    #                    analog=False, ftype='cheby2', fs=Fs,
    #                    output='sos')
	# fsignal 		 = sosfiltfilt(averaging_filter, fsignal)
	# fsignal 		 = fsignal[::downsampling_factor]
	# t 			 = t[::downsampling_factor]
	#   
	# 
	lowm  = prf - (signal_of_interest)
	highm = prf + (signal_of_interest)
	modulation_filter = iirfilter(17, [lowm,highm], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=nFs,
                       output='sos')
	modulated_signal 	= sosfiltfilt(modulation_filter, fsignal)
	demodulated_signal  = demodulate(modulated_signal,prf)
	#    
	# Let's try demodulating with the carrier signal. This works too. 
	# carrier_filter 		= iirfilter(17, [prf-10,prf+10], rs=60, btype='bandpass',
    #                			analog=False, ftype='cheby2', fs=Fs,
    #                			output='sos')	
	# carrier_signal 		= sosfiltfilt(carrier_filter, fsignal)
	# demodulated_signal  = demodulate2(modulated_signal,carrier_signal)	
	# 
	# Let's try demodulating with the hilbert transform. 
	# demodulated_signal = np.abs(hilbert(modulated_signal))
	# #  
	signal_isolation_filter = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=nFs,
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

filtered_real_signal     = sosfiltfilt(signal_isolation_filter, fsignal)

demodulated_signal 		= outputs
filtered_demod_signal 	= f_outputs

# Now downsample the signal so the spectrogram can be computed more easily. 
new_Fs                        	= 1000 
newN 							= int(duration*new_Fs)
downsampling_factor           	= int(nFs/new_Fs)

low_filter = iirfilter(17, [new_Fs], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=nFs,
                       output='sos')

reals     	= sosfiltfilt(low_filter, filtered_real_signal)
demods     	= sosfiltfilt(low_filter, filtered_demod_signal)

realsignal 	= reals[::downsampling_factor]
demodsignal = demods[::downsampling_factor]
t 			= t[::downsampling_factor]

# 
# Calculate FFT spectrums. 
timestep        = 1.0/new_Fs
start_pause     = int((0.125/2) * newN) #  start 1 second in to exclude the settling of the filter. 
end_pause       = int(newN)

fft_data        = fft(realsignal[start_pause:end_pause])
fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]

fft_ddata       = fft(demodsignal[start_pause:end_pause])
fft_ddata       = np.abs(2.0/(end_pause-start_pause) * (fft_ddata))[1:(end_pause-start_pause)//2]
xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies     = xf[1:(end_pause-start_pause)//2]
# 
# Before correlation, must account for the lag accurately. 
x               = realsignal
y               = demodsignal
t_cut           = t
# 
# print ('lens:',len(x),len(y),len(t))
correlation = signal.correlate(x-np.mean(x), y - np.mean(y), mode="full")
lags = signal.correlation_lags(len(x), len(y), mode="full")
idx_lag = -lags[np.argmax(correlation)]
print ('lag is',idx_lag)
# zero it if it is so wrong. 
if idx_lag > new_Fs:
	idx_lag = 0 
idx_lag = 0 
if idx_lag > 0:
    x 		= x[:(len(y)-idx_lag)]
    t_cut  	= t_cut[idx_lag:]
    y 		= y[idx_lag:]
else: 
    x 		= x[-idx_lag:]
    t_cut   = t_cut[-idx_lag:]
    y 		= y[:(len(demodsignal)+idx_lag)]
# 
# 
df 				= pd.DataFrame({'x': x-np.mean(x), 'y': y-np.mean(y) })
window          = int(new_Fs/5)
rolling_corr    = df['x'].rolling(window).corr(df['y'])
# 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(611)
plt.plot(t[start_pause:end_pause],realsignal[start_pause:end_pause],'k')
plt.xlim([start_pause_in_seconds,duration])
plt.legend(['lfp signal'],loc='upper right')

ax2 = fig.add_subplot(612)
plt.plot(t[start_pause:end_pause],demodsignal[start_pause:end_pause],'r')
plt.xlim([start_pause_in_seconds,duration])
plt.legend(['demodulated signal'],loc='upper right')

ax3 = fig.add_subplot(613) # rolling correlation metric. 
plt.plot(t_cut,rolling_corr*rolling_corr,'k')
ax3.set_xlim([start_pause_in_seconds,duration])
ax3.set_ylim([0,1])
plt.legend(['rolling correlation'],loc='upper right')

ax3 = fig.add_subplot(223)
plt.plot(frequencies,fft_data,'k')
ax3.set_xlim([0,signal_of_interest])
plt.legend(['lfp signal'],loc='upper right')

ax4 = fig.add_subplot(224)
plt.plot(frequencies,fft_ddata,'r')
plt.legend(['demodulated signal'],loc='upper right')
ax4.set_xlim([0,signal_of_interest])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
plot_filename = '_demod_result.png'
plt.savefig(plot_filename)
plt.show()
# 
# Do Spectrograms of both the original 10-100 Hz signal, and the demodulated one. 
# 
# 
Oz = realsignal
fs = new_Fs
print ('total available to nperseg:',len(Oz))  # 120000
nperseg 	= len(Oz)-1
nperseg 	= 1000
noverlap 	= nperseg-1
# 

flfp, tlfp, Sxx = signal.spectrogram(Oz, fs, nperseg=nperseg, noverlap=noverlap,window=signal.get_window('hann',nperseg),mode='psd',scaling='density')
vvmin = 0 
vvmax = np.max(Sxx/factor_1)

fd, td, Sxxd = signal.spectrogram(demodsignal, fs, nperseg=nperseg, noverlap=noverlap,window=signal.get_window('hann',nperseg),mode='psd',scaling='density')
dvvmin = 0 
dvvmax = np.max(Sxxd/factor_2)

# I could also vertically add the power bands and plot frequency vs total power. 
print ('spectrogram dimensions:',Sxxd.shape,len(flfp) ) # 1501, 117001
demod_totals = np.sum(Sxxd,axis=1)      #  sum all time per unit frequency. 
lfp_totals = np.sum(Sxx,axis=1)
demod_time_totals = np.sum(Sxxd,axis=0)  # sum all frequencies per unit time. 
lfp_time_totals = np.sum(Sxx,axis=0)
# 
# Create a correlation metric for this demod_time_totals. 
# 
df = pd.DataFrame({'x': lfp_time_totals -np.mean(lfp_time_totals), 'y': demod_time_totals-np.mean(demod_time_totals) })
window          = 400
# window        = len(demodulated_signal)
rolling_corr2   = df['x'].rolling(window).corr(df['y'])
# 
# What if I calculated the correlation with time per frequency bin? i.e. some frequency bins will trend better than others. 
# i.e. the 50 Hz one will not. 
# spec_freqs,b=Sxxd.shape
# corr_per_freq = []
# for i in range(len(flfp)):
# 	demodulated = Sxxd[i,:]
# 	lfp_signal  = Sxx[i,:]	
# 	df = pd.DataFrame({'x': lfp_signal -np.mean(lfp_signal), 'y': demodulated-np.mean(demodulated) })
# 	window          = 4000
# 	rolling_corr_per_freq   = df['x'].rolling(window).corr(df['y'])
# 	corr_per_freq.append(np.mean(rolling_corr_per_freq*rolling_corr_per_freq))
# 
# 
# 
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(flfp,corr_per_freq)
# ax.set_xlim([low,high])
# ax.set_xlabel('Frequencies(Hz)')
# plt.show()
# 
# 
# 
# 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
plt.plot(td,(lfp_time_totals-np.min(lfp_time_totals))/(np.max(lfp_time_totals)- np.min(lfp_time_totals) ),'k')
plt.plot(td,(demod_time_totals-np.min(demod_time_totals))/(np.max(demod_time_totals) -np.min(demod_time_totals)),'r')
plt.legend(['original lfp gamma summation','demodulated gamma summation'],loc='upper right')
ax.set_xlabel('Time(s)')
ax.set_xlim([0,duration])
ax.set_ylabel('Normalized amplitudes')
ax.set_title('Ketamine Gamma Demodulation')
ax2 = fig.add_subplot(212)
plt.plot(td,rolling_corr2*rolling_corr2,'k')
ax2.set_xlim([0,duration])
ax2.set_ylabel('Correlation')
ax2.set_xlabel('Time(s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plot_filename = 'ketamine_gamma_demodulation_trends.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.show()
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# im = plt.pcolormesh(tlfp, flfp, Sxx, shading='auto',cmap = 'inferno',vmin=dvvmin,vmax=dvvmax/100)
# plt.ylim([90,180+90])
# plt.xlim([0,duration])
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (s)')
# plt.legend(['original lfp between 90-180'],loc='upper right')
# fig.colorbar(im).set_label('Intensity (dB)')
# plt.show()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
im = plt.pcolormesh(tlfp, flfp, Sxx, shading='auto',cmap = 'inferno',vmin=vvmin,vmax=vvmax)
plt.ylim([low,high])
plt.xlim([0,duration])
plt.ylabel('Frequency (Hz)')
ax.set_title('Original Signal')
# plt.xlabel('Time (s)')
# plt.legend(['demodulated'],loc='upper right')
fig.colorbar(im).set_label('Intensity (dB)')

ax2 = fig.add_subplot(212)
im = plt.pcolormesh(td, fd, Sxxd, shading='auto',cmap = 'inferno',vmin=dvvmin,vmax=dvvmax)
plt.ylim([low,high])
plt.xlim([0,duration])
ax2.set_title('Demodulated')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
# plt.legend(['demodulated'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
fig.colorbar(im).set_label('Intensity (dB)')
plot_filename = 'ketamine_gamma_demodulation_spectrograms.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.show()


# 

