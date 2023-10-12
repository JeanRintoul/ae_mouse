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


# savepath       		= 'D:\\ae_mouse\\e105_rfae_meps\\t6_phantom_demodulation_amplitudes\\'
# # savepath       		= 'D:\\ae_mouse\\e105_rfae_meps\\t6_phantom_demodulation_amplitudes\\day1\\'
# file_number    		= 3
# Fs             		= 5e6
# duration       		= 8.0	
# prf_list 			= [4020]
# signal_of_interest 	= 90
# low 				= 1


savepath       		= 'D:\\ae_mouse\\e105_rfae_meps\\t2_phantom_detangling\\demodulation_challenge\\'
file_number    		= 10 # file 1-10
Fs              	= 5e3
# Fs              	= 5e6
duration        	= 12.0	
prf_list 			= [80]
signal_of_interest 	= 40
low 				= 5

# 
# signal_of_interest 	= 90.0
# low 				= 30

# savepath        = 'D:\\ae_mouse\\e107_revision\\t5_mouse\\'
# file_number     = 11
# Fs              = 5e6
# # Fs              = 1e4
# duration        = 12	
# prf_list 		= [1020*1, 1020*3,500000*2] # 1020*2 no  # 1020*1, 1020*3
# prf_list 		= [180]
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

# temp filter
ick_filter = iirfilter(17, [28,30], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=nFs,
                       output='sos')
reals     	= sosfiltfilt(ick_filter, reals)
demods     	= sosfiltfilt(ick_filter, demods)

# 
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
idx_lag = 0 
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
t = t_cut
# 
# df 				= pd.DataFrame({'x': x-np.mean(x), 'y': y-np.mean(y) })
# window          = int(new_Fs/5)
# rolling_corr    = df['x'].rolling(window).corr(df['y'])
# # 

# Do Spectrograms of both the original 10-100 Hz signal, and the demodulated one. 
#
print ('total available to nperseg:',len(x))  # 120000
nperseg 	= len(x)-1
nperseg 	= 3000
# nperseg 	= 5000
noverlap 	= nperseg-1
# 

flfp, tlfp, Sxx = signal.spectrogram(x, new_Fs, nperseg=nperseg, noverlap=noverlap,window=signal.get_window('hann',nperseg),mode='psd',scaling='density')
vvmin = 0 
vvmax = np.max(Sxx/factor_1)

fd, td, Sxxd = signal.spectrogram(y, new_Fs, nperseg=nperseg, noverlap=noverlap,window=signal.get_window('hann',nperseg),mode='psd',scaling='density')
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
window          = 1000
# window = 800
# window        = len(demodulated_signal)
rolling_corr2   = df['x'].rolling(window).corr(df['y'])
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
plt.plot(frequencies,fft_data,'k')
ax.set_xlim([0,signal_of_interest])
# plt.legend(['lfp signal'],loc='upper right')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = str(file_number)+'_lfp_fftresult.png'
plt.savefig(plot_filename)
plt.show()

# ax.set_ylabel('Volts ($\mu$V)')
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
plt.plot(frequencies,fft_ddata,'k')
ax.set_xlim([0,signal_of_interest])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.legend(['demodulated signal'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = str(file_number)+'_demod_fftresult.png'
plt.savefig(plot_filename)
plt.show()

# 
# 
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
plt.plot(td,(lfp_time_totals-np.min(lfp_time_totals))/(np.max(lfp_time_totals)- np.min(lfp_time_totals) ),'k')
plt.plot(td,(demod_time_totals-np.min(demod_time_totals))/(np.max(demod_time_totals) -np.min(demod_time_totals)),'r')
plt.legend(['LFP Gamma','Demodulated Gamma'],loc='upper right',frameon=0,fontsize=16)
# ax.set_xlabel('Time(s)')
ax.set_xlim([0,duration])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# ax.set_ylabel('Normalized amplitudes')
# ax.set_title('Ketamine Gamma Demodulation')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = str(file_number)+'ketamine_gamma_demodulation_comparison_correlation.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.show()
# 

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
plt.plot(td,rolling_corr2*rolling_corr2,'k')
# plt.plot(t,rolling_corr,'r')
ax.set_xlim([0,duration])
ax.set_ylim([0,1])
# ax.set_ylabel('Correlation')
# ax.set_xlabel('Time(s)')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = str(file_number)+'ketamine_gamma_demodulation_correlation.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.show()
# 
print ('maxes',vvmax,dvvmax)

factor_lfp = 4 
factor_demod = 1.5 

# extent_params = [0.3,11.7,5,40]

# print (extent_params)
print ('td',np.min(td),np.max(td) )
print ('fd',np.min(fd),np.max(fd) ) 

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
# im = plt.imshow(Sxx,  interpolation = 'bicubic',cmap = 'Reds',vmin=vvmin,vmax=vvmax/factor_lfp,extent=extent_params,aspect='auto')
im = plt.pcolormesh(td, fd, Sxx, shading='auto',cmap = 'Reds',vmin=vvmin,vmax=vvmax/factor_lfp)
# plt.ylim([low,high])
plt.ylim([low,high])
plt.xlim([0,duration])
plt.xticks([]) 
plt.yticks([]) 
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (s)')
# cb=plt.colorbar(im, drawedges=False).set_label('Intensity (dB)')
# cb = plt.colorbar(im).set_label('Intensity (dB)')
# cb.drawedges(False)
# cb.outline.set_color('white')
# cb.outline.set_linewidth(0)

# cb.outline.set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
plot_filename = str(file_number)+'ketamine_gamma_spectrograms_lfp'+str(np.round(vvmax,2) )+'.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
# im = plt.imshow(Sxxd,  interpolation = 'bicubic',cmap = 'Reds',vmin=dvvmin,vmax=dvvmax/factor_demod,extent=extent_params,aspect='auto')
im = plt.pcolormesh(td, fd, Sxxd, shading='auto',cmap = 'Reds',vmin=dvvmin,vmax=dvvmax/factor_demod)
plt.ylim([low,high])
plt.xlim([0,duration])
plt.xticks([]) 
plt.yticks([]) 
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (s)')
# fig.colorbar(im).set_label('Intensity (dB)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()

plot_filename = str(file_number)+'ketamine_gamma_spectrograms_demod'+str(np.round(dvvmax,2) )+'.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.show()


# 

