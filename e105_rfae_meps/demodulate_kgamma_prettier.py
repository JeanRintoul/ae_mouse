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
# 
# in the mouse yesterday. 
savepath       = 'D:\\ae_mouse\\e105_rfae_meps\\t6_phantom_demodulation_amplitudes\\'
file_number    = 9
Fs             = 5e6
duration       = 8.0	
prf_list 			= [180]
signal_of_interest 	= 90
low 				= 10

# savepath       		= 'D:\\ae_mouse\\e105_rfae_meps\\t2_phantom_detangling\\demodulation_challenge\\'
# file_number    		= 2  # file 1-10
# Fs              	= 5e3
# duration        	= 12.0	
# prf_list 			= [80]
# signal_of_interest 	= 40
# low 				= 5

# savepath       = 'D:\\ae_mouse\\e105_rfae_meps\\t6_phantom_demodulation_amplitudes\\'
# file_number    = 4
# Fs             = 5e6
# duration       = 8.0	
# 
# savepath        = 'D:\\ae_mouse\\e105_rfae_meps\\t4_mouse\\'
# file_number     = 7
# # file_number     = 7
# # savepath        = 'D:\\ae_mouse\\e105_rfae_meps\\t3_mouse\\'
# # file_number     = 11
# Fs              = 1e4
# duration        = 12.0	

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
# 

# 
# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(211)
# plt.plot(t,data[m_channel],'k')
# ax2 = fig.add_subplot(212)
# plt.plot(t,data[rf_channel],'k')
# plt.show()
# 
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
# def demodulate2(in_signal,carrier_signal): 
#     return in_signal*carrier_signal
# 



high 				= signal_of_interest
# 
demod_outputs 		= []
f_outputs           = []
for i in range(len(prf_list)):
	print ('PRF:',prf_list[i])
	prf             	= prf_list[i]
	lowm  = prf -(signal_of_interest)
	highm = prf +(signal_of_interest)
	modulation_filter = iirfilter(17, [lowm,highm], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=new_Fs,
                       output='sos')
	modulated_signal 	= sosfiltfilt(modulation_filter, dsignal)
	demodulated_signal  = demodulate(modulated_signal,prf)

	# Now finally filter both original and new signals? 
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

# Before correlation, must account for the lag accurately. 
x               = filtered_real_signal
y               = filtered_demod_signal
t_cut           = t
correlation = signal.correlate(x-np.mean(x), y - np.mean(y), mode="full")
lags = signal.correlation_lags(len(x), len(y), mode="full")
idx_lag = -lags[np.argmax(correlation)]
print ('lag is',idx_lag)
if idx_lag > 0:
    x = x[:(len(y)-idx_lag)]
    t_cut              = t_cut[idx_lag:]
    y = y[idx_lag:]
else: 
    x = x[-idx_lag:]
    t_cut              = t_cut[-idx_lag:]
    y = y[:(len(demodulated_signal)+idx_lag)]
    
df = pd.DataFrame({'x': x-np.mean(x), 'y': y-np.mean(y) })
window          = 10000
window          = 50000
# window          = len(demodulated_signal)
rolling_corr    = df['x'].rolling(window).corr(df['y'])


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(611)
# plt.plot(t[start_pause:end_pause],filtered_real_signal[start_pause:end_pause],'k')
# plt.xlim([2,duration])
# plt.legend(['lfp signal'],loc='upper right')
# ax2 = fig.add_subplot(612)
# plt.plot(t[start_pause:end_pause],filtered_demod_signal[start_pause:end_pause],'r')
# plt.xlim([2,duration])
# plt.legend(['demodulated signal'],loc='upper right')

# ax3 = fig.add_subplot(613) # rolling correlation metric. 
# plt.plot(t_cut,rolling_corr*rolling_corr,'k')
# plt.xlim([2,duration])
# plt.ylim([0,1])
# plt.legend(['rolling correlation'],loc='upper right')

# ax3 = fig.add_subplot(223)
# plt.plot(frequencies,fft_data,'k')
# ax3.set_xlim([0,signal_of_interest])
# plt.legend(['lfp signal'],loc='upper right')
# ax4 = fig.add_subplot(224)
# plt.plot(frequencies,fft_ddata,'r')
# plt.legend(['demodulated signal'],loc='upper right')
# ax4.set_xlim([0,signal_of_interest])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# ax4.spines['right'].set_visible(False)
# ax4.spines['top'].set_visible(False)
# plot_filename = '_demod_result.png'
# plt.savefig(plot_filename)
# plt.show()
# 
# Do Spectrograms of both the original 10-100 Hz signal, and the demodulated one. 
# 
# 
Oz = filtered_real_signal
fs = new_Fs
print ('total available to nperseg:',len(Oz))  # 120000
nperseg 	= len(Oz)-1
nperseg 	= 3000
noverlap 	= nperseg-1
# 
flfp, tlfp, Sxx = signal.spectrogram(Oz, fs, nperseg=nperseg , noverlap=noverlap,window=signal.get_window('hann',nperseg),mode='psd',scaling='density')
vvmin = 0 
vvmax = np.max(Sxx/1)

fd, td, Sxxd = signal.spectrogram(filtered_demod_signal, fs, nperseg=nperseg , noverlap=noverlap,window=signal.get_window('hann',nperseg),mode='psd',scaling='density')
dvvmin = 0 
dvvmax = np.max(Sxxd/1)

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
window          = 10000
# window          = 5000
window          = 40000
# window        = len(demodulated_signal)
rolling_corr2   = df['x'].rolling(window).corr(df['y'])

print ('mean correlation:',np.mean(rolling_corr2))
# 
# What if I calculated the correlation with time per frequency bin? i.e. some frequency bins will trend better than others. 
# i.e. the 50 Hz one will not. 
# spec_freqs,b=Sxxd.shape
corr_per_freq = []
for i in range(len(flfp)):
	demodulated = Sxxd[i,:]
	lfp_signal  = Sxx[i,:]	
	df = pd.DataFrame({'x': lfp_signal -np.mean(lfp_signal), 'y': demodulated-np.mean(demodulated) })
	window          = 4000
	rolling_corr_per_freq   = df['x'].rolling(window).corr(df['y'])
	corr_per_freq.append(np.mean(rolling_corr_per_freq*rolling_corr_per_freq))
# 
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
plot_filename = '_lfp_fftresult.png'
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
plot_filename = '_demod_fftresult.png'
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
plt.plot(td,rolling_corr2,'k')
ax.set_xlim([0,duration])
ax.set_ylim([-1,1])
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

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
im = plt.pcolormesh(tlfp, flfp, Sxx, shading='auto',cmap = 'Reds',vmin=vvmin,vmax=vvmax/factor_lfp)
plt.ylim([low,high])
plt.xlim([0,duration])
# plt.xticks([]) 
# plt.yticks([]) 
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')

fig.colorbar(im).set_label('Intensity (dB)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = str(file_number)+'ketamine_gamma_spectrograms_lfp'+str(np.round(vvmax,2) )+'.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
im = plt.pcolormesh(td, fd, Sxxd, shading='auto',cmap = 'Reds',vmin=dvvmin,vmax=dvvmax/factor_demod)
plt.ylim([low,high])
plt.xlim([0,duration])
# plt.xticks([]) 
# plt.yticks([]) 
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
fig.colorbar(im).set_label('Intensity (dB)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()

plot_filename = str(file_number)+'ketamine_gamma_spectrograms_demod'+str(np.round(dvvmax,2) )+'.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.show()


# 

