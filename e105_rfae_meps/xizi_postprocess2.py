'''

Copy Xizi's post-processing pipeline as exactly and unimaginatively as possible. 
Author: Jean Rintoul
Date: 31/07/2023

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
from scipy.signal import hilbert
import pandas as pd
# 
# files 1-10. 

filename      = 'xizi_phantom_demodulation.npz'
data          = np.load(filename)
raw_signal    = data['outputs']
Fs            = data['new_Fs']
a             = raw_signal.shape
print (a[0], Fs)
prf             = 80
signal_of_interest = 26
# gain            = 2000
timestep        = 1.0/Fs
duration        = a[0]/Fs
print ('duration',duration)
N = a[0]
# 
# find the fft of the data. 
start_pause     = int(0.0 * N)
end_pause       = int( N)

fft_data        = fft(raw_signal[start_pause:end_pause])
fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]

xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies     = xf[1:(end_pause-start_pause)//2]
# # 
t = np.linspace(0, duration, N, endpoint=False)

low  = prf - 1
high = prf + 1
prf_filter = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

low  = signal_of_interest-1 
high = signal_of_interest+1
signal_filter = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

low  = prf -(signal_of_interest +5)
high = prf +(signal_of_interest +5)
modulation_filter = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')


modulated_signal = sosfiltfilt(modulation_filter, raw_signal)
carrier_signal   = sosfiltfilt(prf_filter, raw_signal)

def demodulate(in_signal,carrier_f): 
    idown = in_signal*np.cos(2*np.pi*carrier_f*t)
    qdown = in_signal*np.sin(2*np.pi*carrier_f*t)   
    demodulated_signal = idown + 1j*qdown
    return np.abs(demodulated_signal)

def demodulate2(in_signal,carrier_signal): 
    return in_signal*carrier_signal

# To do - correct for the issue by performing hilbert tx
analytical_signal   = hilbert(modulated_signal)
demodulated_signal  = demodulate(modulated_signal,prf)
demodulated_signal2 = demodulate(modulated_signal,carrier_signal)

fft_hdata        = fft(np.abs(analytical_signal))
fft_hdata        = np.abs(2.0/(end_pause-start_pause) * (fft_hdata))[1:(end_pause-start_pause)//2]

fft_ddata        = fft(demodulated_signal)
fft_ddata        = np.abs(2.0/(end_pause-start_pause) * (fft_ddata))[1:(end_pause-start_pause)//2]

fft_d2data        = fft(demodulated_signal2)
fft_d2data        = np.abs(2.0/(end_pause-start_pause) * (fft_d2data))[1:(end_pause-start_pause)//2]
# 
filtered_signal      = sosfiltfilt(signal_filter, raw_signal)
filtered_asignal     = sosfiltfilt(signal_filter, analytical_signal)
filtered_dsignal     = sosfiltfilt(signal_filter, demodulated_signal)
filtered_dsignal2    = sosfiltfilt(signal_filter, demodulated_signal2)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(311)
plt.plot(t,carrier_signal,'k')
plt.legend(['carrier signal'],loc='upper right')
# ax.set_xlim([2,10])
ax2 = fig.add_subplot(312)
# plt.plot(t,filtered_dsignal,'k')
plt.plot(t,modulated_signal,'k')
plt.legend(['modulated signal'],loc='upper right')
ax3 = fig.add_subplot(313)
plt.plot(t,raw_signal,'k')
# ax.set_xlim([2,10])
plt.legend(['raw signal'],loc='upper right')
plt.show()

# Before correlation, must account for the lag accurately. 
x               = filtered_signal
y               = filtered_dsignal
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
    
df = pd.DataFrame({'x': x, 'y': y })
window          = 1000
# window          = len(demodulated_signal)
rolling_corr    = df['x'].rolling(window).corr(df['y'])

# phase aberration plot. 
# It seems that the modulated signal has a slight lag, which is correct because it undergoes an acoustic conversion. 
offset = 2.01783-2.00395
offset = 0
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
plt.plot(t,filtered_signal/np.max(filtered_signal),'k')
plt.plot(t-offset,filtered_dsignal2/np.max(filtered_dsignal2),'m')
plt.legend(['actual signal','recovered signal'],loc='upper right')
ax.set_xlim([0,np.max(t)])
ax2 = fig.add_subplot(212)
plt.plot(t_cut,rolling_corr,'k')
ax2.set_ylim([-1,1])
ax2.set_xlim([0,np.max(t)])
ax2.set_xlabel('Time(s)')
plt.legend(['correlation plot(1 second)'],loc='lower right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plot_filename = 'demodulation_results.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(t,filtered_signal/np.max(filtered_signal),'k')
plt.plot(t-offset,filtered_dsignal2/np.max(filtered_dsignal2),'m')
plt.legend(['actual signal','recovered signal'],loc='upper right')
ax.set_xlim([6,7])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = 'demodulation_closeup.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(frequencies,fft_data,'k')
# ax.set_xlim([0,1e6+100])
plt.legend(['fft of all data'],loc='upper right')
plt.show()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(121)
plt.plot(frequencies,fft_data,'k')
# plt.plot(frequencies,fft_ddata,'r')
ax.set_xlim([0,60])
ax.set_ylim([0,90])
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Frequencies(Hz)')
plt.legend(['Original','IQ demodulation'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2 = fig.add_subplot(122)
plt.plot(frequencies,fft_ddata,'r')
plt.legend(['IQ demodulation result'],loc='upper right')
ax2.set_xlim([0,60])
ax2.set_ylim([0,0.9])
ax2.set_ylabel('Volts ($\mu$V)')
ax2.set_xlabel('Frequencies(Hz)')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plot_filename = 'demod_ffts.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.show()

# # 
# ax.set_ylabel('Volts ($\mu$V)')
