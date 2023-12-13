'''

Title: field_view.py
Function: View a stream code data file. 

Author: Jean Rintoul
Date: 18.03.2022

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
from scipy.stats import pearsonr
from scipy.signal import kaiserord
from scipy.signal import spectrogram
from scipy.signal import decimate
from scipy.signal import resample
import scipy.signal
# t4/f7,10,11 is pretty good. 
# try file 11 for demodulation. 
file_number    = 1
gain           = 1000
savepath       = 'D:\\ae_mouse\\e107_revision\\t1_mouse\\'
Fs              = 5e6
duration        = 8.0   
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 4

def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx

filename    = savepath + 't'+str(file_number)+'_stream.npy'
data = np.load(filename)
a,b = data.shape
t = np.linspace(0, duration, N, endpoint=False)
# convert it to microvolts by taking the gain into account. 
fsignal = 1e6*data[m_channel]/gain
rfsignal = 10*data[rf_channel] 
fsignal = fsignal - np.mean(fsignal)

start_pause     = int(0.0 * N)
end_pause       = int( N)
fft_data        = fft(fsignal[start_pause:end_pause])
fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]
xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies     = xf[1:(end_pause-start_pause)//2]

# 
# (f, S) = scipy.signal.periodogram(fsignal, Fs, scaling='density')
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.semilogy(f, S,'k')
# # plt.ylim([1e-7, 1e2])
# plt.xlim([0,100])
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()
# 
# 
# Downsample the signal to 200 Hz. 
new_Fs                        = 1000
downsampling_factor           = int(Fs/new_Fs)
high = new_Fs 
low_filter = iirfilter(17, [high], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')


fsignal = sosfiltfilt(low_filter, fsignal)
nsignal = fsignal[::downsampling_factor]
t       = t[::downsampling_factor]

low  = 10
high = 100
gamma_filter = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=new_Fs,
                       output='sos')
gamma_signal = sosfiltfilt(gamma_filter, nsignal)
gamma_signal = gamma_signal - np.mean(np.abs(gamma_signal))


delta_filter = iirfilter(17, [0.5,10], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=new_Fs,
                       output='sos')
delta_signal = sosfiltfilt(delta_filter, nsignal)
delta_signal = delta_signal - np.mean(np.abs(delta_signal))

# Now do a PSD of the gamma signal. 
Oz = gamma_signal
fs = new_Fs
#print ('total available to nperseg:',len(Oz))  # 120000
nperseg = len(Oz)-1
nperseg = 600
noverlap = nperseg-1
f350, t350, Sxx = signal.spectrogram(Oz, fs, nperseg=nperseg , noverlap=noverlap,window=signal.get_window('hann',nperseg),mode='psd',scaling='density')
vvmin = 0 
# vvmax = np.max(Sxx)
vvmax = np.max(Sxx/10)


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(311)
plt.plot(t,gamma_signal,'k')
ax.set_xlim([0,duration])
ax2 = fig.add_subplot(312)
plt.plot(t,delta_signal,'k')
ax2.set_xlim([0,duration])
ax3 = fig.add_subplot(313)
im = plt.pcolormesh(t350, f350, Sxx, shading='auto',cmap = 'inferno',vmin=vvmin,vmax=vvmax)
plt.ylim([10,100])
plt.xlim([0,duration])
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
# fig.colorbar(im).set_label('Intensity (dB)')
plot_filename = 'ketamine_gamma_psd.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.show()


Oz = nsignal
fs = new_Fs
print ('total available to nperseg:',len(Oz))  # 120000
nperseg = len(Oz)-1
# nperseg = 6000
nperseg = 1000
noverlap = nperseg-1
f350, t350, Sxx = signal.spectrogram(Oz, fs, nperseg=nperseg , noverlap=noverlap,window=signal.get_window('hann',nperseg),mode='psd',scaling='density')
vvmin = 0 
# vvmax = 10  # np.max(Sxx/100)
vvmax = np.max(Sxx/40)

print ('max db?',np.max(Sxx),Sxx.shape)
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(311)
plt.plot(t,nsignal,'k')
ax2 = fig.add_subplot(312)
plt.plot(frequencies,fft_data,'k')
ax2.set_xlim([0,100])
ax2.set_xlabel('Frequencies(Hz)')

ax2.set_ylabel('Volts ($\mu$V)')
ax3 = fig.add_subplot(313)
im = plt.pcolormesh(t350, f350, Sxx, shading='auto',cmap = 'inferno',vmin=vvmin,vmax=vvmax)
plt.ylim([0,100])
plt.xlim([0,duration])

plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
# fig.colorbar(im).set_label('Intensity (dB)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
plot_filename = 'ketamine_psd.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.show()

# If I can scale it suitably, there should be a 3.088hz peak, 43, 50, 55 Hz little peaks


