'''

Title: what is the current? 
Author: Jean Rintoul
Date: 02.02.2023

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
from scipy.signal import kaiserord, lfilter, firwin, freqz
import pandas as pd
import sys
sys.path.append('D:\\mouse_aeti')  #  so that we can import from the parent folder. 
import mouse_library as m
# 
# Files 5-9
# 
# 
file_number     = 31
savepath        = 'D:\\mouse_aeti\\e100_neural_recording_pat_e_mouse\\t2\\'
Fs              = 5e6
duration        = 4.0
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0   
rf_channel      = 4 
# Load in the file. 
filename    = savepath + 't'+str(file_number)+'_stream.npy'
data = np.load(filename)
a,b = data.shape
t = np.linspace(0, duration, N, endpoint=False)
gain = 500 
fsignal = 1e6*data[m_channel]/gain
rfsignal = 10*data[rf_channel]
#  filter parameters
# low  = 0.5
# high = 1000 
# high = 1
# sos_low = iirfilter(17, [high], rs=60, btype='lowpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# dcsignal  = sosfiltfilt(sos_low, fsignal)

low  = 1020 - 15
high = 1020 + 15
sos_df = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
dfsignal  = sosfiltfilt(sos_df, fsignal)
rfdfsignal  = sosfiltfilt(sos_df, rfsignal)

low  = 1e6+1020 - 15
high = 1e6+1020 + 15
sos_sf = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
sfsignal  = sosfiltfilt(sos_sf, fsignal)
rfsfsignal  = sosfiltfilt(sos_sf, rfsignal)


start_pause     = int(0.25 * N+1)
end_pause       = int(0.875 * N-1)
window          = np.hanning(end_pause-start_pause)
fft_data        = fft(fsignal[start_pause:end_pause])
fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]

fft_rfdata        = fft(rfsignal[start_pause:end_pause])
fft_rfdata        = np.abs(2.0/(end_pause-start_pause) * (fft_rfdata))[1:(end_pause-start_pause)//2]

xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies     = xf[1:(end_pause-start_pause)//2]
# 
# 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
plt.plot(t,rfsignal,color='r')
plt.plot(t,sfsignal,color='m')
plt.plot(t,dfsignal,color='k')
ax.set_xlim([0,duration])
ax2 = fig.add_subplot(212)
# plt.plot(t,rfsignal,color='r')
plt.plot(t,rfsfsignal,color='m')
plt.plot(t,rfdfsignal,color='k')
ax2.set_xlim([0,duration])
plt.show()
# 
# 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
plt.plot(frequencies,fft_data,color='k')
ax.set_xlim([0,5000])
ax.set_ylim([0,200])
plt.legend(['v chan'],loc='upper right')
ax2 = fig.add_subplot(212)
plt.plot(frequencies,fft_rfdata,color='r')
ax2.set_xlim([0,5000])
ax2.set_ylim([0,0.005])
ax.set_ylabel('Volts ($\mu$V)')
ax2.set_ylabel('Volts (V)')
plt.legend(['rf chan'],loc='upper right')
# ax.set_xlabel('Time(s)')
# plt.autoscale(enable=True, axis='x', tight=True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
# plt.title('VEP')
# plot_filename = 'VEP.png'
# plt.savefig(plot_filename)
plt.show()