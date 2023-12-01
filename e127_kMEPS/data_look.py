'''

Title: compare the data going into generator with the data coming out of generator. 

Author: Jean Rintoul
Date:   26.10.2023

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 
# 
carrier_frequency = 1000
savepath       = 'D:\\ae_mouse\\e127_kMEPS\\transmission_spectrum_mouse\\'


#
duration        = 6
Fs              = 5e6
timestep        = 1.0/Fs
N               = int(Fs*duration)
channel_of_interest = 6
# # 
filename    = savepath + 't'+str(carrier_frequency)+'_stream.npy'
data = np.load(filename)
a,b = data.shape
print ('shape',a,b)

signal = data[channel_of_interest]

t = np.linspace(0, duration, N, endpoint=False)
fft_m = fft(signal)
fft_m = np.abs(2.0/(N) * (fft_m))[1:(N)//2]
xf = np.fft.fftfreq( (N), d=timestep)[:(N)//2]
frequencies = xf[1:(N)//2]


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
plt.plot(t,data[channel_of_interest],'k')
plt.legend(['v chan'],loc='upper right')
ax2 = fig.add_subplot(212)
plt.plot(frequencies,fft_m,'k')
ax2.set_xlim([0,carrier_frequency*3])
# ax2.set_xlim([0,carrier_frequency+carrier_frequency/2])
# plt.legend(['from recording chan'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plot_filename = 'carrier_data.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(t,from_generator,'k')
# ax.set_xlim([start,stop])
# plt.legend(['from generator'],loc='upper right')
# ax2 = fig.add_subplot(212)
# plt.plot(t,from_recording_chan,'k')
# ax2.set_xlim([start,stop])
# plt.legend(['from recording chan'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = 'zoom_file_comparison.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()

