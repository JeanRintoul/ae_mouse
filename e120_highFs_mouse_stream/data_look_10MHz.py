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
savepath       = 'D:\\ae_mouse\\e120_highFs_mouse_stream\\'

file_number    = 1
#
duration        = 3.5	
# 
Fs              = 1e7
timestep        = 1.0/Fs
N               = int(Fs*duration)
channel_of_interest = 1 
# # 
filename    = savepath + 't'+str(file_number)+'_stream_1MHz.npy'
data = np.load(filename)
a,b = data.shape
print ('shape',a,b)
from_recording_chan = data[channel_of_interest]

dfilename    = 'raw_sig_block.npy'
from_generator = np.load(dfilename)
print ('shape',len(from_generator) )

t = np.linspace(0, duration, N, endpoint=False)


start = 2
stop  = 2.00001
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
plt.plot(t,from_generator,'k')
# ax.set_xlim([])
plt.legend(['from generator'],loc='upper right')
ax2 = fig.add_subplot(212)
plt.plot(t,from_recording_chan,'k')
plt.legend(['from recording chan'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plot_filename = 'whole_file_comparison.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
plt.plot(t,from_generator,'k')
ax.set_xlim([start,stop])
plt.legend(['from generator'],loc='upper right')
ax2 = fig.add_subplot(212)
plt.plot(t,from_recording_chan,'k')
ax2.set_xlim([start,stop])
plt.legend(['from recording chan'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plot_filename = 'zoom_file_comparison.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

