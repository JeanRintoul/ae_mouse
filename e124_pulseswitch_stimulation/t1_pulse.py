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
from scipy.stats import ttest_ind
from scipy.signal import hilbert
from scipy.signal import find_peaks
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16

file_number = 18
savepath    = 'D:\\ae_mouse\\e124_pulseswitch_stimulation\\t1\\'
gain        = 500
m_channel   = 0 
rf_channel  = 2
filename    = savepath + 't'+str(file_number)+'_stream.npy'
data        = np.load(filename)
fsignal     = 1e6*data[m_channel]/gain
rfsignal    = 10*data[rf_channel]  
Fs          = 1e7
timestep    = 1.0/Fs
duration    = 6 
N           = int(Fs*duration)
t           = np.linspace(0, duration, N, endpoint=False)


new_Fs = 1e5

# Hilbert transform c_data, so I can later downsample 
c_signal  = hilbert(fsignal)
c_envelope = np.abs(c_signal)
sos_lp = iirfilter(17, [new_Fs/2], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
c_hilberted             = sosfiltfilt(sos_lp, c_envelope)
  

sos_df_band = iirfilter(17, [2000], rs=60, btype='highpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
n_data              = sosfiltfilt(sos_df_band, fsignal)

s1 = 0 
s2 = duration 

# height of each pulse? 
pulse_size = np.max(n_data)-np.min(n_data)

print ('pulse size:', pulse_size)

fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111)
plt.plot(t,n_data,'k')
plt.show()

# 
# downsample for easier plotting. 
# 
# downsampling_factor  = int(Fs/new_Fs)
# dt                   = t[::downsampling_factor]
# d_df_data            = df_data[::downsampling_factor]
# d_c_hilberted        = c_hilberted[::downsampling_factor]
# mains_list = [50,100,150,200,250,300,350,400,450,500,1000]
# for i in range(len(mains_list)):
#     sos_mains_stop = iirfilter(17, [mains_list[i]-2 , mains_list[i]+2 ], rs=60, btype='bandstop',
#                            analog=False, ftype='cheby2', fs=new_Fs,
#                            output='sos')
#     d_df_data               = sosfiltfilt(sos_mains_stop , d_df_data)
# # 
# # 
# s1 = 0 
# s2 = duration 

# fig = plt.figure(figsize=(5,5))
# ax  = fig.add_subplot(111)
# plt.plot(t,fsignal,'k')
# plt.show()


# fig = plt.figure(figsize=(5,5))
# ax  = fig.add_subplot(111)
# plt.plot(dt,d_df_data/np.max(df_data),'k')
# plt.plot(dt,d_c_hilberted/np.max(d_c_hilberted),'r')
# ax.set_xlim([s1,s2])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.show()
# # 
