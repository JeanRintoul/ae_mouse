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
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16
#
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx

dfs 		= [161.8,293,316.5,357.3,296.6]
carriers 	= [2.17,4.4,1.42,0.78,2.2 ]

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
plt.plot(dfs,carriers,'k')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()



savepath    = 'D:\\ae_mouse\\e118_ionic_mixing_test\\t1\\'

file_number  = 21
# 
duration        = 6.0   
m_channel       = 0 
rf_channel      = 2 
gain            = 1000 
Fs              = 5e6
timestep        = 1.0/Fs
N               = int(Fs*duration)
t               = np.linspace(0, duration, N, endpoint=False)
#
# 
filename    = savepath + 't'+str(file_number)+'_stream.npy'
data        = np.load(filename)
fsignal     = 1e6*data[m_channel]/gain
rfsignal    = 10*data[rf_channel]  
# 
start_section =  1.5 
end_section   =  2.64

fft_start       = int(start_section*Fs)
fft_end         = int(end_section*Fs) 
N               = fft_end - fft_start
fft_data        = fft(fsignal[fft_start:fft_end] )
fft_data        = np.abs(2.0/(N-1) * (fft_data))[1:(N-1)//2]
xf              = np.fft.fftfreq( (N-1), d=timestep)[:(N-1)//2]
frequencies     = xf[1:(N-1)//2]

carrier_idx = find_nearest(frequencies,2e6)
df_idx = find_nearest(frequencies,10)

print ('df and carrier height',2*fft_data[df_idx],2*fft_data[carrier_idx])
#
# 
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(211)
plt.plot(t,fsignal,'k')
# ax.set_xlim([1.5,2.5])
ax2 = fig.add_subplot(212)
plt.plot(frequencies,fft_data,'k')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.show()

