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
import scipy.stats
from scipy.stats import pearsonr
import pandas as pd
from scipy import signal
# 
# 
# Try with a differently generated VEP filter. 
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16
# 
saveprefix  = './/images//'
# 
savepath    = 'D:\\ae_mouse\\e142_ae_neural_decoding\\t6_mouse\\'

# savepath    = 'D:\\ae_mouse\\e142_ae_neural_decoding\\t6_mouse\\ae8Hz_VEP_g5000_good\\'
# savepath    = 'D:\\ae_mouse\\e142_ae_neural_decoding\\t4_mouse_g2000\\ae10hz_g2000\\'
# 
# savepath    = 'D:\\ae_mouse\\e142_ae_neural_decoding\\t1_g10000\\10Hz_aevep_g10000\\'
# 
# 
# file_number   = 8
# filename      = 't'+str(file_number)+'_stream.npy'
# filename      = 'ae10hz_g5000.npy'  # really bad. heat mat noise
# filename      = 'ae10hz_wF21_g10000.npy' # strange frequency shifting.
# filename      = '10hz_g5000_noF21.npy' # strange frequency shifting. 
# 
# can I demodulate this? 0.02 amplitude, with vep at 10 hz being 9 microvolts, e field being 
filename = 'ae10Hz_g5000_F21.npy'  # really good. I can see everything in a single file. 
# 

# alternate comparison file . 


# # 
gain            = 5000
band_limit      = 35
Fs              = 2e6
timestep        = 1.0/Fs
duration        = 30

m_channel       = 0 
rf_channel      = 4
marker_channel  = 7 
pulse_length    = 0.0 # pulse length in seconds. 
N               = int(Fs*duration)
carrier         = 500000
t               = np.linspace(0, duration, N, endpoint=False)
# 
# 

data        = np.load(savepath+filename)
fsignal     = (1e6*data[m_channel]/gain)
fsignal     = fsignal - np.mean(fsignal)
rfsignal    = 10*data[rf_channel]
marker      = data[marker_channel]
rbeta           = 14
raw_timestep    = 1/Fs 
rawN            = len(fsignal)
raw_window      = np.kaiser( (rawN), rbeta )
xf              = np.fft.fftfreq( (rawN), d=raw_timestep)[:(rawN)//2]
frequencies = xf[1:(rawN)//2]

fft_raw         = fft(fsignal)
fft_raw         = np.abs(2.0/(rawN) * (fft_raw))[1:(rawN)//2]
fft_rawk         = fft(fsignal*raw_window)
fft_rawk         = np.abs(2.0/(rawN) * (fft_rawk))[1:(rawN)//2]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   

# 
# fontsize control. 
f = 18

fig = plt.figure(figsize=(8,3))
ax  = fig.add_subplot(111)
plt.plot(t,fsignal,'k')
plt.show()
# 
fig = plt.figure(figsize=(8,3))
ax  = fig.add_subplot(211)
# plt.plot(frequencies,fft_raw,'r')
plt.plot(frequencies,fft_rawk,'k')
ax.set_xlim([carrier-band_limit,carrier+band_limit])
ax.set_ylim([0,0.15])
plt.xticks([carrier-10,carrier+10,carrier-20,carrier+20,carrier-30,carrier+30],fontsize=f)
plt.yticks(fontsize=f)

ax2  = fig.add_subplot(212)
plt.plot(frequencies,fft_rawk,'k')
ax2.set_xlim([carrier-band_limit,carrier+band_limit])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.ticklabel_format(useOffset=False)
plt.locator_params(axis='y', nbins=6)
# plt.locator_params(axis='x', nbins=4)
plt.tight_layout()
plot_filename = savepath+'_f1.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_raw,'k')
# plt.plot(frequencies,fft_rawk,'k')

ax.set_xlim([0,band_limit])
ax.set_ylim([0,10])
plt.xticks([10,20,30],fontsize=f)
plt.yticks(fontsize=f)
plt.locator_params(axis='y', nbins=6)
# plt.locator_params(axis='x', nbins=5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = savepath+'_f1_vep.png'
plt.savefig(plot_filename)
plt.show()

