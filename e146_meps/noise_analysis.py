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
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16
# 
savepath             = 'D:\\ae_mouse\\e146_meps\\t8_mouse\\aedc2\\'
outpath              = savepath 
# 
ae_filename   = savepath + 'tae_g500_stream.npy'     # duration 6
# ae_filename   = savepath + 'tp_p0.15_g500_stream.npy'     # duration 6
# ae_filename   = savepath + 'tv_v0.12_g500_stream.npy'     # duration 6

# 
duration        = 6
# 
frequency       = 1

m_channel       = 0 
rf_channel      = 4
v_channel       = 6
i_channel       = 5 
emg_channel     = 2 


brain_gain      = 10
emg_gain        = 500 


band_limit      = 80 
Fs              = 5e6 
timestep        = 1/Fs
N               = int(duration*Fs)
t               = np.linspace(0, duration, N, endpoint=False)
cut             = 1000
sos_low_band    = iirfilter(17, [cut], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
sos_emg_band    = iirfilter(17, [20,500], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
mains_fs = np.arange(50,1000,50)
# print('mains:',mains_fs)
# mains_fs = np.arange(50,300,50)
def mains_stop(signal):
    mains = mains_fs
    # mains = [50]    
    for i in range(len(mains)):
        sos_mains_stop    = iirfilter(17, [mains[i]-4,mains[i]+4], rs=60, btype='bandstop',
                               analog=False, ftype='cheby2', fs=Fs,
                               output='sos')
        signal = sosfiltfilt(sos_mains_stop, signal)
    return signal


# get the frequencies ready
start_idx       = int(0*Fs) 
end_idx         = int(duration*Fs) 
# 
data           = np.load(ae_filename)
ae_fsignal     = (1e6*data[m_channel]/brain_gain)
ae_emgsignal   = (1e6*data[emg_channel]/emg_gain)
ae_rfsignal    = 10*data[rf_channel]
ae_vsignal     = 10*data[v_channel]
ae_low_signal  = sosfiltfilt(sos_low_band, ae_fsignal)
ae_emg  = sosfiltfilt(sos_emg_band, ae_emgsignal)
ae_emg_signal  = mains_stop(ae_emg)
# ae_emg_signal  = sosfiltfilt(sos_emg_stop, ae_emg_signal)
ae_pp_amp      = np.max(ae_low_signal[start_idx:end_idx]) - np.min(ae_low_signal[start_idx:end_idx])
print ('ae amplitude:',ae_pp_amp)

start_pause = int(0*Fs)
end_pause = int(duration*Fs)
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx
# 
# 
df_idx = find_nearest(frequencies,frequency)
fft_m = fft(ae_fsignal[start_pause:end_pause])
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]

fft_v = fft(ae_vsignal[start_pause:end_pause])
fft_v = np.abs(2.0/(end_pause-start_pause) * (fft_v))[1:(end_pause-start_pause)//2]

fft_rf = fft(ae_rfsignal[start_pause:end_pause])
fft_rf = np.abs(2.0/(end_pause-start_pause) * (fft_rf))[1:(end_pause-start_pause)//2]

fft_ae_emg = fft(ae_emgsignal[start_pause:end_pause])
fft_ae_emg = np.abs(2.0/(end_pause-start_pause) * (fft_ae_emg))[1:(end_pause-start_pause)//2]
# 
# 
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(311)
plt.plot(t, ae_rfsignal,'k')
# ax.set_xlim([0,1000])
# ax.set_xlim([0,600000])
ax2 = fig.add_subplot(312)
plt.plot(frequencies,fft_rf,'k')
# ax.set_xlim([0,1000])
# ax2.set_xlim([0,600000])
# ax3 = fig.add_subplot(313)
# plt.plot(t, ae_emg_signal,'k')
plt.tight_layout()
# plot_filename = outpath+'noise_rf.png'
# plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(311)
plt.plot(t, ae_emgsignal,'k')
# ax.set_xlim([0,1000])
# ax.set_xlim([0,600000])
ax2 = fig.add_subplot(312)
plt.plot(frequencies,fft_ae_emg,'k')
# ax.set_xlim([0,1000])
# ax2.set_xlim([0,600000])
ax3 = fig.add_subplot(313)
plt.plot(frequencies,fft_v,'k')
plt.tight_layout()
plot_filename = outpath+'noise_volts.png'
plt.savefig(plot_filename)
plt.show()
#

# fontsize on plots
# f       = 18
# vlim    = 100
# plim    = 100 
# siglim  = 50000
# emglim  = 300
# # Turn interactive plotting off
# # plt.ioff()

# # # # # # AE plots # # # # 
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# plt.plot(t, ae_vsignal,'grey')

# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
# ax.set_ylim([-plim,plim])
# ax.set_xlim([0,duration])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'ae_rf.png'
# plt.savefig(plot_filename)
# plt.show()
# plt.close(fig)
