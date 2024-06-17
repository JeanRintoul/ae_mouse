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
savepath             = 'D:\\ae_mouse\\e146_meps\\t3_mouse\\loo_test1\\'
outpath              = 'D:\\ae_mouse\\e146_meps\\t3_mouse\\'
ae_filename   = savepath + 'tae_g10_stream.npy'     # duration 10
# 
# savepath      = 'D:\\ae_mouse\\e146_meps\\t2_mouse_ae2RFAMP\\ae_loo_test7_lowp\\'
# ae_filename   = savepath + 'tae_p_g500_stream.npy'     # duration 10
# 
# savepath             = 'D:\\ae_mouse\\e146_meps\\t2_mouse_ae2RFAMP\\ae_loo_test5\\'
# ae_filename   = savepath + 'tae_p_g500_stream.npy'     # duration 10
# # 
# # 
# savepath             = 'D:\\ae_mouse\\e146_meps\\t2_mouse_ae2RFAMP\\epulse_loo_test3\\'
# ae_filename   = savepath + 'tae_epulse_p_g500_stream.npy'     # duration 10
# 
# 
frequency       = 0.5
m_channel       = 0 
rf_channel      = 4
v_channel       = 6
i_channel       = 5 
emg_channel     = 2 
gain            = 500
duration        = 12
band_limit      = 80 
Fs              = 5e6 
timestep        = 1/Fs
N               = int(duration*Fs)
t               = np.linspace(0, duration, N, endpoint=False)
cut             = 1000
sos_low_band    = iirfilter(17, [cut], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
sos_emg_band    = iirfilter(17, [20,2000], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

# get the frequencies ready
start_idx       = int(0*Fs) 
end_idx         = int(duration*Fs) 
# 
data           = np.load(ae_filename)
ae_fsignal     = (1e6*data[m_channel]/gain)
ae_emgsignal   = (1e6*data[emg_channel]/gain)
ae_rfsignal    = 10*data[rf_channel]
ae_vsignal     = 10*data[v_channel]
ae_low_signal  = sosfiltfilt(sos_low_band, ae_fsignal)
ae_emg_signal  = sosfiltfilt(sos_emg_band, ae_emgsignal)
ae_pp_amp      = np.max(ae_low_signal[start_idx:end_idx]) - np.min(ae_low_signal[start_idx:end_idx])
print ('ae amplitude:',ae_pp_amp)

start_pause = int(3*Fs)
end_pause = int((7)*Fs)
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx
#
# mains frequencies to remove. 
# 
mains = [50,100,150,250,350,400,450,500,550]
for i in range(len(mains)):
    sos_mains_stop    = iirfilter(17, [mains[i]-2,mains[i]+2], rs=60, btype='bandstop',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
    ae_emg_signal = sosfiltfilt(sos_mains_stop, ae_emg_signal)
#
#
analytical_signal     = hilbert(ae_emg_signal) # Hilbert demodulate.  
h_signal              = np.abs(analytical_signal)
sos_hlp    = iirfilter(17, [20], rs=60, btype='lowpass',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
hilblp = sosfiltfilt(sos_hlp, h_signal)

df_idx = find_nearest(frequencies,frequency)
fft_emg = fft(ae_emg_signal[start_pause:end_pause])
fft_emg = np.abs(2.0/(end_pause-start_pause) * (fft_emg))[1:(end_pause-start_pause)//2]
#

fft_h = fft(hilblp[start_pause:end_pause])
fft_h = np.abs(2.0/(end_pause-start_pause) * (fft_h))[1:(end_pause-start_pause)//2]

print ('ae df:',2*fft_emg[df_idx])
# fontsize on plots
f       = 18
vlim    = 100
plim    = 100 
siglim  = 8000
emglim  = 200
# 
# Turn interactive plotting off
# plt.ioff()
# 
# # # # # EMG hilbert transform plots # # # # 
factor = 50 
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(211)
plt.plot(t,factor*ae_emg_signal,'r')
plt.plot(t,factor*hilblp,'b')
plt.plot(t,ae_low_signal,'k')
ax.set_ylim([-siglim,siglim])
ax.set_xlim([0,duration])
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax2 = fig.add_subplot(212)
plt.plot(frequencies,fft_h,'k')
ax2.set_xlim([0,50])
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'hilbert_ae_signal.png'
plt.savefig(plot_filename)
plt.show()
# 
# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(t,ae_emg_signal,'k')
# ax.set_ylim([-emglim,emglim])
# ax.set_xlim([0,duration])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'hilbert_ae_emg_signal.png'
# plt.savefig(plot_filename)
# plt.show()

# # # # # # AE plots # # # # 
