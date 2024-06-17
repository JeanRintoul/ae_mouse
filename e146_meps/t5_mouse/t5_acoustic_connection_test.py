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
savepath             = 'D:\\ae_mouse\\e146_meps\\t5_mouse\\acoustic_connection_test\\'
# 
outpath              = savepath
# 
# 
n_repeats       = 1
m_channel       = 0 
rf_channel      = 4
v_channel       = 6
gain            = 500 
duration        = 6
band_limit      = 80 
Fs              = 5e6 
timestep        = 1/Fs
N               = int(duration*Fs)
t               = np.linspace(0, duration, N, endpoint=False)
# 
cut             = 1000
sos_low_band    = iirfilter(17, [cut], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
sos_emg_band    = iirfilter(17, [20,5000], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

# sos_mains_band    = iirfilter(17, [45,55], rs=60, btype='bandstop',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# 
start_idx       = int(0*Fs) 
end_idx         = int(6*Fs) 
newN            = int(end_idx-start_idx)
xf              = np.fft.fftfreq( (newN), d=timestep)[:(newN)//2]
frequencies     = xf[1:(newN)//2]
# 

def mains_stop(signal):
    mains = [50,100,150,250,350,400,450,500,550]
    for i in range(len(mains)):
        sos_mains_stop    = iirfilter(17, [mains[i]-2,mains[i]+2], rs=60, btype='bandstop',
                               analog=False, ftype='cheby2', fs=Fs,
                               output='sos')
        signal = sosfiltfilt(sos_mains_stop, signal)
    return signal

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   

vs      = [] 
amps    = []
flist   = []
dcs     = []
outname = 'ac_rep1'
frequency = 0.5 
filename = savepath + 'tp_p_0.1_acdc_rep1_stream.npy' 
filename = savepath + 'tp_p_0.1_ac_rep1_stream.npy' 

brain_gain  = 10 
emg_gain    = 500 
emg_channel = 2 

data        = np.load(filename)
fsignal     = (1e6*data[m_channel]/brain_gain)
emg_signal     = (1e6*data[emg_channel]/emg_gain)
rfsignal    = 10*data[rf_channel]
vsignal     = data[v_channel]

fft_raw     = fft(fsignal[start_idx:end_idx])
fft_raw     = np.abs(2.0/(newN) * (fft_raw))[1:(newN)//2]

df_idx      = find_nearest(frequencies,frequency)

final_idx   = find_nearest(frequencies,50)
dc_idx      = find_nearest(frequencies,0)
# 
low_signal        = sosfiltfilt(sos_low_band, fsignal)
nomains_signal    = mains_stop(emg_signal)
emg_signal        = sosfiltfilt(sos_emg_band, nomains_signal)

st = int(2*Fs)
et = int(6*Fs)
pp_amp = np.max(low_signal[st:et]) - np.min(low_signal[st:et])

# 
amplitude   = fft_raw[df_idx]
dc          = fft_raw[dc_idx]
print ('amplitude:',amplitude*2,pp_amp)
dcs.append(dc)
amps.append(pp_amp)
flist.append(fft_raw[0:final_idx])
# 
f = 18
# 
fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t, rfsignal,'grey')
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.set_xlim([0,duration])
ax.set_ylim([-25,25])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = savepath + outname+'_rf.png'
plt.savefig(plot_filename)
plt.show()

# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(t, vsignal,'pink')
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
# ax.set_ylim([-50,50])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = savepath + outname+'_v.png'
# plt.savefig(plot_filename)
# plt.show()

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t,low_signal,'k')
ax.set_xlim([0,duration])
ax.set_ylim([-1000,5000])
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = savepath + outname+'_signal.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t,emg_signal,'k')
ax.set_xlim([0,duration])
ax.set_ylim([-15,15])
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = savepath + outname+'_emg_signal.png'
plt.savefig(plot_filename)
plt.show()


# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_raw,'k')
# ax.set_xlim([0,20])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = savepath + outname+'_fft.png'
# plt.savefig(plot_filename)
# plt.show()






