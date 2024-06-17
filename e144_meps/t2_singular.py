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
savepath             = 'D:\\ae_mouse\\e144_meps\\t2_mouse_good\\leave_one_out\\'
# 
outpath              = 'D:\\ae_mouse\\e144_meps\\'
# 
outname = 'loo_test'
# 
# 


n_repeats       = 1
m_channel       = 0 
rf_channel      = 4
gain            = 200 
duration        = 6
band_limit      = 80 
Fs              = 5e6 
timestep        = 1/Fs
N               = int(duration*Fs)
t               = np.linspace(0, duration, N, endpoint=False)
# 
cut             = 50
sos_low_band    = iirfilter(17, [0.3,cut], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
start_idx       = int(0*Fs) 
end_idx         = int(6*Fs) 
newN            = int(end_idx-start_idx)
xf              = np.fft.fftfreq( (newN), d=timestep)[:(newN)//2]
frequencies     = xf[1:(newN)//2]
# 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   

vs      = [] 
amps    = []
flist   = []
dcs     = []

frequency = 1 

filename    = savepath + 'p0_v12.npy'    # duration 6
# filename    = savepath + 'v12_p0.2.npy'  # duration 6 
# filename    = savepath + 'v0p0.2.npy'  # duration 10
filename = savepath + 'p0.2v12_3_movements.npy' 


data        = np.load(filename)
fsignal     = (1e6*data[m_channel]/gain)
rfsignal    = 10*data[rf_channel]

fft_raw     = fft(fsignal[start_idx:end_idx])
fft_raw     = np.abs(2.0/(newN) * (fft_raw))[1:(newN)//2]

df_idx      = find_nearest(frequencies,frequency)

final_idx   = find_nearest(frequencies,50)
dc_idx      = find_nearest(frequencies,0)
# 
low_signal    = sosfiltfilt(sos_low_band, fsignal)

st = int(2*Fs)
et = int(8*Fs)
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
fig = plt.figure(figsize=(4,1))
ax = fig.add_subplot(111)
plt.plot(t, rfsignal,'grey')
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = savepath + 'rf.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(211)
plt.plot(t,low_signal,'k')
ax.set_ylim([-250,350])
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax2 = fig.add_subplot(212)
plt.plot(frequencies,fft_raw,'k')
ax2.set_xlim([0,45])
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = savepath + 'stuff.png'
plt.savefig(plot_filename)
plt.show()






