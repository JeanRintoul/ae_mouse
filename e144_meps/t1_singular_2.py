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
savepath             = 'D:\\ae_mouse\\e144_meps\\t3_mouse_badcupelectrodeconnection\\leave_one_out\\6v\\'
savepath             = 'D:\\ae_mouse\\e144_meps\\t1_phantom\\proposed_dualpulse_waveform\\'

# 
outpath              = 'D:\\ae_mouse\\e144_meps\\t1_phantom\\'
# 
outname = 'ae'
# 

n_repeats       = 1
m_channel       = 0 
rf_channel      = 4
v_channel       = 6
gain            = 200 
duration        = 10
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

frequency = 0.5 


# filename    = savepath + 'v6.npy'  # duration 10

filename    = savepath + 'pressure0.2_v12.npy' # ae duraiton 10
# filename    = savepath + 'pressure0.2_v0.npy' # pressure 10s
# filename    = savepath + 'nopressure_v12.npy' # voltage only 10s


# filename    = savepath + 'ae_p0.2_v6.npy' # ae duraiton 6
# filename    = savepath + 'p_p0.2_v0.npy' # pressure 10s
# filename    = savepath + 'v_v6_p0.npy' # voltage only 10s


data        = np.load(filename)
fsignal     = (1e6*data[m_channel]/gain)
rfsignal    = 10*data[rf_channel]
vsignal     = data[v_channel]

fft_raw     = fft(fsignal[start_idx:end_idx])
fft_raw     = np.abs(2.0/(newN) * (fft_raw))[1:(newN)//2]

df_idx      = find_nearest(frequencies,frequency)

final_idx   = find_nearest(frequencies,50)
dc_idx      = find_nearest(frequencies,0)
# 
low_signal    = sosfiltfilt(sos_low_band, fsignal)

st = int(2*Fs)
et = int(4*Fs)
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
ax.set_ylim([-100,100])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = outpath + outname+'_rf.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t, vsignal,'pink')
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.set_ylim([-20,20])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = outpath + outname+'_v.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t,low_signal,'k')
ax.set_ylim([-300,300])
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = outpath + outname+'_signal.png'
plt.savefig(plot_filename)
plt.show()



fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(frequencies,fft_raw,'k')
ax.set_xlim([0,20])
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = outpath + outname+'_fft.png'
plt.savefig(plot_filename)
plt.show()






