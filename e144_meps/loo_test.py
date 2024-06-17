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
savepath             = 'D:\\ae_mouse\\e144_meps\\t4_mouse_pulse\\'
outpath              = 'D:\\ae_mouse\\e144_meps\\images\\'
# 
# ae_filename   = savepath + '2-54-p0.35_v12_g500_4.npy'     # duration 6
ae_filename   = savepath + '2-49-ae_p0.35_v12_g500.npy'     # duration 6
p_filename    = savepath + '2-48-p0.35_g500_x.npy'       # duration 6 
v_filename    = savepath + '2-53-v12_p0_g500.npy'           # duration 6

frequency       = 0.5 
m_channel       = 0 
rf_channel      = 4
v_channel       = 6
i_channel       = 5 
gain            = 500 
duration        = 6
band_limit      = 80 
Fs              = 5e6 
timestep        = 1/Fs
N               = int(duration*Fs)
t               = np.linspace(0, duration, N, endpoint=False)
cut             = 1000
sos_low_band    = iirfilter(17, [cut], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# get the frequencies ready
start_idx       = int(0*Fs) 
end_idx         = int(6*Fs) 
# 
data           = np.load(ae_filename)
ae_fsignal     = (1e6*data[m_channel]/gain)
ae_rfsignal    = 10*data[rf_channel]
ae_vsignal     = data[v_channel]
ae_low_signal  = sosfiltfilt(sos_low_band, ae_fsignal)
ae_pp_amp = np.max(ae_low_signal[start_idx:end_idx]) - np.min(ae_low_signal[start_idx:end_idx])
print ('ae amplitude:',ae_pp_amp)

data           = np.load(p_filename)
p_fsignal     = (1e6*data[m_channel]/gain)
p_rfsignal    = 10*data[rf_channel]
p_vsignal     = data[v_channel]
p_low_signal  = sosfiltfilt(sos_low_band, p_fsignal)
p_pp_amp = np.max(p_low_signal[start_idx:end_idx]) - np.min(p_low_signal[start_idx:end_idx])
print ('p amplitude:',p_pp_amp)

data           = np.load(v_filename)
v_fsignal     = (1e6*data[m_channel]/gain)
v_rfsignal    = 10*data[rf_channel]
v_vsignal     = data[v_channel]
v_low_signal  = sosfiltfilt(sos_low_band, v_fsignal)
v_pp_amp = np.max(v_low_signal[start_idx:end_idx]) - np.min(v_low_signal[start_idx:end_idx])
print ('v amplitude:',v_pp_amp)

# fontsize on plots
f       = 18
vlim    = 30
plim    = 100 
siglim  = 500
# Turn interactive plotting off
# plt.ioff()

# # # # # AE plots # # # # 
fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t, ae_rfsignal,'grey')
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.set_ylim([-plim,plim])
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'ae_rf.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t, ae_vsignal,'pink')
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.set_ylim([-vlim,vlim])
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'ae_v.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t,ae_low_signal,'k')
ax.set_ylim([-siglim,siglim])
ax.set_xlim([0,duration])
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'ae_signal.png'
plt.savefig(plot_filename)
plt.show()
# plt.close(fig)

# # # # # P plots # # # # 
fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t, p_rfsignal,'grey')
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.set_ylim([-plim,plim])
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'p_rf.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t, p_vsignal,'pink')
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.set_ylim([-vlim,vlim])
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'p_v.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t,p_low_signal,'k')
ax.set_ylim([-siglim,siglim])
ax.set_xlim([0,duration])
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'p_signal.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)

# # # # # V plots # # # # 
fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t, v_rfsignal,'grey')
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.set_ylim([-plim,plim])
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'v_rf.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t, v_vsignal,'pink')
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.set_ylim([-vlim,vlim])
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'v_v.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t,v_low_signal,'k')
ax.set_ylim([-siglim,siglim])
ax.set_xlim([0,duration])
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'v_signal.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)

# # # # # END PLOTS # # # # 