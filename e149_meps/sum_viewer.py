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
savepath             = 'D:\\ae_mouse\\e149_meps\\t2_mouse_temperature\\averaging_0.5hz\\'

savepath             = 'D:\\ae_mouse\\e149_meps\\t1_temperature_firstattempt\\lotsoaveraging_nobrainsig\\'

outpath              = savepath 
# 
ae_filename   = savepath + 'tae_g500_stream.npy'     # duration 6
p_filename    = savepath + 'tp_p0.15_g500_stream.npy'   # duration 6 
v_filename    = savepath + 'tv_v8_g500_stream.npy'  # duration 6
# 

data_list = [1,2,3,4,8,9,10,11,12,13,14,15,16,17]
data_list = [10,13,14]

data_list = [11,12,12]

data_list = [3,4,5,7,8,9]
data_list = [7,8,9]

duration        = 6
# 
frequency       = 1

m_channel       = 0 
rf_channel      = 4
v_channel       = 6
i_channel       = 5 
emg_channel     = 2 
# 
# gain          = 500
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
sos_emg_band    = iirfilter(17, [1000], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

def mains_stop(signal):
    mains = [50]    
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

for i in range(len(data_list)): 
    file = data_list[i]
    ae_filename = savepath +'ae'+str(file)+'\\'+'tae_g500_stream.npy' 
    print ('ae_filename',ae_filename)

    data           = np.load(ae_filename)
    ae_fsignal     = (1e6*data[m_channel]/brain_gain)
    ae_emgsignal   = (1e6*data[emg_channel]/emg_gain)
    ae_rfsignal    = 10*data[rf_channel]
    ae_vsignal     = 10*data[v_channel]
    ae_isignal     = -5*data[i_channel]/49.9 

    ae_low_signal  = sosfiltfilt(sos_low_band, ae_fsignal)
    ae_emg          = sosfiltfilt(sos_emg_band, ae_emgsignal)
    ae_emg_signal  = mains_stop(ae_emg)
    # ae_emg_signal = ae_emg
    ae_pp_amp      = np.max(ae_low_signal[start_idx:end_idx]) - np.min(ae_low_signal[start_idx:end_idx])
    print ('ae amplitude:',ae_pp_amp)

    if i == 0:
        emg_sum = ae_emg_signal 
        brain_sum = ae_low_signal
    else:
        emg_sum = emg_sum + ae_emg_signal
        brain_sum = brain_sum + ae_low_signal

    # p_filename = savepath + 'ae'+str(file)+'\\'+'tp_p0.15_g500_stream.npy' 
    # print ('p_filename',p_filename)

    # data          = np.load(p_filename)
    # p_fsignal     = (1e6*data[m_channel]/brain_gain)
    # p_emgsignal   = (1e6*data[emg_channel]/emg_gain)
    # p_rfsignal    = 10*data[rf_channel]
    # p_vsignal     = 10*data[v_channel]
    # p_low_signal  = sosfiltfilt(sos_low_band, p_fsignal)
    # p_emg  = sosfiltfilt(sos_emg_band, p_emgsignal)
    # p_emg_signal  = mains_stop(p_emg)
    # # p_emg_signal  = sosfiltfilt(sos_emg_stop, p_emg_signal)
    # p_pp_amp      = np.max(p_low_signal[start_idx:end_idx]) - np.min(p_low_signal[start_idx:end_idx])
    # print ('p amplitude:',p_pp_amp)

    # if i == 0:
    #     emg_psum = p_emg_signal 
    # else:
    #     emg_psum = emg_psum + p_emg_signal

n = len(data_list)

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(211)
plt.plot(t,emg_sum/n,'k')
ax.set_xlim([0,duration])
ax.set_ylim([-100,100])
plt.legend(['ae'],loc='upper right')
ax2 = fig.add_subplot(212)
# plt.plot(t,emg_psum,'k')
plt.plot(t,brain_sum/n,'r')
ax2.set_xlim([0,duration])
# ax2.set_ylim([-100,100])
plt.legend(['p'],loc='upper right')
plt.tight_layout()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plot_filename = outpath+'sum_viewer.png'
plt.savefig(plot_filename)
plt.show()
#
