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
template_savepath            = 'D:\\ae_mouse\\e139_ae_neural_decoding\\'
# template_filename            = 'vep_template_t2.npz'
# print ('filename: ', template_filename)
# data                         = np.load(template_savepath+template_filename)
# template                     = data['template']
# 
l_cut       = 0.5
h_cut       = 40
# # # # # # # # # # # # #
t_series      = 'mouse'
savepath      = 'D:\\ae_mouse\\e139_ae_neural_decoding\\t4_mouse\\7hz_aevep\\'

prf                         = 7
match_frequency             = 7
# 
# 
file_num      = 30
# 
# 
Fs              = 2e6
# new_Fs          = 1e4 
timestep        = 1.0/Fs
duration        = 30
gain            = 1000
m_channel       = 0 
rf_channel      = 4
marker_channel  = 7 
pulse_length    = 0.0 # pulse length in seconds. 
N               = int(Fs*duration)

carrier         = 500000
t               = np.linspace(0, duration, N, endpoint=False)

filename    = savepath + 't'+str(file_num)+'_stream.npy'
data        = np.load(filename)
fsignal     = (1e6*data[m_channel]/gain)
fsignal     = fsignal - np.mean(fsignal)
rfsignal    = 10*data[rf_channel]
marker      = data[marker_channel]
# 

# 
fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(111)
plt.plot(t, fsignal,'k')
plt.show()
# 
