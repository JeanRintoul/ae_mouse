'''

Title: vep signal inspection
Function: takes a single file, and averages the veps to see them better. 

Author: Jean Rintoul
Date:   23.10.2023

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
import scipy.stats
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 
# 
savepath       = 'D:\\ae_mouse\\e116_estim\\t4_epulse_mouse\\'

# prf = 0.5
# pulse_length   = 0.01
# file_list      = [26,27,28,29,30,31,32,33,34,35]  # 100ms
# # 'current_burst_length':0.0002,  # 200 microseconds.
# 'current_burst_length':0.001,   # 1ms 
# 'current_burst_length':0.050,   # 50ms 
# 'current_burst_length':0.1,     # 100ms 
# 'current_burst_length':0.01,     # 10ms 

prf            = 2
pulse_length   = 0.0002
file_list      = [1,2,3,4,5]       # 200 microsecond
pulse_length   = 0.001
file_list      = [6,7,8,9,10]      # 1ms
pulse_length   = 0.01
file_list      = [21,22,23,24,25]  # 10ms
# pulse_length   = 0.05
# file_list      = [11,12,13,14,15]  # 50ms
# pulse_length   = 0.1
# file_list      = [16,17,18,19,20]  # 100ms

# 
# 
gain            = 100
duration        = 6.0	
# 
Fs              = 5e6
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 4
start                           = 0.2*(1.0/prf)  # 0.2 seconds
end                             = 0.8*(1.0/prf)
pre_event_idxcount              = int(start*Fs)
post_event_idxcount             = int(end*Fs)
t = np.linspace(0, duration, N, endpoint=False)
print ('len t',len(t))
df_l = 0.1 # dfx-2
df_l = 2 # dfx-2
df_h = 10 
# df_h = 1500
# df_h = 40
if df_l <= 0: 
    df_l = 0.05 
sos_lfp_band = iirfilter(17, [df_l,df_h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
pulse_indices = int(pulse_length*Fs)
print ('pulse indices',pulse_indices)

for n in range(len(file_list)):
    file_number = file_list[n] 
    filename    = savepath + 't'+str(file_number)+'_stream.npy'
    data = np.load(filename)
    a,b = data.shape    
    # print ('shape',a,b)
    # convert it to microvolts by taking the gain into account. 
    fsignal     = 1e6*data[m_channel]/gain
    vsignal     = data[6] 
    # 
    marker  = vsignal/np.max(vsignal)
    diffs   = np.diff(marker)
    # print ('diffs max: ', np.max(diffs))
    zarray  = t*[0]
    indexes = np.argwhere(diffs > 0.2)[:,0]
    X       = np.insert(indexes, 0, 0)
    m       = np.diff(X)
    # print ('m',m)
    m       = np.argwhere(m > 1000)[:,0]
    alignment_indices = indexes[m]
    # print ('the alignment_indices are:',alignment_indices)
    zarray[alignment_indices] = -200 
    # print ('len',len(alignment_indices))
    start_pause = int(alignment_indices[1]-1000)
    end_pause   = int(start_pause + alignment_indices[len(alignment_indices)-1]+1000)
    # 
    # set the pulse data to zero. 
    for i in range(len(alignment_indices)):  # 
        # fsignal[alignment_indices[i]:(alignment_indices[i]+pulse_indices+400)] = 0 
        fsignal[alignment_indices[i]:(alignment_indices[i]+pulse_indices+400)] = fsignal[(alignment_indices[i]-(pulse_indices+400) ):alignment_indices[i]]


    
    dbit = fsignal[start_pause:end_pause]
    # print ('len dbit: ', len(dbit) )
    if n == 0: 
        nffty = dbit
    else:
        nffty = np.concatenate((nffty,dbit))
# 
print ('fft ',len(nffty))
fft_data        = fft(nffty)
N               = len(nffty)
fft_data        = np.abs(2.0/(N-1) * (fft_data))[1:(N-1)//2]
xf              = np.fft.fftfreq( (N-1), d=timestep)[:(N-1)//2]
frequencies     = xf[1:(N-1)//2]
# 
# 
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_data,'k')
ax.set_xlim([0,40])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)    
plot_filename = 'direct_stim_prf_'+str(prf)+'_fft_'+str(pulse_length)+'.png'
plt.savefig(plot_filename)
plt.show()

