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
# 
prf            = 2
pulse_length   = 0.0002
file_list      = [1,2,3,4,5]       # 200 microsecond
# pulse_length   = 0.001
# file_list      = [6,7,8,9,10]      # 1ms
# pulse_length   = 0.01
# file_list      = [21,22,23,24,25]  # 10ms
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
# df_l = 3 # dfx-2
df_h = 300
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
    start_pause = int(alignment_indices[1]-100000)
    end_pause   = int(start_pause + 5*Fs+1000)
    # # 
    # fig = plt.figure(figsize=(10,6))
    # ax  = fig.add_subplot(111)
    # plt.plot(t,fsignal,'k')
    # plt.show()
    # set the pulse data to zero. 
    for i in range(len(alignment_indices)):  # 
        # fsignal[alignment_indices[i]:(alignment_indices[i]+pulse_indices+400)] = 0 
        fsignal[alignment_indices[i]:(alignment_indices[i]+pulse_indices+400)] = fsignal[(alignment_indices[i]-(pulse_indices+400) ):alignment_indices[i]]
    # 
    lfp_data = sosfiltfilt(sos_lfp_band, fsignal)
    dbit = lfp_data[start_pause:end_pause]

    new_Fs                        = 5e6
    downsampling_factor           = int(Fs/new_Fs)
    # nsignal = lfp_data[::downsampling_factor]
    tt       = t[start_pause:end_pause][::downsampling_factor]
    dbit     = dbit[::downsampling_factor]
    vsignalt = vsignal[start_pause:end_pause][::downsampling_factor]
    # print ('len dbit: ', len(dbit) )
    if n == 0: 
        nffty = dbit
    else:
        nffty = np.vstack(( nffty ,dbit))
    print (nffty.shape)

# Downsample the signal 
new_Fs                        = 1000
downsampling_factor           = int(Fs/new_Fs)
#
#
# average_lfp  = average_lfp[::downsampling_factor]


time_segment    = tt[::downsampling_factor] 
vbit            = vsignalt[::downsampling_factor]


print ('what shape',np.array(nffty).shape )
average_lfp     = np.mean(nffty,axis=0)
std_lfp         = np.std(nffty,axis=0)
sem_lfp         = np.std(nffty,axis=0)/np.sqrt(len(file_list))
std_lfp         = sem_lfp
average_lfp =average_lfp[::downsampling_factor]
std_lfp =std_lfp[::downsampling_factor]
start_time      = 0
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(time_segment,average_lfp,color='r')
plt.fill_between(time_segment, average_lfp - std_lfp, average_lfp + std_lfp,
                  color='gray', alpha=0.2)
plt.plot(time_segment,10*vbit,'k')
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Time(s)')
# plt.autoscale(enable=True, axis='x', tight=True)
# ax2 = fig.add_subplot(212)
# plt.plot(time_segment,nfiltered_segmented_array.T)
# plt.axvline(x=start_time,color ='k')
# # ax2.text(0.5, 100, 'individual trials', color='black',fontsize = 6, ha='center')
# plt.autoscale(enable=True, axis='x', tight=True)
# ax2.set_ylabel('Individual trials ($\mu$V)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plot_filename = 'direct_stim_prf_'+str(prf)+'_wholefile_'+str(n_events)+'.png'
# plt.savefig(plot_filename)
plt.show()
# # 
# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_data,'k')
# ax.set_xlim([0,40])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)    
# plot_filename = 'direct_stim_prf_'+str(prf)+'_fft_'+str(pulse_length)+'.png'
# plt.savefig(plot_filename)
# plt.show()

