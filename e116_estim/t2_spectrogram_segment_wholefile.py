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
# df frequencies. 
# 
# 
savepath       = 'D:\\ae_mouse\\e116_estim\\t2\\'
# dfx = 1 
# file_list       = [43,49,56]  # df = 1
# dfx = 2 
# file_list       = [44,50,57]  # df = 2 can see easily in the FFT. 
# dfx = 5
# file_list       = [45,51,58]  # df = 5 
# dfx = 10 
# file_list       = [46,52,59]  # df = 10
# dfx = 40 
# file_list       = [47,53,60]  # df = 40
dfx = 100 
file_list       = [48,54,61]  # df = 100
# df = 4

# file_list       = [25,26,27,28,29] # prf = 5
# file_list       = [20,21,22,23,24] # prf = 2
# file_list       = [14,15,16,17,18,19] # prf = 1
# 
gain            = 1000
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

rf_channel = 6 

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

# pulse_indices = int(pulse_length*Fs)
# print ('pulse indices',pulse_indices)

for n in range(len(file_list)):
    file_number = file_list[n] 
    filename    = savepath + 't'+str(file_number)+'_stream.npy'
    data = np.load(filename)
    a,b = data.shape    
    # print ('shape',a,b)
    # convert it to microvolts by taking the gain into account. 
    fsignal      = 1e6*data[m_channel]/gain
    rfsignal     = data[rf_channel] 
    # 
    # mains_harmonics = [50,100,150,200,250]
    # for i in range(len(mains_harmonics)):
    #     mains_low  = mains_harmonics[i] - 2
    #     mains_high = mains_harmonics[i] + 2
    #     mains_sos = iirfilter(17, [mains_low,mains_high], rs=60, btype='bandstop',
    #                        analog=False, ftype='cheby2', fs=Fs,
    #                        output='sos')
    #     fsignal = sosfiltfilt(mains_sos, fsignal)
    df_l = 0.1 # dfx-2
    # df_l = 3 # dfx-2    
    # df_h = 10
    df_h = 300
    # df_h = 40
    if df_l <= 0: 
        df_l = 0.05 
    sos_lfp_band = iirfilter(17, [df_l, df_h], rs=60, btype='bandpass',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
    lfp_data     = sosfiltfilt(sos_lfp_band, fsignal)
    # 
    marker  = rfsignal/np.max(rfsignal)
    # diffs   = np.diff(marker)
    # # print ('diffs max: ', np.max(diffs))
    # zarray  = t*[0]
    # indexes = np.argwhere(diffs > 0.2)[:,0]
    # X       = np.insert(indexes, 0, 0)
    # m       = np.diff(X)
    # # print ('m',m)
    # m       = np.argwhere(m > 1000)[:,0]
    # alignment_indices = indexes[m]
    # print ('the alignment_indices are:',alignment_indices)
    # zarray[alignment_indices] = -200 
    # 
    start_pause = 0 
    end_pause = int(N)
    # # 
    # fig = plt.figure(figsize=(10,6))
    # ax  = fig.add_subplot(111)
    # plt.plot(t,fsignal,'k')
    # plt.show()
    # set the pulse data to zero. 
    # for i in range(len(alignment_indices)):  # 
    #     # fsignal[alignment_indices[i]:(alignment_indices[i]+pulse_indices+400)] = 0 
    #     fsignal[alignment_indices[i]:(alignment_indices[i]+pulse_indices+400)] = fsignal[(alignment_indices[i]-(pulse_indices+400) ):alignment_indices[i]]
    # 
    dbit = lfp_data[start_pause:end_pause]
    # 
    new_Fs                        = 5e6
    downsampling_factor           = int(Fs/new_Fs)
    # nsignal = lfp_data[::downsampling_factor]
    tt       = t[start_pause:end_pause][::downsampling_factor]
    dbit     = dbit[::downsampling_factor]
    vsignalt = rfsignal[start_pause:end_pause][::downsampling_factor]
    # print ('len dbit: ', len(dbit) )
    if n == 0: 
        nffty = dbit
    else:
        nffty = np.vstack(( nffty ,dbit))
    print (nffty.shape)

time_segment    = tt 
vbit            = vsignalt
# times           = t[alignment_indices]

# print ('times',times)
print ('what shape',np.array(nffty).shape )
average_lfp     = np.mean(nffty,axis=0)
std_lfp         = np.std(nffty,axis=0)
sem_lfp         = np.std(nffty,axis=0)/np.sqrt(len(file_list))
std_lfp         = sem_lfp
start_time      = 0

fft_data        = fft(average_lfp)
N               = len(average_lfp)
fft_data        = np.abs(2.0/(N-1) * (fft_data))[1:(N-1)//2]
xf              = np.fft.fftfreq( (N-1), d=timestep)[:(N-1)//2]
frequencies     = xf[1:(N-1)//2]
#
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_data,'k')
ax.set_xlim([0,110])
plot_filename = 'rf_fft_dfx_'+str(dfx)+'_wholefile_.png'
plt.savefig(plot_filename)
plt.show()
#
print ('lengths',len(time_segment),len(vbit))
#
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(time_segment,vbit,'k')
plt.plot(time_segment,average_lfp,color='r')
# plt.fill_between(time_segment, average_lfp - std_lfp, average_lfp + std_lfp,
#                   color='gray', alpha=0.2)

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
plt.tight_layout()
plot_filename = 'rf_stim_dfx_'+str(dfx)+'_wholefile_.png'
plt.savefig(plot_filename)
plt.show()

new_Fs                        = 10000
downsampling_factor           = int(Fs/new_Fs)
average_lfp  = average_lfp[::downsampling_factor]
time_segment = time_segment[::downsampling_factor]
vbit         = vbit[::downsampling_factor]
Oz = average_lfp
fs = new_Fs
print ('total available to nperseg:',len(Oz))  # 120000
nperseg = len(Oz)-1
nperseg = 1000
noverlap = nperseg-1

noverlap = 0
f350, t350, Sxx = signal.spectrogram(Oz, fs, nperseg=nperseg , noverlap=noverlap,window=signal.get_window('hann',nperseg),mode='psd',scaling='density')
vvmin = 0 
vvmax = np.max(Sxx)/2
# # 
# # start_time      = 0.2*(1.0/prf)  # start of first pulse
# # start2_time     = 0.2*(1.0/prf)+(1.0/prf)  # start of second pulse
# # 
print(Sxx.shape,t350.shape,f350.shape)
vsum = np.sum(Sxx,axis=0)
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(time_segment,10*vbit-10,'k')
plt.plot(t350,vsum)

# # plot the alignment indices
# for i in range(len(alignment_indices)):
#     plt.axvline(x=t[alignment_indices[i]],color ='r')
plt.show()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
im = plt.pcolormesh(t350, f350, Sxx, shading='auto',cmap = 'inferno',vmin=vvmin,vmax=vvmax)
plt.ylim([0,110])
plt.plot(time_segment,vbit+80,'white')
plt.xlim([0,np.max(time_segment) ])
# plot the start time of the pulse. 
# plt.axvline(x=start_time,color ='r')
# plt.axvline(x=start2_time,color ='r')
# 
# plot the alignment indices
# for i in range(len(alignment_indices)):
#     plt.axvline(x=t[alignment_indices[i]],color ='r')
# 
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# fig.colorbar(im).set_label('Intensity (dB)')
plot_filename = 'dfx_'+str(dfx)+'_spectrogram.png'
plt.savefig(plot_filename, bbox_inches='tight')
plt.show()
# 
# vertically summ the spectral power. 






