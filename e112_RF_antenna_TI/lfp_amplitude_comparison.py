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
# 
# 
phantom_savepath    = 'D:\\ae_mouse\\e112_RF_antenna_TI\\t1_phantom\\'
pfile_number        = 20
# 
# 
savepath       	    = 'D:\\ae_mouse\\e112_RF_antenna_TI\\t2_mouse\\'
file_number         = 27
# 
# frequency trend in fft. 
# file: 
# f: 3	/ 5  /  10   /  40 / 100
# a: 154,  / 95,  / 46.2  / 40.3  / 41
# It is highly dependent on the timing of the k wearing off. 
# 
# 
duration        = 8.0	
gain            = 1000
Fs              = 5e6
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 4

phantom_filename    = savepath + 't'+str(pfile_number)+'_stream.npy'
pdata = np.load(phantom_filename)
# 
filename    = savepath + 't'+str(file_number)+'_stream.npy'
data = np.load(filename)
a,b = data.shape
print ('shape',a,b)
t = np.linspace(0, duration, N, endpoint=False)
print ('len t',len(t))
# convert it to microvolts by taking the gain into account. 
fsignal  = 1e6*data[m_channel]/gain
pfsignal = 1e6*pdata[m_channel]/gain
vsignal  = data[v_channel]
pvsignal = pdata[v_channel]
# rfsignal = 10*data[rf_channel] 
# 
# find the fft of the data. 
start_pause     = int(1 * Fs)
end_pause       = int(5 * Fs)
# 
# 
fft_data        = fft(fsignal[start_pause:end_pause])
fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]

fft_pdata        = fft(pfsignal[start_pause:end_pause])
fft_pdata        = np.abs(2.0/(end_pause-start_pause) * (fft_pdata))[1:(end_pause-start_pause)//2]


# fft_rfdata        = fft(rfsignal[start_pause:end_pause])
# fft_rfdata        = np.abs(2.0/(end_pause-start_pause) * (fft_rfdata))[1:(end_pause-start_pause)//2]

fft_vdata        = fft(vsignal[start_pause:end_pause])
fft_vdata        = np.abs(2.0/(end_pause-start_pause) * (fft_vdata))[1:(end_pause-start_pause)//2]
xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies     = xf[1:(end_pause-start_pause)//2]
# 
# Downsample the signal for faster plotting times.  
new_Fs                        = 5e6
downsampling_factor           = int(Fs/new_Fs)
fsignal = fsignal[::downsampling_factor]
pfsignal = pfsignal[::downsampling_factor]
vsignal = vsignal[::downsampling_factor]
pvsignal = pvsignal[::downsampling_factor]
t = t[::downsampling_factor]
Fs = new_Fs
#
df_l = 0.5 # dfx-2
df_h = 300
if df_l <= 0: 
	df_l = 0.05 
sos_lfp_band = iirfilter(17, [df_l, df_h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
lfp_data     	= sosfiltfilt(sos_lfp_band, fsignal)
lfp_pdata        = sosfiltfilt(sos_lfp_band, pfsignal)
lfp_v_data      = sosfiltfilt(sos_lfp_band, vsignal)
# 

df_l = 300 
df_h = 1500
spike_band = iirfilter(17, [df_l, df_h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

spike_data        = sosfiltfilt(spike_band, fsignal)
spike_pdata        = sosfiltfilt(spike_band, pfsignal)

mains = np.linspace(300,1500,25)
print ('mains',mains)
for i in range(len(mains)):
    f           = mains[i]
    mains_stop  = iirfilter(17, [f-2, f+2], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
    spike_data  = sosfiltfilt(mains_stop, spike_data)
    spike_pdata  = sosfiltfilt(mains_stop, spike_pdata)

# 
label1 = 'early'
label2 = 'late'
# Do some good sub-sampling. 
# 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(411)
plt.plot(frequencies,fft_pdata,'m')
plt.plot(frequencies,fft_data,'k')
# ax.set_xlim([0,110])
ax.set_xlim([0,1500])
plt.legend([label1,label2],loc='upper right')

ax2 = fig.add_subplot(412)
plt.plot(t,lfp_pdata,'m')
plt.plot(t,lfp_data,'k')
# plt.plot(t,fsignal,'k')
# plt.plot(t,pfsignal,'m')
plt.legend([label1,label2],loc='upper right')

ax3 = fig.add_subplot(413)
plt.plot(t,spike_data,'k')
plt.plot(t,spike_pdata,'m')

ax3.set_xlim([0,8])
ax4 = fig.add_subplot(414)
plt.plot(t,vsignal,'r')
# plt.plot(t,pvsignal,'k')
plt.legend(['applied signal'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

plot_filename = 'mep_fft.png'
plt.savefig(plot_filename)
plt.show()
# 
# 
# isolate the spikes. 
spike_threshold = 20 
spike_dots      = t*[0]
spike_dots[spike_data > spike_threshold] = 1
# 
# start and end point to focus on. 
s = 1.1
e = 6.7

s = 0.0
e = 8.0

# s = 2.0
# e = 3.5
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# # plt.plot(t,df_data,'k')
# plt.plot(t,lfp_data,'k')
# plt.legend(['lfp'],loc='upper right')

# ax.set_xlim([s,e])
# ax2 = fig.add_subplot(312)
# plt.plot(t,spike_data,'k')
# plt.legend(['spikes'],loc='upper right')

# ax2.set_xlim([s,e])
# ax3 = fig.add_subplot(313)
# plt.plot(t,spike_dots,'.k')
# plt.legend(['spike dot'],loc='upper right')

# ax3.set_xlim([s,e])
# # Plot spike plot of time, versus period. 
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# # plot_filename = 'mep_artefact_data.png'
# # plt.savefig(plot_filename)
# plt.show()

# 
# Now divide the data up into periods, so I can see any trend over a period. 
# 
# find the peaks. Align peak to peak of 10Hz data? 


# 
