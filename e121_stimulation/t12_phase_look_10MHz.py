'''

Title: compare the data going into generator with the data coming out of generator. 

Author: Jean Rintoul
Date:   26.10.2023

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
from scipy.signal import hilbert
from scipy.signal import find_peaks
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 
# There appears to be no phase offset. 
# I want to save out the:  dlfp_data, and the 3 peak to peak measures, dfx, filenumber. 
# I also want to group it into an array, so I think I need a nested loop. 
# There would be 2 output npzs. One for mouse, one for phantom. 
# 
df_ramp_frequencies = [1,2,5,10,40,100,300,1000]
# savepath            = 'D:\\ae_mouse\\e121_stimulation\\t1_mouse\\'
# # mouse ramp files. 
# df_ramp_files1       = [3,4,5,6,7,8,9,10]
# df_ramp_files2       = [18,19,20,21,22,23,24,25]
# df_ramp_files3       = [26,27,28,29,30,31,32,33]
# df_ramp_files4       = [34,35,36,37,38,39,40,41]


element = 1

# phantom ramp files. 
savepath       = 'D:\\ae_mouse\\e121_stimulation\\t2_phantom\\'
df_ramp_files1       = [1,2,3,4,5,6,7,8]
df_ramp_files2       = [9,10,11,12,13,14,15,16]
df_ramp_files3       = [17,18,19,20,21,22,23,24]
df_ramp_files4       = [25,26,27,28,29,30,31,32]


duration        = 6.0   
m_channel       = 0 
rf_channel      = 2 
gain            = 500 
Fs              = 1e7
timestep        = 1.0/Fs
N               = int(Fs*duration)
t = np.linspace(0, duration, N, endpoint=False)




file_number=df_ramp_files1[element]
dfx = df_ramp_frequencies[element] 
print ('dfx/file',dfx,file_number)
filename    = savepath + 't'+str(file_number)+'_stream.npy'
data = np.load(filename)
a,b = data.shape
# print ('shape',a,b)
fsignal             = 1e6*data[m_channel]/gain
rfsignal            = 10*data[rf_channel]  
# 
fft_start = int(1.6*Fs)
fft_end   = int(4.3*Fs) 
fft_data        = fft(fsignal[fft_start:fft_end])
N = fft_end - fft_start
fft_data        = np.abs(2.0/(N-1) * (fft_data))[1:(N-1)//2]
xf              = np.fft.fftfreq( (N-1), d=timestep)[:(N-1)//2]
frequencies     = xf[1:(N-1)//2]
# dfx band filter. 
bandwidth = 3
bl = dfx - bandwidth
if (dfx-bandwidth) <= 0:
    bl = 0.1
bh = dfx + bandwidth
sos_dfx_band = iirfilter(17, [bl,bh], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# lfp band filter. 
df_h = 300
sos_lfp_band = iirfilter(17, [df_h], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# spike band filter. 
df_l = 300
df_h = 1500
sos_spike_band = iirfilter(17, [df_l, df_h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
lfp_data              = sosfiltfilt(sos_lfp_band, fsignal)
spike_data            = sosfiltfilt(sos_spike_band, fsignal)
dfx_data              = sosfiltfilt(sos_dfx_band, fsignal)

# decimate the data so it plots faster. 
new_Fs                        = 1e4
downsampling_factor           = int(Fs/new_Fs)
dt           = t[::downsampling_factor]
dlfp_data    = lfp_data[::downsampling_factor]
dspike_data  = spike_data[::downsampling_factor]
ddfx_data = dfx_data[::downsampling_factor]
# 
df_idx = find_nearest(frequencies,dfx)
lfp_height = np.max(lfp_data[fft_start:fft_end]) - np.min(lfp_data[fft_start:fft_end])
dfx_filtered_height = np.max(ddfx_data) - np.min(ddfx_data)
print ('fft df p-p, measured p-p',np.round(2*fft_data[df_idx],2),np.round(lfp_height,2 ),np.round(dfx_filtered_height,2) )

 

start = 0 
stop  = duration
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(dt,dlfp_data,'k')
# ax.set_xlim([start,stop])
# ax2 = fig.add_subplot(312)
# plt.plot(frequencies,fft_data,'k')
# ax2.set_xlim([0,dfx+40])
# # plt.legend(['from generator'],loc='upper right')
# ax3 = fig.add_subplot(313)
# plt.plot(dt,ddfx_data,'k')
# # ax3.set_xlim([start,stop])
# # plt.legend(['from recording chan'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plot_filename = 'whole_file_comparison.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()
# 

rfanalytical_signal  = hilbert(rfsignal )
rfamplitude_envelope = np.abs(rfanalytical_signal)
sos_lp = iirfilter(17, [2000], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
rfhilberted             = sosfiltfilt(sos_lp, rfamplitude_envelope)
rfh_data = rfhilberted [::downsampling_factor]

# 
# 
# 
# print ('peaks',peaks)
# 
# Find the average peak position for this frequency with respect to the original data. 
# 
start_time = 1.6
end_time   = 4.4
start_idx  = find_nearest(dt,start_time)
end_idx    = find_nearest(dt,end_time)
# 
r2 = rfh_data[start_idx:end_idx]
t2 = dt[start_idx:end_idx]
d2 = ddfx_data[start_idx:end_idx]
peaksr, _ = find_peaks(r2, prominence=5)
peaksd, _ = find_peaks(d2, prominence=5)
print ('lengths:', len(peaksr),len(peaksd) )
# 
print ('peaks: ',peaksr)
print ('peaks2: ',peaksd)


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(t2[peaksr], r2[peaksr],'or')
plt.plot(t2, r2,'r')
plt.plot(t2[peaksd], d2[peaksd],'ob')
plt.plot(t2, d2,'b')
ax.set_xlim([2,4])
plt.show()
# 
# positive means that the red peak is further to the right of the blue peak
# Mouse results. 
# dfx = 1, degree offset = + 30.22
# dfx = 2, degree offset = + 12.45
# dfx = 5, degree offset = + 5.73
# dfx = 5, degree offset = + 5.73
# dfx = 10, degree offset = -8.14
# dfx = 40, degree offset = -2.19
# 
# saline
# dfx = 1  degree offset = +7 degrees. 
# dfx = 2  degree offset = +0 degrees. 
# dfx = 5 
# dfx = 10  
# 
peak_offsets        = t2[peaksr]-t2[peaksd][1:] 
timestep            = t2[2]-t2[1]
peak_times          = peak_offsets
print ('peak times',peak_times)
period              = 1/dfx
peaks_degree_offset = 360*peak_times/period
mean_offset         = np.mean(peaks_degree_offset)
print ('peak_times degree offset: ', peaks_degree_offset,mean_offset)
print ('dfx: ', dfx)
print ('mean degree offset: ', mean_offset)
# 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(t2[peaksr], r2[peaksr],'or')
plt.plot(t2, r2,'r')
# 
plt.plot(t2[peaksd], d2[peaksd],'ob')
plt.plot(t2, d2,'b')
# 
ax.set_xlim([2,4])
plt.show()


# # How do we analyze the phase offsets? 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(dt,ddfx_data/np.max(ddfx_data),'k')
# plt.plot(dt,rfh_data/np.max(rfh_data),'r')

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # ax2.spines['right'].set_visible(False)
# # ax2.spines['top'].set_visible(False)
# # ax3.spines['right'].set_visible(False)
# # ax3.spines['top'].set_visible(False)
# plt.show()



# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(t,from_generator,'k')
# ax.set_xlim([start,stop])
# plt.legend(['from generator'],loc='upper right')
# ax2 = fig.add_subplot(212)
# plt.plot(t,from_recording_chan,'k')
# ax2.set_xlim([start,stop])
# plt.legend(['from recording chan'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = 'zoom_file_comparison.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()

