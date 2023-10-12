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
savepath       = 'D:\\ae_mouse\\e107_revision\\t6_phantom\\'
dfx  		   = 5
file_number    = 5
# duration       = 6.0	
# 
# 
# 
# savepath       	= 'D:\\ae_mouse\\e107_revision\\t5_mouse\\'
# dfx  		   	= 10
# # file_number   = 32 # 10Hz, looks like it works. v out was 4v. 600 microvolts. However, I think harmonics make this up and the actual height of the time series signal is bigger. i.e. 1000 microvolts. 
# # file_number   = 33 # 1Hz, v out was 4v. cannot see it. 
# # file_number   = 30 # 30Hz, v out was 4v. I can see some response at the start, but it is a bit hard to follow. 560 microvolts height in FFT. 840 microvolts heigh by eye. 
# # file_number 	= 31 # 100Hz, 584 microvolts in fft. 620 microvolts in time series. Where the amplitude changes there are spikes, but there is no spiking with the beat frequency. 
# # file_number 	= 29 # 5Hz, there is 5 hz but also a big 25hz. weird extra crap is here. 790 microvolts amplitude in fft at 5hz. The actual time series is all messed up due to the 25hz weirdness. 
# # file_number 	= 27 # 3Hz, 15 Hz is dominating. Cannot see it in FFT. 
# dfx             = 10
# # vouts at 2v. 
# # file_number 	= 17 # 10Hz, it is there, but I am not reaching threshold. 50 microvolts only. 
# # # Idea, am I hitting the charge injection limit at the lower frequencies, is the charge injection limit frequency dependent? 
# # file_number 	= 14 # dfx is 4. Big 20Hz... I am having this same weird problem. Think it is CIL 
# # file_number 	= 21 # dfx = 10, vout = 3 , doesnt look as good. not sure why. 
# # file_number 	= 26 # dfx = 5, just non linear artefact. 
# file_number 	= 32
# file_number   = 29
# file_number   = 25
# file_number 	= 32  # dfx = 2 

# frequency trend in fft. 
# file: 
# f: 	5   /  10    /  30   / 100
# a: 	790 /  600   / 560   / 584  in fft 
#

duration        = 6.0	
gain            = 100
Fs              = 5e6
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 4
# 
filename    = savepath + 't'+str(file_number)+'_stream.npy'
data = np.load(filename)
a,b = data.shape
print ('shape',a,b)
t = np.linspace(0, duration, N, endpoint=False)
print ('len t',len(t))
# convert it to microvolts by taking the gain into account. 
fsignal = 1e6*data[m_channel]/gain
vsignal = data[v_channel]
rfsignal = 10*data[rf_channel] 
# 
# find the fft of the data. 
start_pause     = int(0.25 * N+1)
end_pause       = int(0.75 * N-1)
start_pause     = int(1.4 * Fs)
end_pause       = int(3.2 * Fs)
# 
# 
fft_data        = fft(fsignal[start_pause:end_pause])
fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]
fft_rfdata        = fft(rfsignal[start_pause:end_pause])
fft_rfdata        = np.abs(2.0/(end_pause-start_pause) * (fft_rfdata))[1:(end_pause-start_pause)//2]
fft_vdata        = fft(vsignal[start_pause:end_pause])
fft_vdata        = np.abs(2.0/(end_pause-start_pause) * (fft_vdata))[1:(end_pause-start_pause)//2]
xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies     = xf[1:(end_pause-start_pause)//2]
# 
# 
start_pause     = int(3.26 * Fs)
end_pause       = int(4.6 * Fs)
fft_data2        = fft(fsignal[start_pause:end_pause])
fft_data2        = np.abs(2.0/(end_pause-start_pause) * (fft_data2))[1:(end_pause-start_pause)//2]
fft_rfdata2        = fft(rfsignal[start_pause:end_pause])
fft_rfdata2        = np.abs(2.0/(end_pause-start_pause) * (fft_rfdata2))[1:(end_pause-start_pause)//2]
fft_vdata2        = fft(vsignal[start_pause:end_pause])
fft_vdata2        = np.abs(2.0/(end_pause-start_pause) * (fft_vdata2))[1:(end_pause-start_pause)//2]
xf2              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies2     = xf2[1:(end_pause-start_pause)//2]




# 
# 
df_l = 0.5 # dfx-2
df_h = 100
if df_l <= 0: 
	df_l = 0.05 
sos_lfp_band = iirfilter(17, [df_l, df_h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

lfp_data     	= sosfiltfilt(sos_lfp_band, fsignal)
lfp_v_data      = sosfiltfilt(sos_lfp_band, vsignal)


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(311)
plt.plot(t,fsignal,'k')
plt.plot(t,lfp_data,'m')
ax2 = fig.add_subplot(312)
plt.plot(t,vsignal,'k')
plt.plot(t,lfp_v_data,'m')
ax3 = fig.add_subplot(313)
plt.plot(t,rfsignal,'k')
plot_filename = 'mep_artefact_data.png'
plt.savefig(plot_filename)
plt.show()

max_lim = 205

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(221)
plt.plot(frequencies,fft_data,'k')
ax.set_xlim([0,max_lim])
plt.legend(['ae signal'],loc='upper right')
ax2 = fig.add_subplot(222)
plt.plot(frequencies,fft_rfdata,'k')
ax2.set_xlim([0,max_lim])
plt.legend(['rf signal'],loc='upper right')



ax3 = fig.add_subplot(223)
plt.plot(frequencies2,fft_data2,'k')
ax3.set_xlim([0,max_lim])
plt.legend(['ae signal2'],loc='upper right')
ax4 = fig.add_subplot(224)
plt.plot(frequencies2,fft_rfdata2,'k')
ax4.set_xlim([0,max_lim])
plt.legend(['rf signa2l'],loc='upper right')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)


plot_filename = 'mep_artefact_data.png'
plt.savefig(plot_filename)
plt.show()

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(t,spike_data/np.max(spike_data),'k')
# plt.plot(t,df_data/np.max(df_data),'r')
# ax.set_xlim([0,duration])
# ax2 = fig.add_subplot(312)
# plt.plot(frequencies,fft_data,'k')

# ax3 = fig.add_subplot(313)
# plt.plot(t,df_data,'k')
# ax3.set_xlim([0,duration])

# plot_filename = 'spike_data.png'
# plt.savefig(plot_filename)
# plt.show()
# 
# 400 p-p was giving a good amplitude signal in dual acoustic meps. 
# 
# demodulated_signal = rfsignal * vsignal
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(t,1+rfsignal/np.max(rfsignal),'r')
# plt.plot(t,vsignal/np.max(vsignal),'m')
# plt.plot(t,-1-fsignal/np.max(fsignal),'k')
# plot_filename = 'draw_data.png'
# plt.savefig(plot_filename)
# plt.show()
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(t,1+rfsignal/np.max(rfsignal),'r')
# plt.plot(t,vsignal/np.max(vsignal),'m')
# plt.plot(t,-1-fsignal/np.max(fsignal),'k')
# ax2 = fig.add_subplot(312)
# plt.plot(t,demodulated_signal,'k')
# ax3 = fig.add_subplot(313)
# plt.plot(frequencies,fft_data,'k')
# plt.plot(frequencies,fft_rfdata,'r')
# # ax2.set_xlim([0,1100])
# # ax2.set_xlim([0,5])
# # ax2.set_ylim([0,20])
# # plot_filename = 'draw_data.png'
# # plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(frequencies,fft_data,'k')
# # plt.plot(frequencies,fft_rfdata,'r')
# ax.set_xlim([1000000-200,1000000+200])
# ax2 = fig.add_subplot(212)
# plt.plot(frequencies,fft_data,'k')
# # plt.plot(frequencies,fft_rfdata,'r')
# ax2.set_xlim([500000-200,500000+200])
# # ax2.set_xlim([0,5])
# # ax2.set_ylim([0,20])
# plt.show()

# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(t,1e6*data[m_channel]/gain,'k')
# ax.set_xlim([0,duration])
# # ax.set_xlim([1.755,1.756])
# ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xlabel('Time(s)')
# plt.legend(['measurement channel at focus'],loc='upper right')
# ax2 = fig.add_subplot(212)
# plt.plot(t,10*data[rf_channel],'r')
# ax2.set_xlim([0,duration])
# ax2.set_ylabel('Volts (V)')
# plt.legend(['rf monitor channel'],loc='upper right')
# # ax2.set_xlim([1.755,1.756])
# ax.set_xlabel('time (s)')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = 'draw_data.png'
# plt.savefig(plot_filename)
# plt.show()
# 

# 
# low  = 1000-5
# high = 1000 +5

# sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# f1signal  = sosfiltfilt(sos_low, fsignal)


# low  = 3000-5
# high = 3000 +5

# sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# f2signal  = sosfiltfilt(sos_low, fsignal)


# low  = 1001000 -5
# high = 1001000 +5

# sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# f3signal  = sosfiltfilt(sos_low, fsignal)


# low  = 500000 -5000
# high = 500000 +5000
# high = 10
# sos_low = iirfilter(17, [high], rs=60, btype='lowpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# f4signal  = sosfiltfilt(sos_low, fsignal)

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(t,f4signal,'k')
# # ax.set_xlim([1.755,1.756])
# plot_filename = 'dc_components.png'
# plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(t,f1signal,'k')
# plt.legend(['1kHz filtered'],loc='upper right')
# ax.set_xlim([1.755,1.756])
# ax2 = fig.add_subplot(212)
# plt.plot(t,fsignal,'m')
# ax2.set_xlim([1.755,1.756])
# plt.legend(['raw single pulse'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = 'dcomponents.png'
# plt.savefig(plot_filename)
# plt.show()
# # 
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_data,'k')
# # ax.set_xlim([0,1000000])
# ax.set_xlim([0,5000])
# ax.set_ylim([0,2.5])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = 'dFFT.png'
# plt.savefig(plot_filename)
# plt.show()
# # 
# 

