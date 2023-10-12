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

plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 
# savepath      = 'D:\\ae_mouse\\e107_revision\\t5_mouse\\'
# savepath      = 'D:\\ae_mouse\\e113_meps\\t1\\'
# 
savepath       = 'D:\\ae_mouse\\e115_F21_hydrophone\\t4_F21_phantom_ramp_test_with_gain100\\'
# 
# dfx  		   = 2
# file_number  = 21
dfx            = 5
file_number    = 2

# dfx            = 1
# file_number    = 12
# duration       = 6.0	
# 
# 
# 
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
fsignal = -1e6*data[m_channel]/gain
vsignal = data[v_channel]
rfsignal = 10*data[rf_channel] 
# 
# find the fft of the data. 
start_pause     = int(1.4 * Fs)
end_pause       = int(4.6 * Fs)
# 
# 
fft_data        = fft(fsignal[start_pause:end_pause])
fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]
fft_rfdata      = fft(rfsignal[start_pause:end_pause])
fft_rfdata      = np.abs(2.0/(end_pause-start_pause) * (fft_rfdata))[1:(end_pause-start_pause)//2]
fft_vdata       = fft(vsignal[start_pause:end_pause])
fft_vdata       = np.abs(2.0/(end_pause-start_pause) * (fft_vdata))[1:(end_pause-start_pause)//2]
xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies     = xf[1:(end_pause-start_pause)//2]
# 
# 
df_l = 0.5 # dfx-2
df_h = 300
if df_l <= 0: 
	df_l = 0.05 
sos_lfp_band = iirfilter(17, [df_l, df_h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

lfp_data     	= sosfiltfilt(sos_lfp_band, fsignal)
lfp_v_data      = sosfiltfilt(sos_lfp_band, 1e6*vsignal)
lfp_rf_data      = sosfiltfilt(sos_lfp_band, 1e6*rfsignal)

df_l = 400
df_h = 900
sos_spike_band = iirfilter(17, [df_l, df_h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

spike_data        = sosfiltfilt(sos_spike_band, fsignal)
spike_v_data      = sosfiltfilt(sos_spike_band, vsignal)
spike_rf_data      = sosfiltfilt(sos_spike_band, rfsignal)




fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(211)
plt.plot(frequencies,fft_data,'k')
# plt.plot(frequencies,fft_data,'k')
ax.set_xlim([0,1000])

ax2 = fig.add_subplot(212)
plt.plot(t,spike_data,'k')

# ax.set_xlim([499850,500050])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = 'fft_data.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# # plt.plot(t,fsignal,'k')
# plt.plot(t,lfp_data,'m')
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.set_xlim([0,duration])
# ax2 = fig.add_subplot(312)
# #plt.plot(t,vsignal,'k')
# plt.plot(t,lfp_v_data,'m')
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax2.set_xlim([0,duration])
# ax3 = fig.add_subplot(313)
# #plt.plot(t,rfsignal,'k')
# plt.plot(t,lfp_rf_data,'m')
# plot_filename = 'mep_artefact_data.png'
# ax3.set_xlim([0,duration])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plt.tight_layout()
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()

# fig_shape_1 = 6 
# fig_shape_2 = 2 

# fig = plt.figure(figsize=(fig_shape_1,fig_shape_2))
# ax = fig.add_subplot(111)
# plt.plot(t,vsignal,'k')
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = 'voltage_data.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()

# fig = plt.figure(figsize=(fig_shape_1,fig_shape_2))
# ax = fig.add_subplot(111)
# plt.plot(t,rfsignal,'k')
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = 'rf_data.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()


# fig = plt.figure(figsize=(fig_shape_1,fig_shape_2))
# ax = fig.add_subplot(111)
# plt.plot(t,spike_data,'k')
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = 'spike_data.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()


# fig = plt.figure(figsize=(fig_shape_1,fig_shape_2))
# ax = fig.add_subplot(111)
# plt.plot(t,lfp_data,'m')
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = 'lfp_data.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()


# ax3 = fig.add_subplot(313)
# plt.plot(frequencies,fft_data,'k')

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(221)
# plt.plot(frequencies,fft_data,'k')
# ax.set_xlim([0,max_lim])
# plt.legend(['ae signal'],loc='upper right')
# ax2 = fig.add_subplot(222)
# plt.plot(frequencies,fft_rfdata,'k')
# ax2.set_xlim([0,max_lim])
# plt.legend(['rf signal'],loc='upper right')



# ax3 = fig.add_subplot(223)
# plt.plot(frequencies2,fft_data2,'k')
# ax3.set_xlim([0,max_lim])
# plt.legend(['ae signal2'],loc='upper right')
# ax4 = fig.add_subplot(224)
# plt.plot(frequencies2,fft_rfdata2,'k')
# ax4.set_xlim([0,max_lim])
# plt.legend(['rf signa2l'],loc='upper right')

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# ax4.spines['right'].set_visible(False)
# ax4.spines['top'].set_visible(False)


# plot_filename = 'mep_artefact_data.png'
# plt.savefig(plot_filename)
# plt.show()

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

