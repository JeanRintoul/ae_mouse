'''

Title: what is the current? 
Author: Jean Rintoul
Date: 02.02.2023

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
from scipy.signal import kaiserord, lfilter, firwin, freqz
import pandas as pd
import sys
sys.path.append('D:\\mouse_aeti')  #  so that we can import from the parent folder. 
import mouse_library as m
# 
#
# 
# MOUSE: PRF with and without acoustic connection. 
# ac_file_number     = 23  # Connected 22-25
# nac_file_number     = 27  # Not Connected 26-29
# savepath        = 'D:\\mouse_aeti\\e99_frequency_mixing_analysis\\t2_mouse\\'
# 
# SALINE: PRF with and without acoustic connection. 
ac_file_number     = 3  # Connected 22-25
nac_file_number    = 4  # Not Connected 33-36
savepath        = 'D:\\mouse_aeti\\e99_frequency_mixing_analysis\\'
# 
# 
gain            = 1000
Fs              = 5e6
duration        = 4.0
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0   
rf_channel      = 4 
# Load in the file. 
ac_filename      = savepath + 't'+str(ac_file_number)+'_stream.npy'
ac_data          = np.load(ac_filename)
a,b              = ac_data.shape
# 
nac_filename     = savepath + 't'+str(nac_file_number)+'_stream.npy'
nac_data         = np.load(nac_filename)
a,b              = nac_data.shape
# 

t = np.linspace(0, duration, N, endpoint=False)
# 
ac_fsignal   = 1e6*ac_data[m_channel]/gain
ac_rfsignal  = 10*ac_data[rf_channel]
nac_fsignal  = 1e6*nac_data[m_channel]/gain
nac_rfsignal = 10*nac_data[rf_channel]
# 
low  = 1000 - 2
high = 1000 + 2
sos_df = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
ac_dfsignal    = sosfiltfilt(sos_df, ac_fsignal)
ac_rfdfsignal  = sosfiltfilt(sos_df, ac_rfsignal)
nac_dfsignal    = sosfiltfilt(sos_df, nac_fsignal)
nac_rfdfsignal  = sosfiltfilt(sos_df, nac_rfsignal)

low  = 1e6+1000 - 2
high = 1e6+1000 + 2
# low  = 50 - 5 # look at the mains noise too. 
# high = 50 + 5

sos_sf = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
ac_sfsignal     = sosfiltfilt(sos_sf, ac_fsignal)
ac_rfsfsignal   = sosfiltfilt(sos_sf, ac_rfsignal)
nac_sfsignal    = sosfiltfilt(sos_sf, nac_fsignal)
nac_rfsfsignal  = sosfiltfilt(sos_sf, nac_rfsignal)
# 

start_pause     = int(0.25 * N+1)
end_pause       = int(0.875 * N-1)
window          = np.hanning(end_pause-start_pause)
ac_fft_data        = fft(ac_fsignal[start_pause:end_pause])
ac_fft_data        = np.abs(2.0/(end_pause-start_pause) * (ac_fft_data))[1:(end_pause-start_pause)//2]
nac_fft_data       = fft(nac_fsignal[start_pause:end_pause])
nac_fft_data       = np.abs(2.0/(end_pause-start_pause) * (nac_fft_data))[1:(end_pause-start_pause)//2]

ac_fft_rfdata        = fft(ac_rfsignal[start_pause:end_pause])
ac_fft_rfdata        = np.abs(2.0/(end_pause-start_pause) * (ac_fft_rfdata))[1:(end_pause-start_pause)//2]
nac_fft_rfdata       = fft(nac_rfsignal[start_pause:end_pause])
nac_fft_rfdata       = np.abs(2.0/(end_pause-start_pause) * (nac_fft_rfdata))[1:(end_pause-start_pause)//2]



xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies     = xf[1:(end_pause-start_pause)//2]
# 
# 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
# plt.plot(t,ac_rfsignal,color='r')
plt.plot(t,ac_dfsignal,color='k')
plt.plot(t,ac_sfsignal,color='m')
plt.legend(['ac df','ac sf'],loc='upper right')
ax.set_xlim([0,duration])
ax.set_title('acoustically connected')
ax2 = fig.add_subplot(212)
# plt.plot(t,nac_rfsignal,color='r')
plt.plot(t,nac_dfsignal,color='k')
plt.plot(t,nac_sfsignal,color='m')
ax2.set_title('acoustically not connected')
plt.legend(['nac df','nac sf'],loc='upper right')
ax2.set_xlim([0,duration])
plot_filename = 'fmixtimeseries.png'
plt.savefig(plot_filename)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.show()
# 
# 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
plt.plot(frequencies,ac_fft_data,color='k')
plt.plot(frequencies,nac_fft_data,color='r')
plt.legend(['ac measurement data','nac measurement data'],loc= 'upper right')
ax.set_xlim([0,3500])
ax.set_ylim([0,5])
# plt.legend(['m chan'],loc='upper right')
ax2 = fig.add_subplot(212)
plt.plot(frequencies,ac_fft_rfdata,color='k')
plt.plot(frequencies,nac_fft_rfdata,color='r')
plt.legend(['rf ac','rf nac'],loc= 'upper right')
ax2.set_xlim([0,3500])
ax2.set_ylim([0,0.003])
ax.set_ylabel('Volts ($\mu$V)')
ax2.set_ylabel('Volts (V)')
# plt.legend(['rf ac','rf nac'],loc='upper right')
# ax.set_xlabel('Time(s)')
# plt.autoscale(enable=True, axis='x', tight=True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.title('ac vs nac')
plot_filename = 'acvnac.png'
plt.savefig(plot_filename)
plt.show()


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(frequencies,ac_fft_data,color='k')
# plt.plot(frequencies,nac_fft_data,color='r')
# ax.set_xlim([999990,1004010])
# ax.set_ylim([0,500])
# plt.legend(['v chan'],loc='upper right')
# ax2 = fig.add_subplot(212)
# plt.plot(frequencies,ac_fft_rfdata,color='k')
# plt.plot(frequencies,nac_fft_rfdata,color='r')
# ax2.set_xlim([999990,1004010])
# ax2.set_ylim([0,0.02])
# ax.set_ylabel('Volts ($\mu$V)')
# ax2.set_ylabel('Volts (V)')
# plt.legend(['rf chan'],loc='upper right')
# # ax.set_xlabel('Time(s)')
# # plt.autoscale(enable=True, axis='x', tight=True)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# # plt.title('VEP')
# # plot_filename = 'VEP.png'
# # plt.savefig(plot_filename)
# plt.show()