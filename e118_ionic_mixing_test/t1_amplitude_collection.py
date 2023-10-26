'''

Save out 
df amplitude, 

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx

plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 
savepath       = 'D:\\ae_mouse\\e118_ionic_mixing_test\\t1\\'
dfx            = 10
file_number    = 31
# 
file_list = [26,27,28,29,30] # Direct 10Hz. 
# file_list = [31,32,33,34,35] # 2MHz modulated 10Hz. 

# print ('file_number:',file_number)
for i in range(len(file_list)):
    file_number = file_list[i]


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
    # 
    filename    = savepath + 't'+str(file_number)+'_stream.npy'
    data = np.load(filename)
    a,b = data.shape
    # print ('shape',a,b)
    t = np.linspace(0, duration, N, endpoint=False)
    # print ('len t',len(t))
    # convert it to microvolts by taking the gain into account. 
    fsignal     = 1e6*data[m_channel]/gain
    vsignal     = data[6] 

    # 
    # find the fft of the data. 
    start_pause     = int(1.4 * Fs)
    end_pause       = int(4.6 * Fs)
    # 
    # 
    fft_data        = fft(fsignal[start_pause:end_pause])
    fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]
    fft_vdata       = fft(vsignal[start_pause:end_pause])
    fft_vdata       = np.abs(2.0/(end_pause-start_pause) * (fft_vdata))[1:(end_pause-start_pause)//2]
    xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
    frequencies     = xf[1:(end_pause-start_pause)//2]
    # 
    # 
    carrier_idx = find_nearest(frequencies,500000)
    carrier_2idx = find_nearest(frequencies,500000+dfx)
    df_idx      = find_nearest(frequencies,dfx)

    print ('file number',str(file_number))
    print('df amplitude(microvolts)',2*fft_data[df_idx]) 



# # 
# # 
# df_l = 0.5 # dfx-2
# # df_l = 10 
# df_h = 300
# if df_l <= 0: 
# 	df_l = 0.05 
# sos_lfp_band = iirfilter(17, [df_l, df_h], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')

# lfp_data     	= sosfiltfilt(sos_lfp_band, fsignal)
# lfp_v_data      = sosfiltfilt(sos_lfp_band, 1e6*vsignal)
# # lfp_rf_data      = sosfiltfilt(sos_lfp_band, 1e6*rfsignal)
# # 
# df_l = 300
# df_h = 1500
# sos_spike_band = iirfilter(17, [df_l, df_h], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# spike_data        = sosfiltfilt(sos_spike_band, fsignal)
# spike_v_data      = sosfiltfilt(sos_spike_band, vsignal)
# spike_rf_data      = sosfiltfilt(sos_spike_band, rfsignal)
# 

# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(211)
# plt.plot(t,vsignal,'k')
# ax2  = fig.add_subplot(212)
# plt.plot(t,i_data,'r')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = 'rfti_impedance.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()


# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_data,'k')
# # ax.set_xlim([0,110])
# ax.set_xlim([0,110])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = 'rfti_fft.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()

# # Plots
# start = 1.5 
# stop  = 4.5

# start = 0
# stop  = duration
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(t,lfp_data,'k')
# # plt.plot(t,10*spike_data,'r')
# plt.plot(t,-250+100*vsignal/np.max(vsignal),'r')
# plt.legend(['lfp','applied'],loc='upper right')
# ax.set_xlim([start,stop])
# ax2 = fig.add_subplot(312)
# plt.plot(t,spike_data,'k')
# plt.plot(t,-100+10*vsignal/np.max(vsignal),'r')
# # 
# ax2.set_xlim([start,stop])
# plt.legend(['spikes','applied'],loc='upper right')
# ax3 = fig.add_subplot(313)
# plt.plot(t,fsignal/np.max(fsignal),'k')
# # plt.plot(t,vsignal/np.max(vsignal),'r')
# plt.legend(['raw measured signal'],loc='upper right')
# ax3.set_xlim([start,stop])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plot_filename = 'rfti_debugging.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()
# 
# 
# 

