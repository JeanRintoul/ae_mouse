"""
Aggregate summary files. 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import iirfilter,sosfiltfilt,sosfilt
from scipy.fft import fft,fftshift,ifft
import sys
sys.path.append('D:\\mouse_aeti')  #  so that we can import from the parent folder. 
import mouse_library as m

Fs = 1e5
frequency_of_interest = 70
sos_high               = iirfilter(17, [frequency_of_interest-2,frequency_of_interest+2], rs=60, btype='bandpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')

#  this data is somewhat filtered. 
filename          	= 'saline_acoustic_connection_summary_data.npz'
data              	= np.load(filename)
a_total_raw       	= np.array(data['total_rawdata'])
a_total_demod       = np.array(data['total_demod'])
a_frequencies       = data['frequencies']
a_total_rfft        = np.array(data['total_rfft'])
a_total_dfft        = np.array(data['total_dfft'])
a_total_rffft        = np.array(data['total_rffft'])
a_td       		    = data['td']

a,b = a_total_raw.shape
print ('acoustic connection shape',a_total_raw.shape)

nfilename          	= 'saline_no_acoustic_connection_summary_data.npz'
ndata              	= np.load(nfilename)
n_total_raw       	= np.array(ndata['total_rawdata'])
n_total_demod       = np.array(ndata['total_demod'])
n_frequencies       = ndata['frequencies']
n_total_rfft        = np.array(ndata['total_rfft'])
n_total_dfft        = np.array(ndata['total_dfft'])
n_total_rffft       = np.array(ndata['total_rffft'])
n_td      		    = ndata['td']

c,d = n_total_raw.shape
print ('no acoustic connection shape',n_total_raw.shape)
n_events,N 	= a_total_raw.shape
# 
a_average_raw       = np.mean(a_total_raw,axis=0)
a_average_demod     = np.mean(a_total_demod,axis=0)
a_std_raw           = np.std(a_total_raw,axis=0)
a_std_demod         = np.std(a_total_demod,axis=0)
# 
a_average_rfft      = np.mean(a_total_rfft,axis=0)
a_average_dfft      = np.mean(a_total_dfft,axis=0)
a_average_rffft     = np.mean(a_total_rffft,axis=0)
a_std_rfft          = np.std(a_total_rfft,axis=0)
a_std_dfft          = np.std(a_total_dfft,axis=0)
a_std_rffft          = np.std(a_total_rffft,axis=0)
#
n_average_raw        = np.mean(n_total_raw,axis=0)
n_average_demod      = np.mean(n_total_demod,axis=0)
n_std_raw            = np.std(n_total_raw,axis=0)
n_std_demod          = np.std(n_total_demod,axis=0)
#
n_average_rfft       = np.mean(n_total_rfft,axis=0)
n_average_dfft       = np.mean(n_total_dfft,axis=0)
n_average_rffft      = np.mean(n_total_rffft,axis=0)
n_std_rfft           = np.std(n_total_rfft,axis=0)
n_std_dfft           = np.std(n_total_dfft,axis=0)
n_std_rffft          = np.std(n_total_rffft,axis=0)


# A bar plot with mean and standard deviation at 7Hz. 
frequency_idx           = m.find_nearest(n_frequencies,frequency_of_interest)
print ('frequency_idx',frequency_idx)
# 
# start past the beginning, as I sometimes get a transient at the start which creates ringing. 

a_filtered_rawdata  = sosfiltfilt(sos_high, a_average_raw[int(Fs/2) :] - np.mean(a_average_raw[int(Fs/8):int(Fs)])   )
a_filtered_demod    = sosfiltfilt(sos_high, a_average_demod[int(Fs/2) :]- np.mean(a_average_demod[int(Fs/8):int(Fs)])  )
n_filtered_rawdata  = sosfiltfilt(sos_high, n_average_raw[int(Fs/2) :] - np.mean(n_average_raw[int(Fs/8):int(Fs)])   )
n_filtered_demod    = sosfiltfilt(sos_high, n_average_demod[int(Fs/2) :] - np.mean(a_average_demod[int(Fs/8):int(Fs)]) )
# 
# the passband-stopband is so small I can't really design a FIR filter in this space. 
# 
# 



idx = 0

# xf                      = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# frequencies             = xf[1:(end_pause-start_pause)//2]
# frequency_idx           = m.find_nearest(frequencies,high)

# a_signal = a_total_demod[idx,:]
# timestep = 1/Fs
# N = b
# xf                      = np.fft.fftfreq(N,d=timestep)
# fidx = m.find_nearest(xf,10)
# print ('fidx',fidx)
# X = fft(a_signal)
# X[0:40] = 0 
# X[-40:] = 0 
# X[80:50000] = 0 
# X[-50000:-80] = 0 
# y = ifft(X)
# inverse fft filter gives poor resolution in time domain. 

# window = np.hanning(newN)

# fig = plt.figure(figsize=(5,5))
# ax  = fig.add_subplot(211)
# plt.plot(xf,X)
# # plt.plot(X)
# ax  = fig.add_subplot(212)
# # plt.plot(a_td,y/np.max(y))
# plt.plot(a_td[int(Fs/2) : int(Fs*7.5) ],0.05*y[int(Fs/2) : int(Fs*7.5) ],'b')
# plt.plot(a_td[int(Fs/2): ],a_filtered_demod/np.max(a_filtered_demod),'r')
# # plt.plot(n_td[int(Fs/2) :],a_filtered_rawdata/np.max(a_filtered_rawdata),'k')
# ax.set_xlim([0,8])
# plt.show()

fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(411)
# plt.plot(n_td,n_total_demod[idx,:],'k')
# plt.plot(a_td,a_total_demod[idx,:],'r')
plt.plot(n_td,n_total_demod[idx,:] ,'k')
plt.plot(a_td,a_total_demod[idx,:],'r')

plt.legend(['no acoustic connection','acoustic connection'])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax2  = fig.add_subplot(412)
plt.plot(n_td,n_total_raw[idx,:],'k')
plt.plot(a_td,a_total_raw[idx,:],'r')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax3  = fig.add_subplot(413)
plt.plot(n_frequencies,n_total_dfft[idx,:],'k')
plt.plot(a_frequencies,a_total_dfft[idx,:],'r')
ax3.set_xlim([0,100])
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

ax4  = fig.add_subplot(414)
plt.plot(n_td[int(Fs/2) :],a_filtered_rawdata/np.max(a_filtered_rawdata),'k')
plt.plot(a_td[int(Fs/2) :],a_filtered_demod/np.max(a_filtered_demod),'r')
# plt.plot(a_td[int(Fs/2) :],conditioned_signal/np.max(conditioned_signal),'b')
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)

plot_filename = 'demod_comparison.png'
plt.savefig(plot_filename)
plt.show()





# Then work on improving the time series solution. It seems clear that I have a phase and amplitude offset. 
# It may be caused by the IIR filter. 
# Maybe I could try applying a FIR filter to the data. Look at some raw single instance data files. 


# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(111)
# plt.plot(a_frequencies,a_average_rfft,'k')
# plt.plot(n_frequencies,n_average_rfft,'r')
# # plt.plot(a_frequencies,n_average_rffft,'c')
# plt.show()

#
# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(111)
# # plt.plot(a_td,a_filtered_demod/np.max(a_filtered_demod),'k') 
# plt.plot(n_td,n_filtered_demod/np.max(n_filtered_demod),'g') 
# plt.plot(a_td,a_filtered_rawdata/np.max(a_filtered_rawdata),'r') 
# plt.plot(a_td,a_average_raw,'r') 
# plt.fill_between(a_td, a_average_raw - a_std_raw, a_average_raw + a_std_raw,
#                   color='r', alpha=0.5)
# 
# plt.plot(a_td,a_average_demod,'b') 
# # plt.fill_between(a_td, a_average_raw - a_std_raw, a_average_raw + a_std_raw,
# # #                   color='k', alpha=0.5)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = 'rawdata.png'
# plt.savefig(plot_filename)
# plt.show()


# # pressure applied with acoustic connection
# average_VEP 		= np.mean(d_total_rd,axis=0)
# std_VEP     		= np.std(d_total_rd,axis=0)
# average_demod_VEP 	= np.mean(d_total_dd,axis=0)
# std_demod_VEP     	= np.std(d_total_dd,axis=0)
# average_VEP_fft 	= np.mean(d_total_rfft,axis=0)
# average_demod_fft 	= np.mean(d_total_dfft,axis=0)
# std_VEP_fft 		= np.std(d_total_rfft,axis=0)
# std_demod_fft 		= np.std(d_total_dfft,axis=0)

# # no acoustic connection 
# n_average_VEP 			= np.mean(n_total_rd,axis=0)
# n_std_VEP     			= np.std(n_total_rd,axis=0)
# n_average_demod_VEP 	= np.mean(n_total_dd,axis=0)
# n_std_demod_VEP     	= np.std(n_total_dd,axis=0)
# n_average_VEP_fft 		= np.mean(n_total_rfft,axis=0)
# n_average_demod_fft 	= np.mean(n_total_dfft,axis=0)
# n_std_VEP_fft 			= np.std(n_total_rfft,axis=0)
# n_std_demod_fft 		= np.std(n_total_dfft,axis=0)
# # 
# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(111)
# ax.plot(t,np.max(average_demod_VEP)*marker/np.max(marker),'g')
# ax.plot(t,-average_demod_VEP/np.max(-average_demod_VEP),'k')
# ax.plot(t,n_average_demod_VEP/np.max(n_average_demod_VEP),'r')
# ax.plot(t,average_VEP/np.max(average_VEP),'b')

# # plt.fill_between(t, n_average_demod_VEP - n_std_demod_VEP, n_average_demod_VEP + n_std_demod_VEP,
# #                   color='r', alpha=0.5)
# # plt.fill_between(t, average_demod_VEP - std_demod_VEP, average_demod_VEP + std_demod_VEP,
# #                   color='k', alpha=0.5)


# # plt.legend(['led on/off','acoustic connection','no acoustic_connection','real VEP'])
# plt.legend(['led on/off','acoustic connection','no acoustic_connection','vep'])
# ax.set_xlim([0,np.max(t)])
# ax.set_ylabel('norm Volts ($\mu$V)')
# ax.set_xlabel('Time(s)')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = 'demod_vep_individual_trials.png'
# plt.savefig(plot_filename)
# plt.show()


# # # FFT 
# fig = plt.figure(figsize=(10,6))
# ax2  = fig.add_subplot(111)
# plt.plot(d_frequencies,average_demod_fft/a,'k')
# plt.plot(n_frequencies,n_average_demod_fft/c,'r')
# plt.legend(['acoustic connection','no acoustic_connection'])
# ax2.set_xlim([0,np.max(d_frequencies)])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = 'vep_individual_trials_fft.png'
# plt.savefig(plot_filename)
# plt.show()
