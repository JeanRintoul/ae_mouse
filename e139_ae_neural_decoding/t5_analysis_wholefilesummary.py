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
from scipy.stats import ttest_ind
from scipy.signal import hilbert
from scipy.signal import find_peaks
import scipy.stats
from scipy.stats import pearsonr
import pandas as pd
from scipy import signal
# 
# 
# Try with a differently generated VEP filter. 
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16
# 
saveprefix  = './/images//'
# 
savepath            = 'D:\\ae_mouse\\e139_ae_neural_decoding\\'
filename            = 'e139_ae_neural_decodingef10.npz'
print ('filename: ', filename)
# x = np.load('mnist.npz', mmap_mode='r')
data                            = np.load(savepath+filename, mmap_mode='r')
# print (data)
rf_summation          = data['rf_summation']
marker_summation      = data['marker_summation']
average_signal        = data['average_signal']
average_corrd         = data['average_corrd']
n          			  = data['n']
# 

rawdata_summation = average_signal

Fs 		 = 2e6 
timestep = 1/Fs
carrier  = 500000
N 		 = len(rawdata_summation)

beta            = 20
window          = np.kaiser( (N), beta )

xf              = np.fft.fftfreq( (N), d=timestep)[:(N)//2]
frequencies     = xf[1:(N)//2]
# #
fft_raw          = fft(rawdata_summation)
fft_raw          = np.abs(2.0/(N) * (fft_raw))[1:(N)//2]
# #
fft_rawk         = fft(rawdata_summation*window)
fft_rawk         = np.abs(2.0/(N) * (fft_rawk))[1:(N)//2]
# #
# #
# # fontsize control. 
f = 14
# # 
# # 
# # fig = plt.figure(figsize=(6,6))
# # ax  = fig.add_subplot(211)
# # plt.plot(time_segment7,rawdata_summation7,'k')
# # ax2 = fig.add_subplot(212)
# # plt.plot(time_segment,rawdata_summation,'k')
# # plt.show()
# # 
# fig = plt.figure(figsize=(3,3))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_rawk,'k')
# plt.plot(frequencies,fft_raw,'r')
# ax.set_xlim([0,20])
# ax.set_ylim([0,15])
# plt.xticks(fontsize=f)
# plt.yticks(fontsize=f)
# plt.tight_layout()
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = savepath+'twotone_lf.png'
# plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(3,3))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_rawk,'k')
# plt.plot(frequencies,fft_raw,'r')
# ax.set_xlim([carrier-20,carrier+20])
# plt.xticks([carrier-20,carrier,carrier+20])
# # ax.set_ylim([0,15])
# ax.ticklabel_format(useOffset=False)
# plt.xticks(fontsize=f)
# plt.yticks(fontsize=f)
# plt.tight_layout()
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = savepath+'twotone_carrier.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(3,3))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_rawk,'k')
# plt.plot(frequencies,fft_raw,'r')
# ax.set_xlim([carrier-20,carrier+20])
# plt.xticks([carrier-20,carrier,carrier+20])
# ax.ticklabel_format(useOffset=False)
# ax.set_ylim([0,10])
# plt.xticks(fontsize=f)
# plt.yticks(fontsize=f)
# plt.tight_layout()
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = savepath+'twotone_carrier_zoom.png'
# plt.savefig(plot_filename)
# plt.show()
# 
# 
fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(211)
plt.plot(frequencies,fft_rawk,'k')
plt.plot(frequencies,fft_raw,'r')
#plt.plot(frequencies7,fft_rawk7,'r')
ax.set_xlim([carrier-40,carrier+40])
# plt.axvline(x=carrier-8,color='b')
# plt.axvline(x=carrier+8,color='b')
# ax.set_ylim([0,0.05])
plt.xticks(fontsize=f)
plt.yticks(fontsize=f)
# plt.legend(['Kaiser','Rectangular'],fontsize=f,loc='upper right',framealpha=0.0)
ax2  = fig.add_subplot(212)
plt.plot(frequencies,fft_rawk,'k')
# plt.plot(frequencies,fft_raw,'r')
# plt.legend(['Kaiser','Rectangular'],fontsize=f,loc='upper right',framealpha=0.0)
ax2.set_xlim([0,40])
ax2.set_ylim([0,3])
plt.xticks(fontsize=f)
plt.yticks(fontsize=f)
plt.tight_layout()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plot_filename = savepath+'gain_comparison.png'
plt.savefig(plot_filename)
plt.show()
# 
# t               = np.linspace(0, duration, N, endpoint=False)
# 
#                     
# print ('stuff shape: ',start_indexes)
# # demodulate the signal.
# analytical_signal       = hilbert(carrier_band_summation) # Hilbert demodulate.  
# h_signal                = -np.abs(analytical_signal)
# demodulated_signal      = h_signal - np.mean(h_signal)
# # temp adjust. 
# demodulated_signal  = np.real(analytical_signal)
# # 
# # 
# def demodulate(in_signal,carrier_f,tt): 
#     # return np.abs(in_signal*np.exp(2*np.pi*1j*carrier_f*tt))
#     return np.imag(in_signal*np.exp(2*np.pi*1j*carrier_f*tt))    
# demodulated_signal2       = demodulate(carrier_band_summation,carrier,time_segment) 
# # 
# # calculate the fft of the recovered signal.
# fft_demod         = fft(demodulated_signal)
# fft_demod         = np.abs(2.0/(rawN) * (fft_demod))[1:(rawN)//2]
# # 
# fft_demod2        = fft(demodulated_signal2)
# fft_demod2        = np.abs(2.0/(rawN) * (fft_demod2))[1:(rawN)//2]   
# print ('N lengths',rawN,N)
# # 
# # 
# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(411)
# plt.plot(time_segment,demodulated_signal,'k')
# plt.plot(time_segment,demodulated_signal2,'r')
# ax2 = fig.add_subplot(412)
# plt.plot(time_segment,lfp_summation,'k')
# ax3 = fig.add_subplot(413)
# plt.plot(time_segment,rawdata_summation,'k')
# ax4 = fig.add_subplot(414)
# plt.plot(time_segment,carrier_band_summation,'k')
# plt.show()
# # 
# # NB: It only makes sense to demodulate after much averaging. 
# # Plot the results. 
# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(311)
# plt.plot(raw_frequencies,fft_carrier_band,'k')    
# ax.set_xlim([carrier-h_cut,carrier+h_cut])
# ax.set_ylim([0,20])
# # 
# ax2  = fig.add_subplot(312)
# plt.plot(raw_frequencies,fft_raw,'k')    
# ax2.set_xlim([0,h_cut])
# ax2.set_ylim([0,10])
# # 
# ax3  = fig.add_subplot(313)
# plt.plot(raw_frequencies,fft_demod,'k')   
# plt.plot(raw_frequencies,fft_demod2,'r')   
# ax3.set_xlim([0,20])
# # ax3.set_ylim([0,10])
# plt.show()
# # 
