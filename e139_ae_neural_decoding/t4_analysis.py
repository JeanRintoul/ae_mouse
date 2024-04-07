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
filename            = '4hz_t4_whole.npz'
print ('filename: ', filename)
data                            = np.load(savepath+filename)
time_segment                    = data['time_segment']
rawdata_summation               = data['rawdata_summation']
lfp_summation                   = data['lfp_summation']
carrier_band_summation          = data['carrier_band_summation']


filename7            = '7hz_t4_whole.npz'
print ('filename: ', filename7)
data7                            = np.load(savepath+filename7)
time_segment7                    = data7['time_segment']
rawdata_summation7               = data7['rawdata_summation']
lfp_summation7                   = data7['lfp_summation']
carrier_band_summation7          = data7['carrier_band_summation']


Fs = 2e6 
timestep = 1/Fs
carrier         = 500000
N = len(rawdata_summation)

N7 = len(rawdata_summation7)

beta            = 20
window          = np.kaiser( (N), beta )

xf              = np.fft.fftfreq( (N), d=timestep)[:(N)//2]
frequencies     = xf[1:(N)//2]
#
# fft_raw          = fft(rawdata_summation)
# fft_raw          = np.abs(2.0/(N) * (fft_raw))[1:(N)//2]

fft_rawk         = fft(rawdata_summation*window)
fft_rawk         = np.abs(2.0/(N) * (fft_rawk))[1:(N)//2]

window7          = np.kaiser( (N7), beta )
xf              = np.fft.fftfreq( (N7), d=timestep)[:(N7)//2]
frequencies7     = xf[1:(N7)//2]
fft_rawk7         = fft(rawdata_summation7*window7)
fft_rawk7         = np.abs(2.0/(N7) * (fft_rawk7))[1:(N7)//2]
#
# fontsize control. 
f = 14
# 
# 
# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(211)
# plt.plot(time_segment7,rawdata_summation7,'k')
# ax2 = fig.add_subplot(212)
# plt.plot(time_segment,rawdata_summation,'k')
# plt.show()

fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(211)

# plt.plot(frequencies,fft_rawk,'k')
plt.plot(frequencies7,fft_rawk7,'r')
ax.set_xlim([carrier-20,carrier+20])
plt.axvline(x=carrier-7,color='b')
plt.axvline(x=carrier-14,color='b')
plt.axvline(x=carrier+14,color='b')
plt.axvline(x=carrier+7,color='b')
plt.axvline(x=carrier-4,color='g')
plt.axvline(x=carrier+4,color='g')
ax.set_ylim([0,0.02])
plt.xticks(fontsize=f)
plt.yticks(fontsize=f)
plt.legend(['4Hz g=500','7Hz g=1000'],fontsize=f,loc='upper right',framealpha=0.0)
ax2  = fig.add_subplot(212)
plt.plot(frequencies,fft_rawk,'k')
plt.plot(frequencies7,fft_rawk7,'r')
plt.legend(['lfp 4Hz g=500','lfp 7Hz g=1000'],fontsize=f,loc='upper right',framealpha=0.0)
ax2.set_xlim([0,20])
ax2.set_ylim([0,5])
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
