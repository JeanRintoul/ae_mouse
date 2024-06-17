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
savepath             = 'D:\\ae_mouse\\e142_ae_neural_decoding\\t6_mouse\\'
# 
filename             = 't6_8Hz_g5000.npz'
filenamest           = 't6_isolation.npz'
# 
# 
print ('filename: ', filename)
data                 = np.load(savepath+filename, mmap_mode='r')
t                    = data['t']
rawdata_summation    = data['rawdata_summation']

# print ('filenamest: ', filenamest)
# datast               = np.load(savepath+filenamest, mmap_mode='r')
# tst                    = datast['t']
# rawdata_summationst    = datast['rawdata_summation']

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   

band_limit 	  = 25
Fs 		 	      = 2e6 
timestep 	    = 1/Fs
carrier  	    = 500000
N 		 	      = len(rawdata_summation)

# beta            = 12
beta            = 20
window          = np.kaiser( (N), beta )

xf              = np.fft.fftfreq( (N), d=timestep)[:(N)//2]
frequencies     = xf[1:(N)//2]
# #
fft_raw          = fft(rawdata_summation)
fft_raw          = np.abs(2.0/(N) * (fft_raw))[1:(N)//2]

# fft_rawst        = fft(rawdata_summationst)
# fft_rawst        = np.abs(2.0/(N) * (fft_rawst))[1:(N)//2]
# # #
fft_rawk         = fft(rawdata_summation*window)
fft_rawk         = np.abs(2.0/(N) * (fft_rawk))[1:(N)//2]

# fft_rawkst       = fft(rawdata_summationst*window)
# fft_rawkst       = np.abs(2.0/(N) * (fft_rawkst))[1:(N)//2]
# #
# signal to noise of the up modulated signal? 
#
frequencies_of_interest 		= [carrier-8,carrier+8]
kaiser_signal_totals            = []
kaiser_sidetest_totals          = []
individual_signal_totals        = []
interest_frequencies         	= []
for n in range(len(frequencies_of_interest)): # sum all frequencies per unit time.
  df_idx = find_nearest(frequencies,frequencies_of_interest[n])    
  interest_frequencies.append(df_idx)    
  kaiser_signal_totals.append(fft_rawk[df_idx])
  # kaiser_sidetest_totals.append(fft_rawkst[df_idx])

start_idx = find_nearest(frequencies,carrier + 5 )   # after the first 5Hz. 
end_idx = find_nearest(frequencies,carrier + band_limit)
dis_kaiser_signal_totals      = []
dis_kaiser_sidetest_signal_totals      = []
for n in range(start_idx, end_idx): # sum all frequencies per unit time.
	if n not in interest_frequencies: 
		dis_kaiser_signal_totals.append(fft_rawk[n])
		# dis_kaiser_sidetest_signal_totals.append(fft_rawkst[n]) 
# 			
kaiser_signal_snr  			= 20*np.log(np.mean(kaiser_signal_totals)/np.mean(dis_kaiser_signal_totals))
# kaiser_sidetest_signal_snr  = 20*np.log(np.mean(kaiser_sidetest_totals)/np.mean(dis_kaiser_sidetest_signal_totals))

print ('kaiser_signal_snr/sidetest snr:', kaiser_signal_snr)

f_to_compare = 8.0
upmodidx = find_nearest(frequencies,carrier+f_to_compare)
vepidx    = find_nearest(frequencies,f_to_compare)
ratio = fft_rawk[vepidx]/fft_rawk[upmodidx]
print ('ratio: ',np.round(ratio,2) )
# #

# fontsize control. 
f = 18



fig = plt.figure(figsize=(8,3))
ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_raw,'r')
plt.plot(frequencies,fft_rawk,'k')
ax.set_xlim([carrier-band_limit,carrier+band_limit])
ax.set_ylim([0,0.02])
plt.xticks([carrier-8,carrier+8,carrier-16,carrier+16],fontsize=f)
plt.yticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.ticklabel_format(useOffset=False)
plt.locator_params(axis='y', nbins=6)
# plt.locator_params(axis='x', nbins=4)

plt.tight_layout()
plot_filename = savepath+'_8hz.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_raw,'k')
# plt.plot(frequencies,fft_rawk,'k')

ax.set_xlim([0,band_limit])
ax.set_ylim([0,10])
plt.xticks([8,16,24],fontsize=f)
plt.yticks(fontsize=f)
plt.locator_params(axis='y', nbins=6)
# plt.locator_params(axis='x', nbins=5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = savepath+'_8hz_vep.png'
plt.savefig(plot_filename)
plt.show()

# fig = plt.figure(figsize=(3,3))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_rawk,'k')
# plt.plot(frequencies,fft_rawkst,'g')

# ax.set_xlim([carrier-band_limit,carrier+band_limit])
# ax.set_ylim([0,600])
# plt.xticks([carrier],fontsize=f)
# plt.yticks(fontsize=f)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.ticklabel_format(useOffset=False)
# plt.tight_layout()
# plot_filename = savepath+'_carrier_sidetest_distant.png'
# plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(6,3))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_rawk,'k')
# plt.plot(frequencies,fft_rawkst,'g')

# ax.set_xlim([carrier-band_limit,carrier+band_limit])
# ax.set_ylim([0,0.08])
# plt.xticks([carrier+10,carrier+20,carrier+30],fontsize=f)
# plt.yticks(fontsize=f)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.ticklabel_format(useOffset=False)
# plt.tight_layout()
# plot_filename = savepath+'_carrier_sidetest.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(6,3))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_rawkst,'g')
# plt.plot(frequencies,fft_rawk,'k')
# ax.set_xlim([carrier+8,carrier+41])
# ax.set_ylim([0,0.06])
# plt.xticks([carrier+10,carrier+20],fontsize=f)
# plt.yticks(fontsize=f)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.ticklabel_format(useOffset=False)
# plt.tight_layout()
# plot_filename = savepath+'_carrier_sidetest_zoom.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(3,3))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_rawk,'k')
# plt.plot(frequencies,fft_rawkst,'g')

# ax.set_xlim([0,band_limit])
# ax.set_ylim([0,3])
# plt.xticks(fontsize=f)
# plt.yticks(fontsize=f)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = savepath+'_vep_sidetest.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(211)
# plt.plot(frequencies,fft_rawkst,'g')
# plt.plot(frequencies,fft_rawk,'k')

# # plt.plot(frequencies,fft_raw,'r')
# ax.set_xlim([carrier-band_limit,carrier+band_limit])
# # plt.axvline(x=carrier-8,color='b')
# # plt.axvline(x=carrier+8,color='b')
# # ax.set_ylim([0,0.05])
# plt.xticks(fontsize=f)
# plt.yticks(fontsize=f)
# # plt.legend(['Kaiser','Rectangular'],fontsize=f,loc='upper right',framealpha=0.0)
# ax2  = fig.add_subplot(212)
# plt.plot(frequencies,fft_rawkst,'g')
# plt.plot(frequencies,fft_rawk,'k')
# # plt.legend(['Kaiser','Rectangular'],fontsize=f,loc='upper right',framealpha=0.0)
# ax2.set_xlim([0,band_limit])
# # ax2.set_ylim([0,3])
# plt.xticks(fontsize=f)
# plt.yticks(fontsize=f)
# plt.tight_layout()
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = savepath+'_comparison.png'
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


# 
# 

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
