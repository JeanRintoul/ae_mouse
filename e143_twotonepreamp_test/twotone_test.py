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
savepath             = 'D:\\ae_mouse\\e143_twotonepreamp_test\\'
# 
# 
# filename             = '140t1_single_file_twotone.npz'
# filenamest           = 't1_twotone_resistor_1000.npz'
# filenamest           = 't1_twotone_g2000.npz'
# filenamest           = 't3_twotone_exvivo.npz'
# filename             = '142t4_single_file_twotone.npz' 
# filenamett           = 't1_single_file_twotonephantom.npz'
# 
# 140t8f7_single_file_mouse_g5000.npz
# filename             = '140t8f7_single_file_mouse_g5000.npz' 
# filenamett           = '142t3f1_single_file_twotone.npz'
# 
# 
# filename             = '142t6f21_single_file_mouse.npz' 
# filenamett           = '142t6nof21_single_file_mouse.npz'

filename             = 'e142t8_10Hz_aeVEP_g2000_noF21_singlefile.npz' 
# filenamett           = 'e143_singlefile_twotonephantom.npz'
filenamett           = 't1_single_file_twotonephantom.npz' 

# 
# 
print ('filename: ', filename)
data                 = np.load(savepath+filename, mmap_mode='r')
t                    = data['t']
rawdata_summation    = data['rawdata_summation']

print ('filenamest: ', filenamett)
datatt               = np.load(savepath+filenamett, mmap_mode='r')
ttt                    = datatt['t']
rawdata_summationtt    = datatt['rawdata_summation']

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   

band_limit 	= 45
Fs 		 	    = 2e6 
timestep 	  = 1/Fs
carrier  	  = 500000
N 		 	    = len(rawdata_summation)

# beta            = 12
beta            = 14
window          = np.kaiser( (N), beta )

xf              = np.fft.fftfreq( (N), d=timestep)[:(N)//2]
frequencies     = xf[1:(N)//2]
# #
fft_raw          = fft(rawdata_summation)
fft_raw          = np.abs(2.0/(N) * (fft_raw))[1:(N)//2]

fft_rawtt        = fft(rawdata_summationtt)
fft_rawtt        = np.abs(2.0/(N) * (fft_rawtt))[1:(N)//2]
# #
fft_rawk         = fft(rawdata_summation*window)
fft_rawk         = np.abs(2.0/(N) * (fft_rawk))[1:(N)//2]

fft_rawktt       = fft(rawdata_summationtt*window)
fft_rawktt       = np.abs(2.0/(N) * (fft_rawktt))[1:(N)//2]
# #
# signal to noise of the up modulated signal? 
#
frequencies_of_interest 		= [carrier-10,carrier+10]
# frequencies_of_interest     = [carrier-20,carrier+20]
kaiser_signal_totals            = []
kaiser_sidetest_totals          = []
individual_signal_totals        = []
interest_frequencies         	= []
for n in range(len(frequencies_of_interest)): # sum all frequencies per unit time.
  df_idx = find_nearest(frequencies,frequencies_of_interest[n])    
  interest_frequencies.append(df_idx)    
  kaiser_signal_totals.append(fft_rawk[df_idx])
  kaiser_sidetest_totals.append(fft_rawktt[df_idx])

start_idx = find_nearest(frequencies,carrier + 5 )   # after the first 5Hz. 
end_idx = find_nearest(frequencies,carrier + band_limit)
dis_kaiser_signal_totals      = []
dis_kaiser_sidetest_signal_totals      = []
for n in range(start_idx, end_idx): # sum all frequencies per unit time.
	if n not in interest_frequencies: 
		dis_kaiser_signal_totals.append(fft_rawk[n])
		dis_kaiser_sidetest_signal_totals.append(fft_rawktt[n]) 
# 			
kaiser_signal_snr  			= 20*np.log(np.mean(kaiser_signal_totals)/np.mean(dis_kaiser_signal_totals))
kaiser_sidetest_signal_snr  = 20*np.log(np.mean(kaiser_sidetest_totals)/np.mean(dis_kaiser_sidetest_signal_totals))

print ('kaiser_signal_snr/sidetest snr:', kaiser_signal_snr,kaiser_sidetest_signal_snr)

f_to_compare = 10.0
upmodidx = find_nearest(frequencies,carrier+f_to_compare)
vepidx    = find_nearest(frequencies,f_to_compare)
ratio = fft_rawk[vepidx]/fft_rawk[upmodidx]
print ('ratio: ',np.round(ratio,2) )
# #

# fontsize control. 
f = 16


fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_raw,'k')
# plt.plot(frequencies,fft_rawkst,'r')
plt.plot(frequencies,fft_rawtt,'r')

ax.set_xlim([carrier-band_limit,carrier+band_limit])
# ax.set_ylim([0,600])
plt.xticks([carrier],fontsize=f)
plt.yticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.ticklabel_format(useOffset=False)
plt.tight_layout()
plot_filename = savepath+'_carrier_twotone_distant.png'
plt.savefig(plot_filename)
plt.show()


fig = plt.figure(figsize=(8,3))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_rawk,'k')
plt.plot(frequencies,fft_rawktt,'r')

ax.set_xlim([carrier-band_limit,carrier+band_limit])
ax.set_ylim([0,0.2])
plt.xticks([carrier-10,carrier-40,carrier+10,carrier+40],fontsize=f)
plt.yticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.ticklabel_format(useOffset=False)
plt.tight_layout()
plot_filename = savepath+'_carrier_twotone.png'
plt.savefig(plot_filename)
plt.show()


fig = plt.figure(figsize=(6,3))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_rawk,'k')
plt.plot(frequencies,fft_rawktt,'r')
ax.set_xlim([carrier+8,carrier+41])
ax.set_ylim([0,0.2])
plt.xticks([carrier+10,carrier+40],fontsize=f)
plt.yticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.ticklabel_format(useOffset=False)
plt.tight_layout()
plot_filename = savepath+'_carrier_twotone_zoom.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_raw,'k')
plt.plot(frequencies,fft_rawtt,'r')
ax.set_xlim([0,band_limit])
ax.set_ylim([0,30])
plt.xticks(fontsize=f)
plt.yticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = savepath+'_vep_twotone.png'
plt.savefig(plot_filename)
plt.show()
