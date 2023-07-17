"""
Aggregate summary files. 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import iirfilter,sosfiltfilt
from scipy.fft import fft,fftshift
import sys
sys.path.append('D:\\mouse_aeti')  #  so that we can import from the parent folder. 
import mouse_library as m

frequency_of_interest   = 70
Fs                      = 1e5
sos_high                = iirfilter(17, [6,9], rs=60, btype='bandpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')
# total_fft_ms
#  this data is somewhat filtered. 
filename          	= 'acoustic_connection_summary_data.npz'
data              	= np.load(filename)
a_total_raw       	= np.array(data['total_rawdata'])
a_total_demod       = np.array(data['total_demod'])
a_frequencies       = data['frequencies']
a_total_rfft        = np.array(data['total_rfft'])
a_total_dfft        = np.array(data['total_dfft'])
a_total_rffft       = np.array(data['total_rffft'])
a_td       		    = data['td']
a_total_mfft        = np.array(data['total_fft_ms'])
a_total_vfft        = np.array(data['total_fft_vs'])
a_f                 = np.array(data['f'])

a,b = a_total_raw.shape
print ('acoustic connection shape',a_total_raw.shape)

nfilename          	= 'no_acoustic_connection_summary_data.npz'
ndata              	= np.load(nfilename)
n_total_raw       	= np.array(ndata['total_rawdata'])
n_total_demod       = np.array(ndata['total_demod'])
n_frequencies       = ndata['frequencies']
n_total_rfft        = np.array(ndata['total_rfft'])
n_total_dfft        = np.array(ndata['total_dfft'])
n_total_rffft       = np.array(ndata['total_rffft'])
n_td      		    = ndata['td']
n_total_mfft        = np.array(ndata['total_fft_ms'])
n_total_vfft        = np.array(ndata['total_fft_vs'])
n_f                 = np.array(ndata['f'])

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
a_std_rffft         = np.std(a_total_rffft,axis=0)
# a_mfft              = np.mean(a_total_mfft,axis=0)

a_average_vfft      = np.mean(a_total_vfft,axis=0)
a_std_vfft          = np.std(a_total_vfft,axis=0)

a_baseline          = np.mean(a_mfft,axis=0)
a_baseline_std      = np.std(a_mfft,axis=0)
print ('basleine mean and std of connected:',a_baseline,a_baseline_std)
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
n_mfft              = np.mean(n_total_mfft,axis=0)

n_average_vfft      = np.mean(n_total_vfft,axis=0)
n_std_vfft          = np.std(n_total_vfft,axis=0)

n_baseline          = np.mean(n_mfft,axis=0)
n_baseline_std      = np.std(n_mfft,axis=0)
print ('basleine mean and std of not connected:',n_baseline,n_baseline_std)
# 
# Add in a baseline measurement to the bar plots. 
# 

# A bar plot with mean and standard deviation at 7Hz. 
frequency_idx           = m.find_nearest(n_frequencies,70)
print ('frequency_idx',frequency_idx)


CTEs        = [n_average_dfft[frequency_idx], a_average_dfft[frequency_idx]]
error       = [n_std_dfft[frequency_idx], a_std_dfft[frequency_idx] ]
b2 = [n_baseline,a_baseline] 
baseline_multiplier = 0.05
e2 = [baseline_multiplier*n_baseline_std,baseline_multiplier*a_baseline_std]

SNRS        = [20*np.log10(CTEs[0]/b2[0]),20*np.log10(CTEs[1]/b2[1])]
print ('SNRs:',SNRS)
test_cases  = ['No Acoustic Connection, SNR:'+str(round(SNRS[0]))+'dB','Acoustic Connection, SNR:'+str(round(SNRS[1]))+'dB' ]
x_pos       = np.arange(len(test_cases))
x = np.arange(2) # number of bars. 
width = 0.40
# Build the plot
fig, ax     = plt.subplots()
ax.bar(x-0.2, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10,width=width)
ax.bar(x+0.2, b2, yerr=e2, align='center', alpha=0.5, ecolor='black', capsize=10,width=width)

ax.set_ylabel('Volts ($\mu$V)')
ax.set_xticks(x_pos)
ax.set_xticklabels(test_cases)
plt.legend(["@$\Delta$f", "@baseline"],loc='upper left',frameon=False)
ax.set_title('Saline Artefact Test @ '+str(frequency_of_interest)+'Hz')
# ax.yaxis.grid(True)
# Save the figure and show
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('saline_artefact_barplot.png')
plt.show()

fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111)
plt.plot(a_frequencies,a_average_dfft,'k')
plt.plot(a_frequencies,n_average_dfft,'r')
ax.set_title('Saline Artefact Test @ '+str(frequency_of_interest)+' Hz')
plt.legend(test_cases,loc='upper left',frameon=False)
ax.set_ylim([0,5])
ax.set_xlim([0,100])
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Frequency(Hz)')
# Save the figure and show
plt.tight_layout()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('saline_artefact_fft.png')
plt.show()

tc = ['Acoustic Connection','No Acoustic Connection']
fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111)
plt.plot(a_f,a_mfft,'k')
plt.plot(a_f,n_mfft,'r')
ax.set_title('Artefact Test @ carrier 500kHz')
plt.legend(tc,loc='upper left',frameon=False)
# ax.set_ylim([0,120])
ax.set_xlim([499800,500200])

plt.xticks([499800,500000,500200])

ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Frequency(Hz)')
# Save the figure and show
plt.tight_layout()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('carrier_comparison_fft.png')
plt.show()

# 
acoustoelectric_cases = ['demodulated signal','applied signal']
efficiency_ratio = a_mfft[frequency_idx]/a_average_dfft[frequency_idx]
print ('acoustoelectric efficiency ratio:', efficiency_ratio)
fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111)
ax2 = ax.twinx()
ax.plot(a_frequencies,a_average_dfft,'r')
ax.plot(a_f,a_mfft,'k')
ax2.plot(a_frequencies,a_average_dfft,'r')
ax.set_title('Acoustoelectric Conversion Ratio: '+str(round(efficiency_ratio) ))
ax.legend(acoustoelectric_cases,loc='upper left',frameon=False)
ax.set_ylim([0,201000])
ax2.set_ylim([0,5])
ax.set_xlim([20,100])
ax.set_ylabel('Volts ($\mu$V)')
ax2.set_ylabel('Volts ($\mu$V)',color='r')
ax.set_xlabel('Frequency(Hz)')
# Save the figure and show
plt.tight_layout()
# ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.savefig('acoustoelectric_conversion_scale.png')
plt.show()



# fig = plt.figure(figsize=(5,5))
# ax  = fig.add_subplot(111)
# plt.plot(a_frequencies,n_total_dfft.T,'r')
# plt.plot(a_frequencies,a_total_dfft.T,'k')
# ax.set_title('Saline Artefact Test @ '+str(frequency_of_interest)+' Hz')
# # plt.legend(test_cases,loc='upper right',frameon=False)
# ax.set_ylim([0,5])
# ax.set_xlim([0,100])
# ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xlabel('Frequency(Hz)')
# # Save the figure and show
# plt.tight_layout()
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('saline_artefact_fftlines.png')
# plt.show()
