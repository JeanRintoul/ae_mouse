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

#  this data is somewhat filtered. 
filename          	= 'acoustic_connection_summary_data.npz'
data              	= np.load(filename)
a_total_vfft        = np.array(data['total_fft_vs'])
a_f                 = np.array(data['f'])

a,b = a_total_vfft.shape
print ('acoustic connection shape',a_total_vfft.shape)
nfilename          	= 'no_acoustic_connection_summary_data.npz'
ndata              	= np.load(nfilename)

n_total_vfft        = np.array(ndata['total_fft_vs'])
n_f                 = np.array(ndata['f'])

c,d = n_total_vfft.shape
print ('no acoustic connection shape',n_total_vfft.shape)
n_events,N 	= n_total_vfft.shape

a_average_vfft      = np.mean(a_total_vfft,axis=0)
a_std_vfft          = np.std(a_total_vfft,axis=0)
a_baseline          = np.mean(a_average_vfft)
a_baseline_std      = np.std(a_std_vfft)
print ('baseline mean and std of connected:',a_baseline,a_baseline_std)

n_average_vfft      = np.mean(n_total_vfft,axis=0)
n_std_vfft          = np.std(n_total_vfft,axis=0)
n_baseline          = np.mean(n_average_vfft)
n_baseline_std      = np.std(n_std_vfft)
print ('baseline mean and std of not connected:',n_baseline,n_baseline_std)
# 
# Add in a baseline measurement to the bar plots. 
# 
carrier_frequency = 500000
difference_frequency = carrier_frequency - frequency_of_interest
# A bar plot with mean and standard deviation at 70Hz. 
frequency_idx  = m.find_nearest(n_f,frequency_of_interest)
diff_frequency_idx  = m.find_nearest(n_f,difference_frequency)
carrier_frequency_idx  = m.find_nearest(n_f,carrier_frequency)
print ('frequency_idx',frequency_idx,diff_frequency_idx)


CTEs        = [n_average_vfft[diff_frequency_idx], a_average_vfft[diff_frequency_idx]]
error       = [n_std_vfft[diff_frequency_idx], a_std_vfft[diff_frequency_idx] ]
b2          = [n_baseline,a_baseline] 
baseline_multiplier = 1.0
e2          = [baseline_multiplier*n_baseline_std,baseline_multiplier*a_baseline_std]

SNRS        = [20*np.log10(CTEs[0]/b2[0]),20*np.log10(CTEs[1]/b2[1])]
print ('SNRs:',SNRS)
test_cases  = ['No Acoustic Connection, SNR:'+str(round(SNRS[0]))+'dB','Acoustic Connection, SNR:'+str(round(SNRS[1]))+'dB' ]
x_pos       = np.arange(len(test_cases))
x           = np.arange(2) # number of bars. 
width       = 0.40
# Build the plot
fig, ax     = plt.subplots()
ax.bar(x-0.2, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10,width=width)
ax.bar(x+0.2, b2, yerr=e2, align='center', alpha=0.5, ecolor='black', capsize=10,width=width)

ax.set_ylabel('Volts ($\mu$V)')
ax.set_xticks(x_pos)
ax.set_xticklabels(test_cases)
ax.set_yscale('log')
plt.legend(["@$\Delta$f", "@baseline"],loc='upper left',frameon=False)
ax.set_title('Artefact Test @ '+str(frequency_of_interest)+'Hz')
# ax.yaxis.grid(True)
# Save the figure and show
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('artefact_barplot.png')
plt.show()

fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111)
plt.plot(a_f,n_average_vfft,'r')
plt.plot(a_f,a_average_vfft,'k')
plt.plot(a_f,n_average_vfft,'r')
ax.set_title('Artefact Test @ '+str(frequency_of_interest)+' Hz')
plt.legend(test_cases,loc='upper left',frameon=False)
ax.set_ylim([0,350])
ax.set_xlim([carrier_frequency -2*frequency_of_interest,carrier_frequency +2*frequency_of_interest])
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Frequency(Hz)')
# Save the figure and show
plt.tight_layout()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('artefact_fft.png')
plt.show()

tc = ['Acoustic Connection','No Acoustic Connection']
fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111)
plt.plot(a_f,a_average_vfft,'k')
plt.plot(a_f,n_average_vfft,'r')
ax.set_title('Artefact Test @ carrier 500kHz')
plt.legend(tc,loc='upper left',frameon=False)
ax.set_ylim([0,350])
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

acoustoelectric_cases = ['modulated signal','applied signal']
efficiency_ratio = a_average_vfft[frequency_idx]/a_average_vfft[diff_frequency_idx]
print ('acoustoelectric efficiency ratio:', efficiency_ratio)
fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111)
ax2 = ax.twinx()
# ax2.plot(a_f[carrier_frequency_idx:]-500000,a_average_vfft[carrier_frequency_idx:],'r')
ax.plot(a_f,a_average_vfft/1e6,'k')
ax2.plot(a_f[carrier_frequency_idx:]-500000,a_average_vfft[carrier_frequency_idx:],'r')
ax2.plot(a_f,a_average_vfft/1e6,'k')

ax.set_title('Acoustoelectric Conversion Ratio: '+str(round(efficiency_ratio) ))
ax2.legend(acoustoelectric_cases,loc='upper left',frameon=False)
# ax.set_ylim([0,201000])
# ax.set_ylim([0,1.4])
# ax2.set_ylim([0,7])
ax.set_xlim([20,100])
ax.set_ylabel('Volts (V)')
ax2.set_ylabel('Volts ($\mu$V)',color='r')
ax.set_xlabel('Frequency(Hz)')
# Save the figure and show
plt.tight_layout()
# ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.savefig('acoustoelectric_conversion_scale.png')
plt.show()


