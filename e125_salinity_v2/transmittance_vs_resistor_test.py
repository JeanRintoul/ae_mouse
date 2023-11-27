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
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16
#


f = np.array([100000,500000,1000000,1500000,2000000])


# df amplitudes at electrodes. 
df_data = np.array([[31,	37.96,	39.66,	34.1],
[101,	100,	100,	100],
[200.16,	199.66,	205.01,	189.1],
[226.31,	229.02,	189.9,	183.73],
[186.1,	182.3,	142,	207]])


# carrier amplitudes at electrodes. 
carrier_data = np.array([[1812,	1820,	1754,	1760],
[5333,	5408,	5337,	5299],
[9719,	9696,	9921,	9670],
[9690,	9583,	8297,	8301],
[5912,	6283,	4623,	6583]])


rf_mon_data = np.array([ [70.8,	70.8,	70.8,	80.76],
[72.7,	72.6,	72.6,	72.5],
[76.9,	77,	76.99	,76.67],
[79.3,	79.15,	78.05,	77.94],
[84.03,	86.14,	83.4,	84.6] ])

resistor_df_data = np.array([[7.85,	4.3,	3.98,	3.57],
[15.6,	13.99,	3.81,	2.93],
[5.28,	8.23,	0.88,	3.49],
[2.2,	8.45,	2.14,	3.23],
[11.1,	6.3,	11.8,	5.88]])


resistor_carrier_data = np.array([[2097,	2088,	2079,	2092],
[5459,	5466,	5641,	5648],
[10203,	10190,	10212,	10198],
[8730,	8740,	8732,	8739],
[4811,	4820,	4835,	4815]])



freqs 	= f/1e6
# 
mean_df = np.mean(df_data,1)
std_df  = np.std(df_data,1)

mean_carrier = np.mean(carrier_data,1)
std_carrier  = np.std(carrier_data,1)
# 
# 
resistor_df_mean = np.mean(resistor_df_data,1)
resistor_df_std = np.std(resistor_df_data,1)
# 
resistor_carrier_mean = np.mean(resistor_carrier_data,1)
resistor_carrier_std = np.std(resistor_carrier_data,1)


ratios 	   = mean_df/mean_carrier
resistor_ratios 	   = resistor_df_mean/resistor_carrier_mean
# 
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(freqs,mean_df,'c', marker='.')
plt.plot(freqs,resistor_df_mean,'k', marker='.')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(freqs)+0.1])
# ax.set_ylim([0,np.max(mean_df)])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ticklabel_format(style='plain') 
plt.legend(['saline','resistor'],loc='upper left',fontsize=16,framealpha=0.0)
plt.tight_layout()
plot_filename ='resistor_transmittance_df_frequencies.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()
# 
# 
# 
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(freqs,mean_carrier,'c', marker='.')
plt.plot(freqs,resistor_carrier_mean,'k', marker='.')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(freqs)+0.1])
# ax.set_ylim([0,np.max(mean_df)])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ticklabel_format(style='plain') 
plt.legend(['saline','resistor'],loc='upper left',fontsize=16,framealpha=0.0)
plt.tight_layout()
plot_filename ='resistor_transmittance_carrier_frequencies.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()
# 


fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(freqs,ratios,'c', marker='.')
plt.plot(freqs,resistor_ratios,'k', marker='.')
# plt.fill_between(freqs, ratios-ratios_std, ratios+ratios_std,alpha=0.2,color='cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.set_xlim([0-0.01,np.max(freqs)+0.1])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ticklabel_format(style='plain') 
plt.tight_layout()
plt.legend(['saline','resistor'],loc='upper left',fontsize=16,framealpha=0.0)
plot_filename ='resistor_transmittance_ratios.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()


