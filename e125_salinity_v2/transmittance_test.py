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


f = np.array([100,10000,100000,500000,1000000,1500000,2000000])


df_data = np.array([[1.2,	0.81,	0.16,	0.32],
[0.8,	1.16,	2,	1.59],
[14,	13.6,	14.6,	18.18],
[37,	37.3,	36.3,	36.3],
[8.01,	6.19,	6.17,	6.4],
[138.2,	138.66,	139.6,	139.78],
[284.4,	284.6,	282.4,	260]])

# carrier amplitudes at electrodes. 
carrier_data = np.array([[19.9,	7,	10.6,	5.35],
[193,	186,	194.48,	184.7],
[1861.44,	1801.13,	1799.96,	1787],
[5363,	5386,	5363,	5435],
[10146,	9873,	9795,	9831],
[8218.7,	8514,	9682.5,	8684],
[6922.68,	6807,	4098,	3733]])


rf_mon_data = np.array([[0,	0,	0,	0],
[34.9,	35.15,	35.13,	35.3],
[72,	71.6,	71.6,	71.65],
[72.9,	72.8,	72.3,	72.9],
[76.7,	76.4,	76.3,	75.9],
[77.27,	76.5,	78.68,	78.12],
[82,	83.4,	82.07,	83.7]])

freqs 	= f/1e6
dfs 	= df_data
cs  	= carrier_data


# 
mean_df = np.mean(dfs,1)
std_df  = np.std(dfs,1)
print (mean_df)
mean_carrier = np.mean(cs,1)
std_carrier  = np.std(cs,1)
# 
mean_mon = np.mean(rf_mon_data,1)
std_mon  = np.std(rf_mon_data,1)

ratios 	   = mean_df/mean_carrier
ratios_std = 0

# 
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(freqs,mean_df,'c', marker='.')
plt.fill_between(freqs, mean_df-std_df,mean_df+std_df,alpha=0.2,color='cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(freqs)+0.1])
ax.set_ylim([0,np.max(mean_df)])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ticklabel_format(style='plain') 
plot_filename ='_transmittance_df_frequencies.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(freqs,mean_carrier,'c', marker='.')
plt.fill_between(freqs, mean_carrier-std_carrier,mean_carrier+std_carrier,alpha=0.2,color='cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(freqs)+0.1])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ticklabel_format(style='plain') 
plot_filename ='_transmittance_rf_frequencies.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(211)
plt.plot(freqs,mean_carrier,'c', marker='.')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

ax2 = fig.add_subplot(212)
plt.plot(freqs,mean_mon,'k', marker='.')
# plt.fill_between(freqs, mean_carrier-std_carrier,mean_carrier+std_carrier,alpha=0.2,color='cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(freqs)+0.1])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ticklabel_format(style='plain') 
plot_filename ='_transmittance_mon_vs_received.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()


print ('ratios',ratios)


fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(freqs,ratios,'c', marker='.')


# plt.plot(salinity,mean_dfs,'c')
# ax.set_ylim([0,0.02])
plt.fill_between(freqs, ratios-ratios_std, ratios+ratios_std,alpha=0.2,color='cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.set_xlim([0-0.01,np.max(freqs)+0.1])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ticklabel_format(style='plain') 
plot_filename ='_transmittance_ratios.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()


