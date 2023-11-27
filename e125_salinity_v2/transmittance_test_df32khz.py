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
plot_filename ='t2_transmittance_df_frequencies.png'
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
plot_filename ='t2_transmittance_rf_frequencies.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()



fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(freqs,mean_carrier,'c', marker='.')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.fill_between(freqs, mean_carrier-std_carrier,mean_carrier+std_carrier,alpha=0.2,color='cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(freqs)+0.1])
plt.ticklabel_format(style='plain') 
plot_filename ='t2_transmittance_received.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(freqs,mean_mon,'c', marker='.')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.fill_between(freqs, mean_mon-std_mon,mean_mon+std_mon,alpha=0.2,color='cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(freqs)+0.1])
plt.ticklabel_format(style='plain') 
plot_filename ='t2_transmittance_monitor.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()


# fig = plt.figure(figsize=(6,5))
# ax = fig.add_subplot(211)
# plt.plot(freqs,mean_carrier,'c', marker='.')
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax2 = fig.add_subplot(212)
# plt.plot(freqs,mean_mon,'k', marker='.')
# plt.fill_between(freqs, mean_carrier-std_carrier,mean_carrier+std_carrier,alpha=0.2,color='cyan')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax.set_xlim([0-0.01,np.max(freqs)+0.1])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.ticklabel_format(style='plain') 
# plot_filename ='t2_transmittance_mon_vs_received.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()


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
plot_filename ='t2_transmittance_ratios.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()


