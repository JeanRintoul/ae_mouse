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
f = [50000, 250000, 500000, 1000000, 2000000]

f = [100,10000,100000,500000,1000000,1500000,2000000]

df_data = np.array([[50000,10.5, 2.28 , 11.52 ,  4.99 , 6.4 ],
	[250000,35.35, 45.9, 46.5, 43.18, 43.5 ],
	[500000, 54.59, 57.89, 52.59, 60.19, 66.41],
	[1000000,49.33,35.43,46.8,47.1,48.08 ],
	[1500000,49.33,35.43,46.8,47.1,48.08 ]])


carrier_data = np.array([ [50000,1334.04, 1326, 1318.68,  1315.46, 1304 ],
	[250000,3924, 3941.2, 3938, 3938.3, 3929.15 ],
	[500000, 6542.34, 6554.98, 6574.3, 6571.35, 6593.66],  
	[1000000,11698, 11672,11728,11681.2,11717.4 ],
	[1500000,11487.4, 11801.2, 11284.479, 12234, 12385 ]])
# 
freqs 	= df_data[:,0] 
dfs 	= df_data[:,1:]
cs  	= carrier_data[:,1:]
# 
freqs = freqs/1e6

mean_df = np.mean(dfs,1)
std_df = np.std(dfs,1)
print (mean_df)
mean_carrier = np.mean(cs,1)
std_carrier = np.std(cs,1)
# 
ratios 	   = mean_carrier/mean_df
ratios_std = 0

# 
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(freqs,mean_df,'c', marker='.')
plt.fill_between(freqs, mean_df-std_df,mean_df+std_df,alpha=0.2,color='cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(freqs)+0.1])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ticklabel_format(style='plain') 
plot_filename ='_transmittance_measured_frequencies.png'
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


