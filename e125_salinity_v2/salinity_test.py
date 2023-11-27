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
# File list:  t1
salinity = [0,1.1,2.2,4.5,9]
salinity_p = [0,0.225,0.45,0.9,1.8]
salinity = salinity_p
impedance = np.array([[700,672],
[268,268],
[240.48,239.66],
[228.27,228.35],
[221,221.9]
])
#
data = np.array([[460.9,	472.39,	486.7,	452.4],
[115.98,	114.34,	119.23,	116.06],
[118.7,	121.1,	125.36,	122.11],
[123.49,	125.857,	121.48,	119.7],
[115,	118,	113,	120.8]])
# 

cdata = np.array([[25230,	25572,	26282,	24823],
[6463,	6560,	6536,	6518],
[6642.66,	6681.8,	6670.58,	6584.29],
[7028.48,	6967.37,	7014.48,	6846.7],
[6679,	6671,	6687,	6722]])


dfs = data
cs = cdata
print (dfs)
mean_dfs = np.mean(dfs,1)
std_dfs = np.std(dfs,1)
mean_cs = np.mean(cs,1)
std_cs = np.std(cs,1)

mean_z = np.mean(impedance,1)
std_z = np.std(impedance,1)
print (cs)

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(311)
plt.plot(salinity,mean_z,'k',marker='.')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax2 = fig.add_subplot(312)
plt.plot(salinity,mean_dfs,'c',marker='.')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

ax3 = fig.add_subplot(313)
plt.plot(salinity,mean_cs,'r',marker='.')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

plt.tight_layout()

# ax.set_xlim([0-0.01,np.max(salinity)+0.1])
plot_filename ='salinity_impedance.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()


fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(salinity,dfs,'.k')
plt.plot(salinity,mean_dfs,'c')
plt.fill_between(salinity, mean_dfs-std_dfs, mean_dfs+std_dfs,alpha=0.2,color='cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(salinity)+0.1])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plot_filename ='salinity.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()


fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(salinity,cs,'.k')
plt.plot(salinity,mean_cs,'c')
plt.fill_between(salinity, mean_cs-std_cs, mean_cs+std_cs,alpha=0.2,color='cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(salinity)+0.1])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plot_filename ='carrier_salinity.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()




