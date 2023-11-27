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
#
data = np.array([[0, 38.3, 41.9 , 42.7 ,  43.47 ,  42.212],
[0.225,54.6, 53.2, 51.63, 51.62, 52.78 ],
[0.45,58.55, 52.76, 54.95, 54.47, 50.28 ],
[0.9,69.07 , 59.8 , 65.7 , 64.23 , 54.8 ],
[1.8,54.59, 57.89, 52.59, 60.19, 66.41 ]])
# 
# 
cdata = np.array([[0, 15128.35, 15720.93 , 16249.87 , 15404.59, 16976.4],
[0.225,6341.68, 6339.187, 6341.51, 6357.8, 6348.5 ],
[0.45,6657.8, 6643.65, 6598.55, 6589.7, 6600.4  ],
[0.9,7581.69, 7560.97, 7528.88, 7528.24, 7524.39 ],
[1.8,6542.34, 6554.98, 6574.3, 6571.35, 6593.66 ]])


salinity = data[:,0] 
dfs = data[:,1:]
cs  = cdata[:,1:]
print (dfs)
mean_dfs = np.mean(dfs,1)
std_dfs = np.std(dfs,1)
mean_cs = np.mean(cs,1)
std_cs = np.std(cs,1)
print (mean_dfs)

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(cs.T,dfs.T,'.')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.set_xlim([0-0.01,np.max(salinity)+0.1])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plot_filename ='_salinity.png'
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
plot_filename ='t1_salinity.png'
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
plot_filename ='t1_carrier_salinity.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

#
# FFT amplitude
# US_carrier          = [12969,13046,12382,12859]
# US_df               = [889.13,867.13,808,947]
# # e115 t3, 7,8,9,10 
# twotone_carrier     = [13106,12932,12989,13013] #
# twotone_df          = [40.5, 64.15,29.5,22.6]  # this is at the baseline 
# US_mean         = np.mean(US_df) 
# US_std          = np.std(US_df) 
# twotone_mean    = np.mean(twotone_df) 
# twotone_std     = np.std(twotone_df) 

# # T-test. 
# sample1 = US_df 
# sample2 = twotone_df
# t_stat, p_value = ttest_ind(sample1, sample2) 
# print('T-statistic value: ', t_stat) 
# print('P-Value: ', p_value)
# # 
# # Now do Bar plot. 
# # 

# # Create lists for the plot
# materials = ['US df', 'TwoTone df']
# x_pos = np.arange(len(materials))
# CTEs = [US_mean,twotone_mean]
# error = [US_std,twotone_std]

# # Build the plot
# fig, ax = plt.subplots()
# ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,color='grey', ecolor='black', capsize=10)
# ax.set_xticks(x_pos)
# ax.set_xticklabels(materials)
# ax.yaxis.grid(True)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # Save the figure and show
# plt.tight_layout()
# plt.savefig('bar_plot_with_error_bars.png')
# plt.show()


