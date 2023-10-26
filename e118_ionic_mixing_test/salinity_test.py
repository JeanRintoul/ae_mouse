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
# File list:  1,2,3,4,5 
#
data = np.array([[0, 83.9,87.36,91.67,91.11,93.28],
[0.225,132.17,155.12,138.46,147.54,143.25 ],
[0.45,161.97,164.99,163.73,166.31,161.47 ],
[0.9,174.55,174.43,173.99,174.22,172.63 ],
[1.8,196.55,195.83,196.86,197.27,193.23 ]])


salinity = data[:,0] 
dfs = data[:,1:]
print (dfs)
mean_dfs = np.mean(dfs,1)
std_dfs = np.std(dfs,1)

print (mean_dfs)


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
plot_filename ='_salinity.png'
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


