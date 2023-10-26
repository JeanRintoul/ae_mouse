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

# 500kHz carrier. 
vout =  [17.34,35.4,53,68.9,86.5,101.3,135,164,208.4]
vout = [x * 2 for x in vout]
df_amplitude = [1.63,3.99,8.8,13.45,18.7,24.9,42.6,286.4,1428.6]
# 2MHz carrier. 
vout2 =  [22,40,64,85,94,123,163,183,228]
vout2 = [x * 2 for x in vout2]
df_amplitude2 = [1.35,2.84,0.59,9.32,33.3,96.6,369.6,512,1600]






fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(vout,df_amplitude,'k',marker='.')
plt.plot(vout2,df_amplitude2,'c',marker='.')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0,np.max(vout2)+1])
ax.set_ylim([0,1605])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.legend(['500kHz carrier','2Mhz carrier'],loc='upper left',fontsize=16,framealpha=0)
plot_filename ='_rfamplifier_df_amplitudes.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(2,2))
ax = fig.add_subplot(111)
plt.plot(vout,df_amplitude,'k',marker='.')
plt.plot(vout2,df_amplitude2,'c',marker='.')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0,200])
ax.set_ylim([0,50])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.legend(['500kHz carrier','2Mhz carrier'],loc='upper left',fontsize=16)
plt.tight_layout()
plot_filename ='_rfamplifier_df_amplitudes_zoom.png'
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


