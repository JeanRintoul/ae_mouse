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

data = [17.9, 173,
18.3, 199 ,
18.6, 174.8,
21.2, 179,
21.8, 188.77, 
23.1, 179.55,
24.1, 182.7 ,
25.0, 194.26,
25.5, 179.75,
26.0, 187.14,
26.7, 206.5,
27.3, 187.49,
27.7, 217.07,
27.8, 197.12,
28.4, 192.85,
28.6, 211.87 ,
28.6, 183,
29.1, 243.9, 
29.1, 171,
29.1, 293,
29.2, 217.05,
29.3, 198.3 ,
29.4, 180.88 ,
29.5, 210.3,
29.5, 197.11, 
29.6, 187.24,
29.5, 191.11,
28.9, 201.67 ,
28.8, 194.2,
27.6, 191.67,
26.8, 200.37]

temperature = data[0::2]
df_amplitudes = data[1::2]

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(temperature,df_amplitudes,'.k')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plot_filename ='_temperature.png'
plt.savefig(plot_filename)
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


