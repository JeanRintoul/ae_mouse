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
# FFT amplitude
direct_10_df        = [6.09,11.23,5.87,0.9,0.09]
# e115 t3, 7,8,9,10 
modulated_10_df     = [171.15,176.38,177.69,178.59,179.35] #

direct_mean     = np.mean(direct_10_df) 
modulated_mean  = np.mean(modulated_10_df) 

direct_std = np.std(direct_10_df) 
modulated_std = np.std(modulated_10_df) 

# T-test. 
sample1 = direct_10_df
sample2 = modulated_10_df
t_stat, p_value = ttest_ind(sample1, sample2) 
print('T-statistic value: ', t_stat) 
print('P-Value: ', p_value)
# 
# Now do Bar plot. 
# 

# Create lists for the plot
materials = ['Direct 10Hz', 'Modulated 10Hz']
x_pos = np.arange(len(materials))
CTEs = [direct_mean,modulated_mean]
error = [direct_std,modulated_std]

# Build the plot
# fig, ax = plt.subplots()
fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111)

ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,color='grey', ecolor='black', capsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(materials)
# ax.yaxis.grid(True)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()


