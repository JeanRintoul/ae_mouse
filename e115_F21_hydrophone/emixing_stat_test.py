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
US_carrier          = [12969,13046,12382,12859]
US_df               = [889.13,867.13,808,947]
# e115 t3, 7,8,9,10 
twotone_carrier     = [13106,12932,12989,13013] #
twotone_df          = [40.5, 64.15,29.5,22.6]  # this is at the baseline 
US_mean         = np.mean(US_df) 
US_std          = np.std(US_df) 
twotone_mean    = np.mean(twotone_df) 
twotone_std     = np.std(twotone_df) 

# T-test. 
sample1 = US_df 
sample2 = twotone_df
t_stat, p_value = ttest_ind(sample1, sample2) 
print('T-statistic value: ', t_stat) 
print('P-Value: ', p_value)
# 
# Now do Bar plot. 
# 

# Create lists for the plot
materials = ['US df', 'TwoTone df']
x_pos = np.arange(len(materials))
CTEs = [US_mean,twotone_mean]
error = [US_std,twotone_std]

# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,color='grey', ecolor='black', capsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(materials)
ax.yaxis.grid(True)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()


