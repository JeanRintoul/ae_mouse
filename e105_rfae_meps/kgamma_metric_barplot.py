'''

Title: dual sine artefact comparison test. 
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
from scipy import stats

save_filename = 'kgamma_metric_t3.png'

filename          	= 'kgamma_correlation_test_USON_t3.npz'
data              	= np.load(filename)
data       			= np.array(data['saved_outputs'])

filename          	= 'kgamma_correlation_test_USOFF_t3.npz'
data2              	= np.load(filename)
data2       		= np.array(data2['saved_outputs'])

correlations_on = data[:,0]
correlations_off = data2[:,0]
print ('saved outputs',correlations_on,correlations_off)

uson_correlations_mean    = np.mean(correlations_on,axis=0)
uson_correlations_std    = np.std(correlations_on,axis=0)
#
usoff_correlations_mean    = np.mean(correlations_off,axis=0)
usoff_correlations_std    = np.std(correlations_off,axis=0)

CTEs        = [uson_correlations_mean,usoff_correlations_mean]
error       = [uson_correlations_std, usoff_correlations_std]

s, p = stats.ttest_ind(correlations_on,correlations_off)
print ('p-value:',p)

plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2

test_cases  = ['Ultrasound ON','Ultrasound OFF' ]
x_pos       = np.arange(len(test_cases))
x 			= np.arange(2) # number of bars. 
width 		= 0.40
# Build the plot
fig, ax     = plt.subplots()
ax.bar(x-0.2, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', color='grey',capsize=10,width=width)
# ax.bar(x+0.2, b2, yerr=e2, align='center', alpha=0.5, ecolor='black', capsize=10,width=width)
plt.yticks(np.arange(0, max(CTEs)+0.1, 0.1))
# ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xticks(x_pos)
# ax.set_xticklabels(test_cases)
# plt.legend(["@$\Delta$f", "@baseline"],loc='upper left',frameon=False)
# ax.set_title('Saline Artefact Test @ '+str(frequency_of_interest)+'Hz')
# # ax.yaxis.grid(True)
# # Save the figure and show
plt.yticks(fontsize=16)
plt.xticks([])
# plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(save_filename)
plt.show()


