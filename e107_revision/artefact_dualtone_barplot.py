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


filename          	= 'dual_sine_artefact_test.npz'
# filename          	= 'dual_sine_artefact_test_PRF1020.npz'
data              	= np.load(filename)
data       			= np.array(data['saved_outputs'])

s1 = data[:,0]
s2 = data[:,1]
df = data[:,2]
sf = data[:,3]
# print (data,s1)
s1_mean    = np.mean(s1,axis=0)
s2_mean    = np.mean(s2,axis=0)
df_mean    = np.mean(df,axis=0)
sf_mean    = np.mean(sf,axis=0)
s1_std     = np.std(s1,axis=0)
s2_std     = np.std(s2,axis=0)
df_std     = np.std(df,axis=0)
sf_std     = np.std(sf,axis=0)

CTEs        = [s1_mean,s2_mean]
error       = [s1_std, s2_std ]

plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2

# CTEs        = [df_mean,sf_mean]
# error       = [df_std, sf_std ]


test_cases  = ['Signal @70Hz SNR(dB)','Signal @1020Hz SNR(dB)' ]
x_pos       = np.arange(len(test_cases))
x 			= np.arange(2) # number of bars. 
width 		= 0.40
# Build the plot
fig, ax     = plt.subplots()
ax.bar(x-0.2, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', color='grey',capsize=10,width=width)
# ax.bar(x+0.2, b2, yerr=e2, align='center', alpha=0.5, ecolor='black', capsize=10,width=width)

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
plt.savefig('twotone_artefact_barplot_applied_signal.png')
plt.show()
