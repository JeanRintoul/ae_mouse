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

data = np.load('mouse_data.npz')
pp1 = data['pp1']
pp2 = data['pp2']
pp3 = data['pp3']
lfp_aggregates = data['lfp_aggregates']
df_ramp_frequencies = data['df_ramp_frequencies']

lfp_height_means = np.mean(pp3,axis=0)
lfp_height_stds = np.std(pp3,axis=0)
lfp_height_means2 = np.mean(pp1,axis=0)
lfp_height_stds2 = np.std(pp1,axis=0)
print (len(lfp_height_means))

pdata = np.load('phantom_data.npz')
ppp1 = pdata['pp1']
ppp2 = pdata['pp2']
ppp3 = pdata['pp3']
plfp_aggregates = pdata['lfp_aggregates']
pdf_ramp_frequencies = pdata['df_ramp_frequencies']

plfp_height_means = np.mean(ppp3,axis=0)
plfp_height_stds = np.std(ppp3,axis=0)
plfp_height_means2 = np.mean(ppp1,axis=0)
plfp_height_stds2 = np.std(ppp1,axis=0)
print (len(plfp_height_means))

fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111)
plt.plot(df_ramp_frequencies,lfp_height_means,'k')
plt.plot(df_ramp_frequencies,plfp_height_means,'b')
plt.plot(df_ramp_frequencies,pp3.T,'.k')

plt.fill_between(df_ramp_frequencies, lfp_height_means-lfp_height_stds, lfp_height_means+lfp_height_stds,alpha=0.2,color='grey')

plt.plot(df_ramp_frequencies,ppp3.T,'.b')

plt.fill_between(df_ramp_frequencies, plfp_height_means-plfp_height_stds, plfp_height_means+plfp_height_stds,alpha=0.2,color='b')
ax.set_xlim([0,1050])
ax.set_ylim([0,np.max(pp3)])
# plt.legend(['mouse','phantom'],loc='lower right')
plt.legend(['mouse','phantom'],loc='lower right',framealpha=0.0,fontsize=16)

plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('dfx_filter.png', bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111)
plt.plot(df_ramp_frequencies,lfp_height_means2,'k')
plt.plot(df_ramp_frequencies,plfp_height_means2,'b')
plt.plot(df_ramp_frequencies,pp1.T,'.k')

plt.fill_between(df_ramp_frequencies, lfp_height_means2-lfp_height_stds2, lfp_height_means2+lfp_height_stds2,alpha=0.2,color='grey')

plt.plot(df_ramp_frequencies,ppp1.T,'.b')
plt.plot(df_ramp_frequencies,plfp_height_means2,'b')
plt.fill_between(df_ramp_frequencies, plfp_height_means2-plfp_height_stds2, plfp_height_means2+plfp_height_stds2,alpha=0.2,color='b')
ax.set_xlim([0,1050])
ax.set_ylim([0,np.max(pp1)])
plt.legend(['mouse','phantom'],loc='lower right',framealpha=0.0,fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.tight_layout()
plt.savefig('fft_pp.png', bbox_inches='tight')
plt.show()


# #
# # FFT amplitude
# direct_10_df        = [6.09,11.23,5.87,0.9,0.09]
# # e115 t3, 7,8,9,10 
# modulated_10_df     = [171.15,176.38,177.69,178.59,179.35] #

# direct_mean     = np.mean(direct_10_df) 
# modulated_mean  = np.mean(modulated_10_df) 

# direct_std = np.std(direct_10_df) 
# modulated_std = np.std(modulated_10_df) 

# # T-test. 
# sample1 = direct_10_df
# sample2 = modulated_10_df
# t_stat, p_value = ttest_ind(sample1, sample2) 
# print('T-statistic value: ', t_stat) 
# print('P-Value: ', p_value)
# # 
# # Now do Bar plot. 
# # 

# # Create lists for the plot
# materials = ['Direct 10Hz', 'Modulated 10Hz']
# x_pos = np.arange(len(materials))
# CTEs = [direct_mean,modulated_mean]
# error = [direct_std,modulated_std]

# # Build the plot
# # fig, ax = plt.subplots()
# fig = plt.figure(figsize=(5,5))
# ax  = fig.add_subplot(111)

# ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,color='grey', ecolor='black', capsize=10)
# ax.set_xticks(x_pos)
# ax.set_xticklabels(materials)
# # ax.yaxis.grid(True)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # Save the figure and show
# plt.tight_layout()
# plt.savefig('bar_plot_with_error_bars.png')
# plt.show()


