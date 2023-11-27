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


Fs 		= 1e4
df_h 	= 1400
sos_lfp_band = iirfilter(17, [df_h], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

mains_cut_band = iirfilter(17, [48,52], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

N 				= int(Fs*6)
duration 		= 3 
t               = np.linspace(0, duration, N, endpoint=False)

experiment_type 	= 'mouse'


data = np.load('mouse_data.npz')
lfp_aggregates = data['lfp_aggregates']
df_ramp_frequencies = data['df_ramp_frequencies']

pdata = np.load('phantom_data.npz')
plfp_aggregates = pdata['lfp_aggregates']
pdf_ramp_frequencies = pdata['df_ramp_frequencies']


frequency_element 	= 3
frequency 			= df_ramp_frequencies[frequency_element]
print ('frequency is:', frequency)
print ('shape is: ',lfp_aggregates.shape)
lfp_aggregatesm     = sosfiltfilt(mains_cut_band, lfp_aggregates[:,frequency_element,])
plfp_aggregatesm   = sosfiltfilt(mains_cut_band, plfp_aggregates[:,frequency_element,])
print ('shape is: ',lfp_aggregates.shape)

mouse_lfp_data     = sosfiltfilt(sos_lfp_band, lfp_aggregatesm[:,]).T
phantom_lfp_data   = sosfiltfilt(sos_lfp_band, plfp_aggregatesm[:,]).T


print ('t',len(t))
mouse_lfp_means = np.mean(mouse_lfp_data,axis=1)
mouse_lfp_stds = np.std(mouse_lfp_data,axis=1)
print (mouse_lfp_means.shape)
print ('mouse lfp data shape',mouse_lfp_data.shape)

phantom_lfp_means = np.mean(phantom_lfp_data,axis=1)
phantom_lfp_stds = np.std(phantom_lfp_data,axis=1)
print (phantom_lfp_means.shape)
print ('phantom lfp data shape',phantom_lfp_data.shape)


fig = plt.figure(figsize=(5,3))
ax  = fig.add_subplot(111)
plt.plot(t,mouse_lfp_data,alpha=0.2)
plt.plot(t,mouse_lfp_means,'k')
plt.fill_between(t,mouse_lfp_means-mouse_lfp_stds, mouse_lfp_means+mouse_lfp_stds,alpha=0.2,color='grey')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.set_xlim([1,1.3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('mouse_lfp_dfx_frequency_'+str(frequency)+'.png', bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(5,3))
ax  = fig.add_subplot(111)
plt.plot(t,phantom_lfp_data,alpha=0.2)
plt.plot(t,phantom_lfp_means,'k')
plt.fill_between(t,phantom_lfp_means-phantom_lfp_stds, phantom_lfp_means+phantom_lfp_stds,alpha=0.2,color='grey')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.set_xlim([1,1.3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('phantom_lfp_dfx_frequency_'+str(frequency)+'.png', bbox_inches='tight')
plt.show()


# plt.fill_between(df_ramp_frequencies, lfp_height_means-lfp_height_stds, lfp_height_means+lfp_height_stds,alpha=0.2,color='grey')
# plt.plot(df_ramp_frequencies,ppp3.T,'.b')

# plt.fill_between(df_ramp_frequencies, plfp_height_means-plfp_height_stds, plfp_height_means+plfp_height_stds,alpha=0.2,color='b')
# ax.set_xlim([5,1050])
# ax.set_ylim([0,np.max(pp3)])
# # plt.legend(['mouse','phantom'],loc='lower right')
# plt.legend(['mouse','phantom'],loc='lower right',framealpha=0.0,fontsize=16)

# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plt.savefig('dfx_filter.png', bbox_inches='tight')
# plt.show()

# fig = plt.figure(figsize=(5,5))
# ax  = fig.add_subplot(111)
# plt.plot(df_ramp_frequencies,lfp_height_means2,'k')
# plt.plot(df_ramp_frequencies,plfp_height_means2,'b')
# plt.plot(df_ramp_frequencies,pp1.T,'.k')

# plt.fill_between(df_ramp_frequencies, lfp_height_means2-lfp_height_stds2, lfp_height_means2+lfp_height_stds2,alpha=0.2,color='grey')

# plt.plot(df_ramp_frequencies,ppp1.T,'.b')
# plt.plot(df_ramp_frequencies,plfp_height_means2,'b')
# plt.fill_between(df_ramp_frequencies, plfp_height_means2-plfp_height_stds2, plfp_height_means2+plfp_height_stds2,alpha=0.2,color='b')
# ax.set_xlim([5,1050])
# ax.set_ylim([0,np.max(pp1)])
# plt.legend(['mouse','phantom'],loc='lower right',framealpha=0.0,fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.tight_layout()
# plt.savefig('fft_pp.png', bbox_inches='tight')
# plt.show()


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


