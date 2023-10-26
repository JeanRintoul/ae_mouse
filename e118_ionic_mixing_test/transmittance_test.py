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

savepath       = 'D:\\ae_mouse\\e118_ionic_mixing_test\\t1\\transmittance_rep1\\ti_wave.npz'
t1 = np.load(savepath)['d']
savepath       = 'D:\\ae_mouse\\e118_ionic_mixing_test\\t1\\transmittance_rep2\\ti_wave.npz'
t2 = np.load(savepath)['d']
savepath       = 'D:\\ae_mouse\\e118_ionic_mixing_test\\t1\\transmittance_rep3\\ti_wave.npz'
t3 = np.load(savepath)['d']
savepath       = 'D:\\ae_mouse\\e118_ionic_mixing_test\\t1\\transmittance_rep4\\ti_wave.npz'
t4 = np.load(savepath)['d']


d_f = t1[:,0]
frequencies = d_f/1e6
print (frequencies)

d_rf = np.array(  [t1[:,1],  t2[:,1],t3[:,1],t4[:,1]])
d_mf = np.array(  [t1[:,2],  t2[:,2],t3[:,2],t4[:,2]])
d_df = np.array(  [t1[:,3],  t2[:,3],t3[:,3],t4[:,3]])
print (d_rf.shape)
mean_rf = np.mean(d_rf,0)
std_rf = np.std(d_rf,0)

mean_mf = np.mean(d_mf,0)
std_mf = np.std(d_mf,0)

mean_df = np.mean(d_df,0)
std_df = np.std(d_df,0)

ratios 	   = mean_mf/mean_df
ratios_std = std_mf/std_df

ratios 	   = mean_df/mean_mf
ratios_std = std_df/std_mf


fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(frequencies,mean_mf,'c', marker='.')
# plt.plot(salinity,mean_dfs,'c')
plt.fill_between(frequencies, mean_mf-std_mf,mean_mf+std_mf,alpha=0.2,color='cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(frequencies)+0.1])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ticklabel_format(style='plain') 
plot_filename ='_transmittance_measured_frequencies.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(frequencies,mean_rf,'c', marker='.')
plt.fill_between(frequencies, mean_rf-std_rf,mean_rf+std_rf,alpha=0.2,color='cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(frequencies)+0.1])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ticklabel_format(style='plain') 
plot_filename ='_transmittance_rf_frequencies.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()


fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(frequencies,mean_df,'c', marker='.')
# plt.plot(salinity,mean_dfs,'c')
plt.fill_between(frequencies, mean_df-std_df,mean_df+std_df,alpha=0.2,color='cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(frequencies)+0.1])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ticklabel_format(style='plain') 
plot_filename ='_transmittance_difference_frequencies.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(frequencies,mean_mf/np.max(mean_mf),'c', marker='.')
plt.plot(frequencies,mean_rf/np.max(mean_rf),'r', marker='.')
# plt.plot(salinity,mean_dfs,'c')
plt.fill_between(frequencies, (mean_mf-std_mf)/np.max(mean_mf-std_mf),(mean_mf+std_mf)/np.max((mean_mf+std_mf)),alpha=0.2,color='cyan')
plt.fill_between(frequencies, (mean_rf-std_rf)/np.max(mean_rf-std_rf),(mean_rf+std_rf)/np.max((mean_rf+std_rf)),alpha=0.2,color='r')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(frequencies)+0.1])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ticklabel_format(style='plain') 
plot_filename ='_transmittance_normalized_transmitted_vs_measured.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()


fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(frequencies,ratios,'c', marker='.')
# plt.plot(salinity,mean_dfs,'c')
ax.set_ylim([0,1])
plt.fill_between(frequencies, ratios-ratios_std, ratios+ratios_std,alpha=0.2,color='cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(frequencies)+0.1])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ticklabel_format(style='plain') 
plot_filename ='_transmittance_ratios.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()


# #
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


