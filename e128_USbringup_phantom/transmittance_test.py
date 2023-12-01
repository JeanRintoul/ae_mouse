'''

Title: transmittance test plotting
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
carriers =[40,100, 500, 1000, 1500, 2000, 4000, 8000, 10000, 30000, 50000,100000,250000,500000,750000,1000000,2000000]
carriers = np.array(carriers)/1e6

data = np.load('D:\\ae_mouse\\e128_USbringup_phantom\\t1\\transmission_1\\transmission_info.npz')
# data = np.load('D:\\ae_mouse\\e128_USbringup_phantom\\t1\\transmission_2\\transmission_info.npz')
thingstosave = data['thingstosave']
things = np.array(thingstosave)
print ('things shape',things.shape)
i_data = things[:,0]
v_data = things[:,1]
zs = things[:,2]
dfs = things[:,3]
carriers_amplitudes = things[:,4]
carriers_amplitudes = np.array(carriers_amplitudes)/1e6
carriers_amplitudes_mon = things[:,5]

scaled_dfs = np.array(dfs)/1e6

ratios 	   = scaled_dfs/carriers_amplitudes


fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
# ax.set_yscale('log')
# 
# plt.yscale('log')
plt.plot(carriers*1e6,dfs,'c', marker='.')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.set_xlim([0-0.01,np.max(carriers)+0.01])
ax.set_xlim([0,100000])
ax.set_ylim([1,np.max(dfs)])
# plt.ticklabel_format(style='plain', axis='y')
# ax.set_yscale('log')
plt.yscale('log')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.ticklabel_format(style='plain') 
plot_filename ='t1_transmittance_logdf_frequencies.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()


# # 
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(carriers,dfs,'c', marker='.')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(carriers)+0.01])
# ax.set_xlim([0-0.01,10000])
ax.set_ylim([0,np.max(dfs)])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ticklabel_format(style='plain') 
plot_filename ='t1_transmittance_df_frequencies.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()


fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(carriers,carriers_amplitudes,'c', marker='.')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(carriers)+0.01])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ticklabel_format(style='plain') 
plot_filename ='t1_transmittance_carrieramplitudes_frequencies.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
# plt.plot(carriers,carriers_amplitudes_mon,'c', marker='.')
plt.plot(carriers,v_data,'c', marker='.')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.fill_between(freqs, mean_carrier-std_carrier,mean_carrier+std_carrier,alpha=0.2,color='cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(carriers)+0.01])
plt.ticklabel_format(style='plain') 
plot_filename ='t1_transmittance_vmonitor.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()


fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(carriers,zs,'c', marker='.')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.fill_between(freqs, mean_carrier-std_carrier,mean_carrier+std_carrier,alpha=0.2,color='cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([0-0.01,np.max(carriers)+0.01])
plt.ticklabel_format(style='plain') 
plot_filename ='t1_transmittance_impedances.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

print ('ratios',ratios)


fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(carriers,ratios,'c', marker='.')


# plt.plot(salinity,mean_dfs,'c')
# ax.set_ylim([0,0.02])
# plt.fill_between(carriers, ratios-ratios_std, ratios+ratios_std,alpha=0.2,color='cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.set_xlim([0-0.01,np.max(freqs)+0.1])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ticklabel_format(style='plain') 
plot_filename ='t1_transmittance_ratios.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()


