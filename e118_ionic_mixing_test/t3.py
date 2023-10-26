'''

Title: vep signal inspection
Function: takes a single file, and averages the veps to see them better. 

Author: Jean Rintoul
Date:   11.10.2023

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 


savepath       = 'D:\\ae_mouse\\e118_ionic_mixing_test\\t3_picture_of5Hz_2MHz_outputsignal\\'
dfx            = 10
file_number    = 2

gain            = 1000
duration        = 6.0    
# 
Fs              = 5e6
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 4
# 
filename    = savepath + 't'+str(file_number)+'_stream.npy'
data = np.load(filename)
a,b = data.shape
# print ('shape',a,b)
t = np.linspace(0, duration, N, endpoint=False)
# print ('len t',len(t))
# # convert it to microvolts by taking the gain into account. 

vsignal_base = data[1]


start = 1.5 
stop  = 2.7
fig = plt.figure(figsize=(6,2))
ax = fig.add_subplot(111)
plt.plot(t,data[1],'k')
ax.set_xlim([start,stop])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks([])
# plt.xticks([])
plt.tight_layout()
plot_filename = savepath + '\\t'+str(file_number)+'_output_wave.png'
plt.savefig(plot_filename)
plt.show()



# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_data,'k')
# # ax.set_xlim([0,110])
# ax.set_xlim([0,110])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = 'rfti_fft.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()

# # Plots
# start = 1.5 
# stop  = 4.5

# start = 0
# stop  = duration
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(t,lfp_data,'k')
# # plt.plot(t,10*spike_data,'r')
# plt.plot(t,-250+100*vsignal/np.max(vsignal),'r')
# plt.legend(['lfp','applied'],loc='upper right')
# ax.set_xlim([start,stop])
# ax2 = fig.add_subplot(312)
# plt.plot(t,spike_data,'k')
# plt.plot(t,-100+10*vsignal/np.max(vsignal),'r')
# # 
# ax2.set_xlim([start,stop])
# plt.legend(['spikes','applied'],loc='upper right')
# ax3 = fig.add_subplot(313)
# plt.plot(t,fsignal/np.max(fsignal),'k')
# # plt.plot(t,vsignal/np.max(vsignal),'r')
# plt.legend(['raw measured signal'],loc='upper right')
# ax3.set_xlim([start,stop])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plot_filename = 'rfti_debugging.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()
# 
# 
# 

