import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
import os
import time 
from scipy.ndimage.filters import gaussian_filter1d

filename    = 'dual_wave.npz'
data = np.load(filename)
d = data['d']
d_f          = d[:,0]
d_rf1        = d[:,1]
d_rf2        = d[:,2]
d_rf_df      = d[:,3]
d_rf_sf      = d[:,4]
d_mf_c1      = d[:,5]
d_mf_c2      = d[:,6]
d_mf_df      = d[:,7]
d_mf_sf      = d[:,8]

savepath = '\\images'
upper_frequency = 1000010 /1000000
d_f = d_f /1000000 

# Also divide the 
x_df =  d_mf_df/d_mf_c1
x_sf  = d_mf_sf/d_mf_c1

sig_val = 2
sd_rf1   = gaussian_filter1d(d_rf1, sigma=sig_val)
sd_rf2   = gaussian_filter1d(d_rf2, sigma=sig_val)
sd_rf_df = gaussian_filter1d(d_rf_df, sigma=sig_val)
sd_rf_sf = gaussian_filter1d(d_rf_sf, sigma=sig_val)
sd_mf_c1  = gaussian_filter1d(d_mf_c1, sigma=sig_val)
sd_mf_c2  = gaussian_filter1d(d_mf_c2, sigma=sig_val)
sd_mf_df  = gaussian_filter1d(d_mf_df, sigma=sig_val)
sd_mf_sf  = gaussian_filter1d(d_mf_sf, sigma=sig_val)

# start = 0 
# step = 2
# d_f= d_f[start::step]
# sd_rf1   = d_rf1[start::step]
# sd_rf2   = d_rf2[start::step]
# sd_rf_df = d_rf_df[start::step]
# sd_rf_sf = d_rf_sf[start::step]
# sd_mf_c1  = d_mf_c1[start::step]
# sd_mf_c2  = d_mf_c2[start::step]
# sd_mf_df  = d_mf_df[start::step]
# sd_mf_sf  = d_mf_sf[start::step]

# endd = 2 
# sd_rf[0:endd] = d_rf[0:endd]
# sd_mf[0:endd] = d_mf[0:endd]
# d_rf = sd_rf  
# d_mf = sd_mf 
# lower = gaussian_filter1d(means - sems, sigma=2) 
# #  
d_rf1 	= sd_rf1
d_rf2 	= sd_rf2
d_rf_df = sd_rf_df
d_rf_sf = sd_rf_sf
d_mf_c1 = sd_mf_c1
d_mf_c2 = sd_mf_c2
d_mf_df = sd_mf_df
d_mf_sf = sd_mf_sf

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
plt.plot(d_f,x_df,'-k.')
plt.plot(d_f,x_sf,'-r.')
plt.suptitle('divided by the carrier amplitude')
plt.legend('ratio between the measured voltage out and the df and sf')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(311)
plt.plot(d_f,d_rf1,'-k.')
plt.plot(d_f,d_rf2,'-r.')
ax.set_ylabel('Volts (V)')
plt.legend(['rf monitor c1','rf monitor c2'],loc='upper right')
ax.set_xlim([0,upper_frequency])
ax2 = fig.add_subplot(312)
plt.plot(d_f,d_mf_c1,'-k.')
plt.plot(d_f,d_mf_c2,'-r.')
plt.legend(['measured carrier 1 amplitudes@electrode', 'measured carrier 1 amplitudes@electrode'],loc='upper right')
ax2.set_ylabel('Volts ($\mu$V)')
ax2.set_xlabel('Frequency(MHz)')
ax2.set_xlim([0,upper_frequency])

ax3 = fig.add_subplot(313)
plt.plot(d_f,d_mf_df,'-k.')
plt.plot(d_f,d_mf_sf,'-r.')
plt.legend(['difference amplitudes@electrode', 'sum amplitudes@electrode'],loc='upper right')
ax3.set_ylabel('Volts ($\mu$V)')
ax3.set_xlabel('Frequency(MHz)')
ax3.set_xlim([0,upper_frequency])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
plt.suptitle('RF versus measured carrier response amplitude')
plot_filename = 'dual_rf_response.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(d_f,d_rf1/np.max(d_rf1),'-k.')
plt.plot(d_f,d_rf2/np.max(d_rf2),'-b.')
plt.plot(d_f,d_mf_c1/np.max(d_mf_c1),'-r.')
plt.plot(d_f,d_mf_c2/np.max(d_mf_c2),'-m.')
ax.set_xlim([0,upper_frequency])
ax.set_ylabel('Normalized Volts (V)')
plt.legend(['rf monitor c1','rf monitor c2','measured carrier 1 amplitude@electrode','measured carrier 2 amplitude@electrode'],loc='upper right')
ax2.set_ylabel('normalized')
ax2.set_xlabel('Frequency(MHz)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.suptitle('RF versus measured carrier response shape(normalized)')
plot_filename = 'dual_rf_response_normalized.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(d_f,d_mf_df/np.max(d_mf_df),'-k.')
plt.plot(d_f,d_mf_sf/np.max(d_mf_sf),'-r.')

plt.plot(d_f,d_rf_df/np.max(d_rf_df),'-b.')
plt.plot(d_f,d_rf_sf/np.max(d_rf_sf),'-m.')
ax.set_xlim([0,upper_frequency])
ax.set_ylabel('Normalized Volts (V)')
plt.legend(['difference measured','sum measured','rf difference','rf sum'],loc='upper right')
ax.set_xlabel('Frequency(MHz)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.suptitle('Difference and sum frequencies')
plot_filename = 'dual_diffsum_electrode_response_normalized.png'
plt.savefig(plot_filename)
plt.show()











