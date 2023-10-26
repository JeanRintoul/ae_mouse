import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq,fftshift,ifft,ifftshift
from scipy.signal import blackman
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.signal import fftconvolve
#
# Design a band pass filter to isolate just the information I want to get the peak to peak amplitude more accurately.
# 
# 
# // Channel identities: 
# // 1. rf amplifier output
# // 2. hydrophone probe
# // 

# // 3. current monitor for e1
# // 4. current monitor for e2

# // 5. v mon e1 x10  
# // 6. v mon e2 x10 

# // 7. tiepie voltage output waveform x10 
# // 8. diff input voltage across 1k resister. x 10
Fs = 5e6
duration = 4.0

N  = Fs*duration
print("expected no. samples:",N)
t  = np.linspace(0, duration, int(N), endpoint=False)
timestep = 1.0/Fs
N  = int(N) 
xf = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies = xf[1:N//2]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

rf_channel          = 4
hydrophone_channel  = 1 
i_channel           = 5
v_channel           = 6 
ae_channel          = 0 
marker_channel      = 7
gain                = 100
sensitivity         = 0.033    #  hydrophone sensitivity is 33mV/MPa at 500kHz

ae_filename    = 'D:\\ae_mouse\\e115_F21_hydrophone\\t6_prf_electric_field_only\\t1_stream.npy'
ae_data = np.load(ae_filename)
a,b = ae_data.shape
print ('ae shape',a,b)

# h_filename    = 'D:\\ae_mouse\\t6_prf_electric_field_only\\prf_hydrophone_stream.npy'
# h_data = np.load(h_filename)
# a,b = h_data.shape
# print ('pressure shape',a,b)

start_pause     = int(1*Fs)
end_pause       = int(3*Fs)

fsignal = 1e6*ae_data[ae_channel]/gain
# pressure = h_data[hydrophone_channel]*1000/sensitivity  # KPa
# print ('pressure(kPa)',np.max(pressure)*2)


# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(211)
# plt.plot(t,fsignal,'k')
# ax2 = fig.add_subplot(212)
# plt.plot(t,pressure,'k')
# plt.show()

# fft_p = fft(pressure[start_pause:end_pause])
# fft_p = np.abs(2.0/(end_pause-start_pause) * (fft_p))[1:(end_pause-start_pause)//2]
# 
fft_m = fft(fsignal[start_pause:end_pause])
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
# 
# fft_us = fft(h_data[rf_channel][start_pause:end_pause])
# fft_us = np.abs(2.0/(end_pause-start_pause) * (fft_us))[1:(end_pause-start_pause)//2]
# 
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]

plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2

fz = 16 

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
ax.plot(frequencies,fft_m,'k')
ax.set_xlim([0,5100])
ax.set_ylim([0,6])
plt.xticks(fontsize=fz)
plt.yticks(fontsize=fz)
# plt.legend(['ae_chan'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig("efield_prf_df_t6.png", bbox_inches="tight") 
plt.show()

fig = plt.figure(figsize=(9,5))
ax = fig.add_subplot(111)
ax.plot(frequencies,fft_m,'k')
# ax.set_xlim([0,5100])
ax.set_xlim([480000,520000])
ax.set_ylim([0,np.max(fft_m)])
plt.xticks(fontsize=fz)
plt.yticks(fontsize=fz)
# plt.legend(['ae_chan'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig("efield_prf_hf_t6.png", bbox_inches="tight") 
plt.show()


fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
ax.plot(frequencies,fft_m,'k')
# ax.set_xlim([0,5100])
# ax.set_xlim([1e6-20000,1e6+20000])
ax.set_xlim([1e6,1e6+5100])
ax.set_ylim([0,1])
plt.xticks(fontsize=fz)
plt.yticks(fontsize=fz)
# plt.legend(['ae_chan'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig("efield_prf_sf_t6.png", bbox_inches="tight") 
plt.show()
# 