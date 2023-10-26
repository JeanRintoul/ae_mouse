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

# ae_filename    = 't2_F21_electrical\\t25_stream.npy'
ae_filename    = 't2_F21_electrical\\t30_stream.npy'
ae_data = np.load(ae_filename)
a,b = ae_data.shape
print ('ae shape',a,b)

# h_filename    = 't1_F21_hydrophone\\t33_stream.npy'
h_filename    = 't1_F21_hydrophone\\t30_stream.npy'
h_data = np.load(h_filename)
a,b = h_data.shape
print ('pressure shape',a,b)

start_pause     = int(1*Fs)
end_pause       = int(3*Fs)

fsignal = 1e6*ae_data[ae_channel]/gain
pressure = h_data[hydrophone_channel]*1000/sensitivity  # KPa
print ('pressure(kPa)',np.max(pressure)*2)


# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(211)
# plt.plot(t,fsignal,'k')
# ax2 = fig.add_subplot(212)
# plt.plot(t,pressure,'k')
# plt.show()

fft_p = fft(pressure[start_pause:end_pause])
fft_p = np.abs(2.0/(end_pause-start_pause) * (fft_p))[1:(end_pause-start_pause)//2]
# 
fft_m = fft(fsignal[start_pause:end_pause])
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
# 
fft_us = fft(h_data[rf_channel][start_pause:end_pause])
fft_us = np.abs(2.0/(end_pause-start_pause) * (fft_us))[1:(end_pause-start_pause)//2]
# 
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]

plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2

fz = 16 

ymax = 0.5
xmax = 10
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
ax.plot(frequencies,fft_m,'k')
ax.set_xlim([0,xmax*2])
ax.set_ylim([0,ymax])
plt.xticks(fontsize=fz)
plt.yticks(fontsize=fz)
# plt.legend(['ae_chan'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig("hydrophone_efield_df.png", bbox_inches="tight") 
plt.show()

fig = plt.figure(figsize=(7,5))
ax2 = fig.add_subplot(111)
ax2.plot(frequencies,fft_p,'k')
# plt.legend(['hydrophone'],loc='upper right')
ax2.set_xlim([0,xmax*2])
ax2.set_ylim([0,ymax])
plt.xticks(fontsize=fz)
plt.yticks(fontsize=fz)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.savefig("hydrophone_pfield_df.png", bbox_inches="tight") 
plt.show()


fig = plt.figure(figsize=(7,5))
ax2 = fig.add_subplot(111)
ax2.plot(frequencies,fft_p,'k')
# plt.legend(['hydrophone'],loc='upper right')
ax2.set_xlim([0,20])
ax2.set_ylim([0,1.25])
plt.xticks(fontsize=fz)
plt.yticks(fontsize=fz)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.savefig("hydrophone_pfield_df_zoom.png", bbox_inches="tight") 
plt.show()

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
ax.plot(frequencies,fft_m,'k')
# ax.set_xlim([0,5100])
ax.set_xlim([500000-xmax*2,500000+xmax*2])
ax.set_ylim([0,np.max(fft_m)])
plt.xticks(fontsize=fz)
plt.yticks(fontsize=fz)
# plt.legend(['ae_chan'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig("hydrophone_efield_carrier.png", bbox_inches="tight") 
plt.show()

fig = plt.figure(figsize=(7,5))
ax2 = fig.add_subplot(111)
ax2.plot(frequencies,fft_p,'k')
# plt.legend(['hydrophone'],loc='upper right')
ax2.set_xlim([500000-xmax*2,500000+xmax*2])
ax2.set_ylim([0,np.max(fft_p)])
plt.xticks(fontsize=fz)
plt.yticks(fontsize=fz)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.savefig("hydrophone_pfield_carrier.png", bbox_inches="tight") 
plt.show()

fig = plt.figure(figsize=(7,5))
ax2 = fig.add_subplot(111)
ax2.plot(frequencies,fft_p,'k')
# plt.legend(['hydrophone'],loc='upper right')
ax2.set_ylim([0,ymax])
ax2.set_xlim([1e6-xmax*2,1e6+xmax*2])
plt.xticks(fontsize=fz)
plt.yticks(fontsize=fz)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.savefig("hydrophone_pfield_sf.png", bbox_inches="tight") 
plt.show()

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
ax.plot(frequencies,fft_m,'k')
# ax.set_xlim([0,5100])
# ax.set_xlim([1e6-5000,1e6+5000])
ax.set_xlim([1e6-xmax*2,1e6+xmax*2])
ax.set_ylim([0,ymax])
plt.xticks(fontsize=fz)
plt.yticks(fontsize=fz)
# plt.legend(['ae_chan'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig("hydrophone_efield_sf.png", bbox_inches="tight") 
plt.show()
# 
# resistor_current_mon = 49.9  #  49.9 Ohms for current monitor, 1k resistor 
# 
# lowcut  = 1e6 -5 
# highcut = 1e6 +5
# sos500  = signal.iirfilter(17, [lowcut, highcut], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')

#  Filter Design 
# nyq_rate = Fs / 2.0
# # # The desired width of the transition from pass to stop,
# # relative to the Nyquist rate.  
# width = 500.0/nyq_rate
# # The desired attenuation in the stop band, in dB.
# ripple_db = 60.0
# Compute the order and Kaiser parameter for the FIR filter.
# Ntaps, beta = kaiserord(ripple_db, width)
# The cutoff frequency of the filter.
# lowcut  = abs(500000-f)- bandwidth/2 # 504000
# highcut = abs(500000-f) + bandwidth/2 #  514000
# sos = signal.iirfilter(17, [lowcut, highcut], rs=60, btype='band',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# w, h = signal.sosfreqz(sos, 2000, fs=Fs)

# e_lowcut  = f-bandwidth/2
# e_highcut = f+bandwidth/2
# sos_efield = signal.iirfilter(17, [e_lowcut, e_highcut], rs=60, btype='band',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# w_e, h_e = signal.sosfreqz(sos, 2000, fs=Fs)                       
# cut = 15
# sos_dc = signal.iirfilter(17, [cut], rs=60, btype='lowpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')

# ae_idx = abs(500000-f)
# e_idx  = f





# element,ae_idx = find_nearest(frequencies,ae_idx)
# # print ('ae idx is: ',element,ae_idx)
# element,e_idx = find_nearest(frequencies,e_idx)
# # print ('e idx is: ',element,e_idx)
 
# rf_channel  = 0
# hydrophone_channel = 1 
# i_channel   = 3
# v_channel   = 4 
# ae_channel  = 6  
# e_channel   = 7 

# # resistor_current_mon = 49.9 
# gain           = 50
# filename = 'transient_stream_data3.npy'
# filename_p = 'ptransient_stream_data.npy'
# d = np.load(filename_p)
# a,b,c = d.shape
# data  = d.transpose(1,0,2).reshape(b,-1) 
# p_d   = data[hydrophone_channel]
# rf_d  = 10*data[rf_channel]

# # load the other file. 
# d = np.load(filename)
# a,b,c = d.shape
# data = d.transpose(1,0,2).reshape(b,-1) 
# # print (data.shape)

# fsignal = 1e6*data[ae_channel]/gain
# #
# fft_rf = fft(rf_d)
# fft_rf= np.abs(2.0/N * (fft_rf))[1:N//2]
# #
# fft_h = fft(p_d)
# fft_h = np.abs(2.0/N * (fft_h))[1:N//2]
# #
# fft_aefield = fft(data[ae_channel])
# fft_ae = np.abs(2.0/N * (fft_aefield))[1:N//2]
# #
# fft_efield = fft(10*data[e_channel])
# fft_e = np.abs(2.0/N * (fft_efield))[1:N//2]
# #
# # dc_filtered = signal.sosfilt(sos_dc, data[ae_channel])
# # ae_filtered = signal.sosfilt(sos, fsignal)
# # e_filtered  = signal.sosfilt(sos_efield, data[e_channel])

# frf_500  = signal.sosfilt(sos500, rf_d)
# fh_500  = signal.sosfilt(sos500, p_d)
# fae_500  = signal.sosfilt(sos500, fsignal)

# upper_xlim = 1e6*2
# element,e_idx = find_nearest(frequencies,2)
# fft_ae[0:e_idx] = 0 

# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(111)
# ax.plot(frequencies/1e6,fft_h,'k')
# ax.set_xlim([0,1.1])
# # plt.legend(['applied e field'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig("hydrophone.png", bbox_inches="tight") 
# plt.show()


# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(111)
# ax.plot(frequencies,fft_e,'k')
# ax.set_xlim([0,upper_xlim])
# plt.legend(['applied e field'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig("efield.png", bbox_inches="tight") 
# plt.show()



# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(311)
# ax.plot(frequencies,fft_rf,'k')
# plt.legend(['rf'],loc='upper right')
# ax.set_xlim([0,upper_xlim])
# ax2 = fig.add_subplot(312)
# ax2.plot(frequencies,fft_h,'k')
# plt.legend(['hydrophone'],loc='upper right')
# ax2.set_xlim([0,upper_xlim])
# ax3 = fig.add_subplot(313)
# ax3.plot(frequencies,fft_ae,'k')
# ax3.set_xlim([0,upper_xlim])
# plt.legend(['ae_channel'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plt.savefig("hydrophone_vs_rf_amplifier.png", bbox_inches="tight") 
# plt.show()


# filter around 500khz on all channels
# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(311)
# ax.plot(t,frf_500,'k')
# plt.legend(['rf'],loc='upper right')
# ax2 = fig.add_subplot(312)
# ax2.plot(t,fh_500,'k')
# plt.legend(['hydrophone'],loc='upper right')
# ax3 = fig.add_subplot(313)
# ax3.plot(t,fae_500,'k')
# plt.legend(['ae_channel'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plt.savefig("hydrophone_vs_rf_amplifier.png", bbox_inches="tight") 
# plt.show()

# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(311)
# ax.plot(t,rf_d,'k')
# plt.legend(['rf'],loc='upper right')
# ax2 = fig.add_subplot(312)
# ax2.plot(t,p_d,'k')
# plt.legend(['hydrophone'],loc='upper right')
# ax3 = fig.add_subplot(313)
# ax3.plot(t,fsignal,'k')
# plt.legend(['ae_channel'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plt.savefig("hydrophone_vs_rf_amplifier.png", bbox_inches="tight") 
# plt.show()

