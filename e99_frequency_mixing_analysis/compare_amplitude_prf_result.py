'''

Title: demodulation and a few metrics. Shows FFT comparison of Demodded and original, and time series. 

Author: Jean Rintoul
Date: 02.02.2023

'''
# 
# 
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
import pandas as pd
import sys
sys.path.append('D:\\mouse_aeti')  #  so that we can import from the parent folder. 
import mouse_library as m
# 
# 
# IQ demodulate function. 
def demodulate(signal,carrier_f,t):
    IQ = signal*np.exp(1j*(2*np.pi*carrier_f*t ))
    # idown = signal*np.cos(2*np.pi*carrier_f*t)
    # qdown = -signal*np.sin(2*np.pi*carrier_f*t)    
    idown = np.real(IQ)
    qdown = np.imag(IQ)
    v = idown + 1j*qdown
    # is this the complex casting issue?
    v = sosfiltfilt(mouse_sos_band, np.abs(v) )
    return v

#
# saline_file  = 'D:\\mouse_aeti\\e82_ae_demod\\t6_saline_phantom\\'+'t'+str(20)+'_stream.npy'

saline_file  = 'Z:\\experiment_history\\e82_ae_demod\\t8\\'+'t'+str(59)+'_stream.npy'

mouse_file   = 'Z:\\experiment_history\\e82_ae_demod\\t4\\'+'t'+str(68)+'_stream.npy'



gain                    = 1000 
saline_Fs               = 1e7
mouse_Fs                = 5e6
carrier_f               = 672800
marker_channel          = 2 
m_channel               = 0 
rf_channel              = 3 

led_frequency           = 4 # in Hz. 
sduration               = 3.0 
mduration               = 8.0  # 4.0 for file 2. 

led_duration = 1/(2*led_frequency)
saline_timestep = 1.0/saline_Fs
mouse_timestep = 1.0/mouse_Fs
saline_N = int(saline_Fs*sduration)
mouse_N = int(mouse_Fs*mduration)

# create time arrays

saline_t  = np.linspace(0, sduration, saline_N, endpoint=False)
mouse_t  = np.linspace(0, mduration, mouse_N, endpoint=False)



# 
low  = 0.1
high = 300
saline_sos_band = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=saline_Fs,
                       output='sos')
mouse_sos_band = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=mouse_Fs,
                       output='sos')

saline_data = np.load(saline_file)
a,b = saline_data.shape
print ('saline shape',a,b)
smarkerdata = saline_data[marker_channel]
smarker     = smarkerdata/np.max(smarkerdata)
srawdata    = 1e6*saline_data[m_channel]/gain
srfdata     = 10*saline_data[rf_channel]
sdemodulated_signal = demodulate(srawdata,carrier_f,saline_t)
sfiltered_rawdata   = srawdata
sfiltered_demoddata = sdemodulated_signal
sfiltered_rawdata    = sosfiltfilt(saline_sos_band, sfiltered_rawdata)
sfiltered_demoddata    = sosfiltfilt(saline_sos_band, sfiltered_demoddata)
print ('demod data shape: ',sfiltered_rawdata.shape,len(sfiltered_rawdata))
# Convert it to microvolts by taking the gain into account. 
sstart = int(0.3*saline_Fs)
send = int(2.5*saline_Fs)
snewN = len(srawdata[sstart:send])

smax_time = saline_t[send] - saline_t[sstart]

# saline_t  = np.linspace(0, smax_time, snewN, endpoint=False)
saline_xf = np.fft.fftfreq(snewN, d=saline_timestep)[:snewN//2]
saline_frequencies = saline_xf[1:snewN//2]

sfft_rawdata        = fft(srawdata[sstart:send])
sfft_rawdata        = np.abs(2.0/snewN * (sfft_rawdata))[1:snewN//2]
sfft_demodrawdata   = fft(sdemodulated_signal[sstart:send])
sfft_demodrawdata   = np.abs(2.0/snewN * (sfft_demodrawdata))[1:snewN//2]




# # # # # #  Now compute the mouse results. 
duration       = 8.0 
mouse_t        = np.linspace(0, duration, mouse_N, endpoint=False)
marker_channel = 2 
m_channel      = 0 
rf_channel     = 7 
mouse_data = np.load(mouse_file)
a,b = mouse_data.shape
print ('shape',a,b)
mouse_markerdata = mouse_data[marker_channel]
mmarker     = mouse_markerdata/np.max(mouse_markerdata)
mrawdata    = 1e6*mouse_data[m_channel]/gain
mrfdata     = 10*mouse_data[rf_channel]
mdemodulated_signal = demodulate(mrawdata,carrier_f,mouse_t) 
mfiltered_rawdata   = mrawdata
mfiltered_demoddata = mdemodulated_signal
# mains_harmonics = [50,100]
# for i in range(len(mains_harmonics)):
#     mains_low  = mains_harmonics[i] -2
#     mains_high = mains_harmonics[i] +2
#     mains_sos = iirfilter(17, [mains_low,mains_high], rs=60, btype='bandstop',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
#     filtered_rawdata  = sosfiltfilt(mains_sos, rawdata)
#     filtered_demoddata = sosfiltfilt(mains_sos, demodulated_signal)
mfiltered_rawdata    = sosfiltfilt(mouse_sos_band, mfiltered_rawdata)
mfiltered_demoddata    = sosfiltfilt(mouse_sos_band, mfiltered_demoddata)
print ('demod data shape: ',mfiltered_rawdata.shape,len(mfiltered_rawdata))
# Convert it to microvolts by taking the gain into account. 
mstart = int(0.3*mouse_Fs)
mend   = int(2.5*mouse_Fs)
mnewN  = len(mrawdata[mstart:mend])

mmax_time = mouse_t[mend] - mouse_t[mstart]

    
window = np.hanning(mnewN)

# mouse_t  = np.linspace(0, mmax_time, mnewN, endpoint=False)
mouse_xf = np.fft.fftfreq(mnewN, d=mouse_timestep)[:mnewN//2]
mouse_frequencies = mouse_xf[1:mnewN//2]

mfft_rawdata = fft(mrawdata[mstart:mend])
mfft_rawdata = np.abs(2.0/mnewN * (mfft_rawdata))[1:mnewN//2]
mfft_demodrawdata = fft(mdemodulated_signal[mstart:mend])
mfft_demodrawdata = np.abs(2.0/mnewN * (mfft_demodrawdata))[1:mnewN//2]
# do rf data to. 
mfft_rfdata = fft(mrfdata[mstart:mend])
mfft_rfdata = np.abs(2.0/mnewN * (mfft_rfdata))[1:mnewN//2]

# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(111)
# plt.axvline(x=(carrier_f),color='r',linestyle='--')
# plt.plot(mouse_frequencies,mfft_rawdata,'k')
# plt.plot(mouse_frequencies,mfft_rfdata,'g')
# ax.set_xlim([carrier_f-10000,carrier_f+10000])
# # ax.set_ylim([0,7])
# ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xlabel('Frequencies(Hz)')
# plt.legend(['carrier','V measurement','RF monitor'],loc='upper left')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = 'mouse_carrier.png'
# plt.savefig(plot_filename)
# plt.show()

fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(111)
plt.axvline(x=(carrier_f*2+1000),color='c',linestyle='--')
plt.plot(mouse_frequencies,mfft_rawdata,'k')
plt.plot(saline_frequencies,sfft_rawdata,'r')
plt.plot(mouse_frequencies,mfft_rfdata,'g')
ax.set_xlim([carrier_f*2,carrier_f*2+5000])
ax.set_ylim([0,7])
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Frequencies(Hz)')

plt.legend(['sum frequency','V t4 measurement','V t8 measurement','rf data'],loc='lower right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = 'mouse_sum_prf.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(111)
plt.axvline(x=(1000),color='c',linestyle='--')
plt.plot(mouse_frequencies,mfft_rawdata,'k')
plt.plot(saline_frequencies,sfft_rawdata,'r')
plt.plot(mouse_frequencies,mfft_rfdata,'g')
ax.set_ylim([0,15])
ax.set_xlim([0,5000])
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Frequencies(Hz)')

plt.legend(['diff frequency','V t4 measurement','V t8 measurement','rf data'],loc='lower right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.suptitle('VEP@'+str(led_frequency)+'Hz LED flashing frequency n='+str(n_events))
plot_filename = 'mouse_diff_prf.png'
plt.savefig(plot_filename)
plt.show()
#
# 
# Compare the time series of the rf data, and the raw data. 
# Filter them both around the sum and difference frequencies. 
mouse_Fs                = 5e6
fsignal = mrawdata
rfsignal = 10*mouse_data[rf_channel]
low  = 1000 - 5
high = 1000 + 5
sos_df = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=mouse_Fs,
                       output='sos')
dfsignal  = sosfiltfilt(sos_df, fsignal)
rfdfsignal  = sosfiltfilt(sos_df, rfsignal)
# 
# 
fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(211)
plt.plot(mouse_t,dfsignal,'k')
ax2  = fig.add_subplot(212)
plt.plot(mouse_t,rfdfsignal,'r')
plt.suptitle('df')
plt.show()



low  = 672800*2 + 1000 - 5
high = 672800*2 + 1000 + 5
sos_sf = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=mouse_Fs,
                       output='sos')
sfsignal  = sosfiltfilt(sos_sf, fsignal)
rfsfsignal  = sosfiltfilt(sos_sf, rfsignal)
# 
fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(211)
plt.plot(mouse_t,sfsignal,'k')
ax2  = fig.add_subplot(212)
plt.plot(mouse_t,rfsfsignal,'r')
plt.suptitle('sf')
plt.show()


low  = 672800 - 5
high = 672800 + 5
sos_carrier = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=mouse_Fs,
                       output='sos')
csignal  = sosfiltfilt(sos_carrier, fsignal)
rfcsignal  = sosfiltfilt(sos_carrier, rfsignal)
# 
fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(211)
plt.plot(mouse_t,csignal,'k')
ax2  = fig.add_subplot(212)
plt.plot(mouse_t,rfcsignal,'r')
plt.suptitle('carrier')
plt.show()


# 
# 
# 
# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(311)
# plt.plot(mouse_frequencies,mfft_rawdata,'k')
# plt.plot(saline_frequencies,sfft_rawdata,'r')
# ax.set_xlim([0,60])
# ax.set_ylim([0,40])
# ax2  = fig.add_subplot(312)
# plt.plot(mouse_frequencies,mfft_rawdata,'k')
# plt.plot(saline_frequencies,sfft_rawdata,'r')
# ax2.set_xlim([1000,1000+60])
# ax2.set_ylim([0,2])
# ax3  = fig.add_subplot(313)
# plt.plot(mouse_frequencies,mfft_rawdata,'k')
# plt.plot(saline_frequencies,sfft_rawdata,'r')
# ax3.set_xlim([carrier_f,carrier_f+60])
# ax3.set_ylim([0,100])
# plt.show()

# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(211)
# plt.plot(mouse_t,mrawdata,'k')
# # plt.plot(saline_t,srawdata,'r')
# # plt.legend(['mouse','saline'])
# ax.set_xlabel('Time(s)')
# ax.set_ylabel('Volts ($\mu$V)')

# ax2  = fig.add_subplot(212)
# plt.plot(mouse_t,mrfdata,'r')
# ax2.set_ylabel('RF amplifier monitor (V)')
# ax2.set_xlabel('Time(s)')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = 'mouse_prf_time_series.png'
# plt.savefig(plot_filename)
# plt.show()
# 
# 
# Plot the comparison of measured data with RF data. 
# are the spectrums the same?
# 
# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(211)
# plt.plot(mouse_frequencies,mfft_rfdata,'b')
# ax  = fig.add_subplot(212)
# plt.plot(mouse_frequencies,mfft_rawdata,'r')
# plt.show()

# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(211)
# plt.plot(mouse_t,mrawdata,'k')
# ax.set_xlabel('Time(s)')
# ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xlim([3,3.01])
# ax2  = fig.add_subplot(212)
# plt.plot(mouse_t,mrfdata,'r')
# ax2.set_ylabel('RF amplifier monitor (V)')
# ax2.set_xlabel('Time(s)')
# ax2.set_xlim([3,3.01])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = 'mouse_prf_time_series_closeup.png'
# plt.savefig(plot_filename)
# plt.show()
