'''

Title: demodulation and a few metrics. Shows FFT comparison of Demodded and original, and time series. 

Author: Jean Rintoul
Date: 02.02.2023

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
from scipy.signal import kaiserord, lfilter, firwin, freqz
import pandas as pd
import sys
sys.path.append('D:\\mouse_aeti')  #  so that we can import from the parent folder. 
import mouse_library as m


# IQ demodulate function. 
# I use constant to get rid of the rectification around zero issue. 
def demodulate(measured_signal,carrier_f,t):
    # That's interesting, if I add a DC offset here, I obtain the result correctly, otherwise the result is rectified into obscurity. 
    offset = 100
    offset_adjustment = offset*np.sin(2 * np.pi * carrier_f * t)
    IQ = (measured_signal+offset_adjustment)*np.exp(1j*(2*np.pi*carrier_f*t )) 
    # idown = measured_signal*np.cos(2*np.pi*carrier_f*t)
    # qdown = -measured_signal*np.sin(2*np.pi*carrier_f*t)    
    idown = np.real(IQ)
    qdown = np.imag(IQ)
    v = idown + 1j*qdown
    mag = np.abs(v)
    mag = mag - np.mean(mag)
    return mag

# 

file_number           = 3 # vep with pressure. 

# file_number           = 51 # vep with no pressure. 
data_filepath         = 'D:\\mouse_aeti\\e95_VEP_test_only\\t4\\'
# 
gain                    = 500
carrier_f               = 500000
led_frequency           = 7 # in Hz. 
Fs                      = 5e6
marker_channel          = 7 
m_channel               = 0 
rf_channel              = 2
duration                = 8.0
print ('duration',duration)
led_duration = 1/(2*led_frequency)
timestep                = 1.0/Fs
N                       = int(Fs*duration)
# print("expected no. samples:",N)

# create time and frequencies arrays.
t  = np.linspace(0, duration, N, endpoint=False)
print (len(t),N)


low      = 1.0
high     = 50
sos_low  = iirfilter(27, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
#
fc = carrier_f
# remove the original VEP signal from the data such that it doesn't interfere with the demodulation. 
l = fc - 2*high 
h = fc + 2*high
sos_carrier_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

# 
filename    = data_filepath + 't'+str(file_number)+'_stream.npy'
data = np.load(filename)
a,b = data.shape
print ('data shape',a,b)
markerdata = data[marker_channel]
marker     = markerdata/np.max(markerdata)
rfdata     = 10*data[rf_channel]
rawdata    = 1e6*data[m_channel]/gain
rawdata    = rawdata-np.mean(rawdata)
# 
fraw                = sosfiltfilt(sos_carrier_band,rawdata)
demodulated_signal  = demodulate(fraw,carrier_f,t) 
# 
filtered_rawdata        = sosfiltfilt(sos_low, rawdata)
filtered_demoddata      = sosfiltfilt(sos_low, demodulated_signal)


carrier_filtered        = sosfiltfilt(sos_carrier_band,rawdata)
print ('finished filtering')
# 
# it would be better to isolate the area of file with markers in it only. 
xf = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies = xf[1:N//2]
window = np.hanning(N)
# Convert it to microvolts by taking the gain into account. 
fft_rawdata = fft(rawdata*window)
fft_rawdata = np.abs(2.0/N * (fft_rawdata))[1:N//2]
fft_demodrawdata = fft(filtered_demoddata*window)
fft_demodrawdata = np.abs(2.0/N * (fft_demodrawdata))[1:N//2]

rf_fft_data = fft(rfdata*window)
rf_fft_data = np.abs(2.0/N * (rf_fft_data))[1:N//2]

# Decimate arrays before plotting so it is less memory intensive. 
desired_plot_sample_rate    = 1e5
decimation_factor           = int(Fs/desired_plot_sample_rate)
marker                      = marker[::decimation_factor]    
filtered_rawdata            = filtered_rawdata[::decimation_factor]      
filtered_demoddata          = filtered_demoddata[::decimation_factor]   
drawdata                    = rawdata[::decimation_factor] 
ddemodulated_signal         = demodulated_signal[::decimation_factor] 
rfdata                      = rfdata[::decimation_factor]
td                          = t[::decimation_factor]  
# 
# 
# 
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
ax2 = ax.twinx()
# plt.plot(frequencies - carrier_f,fft_rawdata,'r')
ax.plot(frequencies,fft_rawdata,'k')
ax2.plot(frequencies,rf_fft_data,'r')
ax2.set_ylabel('Volts(V)')
ax.set_xlim([carrier_f - 50 ,carrier_f+50])
# plt.legend(['measurement data','rf data fft'])
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Frequencies (Hz)')
# ax.set_xlim([0,1000000])
# ax.set_xlim([0,100])
ax.legend(['measurement data'],loc='upper left')
ax2.legend(['rf data'],loc=0)
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.suptitle('acoustic connection')
plot_filename = 'unwanted_mixing2.png'
plt.savefig(plot_filename)
plt.show()


fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(t,carrier_filtered,'blue')
# ax.set_xlim([carrier_f - 50 ,carrier_f+50])
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Time (s)')
plot_filename = 'filtered_carrier.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(311)
plt.plot(td,100*marker,'g')
plt.plot(td,filtered_rawdata,'k')
# plt.plot(td,filtered_demoddata,'r')
# plt.plot(td,drffiltered_demoddata,'purple') 

# ax.set_ylim([-160,100])
# ax.set_xlim([0.5,2.5])
# plt.legend(['filtered m data($\mu$V)','filtered demod data($\mu$V)','no band filter','LED marker'],loc='lower right')
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Time (s)')
ax2=fig.add_subplot(312)
plt.plot(td,drawdata,'k')
# plt.plot(td,rfdata,'b')
ax2.set_xlim([0,duration])
plt.legend(['raw measurement data($\mu$V)','rf data'],loc='upper right')

ax3=fig.add_subplot(313)
plt.plot(frequencies,fft_rawdata,'k')
plt.plot(frequencies,fft_demodrawdata,'r')
# plt.plot(frequencies,rffft_demodrawdata,'purple')
ax3.set_xlim([0.0,100])
ax3.set_ylim([0.0,400])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
# plot_filename = 'transient_spikerandom_single_trial.png'
# plt.savefig(plot_filename)
plt.show()


