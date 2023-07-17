'''

Title: demodulation and a few metrics. Shows FFT comparison of Demodded and original, and time series. 

Author: Jean Rintoul
Date: 02.02.2023

I am having this abs problem in my real data... 

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


# def demodulate2(measured_signal,carrier_f,t):

# IQ demodulate function. 
# I use constant to get rid of the rectification around zero issue. 
def demodulate(measured_signal,carrier_f,t):
    # That's interesting, if I add a DC offset here, I obtain the result correctly, otherwise the result is rectified into obscurity. 
    offset = 10
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
fs                    = 7 
fc                    = 500000
file_number           = 6   # vep with pressure. 
data_filepath         = 'D:\\mouse_aeti\\e86_demod\\t4_saline_twotone\\'
# 
gain                  = 1
# 
# 
rf_channel          = 2    
marker_channel      = 7
m_channel           = 0
# vmeasure_channel    = 5 
# 
print ('carrier_f',fc)
led_frequency           = 5 # in Hz. 
duration                = 8.0

print ('duration',duration)
led_duration            = 1/(2*led_frequency)
Fs                      = 5e6
timestep                = 1.0/Fs
N = int(Fs*duration)
# print("expected no. samples:",N)
# marker_channel = 7 
# m_channel      = 0   
# rf_channel     = 2
# create time and frequencies arrays.
t  = np.linspace(0, duration, N, endpoint=False)
print (len(t),N)


start_pause     = int(2*Fs)
end_pause       = int(6.5*Fs)

start_pause     = int(0)
end_pause       = int(N)
newN            = end_pause -start_pause
print ('new N',newN )


stop     = 300
sos_low  = iirfilter(27, [stop], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

l = fc - 2*stop  
h = fc + 2*stop
sos_carrier_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

l2 = fc - 1
h2 = fc + 1
sos_carrier = iirfilter(30, [l2,h2], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')




# 
filename    = data_filepath + 't'+str(file_number)+'_stream.npy'
# filename    = 'demod_stream.npy'

data = np.load(filename)
a,b = data.shape
print ('data shape',a,b)
markerdata = data[marker_channel]
marker     = markerdata/np.max(markerdata)
rfdata     = 10*data[rf_channel]
rawdata    = 1e6*data[m_channel]/gain


# Create the transmitted signal. 
A_signal        = [1,1,1,1] # These are the relative amplitudes of the electrophysiological signal.  
f_signal        = [7,60,100,200] # these are the frequencies of the electrophysiological signal 
A_signal        = [1] # These are the relative amplitudes of the electrophysiological signal.  
f_signal        = [7] # these are the frequencies of the electrophysiological signal 

e_signal        = np.zeros(len(t))
for i in range(len(f_signal)):
        e_signal = e_signal + A_signal[i]*np.sin(2 * np.pi * f_signal[i] * t)
# here I add a ramped carrier frequency
# rawdata = (e_signal)*np.sin(2 * np.pi * fc * t) + t*1*np.sin(2 * np.pi * fc * t)
# + e_signal 

window = np.hanning(newN)

# What if I multiply it by an out of phase carrier signal the same amplitude. 
# rawdata = rawdata + 
# What is the shape of the 500khz data?
# 
l = fc - 1 
h = fc + 1
sos_rfdata = iirfilter(17, [l,h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

carrier_filtered_raw                 = sosfiltfilt(sos_rfdata,rawdata)

# remove the signal at the carrier frequency via a bandstop filter. 
# fraw                  = sosfiltfilt(sos_carrier,rawdata)
# take away everything on the sides of the signal of interest to be demodulated.
#  subtract the carrier from the raw data.
# rawdata = rawdata - carrier_filtered_raw 
 

fraw                  = sosfiltfilt(sos_carrier_band,rawdata)
demodulated_signal    = demodulate(fraw,fc,t) 
rawdata               = rawdata - np.mean(rawdata)

filtered_rawdata      = sosfiltfilt(sos_low, rawdata)
filtered_demoddata    = sosfiltfilt(sos_low, demodulated_signal)
print ('finished filtering')

xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]


fft_rfdata = fft(rfdata[start_pause:end_pause]*window)
fft_rfdata = np.abs(2.0/(end_pause-start_pause) * (fft_rfdata))[1:(end_pause-start_pause)//2]

rf_amplitudes      = sosfiltfilt(sos_rfdata, rfdata)
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(311)
plt.plot(t,rawdata,'k')
plt.plot(t,rawdata-carrier_filtered_raw,'r')
ax2 = fig.add_subplot(312)
# plt.plot(t,rfdata,'k')
plt.plot(t,rf_amplitudes/np.max(rf_amplitudes),'b')
plt.plot(t,0.5*carrier_filtered_raw/np.max(carrier_filtered_raw) ,'g')

ax3 = fig.add_subplot(313)
plt.plot(frequencies,fft_rfdata)
plt.show()
# 
# Convert it to microvolts by taking the gain into account. 
fft_rawdata = fft(rawdata[start_pause:end_pause]*window)
fft_rawdata = np.abs(2.0/(end_pause-start_pause) * (fft_rawdata))[1:(end_pause-start_pause)//2]

fft_demodrawdata = fft(filtered_demoddata[start_pause:end_pause]*window)
fft_demodrawdata = np.abs(2.0/(end_pause-start_pause) * (fft_demodrawdata))[1:(end_pause-start_pause)//2]


# Decimate arrays before plotting so it is less memory intensive. 
desired_plot_sample_rate    = 1e5
decimation_factor           = int(Fs/desired_plot_sample_rate)
marker                      = marker[::decimation_factor]    
dfiltered_rawdata            = filtered_rawdata[::decimation_factor]      
dfiltered_demoddata          = filtered_demoddata[::decimation_factor]   
drawdata                    = rawdata[::decimation_factor] 
ddemodulated_signal         = demodulated_signal[::decimation_factor] 
drfdata                      = rfdata[::decimation_factor]
td                          = t[::decimation_factor]  
# 
# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(111)
# # plt.plot(frequencies - carrier_f,fft_rawdata,'r')
# plt.axvline(x=fc*3,color='r')
# plt.axvline(x=fc*2,color='r')
# plt.axvline(x=fc,color='r')
# plt.plot(frequencies,fft_rawdata,'k')
# ax.set_xlim([0,fc*3+10000])
# # ax.set_ylim([0,100])
# plt.show()

l = 6 
h = 8
sos_final_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=1e5,
                       output='sos')
# 
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(311)
# 
# signals to compare at the end
# s1 = filtered_rawdata[int(desired_plot_sample_rate):]
s2 = dfiltered_demoddata
# s1 = sosfiltfilt(sos_final_band, s1)
s2 = sosfiltfilt(sos_final_band, s2)
# # 
plt.plot(t,filtered_rawdata/np.max(filtered_rawdata),'k')
# plt.plot(t,rfdata/np.max(rfdata),'b')
# plt.plot(t,2*rfdata/np.max(rfdata),'k')
# plt.plot(t,rf_amplitudes/np.max(rf_amplitudes),'b')
plt.plot(t,0.5*carrier_filtered_raw/np.max(carrier_filtered_raw) ,'g')
plt.plot(td,s2/np.max(s2),'r')
# 
# ax3.set_xlim([0.0,300])
# ax.set_ylim([-5,5])
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Time (s)')

ax2=fig.add_subplot(312)
plt.plot(frequencies,fft_rawdata,'k')
ax2.set_xlim([0.0,10])

ax3=fig.add_subplot(313)
plt.plot(frequencies,fft_demodrawdata,'r')
ax3.set_xlim([0.0,30])
# ax3.set_ylim([0.0,1.0])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
# plot_filename = 'transient_spikerandom_single_trial.png'
# plt.savefig(plot_filename)
plt.show()


