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


# IQ demodulate function. 
# I use constant to get rid of the rectification around zero issue. 
def demodulate(measured_signal,carrier_f,t):
    # That's interesting, if I add a DC offset here, I obtain the result correctly, otherwise the result is rectified into obscurity. 
    offset = 1000
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
fs                     = 7 
# carrier_f             = 672800
fc                     = 500000
# intermodulation products = [] 
# 
# 1st order: 500khz  7Hz    : fc, fs
# 2nd order: 499.93kh 500.07khz   fc + fs, fc - fs   sum + diff
second_order = [fc + fs, fc-fs] # also 2*fc, 2*fs
# 3rd order: 2fc - fs, 2fc + fs, 2fs+ fc, 2fs - fc
third_order = [2*fc-fs, 2*fc + fs, 2*fs +fc,2*fs - fc]
print ('third order', third_order)
# [999993, 1000007, 500014, -499986]
# set the carrier as 2*fc and fc. 
# diff 7hz, 
# Should I try to baseband the harmonics? 
# To do that set the carrier fc, then multiply once by fs.  
#  
# what happens to negative mixing products? 
fourth_order = [2*fc+2*fs, 2*fc - 2*fs]
# carrier is 2*fc, the multiply by fs. 
print ('fourth order', fourth_order)
fifth_order = [3*fc - 2*fs, 3*fc + 2*fs, 3*fs -2*fc, 3*fs + 2*fc]
print ('fifth order', fifth_order)
# [1499986, 1500014, -999979, 1000021]
# carrier is 3*fc, then multiply by fs? 
# carrier is 3*fs, then multiply by fc?
# How to baseband each intermodulation product? 
# I wouldnt both to implement this, until I can clearly see it. 
# can I see this in the Uren data? 

# file_number   = 68  # vep with no pressure. 
file_number           = 30   # vep with pressure. 
 
data_filepath         = 'D:\\mouse_aeti\\e86_demod\\t3_saline_twotone\\'
# 
gain                  = 1
# 
# 
rf_channel          = 0    
marker_channel      = 4
m_channel          = 6
# vmeasure_channel    = 5 
# 
print ('carrier_f',fc)
led_frequency           = 5 # in Hz. 
# duration                = 8.0
duration                = 0.1
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

# start    = 2.0
stop     = 300
sos_low  = iirfilter(27, [stop], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

l = fc - 2*stop  
h = fc + 2*stop
sos_carrier_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')



# 
# filename    = data_filepath + 't'+str(file_number)+'_stream.npy'
filename    = 'demod_stream.npy'

data = np.load(filename)
a,b = data.shape
print ('data shape',a,b)
markerdata = data[marker_channel]
marker     = markerdata/np.max(markerdata)
rfdata     = 10*data[rf_channel]
rawdata    = 1e6*data[m_channel]/gain


# Create the transmitted signal. 
A_signal        = [1,1,1,1] # These are the relative amplitudes of the electrophysiological signal.  
f_signal        = [5,60,100,200] # these are the frequencies of the electrophysiological signal 
e_signal        = np.zeros(len(t))
for i in range(len(f_signal)):
        e_signal = e_signal + A_signal[i]*np.sin(2 * np.pi * f_signal[i] * t)
# rawdata = (e_signal)*np.sin(2 * np.pi * carrier_f * t)  + e_signal 

fraw                  = sosfiltfilt(sos_carrier_band,rawdata)
demodulated_signal    = demodulate(fraw,fc,t) 
rawdata               = rawdata - np.mean(rawdata)

filtered_rawdata      = sosfiltfilt(sos_low, rawdata)
filtered_demoddata    = sosfiltfilt(sos_low, demodulated_signal)
print ('finished filtering')

xf = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies = xf[1:N//2]

xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]

window = np.hanning(newN)
# Convert it to microvolts by taking the gain into account. 
fft_rawdata = fft(rawdata[start_pause:end_pause]*window)
fft_rawdata = np.abs(2.0/(end_pause-start_pause) * (fft_rawdata))[1:(end_pause-start_pause)//2]

fft_demodrawdata = fft(filtered_demoddata[start_pause:end_pause]*window)
fft_demodrawdata = np.abs(2.0/(end_pause-start_pause) * (fft_demodrawdata))[1:(end_pause-start_pause)//2]


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
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
# plt.plot(frequencies - carrier_f,fft_rawdata,'r')
plt.axvline(x=fc*3,color='r')
plt.axvline(x=fc*2,color='r')
plt.axvline(x=fc,color='r')
plt.plot(frequencies,fft_rawdata,'k')

ax.set_xlim([0,fc*3+10000])
# ax.set_ylim([0,100])
plt.show()

fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(311)
plt.plot(td[int(desired_plot_sample_rate):],filtered_rawdata[int(desired_plot_sample_rate):],'k')
plt.plot(td[int(desired_plot_sample_rate):],filtered_demoddata[int(desired_plot_sample_rate):],'r')
# ax3.set_xlim([0.0,300])
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Time (s)')

ax2=fig.add_subplot(312)
plt.plot(frequencies,fft_rawdata,'k')
ax2.set_xlim([0.0,300])

ax3=fig.add_subplot(313)
plt.plot(frequencies,fft_demodrawdata,'r')
ax3.set_xlim([0.0,300])
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


