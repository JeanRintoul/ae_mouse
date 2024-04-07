#!/usr/bin/python
'''
 Author: Jean Rintoul Date: 23/08/2021

USMEP test
Ultrasound motor evoked potential test. 



'''
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib
from scipy import interpolate
import serial, time
from subprocess import check_output
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.fft import fft, fftfreq
import mouse_library as m
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt


# increment this for each test. 
test_no = 1

aeti_variables = {
'type':'impedance',         # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 1e5,                  # this will change just the sample rate, and not the f generated rate if those two lines are left uncommented. 
'USMEP': 1,                 # when usmep == 1, the current and pressure Fs can be different to the recorded Fs. In this case the US is at 5Mhz, but the recording frequency is 100kHz. This decreases the amount of data that needs to be dumped to disk. 
# 
'duration': 13.0,           # 
'position': test_no,
'pressure_amplitude': 0.1,  # 0.15v is about 415 kPa. 
'pressure_frequency': 500000.0,
'pressure_prf':500,           # pulse repetition frequency for the sine wave. Hz. 
'pressure_ISI':5,             # inter trial interval in seconds. 
'pressure_burst_length':0.3,  # burst length in seconds. 
'current_amplitude':0,        # its actually a voltage .. Volts. 0.0002
'current_frequency': 350,
'prf_frequency': 0.5,       # number of times per second the current frequency sine wave is repeated. 
'ae_channel': 4,            # the channel of the measurement probe. 
'e_channel': 6,             # this is the voltage measured between the stimulator probes. 
'rf_monitor_channel': 2,    # this output of the rf amplifier. 
'current_monitor_channel': 5,  # this is the current measurement channel of the transformer. 
'start_pause': 0.1,         # percent of file in ramp mode. 
'end_pause': 0.9,           # 
'no_ramp':1.0,              # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':100,                   # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':10,        # the current and voltage monitor both have attenuators on them 
'marker_channel':2,
'command_c':'mouse_stream',
'save_folder_path':'D:\\mouse_aeti\\e91_usmeps',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
#  

result, data_out            = m.aeti_recording(**aeti_variables)
# print ('impedance:', data_out[0])
data,idx_lag,original_data  = m.align_data_to_marker_channel(**aeti_variables)

# pressure_signal = 10*data[1]
#  
f = aeti_variables['current_frequency']
marker_channel = aeti_variables['marker_channel']
m_channel = aeti_variables['ae_channel'] 
v_channel = aeti_variables['e_channel'] 
i_channel = aeti_variables['current_monitor_channel'] 
rf_channel = aeti_variables['rf_monitor_channel']
current_signal_frequency = aeti_variables['current_frequency'] 
Fs       = aeti_variables['Fs'] 
duration = aeti_variables['duration'] 
savepath = aeti_variables['save_folder_path']
attenuation = aeti_variables['IV_attenuation']
gain        = aeti_variables['gain']
#  
timestep = 1.0/Fs
N = int(Fs*duration)
# print("expected no. samples:",N)
t               = np.linspace(0, duration, N, endpoint=False)
start_pause     = int(aeti_variables['start_pause'] * N+1)
end_pause       = int(aeti_variables['end_pause'] *N-1)
# print ('start and end:',start_pause,end_pause)
resistor_current_mon    = 49.9  # 49.9 Ohms for current monitor, 
i_data         = -5*data[i_channel]/resistor_current_mon
i_data         = 1000*i_data # convert to mA. 
v_data         = -10*data[v_channel]

if aeti_variables['experiment_configuration'] == 'monopolar':
   print ('monopolar mode')
   resistor_current_mon    = 47  # 49.9 Ohms for current monitor, 
   i_data                  = -attenuation*data[i_channel]/resistor_current_mon
   i_data                  = 1000*i_data # convert to mA. 
   v_data                  = -attenuation*data[v_channel]
#  
fft_m = fft(data[m_channel][start_pause:end_pause])
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]

fft_vapplied = fft(v_data[start_pause:end_pause])
fft_vapplied = np.abs(2.0/(end_pause-start_pause) * (fft_vapplied))[1:(end_pause-start_pause)//2]

xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]

# Remove any 50Hz harmonics from the raw signal. 
# 
# convert into microvolts
fsignal = 1e6*data[m_channel]/gain
mains_harmonics = [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950]

mains_harmonics = np.linspace(50,5000,num=99)
for i in range(len(mains_harmonics)):
    mains_low  = mains_harmonics[i] -10
    mains_high = mains_harmonics[i] +5
    mains_sos = iirfilter(17, [mains_low,mains_high], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
    fsignal = sosfiltfilt(mains_sos, fsignal)


low_s  = 50
high_s = 800
sos_high = iirfilter(17, [low_s,high_s], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
high_filtered_data  = sosfiltfilt(sos_high, fsignal)


# fig = plt.figure(figsize=(10,6))
# plt.plot(t, data[1], color = 'purple') # this is the pressure output signal. 
# plt.legend(['pressure_signal'],loc='upper right', fontsize = 8)
# plot_filename = savepath + '\\t'+str(test_no)+'_pressure_signal.png'
# plt.savefig(plot_filename,bbox_inches='tight')
# plt.show()

# Plot the recorded data on the SR560, with gain and filtering, to show the EMG signal. 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(311)
plt.plot(t, data[rf_channel], color = 'purple') # this is the pressure output signal. 
plt.legend(['rf_signal'],loc='upper right', fontsize = 8)
plt.xlim(0,duration)

ax2 = fig.add_subplot(312)
plt.plot(t, fsignal, color = 'purple')
plt.plot(t, high_filtered_data, color = 'orange')
plt.legend(['raw','hp filtered'],loc='upper right', fontsize = 8)
plt.ylabel('Volts ($\\mu$V)')
plt.xlim(0,duration)

ax3 = fig.add_subplot(313)
sample = Fs
NFFT = 50000
noverlap = 20000
x = 1e6*data[m_channel]/gain
Pxx, freqs, bins, im = ax3.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=noverlap,cmap='viridis',mode='magnitude')
# # The `specgram` method returns 4 objects. They are:
# # - Pxx: the periodogram
# # - freqs: the frequency vector
# # - bins: the centers of the time bins
# # - im: the .image.AxesImage instance representing the data in the plot
# # pxx,  freq, t, cax = plt.specgram(data, Fs=32000)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')
plt.ylim(200,1000)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

hspace = 0.4
plt.subplots_adjust( hspace=hspace)
plt.suptitle('MEP with EMG recording')
plot_filename = savepath + '\\t'+str(test_no)+'_mep_data.png'
plt.savefig(plot_filename,bbox_inches='tight')
plt.show()
