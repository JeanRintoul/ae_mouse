#!/usr/bin/python
'''
 Author: Jean Rintoul Date: 23/08/2021

 Signal test: 
 1. Send in a signal in saline, that measures in microvolt range at the measurement electrode. 
    This way I can calibrate the gain of the SR560. 
 2. DC offset removal test. Generate a sine wave across the stimulation electrodes, measure it. Is there a DC offset? 
    Then try sequentially moving pieces of equipment off the ground plane to determine where the problem is. 


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
test_no = 18

aeti_variables = {
'type':'impedance',         # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 1e5,                  # 
'duration': 15.0,           # 
'position': test_no,
'pressure_amplitude': 0.0,
'pressure_frequency': 5.0,
'current_amplitude': 8.0,   #  its actually a voltage .. Volts. 0.0002
'current_frequency':  5,
# 'prf_frequency': 2.0,       # number of times per second the current frequency sine wave is repeated. 
'ae_channel': 4,            # the channel of the measurement probe. 
'rf_monitor_channel': 2,    # this output of the rf amplifier.  
'e_channel': 6,             # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5, # this is the current measurement channel of the transformer. 
'marker_channel':7,
'start_pause': 0.2,         # percent of file in ramp mode. 
'end_pause': 0.8,           # 
'no_ramp':1.0,              # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':1,                   # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':10,        # the current and voltage monitor both have attenuators on them 
'command_c':'mouse_stream',
'save_folder_path':'D:\\mouse_aeti\\e88_meps',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
# 
result, data_out            = m.aeti_recording(**aeti_variables)
print ('impedance:', data_out[0])
data                        =   m.copy_to_folder_and_return_data(**aeti_variables)
f = aeti_variables['current_frequency']
marker_channel = aeti_variables['marker_channel']
m_channel = aeti_variables['ae_channel'] 
v_channel = aeti_variables['e_channel'] 
i_channel = aeti_variables['current_monitor_channel'] 
current_signal_frequency = aeti_variables['current_frequency'] 
Fs       = aeti_variables['Fs'] 
duration = aeti_variables['duration'] 
savepath = aeti_variables['save_folder_path']
attenuation = aeti_variables['IV_attenuation']

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


fft_m = fft(data[m_channel][start_pause:end_pause])
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]

fft_vapplied = fft(v_data[start_pause:end_pause])
fft_vapplied = np.abs(2.0/(end_pause-start_pause) * (fft_vapplied))[1:(end_pause-start_pause)//2]

xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]


low_cut  = f-50  # where do papers normally put this cut-off? 
high_cut = f+50
if low_cut < 0: 
   low_cut = 0.1

sos = signal.iirfilter(17, [low_cut, high_cut], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
current_filtered_data = signal.sosfiltfilt(sos, i_data)
'''
   Implement mains harmonic filter, until I can remove the source of it physically. 

'''
# fsignal = data[m_channel,:]
# hm_start = 50 
# hm_end   = 600 
# mains_harmonics = np.linspace(hm_start,hm_end,int (hm_end/hm_start ) ) 
# for i in range(len(mains_harmonics)):
#     mains_low  = mains_harmonics[i] -3
#     mains_high = mains_harmonics[i] +3
#     mains_sos = iirfilter(17, [mains_low,mains_high], rs=60, btype='bandstop',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
#     fsignal = sosfiltfilt(mains_sos, fsignal)
# 
# filtered_data = fsignal 
# 
# if it is a pulse, it means the it is a spread out function in fft space. 
# 
current_pp = np.max(current_filtered_data) - np.min(current_filtered_data)
print ('current_pp(mA):', current_pp)

# It would be useful to see the raw measured data, and it's fourier transform. 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(311)
plt.plot(t, 10*data[v_channel], color = 'red')
plt.plot(t, data[m_channel], color = 'purple')

# plt.plot(t, sine_wave, color = 'red')
# plt.plot(t, filtered_data, color = 'orange')
plt.legend(['v_output','raw measured data'],loc='upper right', fontsize = 8)

ax2 = fig.add_subplot(312)
# plt.plot(frequencies, fft_test, color = 'red')
plt.plot(frequencies, fft_m, color = 'purple')
# plt.plot(frequencies, fft_vapplied, color = 'blue')
ax2.set_xlim([0,100])
ax2.set_xlabel('Frequency(Hz)')
plt.legend(['measured data'],loc='upper right', fontsize = 8)
# ax2.set_ylim([0,0.0002])
ax3 = fig.add_subplot(313)
plt.plot(t, i_data, color = 'g')
plt.plot(t,current_filtered_data,color='r')
# plt.plot(t, v_data, color = 'b',linewidth=2)
plt.legend(['I(mA)'],loc='upper right', fontsize = 8)
# plt.legend(['I(mA)','V'],loc='upper right', fontsize = 8)
plt.xlabel('time(s)')
# ax3.set_xlim([0,8])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
hspace = 0.4
plt.subplots_adjust( hspace=hspace)
plot_filename = savepath + '\\t'+str(test_no)+'_mep_data.png'
plt.savefig(plot_filename,bbox_inches='tight')
plt.show()
