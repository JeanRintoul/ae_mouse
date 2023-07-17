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
test_no = 1

aeti_variables = {
'type':'impedance',         # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 1e5,                  # 
'duration': 30.0,            # 
'position': test_no,
'pressure_amplitude': 0.0,
'pressure_frequency': 10000.0,
'current_amplitude': 0.5,   #  its actually a voltage .. Volts. 0.0002
'current_frequency': 10000,
'prf_frequency': 2.0,       # number of times per second the current frequency sine wave is repeated. 
'ae_channel': 0,            # the channel of the measurement probe. 
'e_channel': 4,             # this is the voltage measured between the stimulator probes. 
'rf_monitor_channel': 1,    # this output of the rf amplifier. 
'current_monitor_channel': 5,  # this is the current measurement channel of the transformer. 
'start_pause': 0.2,         # percent of file in ramp mode. 
'end_pause': 0.8,           # 
'no_ramp':1.0,              # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':1,                   # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':10,        # the current and voltage monitor both have attenuators on them 
'marker_channel':2,
'command_c':'mouse_stream',
'save_folder_path':'D:\\mouse_aeti\\e74_nr_ketxyl_screw_mouse\\pulse_test',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
#  
result, data_out            = m.aeti_recording(**aeti_variables)
print ('impedance:', data_out[0])
data,idx_lag,original_data  = m.align_data_to_marker_channel(**aeti_variables)

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

raw_pp = np.max(data[m_channel])-np.min(data[m_channel])
# print ('min and max of raw signal:',np.min(data[m_channel]),np.max(data[m_channel])  )
print ('DC offset raw signal:',np.mean(data[m_channel]) )

f_idx = m.find_nearest(frequencies,f)
fft_pp = fft_m[f_idx]*2
fft_baseline = np.median(fft_m)*2
print ('FFT pp median = ',fft_baseline)
print ('FFT pp height = ',fft_pp)
goal_amplitude = 0.000001  #  1 microvolt. 
gain = 1
desired_signal_height = goal_amplitude*gain 
print ('desired_signal_height',desired_signal_height)
current_signal_height = fft_pp/gain
print ('current signal height ', current_signal_height)
print ('current signal to desired ratio ', current_signal_height/desired_signal_height )
print ('raw_pp', raw_pp)

print ('snr: ',20*np.log(fft_pp/fft_baseline) )


sine_wave = 0.2*np.sin(2 * np.pi * f * t)
fft_test = fft(sine_wave[start_pause:end_pause])
fft_test = np.abs(2.0/(end_pause-start_pause) * (fft_test))[1:(end_pause-start_pause)//2]


low_cut  = f-400  # where do papers normally put this cut-off? 
high_cut = f+400

sos = signal.iirfilter(17, [low_cut, high_cut], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
filtered_data = signal.sosfiltfilt(sos, data[m_channel])

'''
   Implement mains harmonic filter, until I can remove the source of it physically. 


'''
fsignal = data[m_channel,:]
hm_start = 50 
hm_end   = 600 
mains_harmonics = np.linspace(hm_start,hm_end,int (hm_end/hm_start ) ) 
for i in range(len(mains_harmonics)):
    mains_low  = mains_harmonics[i] -3
    mains_high = mains_harmonics[i] +3
    mains_sos = iirfilter(17, [mains_low,mains_high], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
    fsignal = sosfiltfilt(mains_sos, fsignal)

filtered_data = fsignal 

# # # 
current_filtered_data = signal.sosfiltfilt(sos, i_data)




filt_pp = np.max(filtered_data)-np.min(filtered_data)

fft_filt = fft(filtered_data[start_pause:end_pause])
fft_filt = np.abs(2.0/(end_pause-start_pause) * (fft_filt))[1:(end_pause-start_pause)//2]
fft_filt_baseline = np.median(fft_filt)*2

print ('filtered data snr: ',20*np.log(filt_pp/fft_filt_baseline)  )
print ('filtered data pp amplitude', filt_pp)

# print ('min and max of raw signal:',)
# It would be useful to see the raw measured data, and it's fourier transform. 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(311)
plt.plot(t, data[m_channel], color = 'purple')
# plt.plot(t, sine_wave, color = 'red')
plt.plot(t, filtered_data, color = 'orange')
plt.legend(['raw measured data','filtered data'],loc='upper right', fontsize = 8)
# ax.set_xlim([0,8])
# plt.xlim([4.00,4.01])
# plt.xlim([4.00,4.8])
ax2 = fig.add_subplot(312)
# plt.plot(frequencies, fft_test, color = 'red')
plt.plot(frequencies, fft_m, color = 'purple')
# plt.plot(frequencies, fft_vapplied, color = 'blue')
ax2.set_xlim([0,5000])
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
plot_filename = savepath + '\\t'+str(test_no)+'_raw_measured_pulse_data.png'
plt.savefig(plot_filename,bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(frequencies, fft_m, color = 'purple')
ax.set_xlim([0,50])
ax.set_xlabel('Frequency(Hz)')
ax.set_ylabel('Volts(V)')
plt.legend(['measured data'],loc='upper right', fontsize = 8)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = savepath + '\\t'+str(test_no)+'_fft_pulse_noise.png'
plt.savefig(plot_filename,bbox_inches='tight')
plt.show()
# 
# 
# 
# 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(t, filtered_data, color = 'orange')
ax.set_ylabel('Volts(V)')
ax.set_xlabel('Time(s)')
plt.legend(['measured data'],loc='upper right', fontsize = 8)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = savepath + '\\t'+str(test_no)+'_pulse.png'
plt.savefig(plot_filename,bbox_inches='tight')
plt.show()