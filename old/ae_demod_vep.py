#!/usr/bin/python
'''
 Author: Jean Rintoul Date: 13/08/2021

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
# 
# increment this for each test. 
# 
# 
test_no = 44

# start to stop file labels. 
file_list_start = 1
file_list_end   = 1

gain            = 1
# 
aeti_variables = {
'type':'demodulation',      # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 5e6,                  # 
'duration': 8.0,            # overflows at 4seconds, 3.5 seconds works. 
'position': test_no,
'pressure_amplitude': 0.4,  # 0.15 is about 400kPa. 
'pressure_frequency': 672800.0,
'current_amplitude': 0.0,   # its actually a voltage .. Volts. 
'current_frequency': 1000,  # 
# 'ti_frequency': 0,        # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'rf_monitor_channel': 7,    # this output of the rf amplifier. 
'marker_channel':7,
'ae_channel': 1,            # the channel of the measurement probe. 
'e_channel': 0,             # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 1, # this is the current measurement channel of the transformer. 
'start_pause': 0.125,         # percent of file in ramp mode. 
'end_pause': 0.875,         # 
'start_null': 0.125,        # percent of file set to zero at the beginning. 
'start_pause': 0.25,        # percent of file in ramp mode or null at start. 
# 'end_pause': 0.3,         # percent of file where down ramp starts. 
# 'end_null': 0.65,         # percent of file set to zero at the end. 
'no_ramp':0.0,              # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':10,        # the current and voltage monitor both have attenuators on them 
'command_c':'mouse_stream',
'save_folder_path':'D:\\mouse_aeti\\e86_demod',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}


# IQ demodulate function. 
def demodulate(measured_signal,carrier_f,t):
    IQ = measured_signal*np.exp(1j*(2*np.pi*carrier_f*t ))
    # idown = signal*np.cos(2*np.pi*carrier_f*t)
    # qdown = -signal*np.sin(2*np.pi*carrier_f*t)    
    idown = np.real(IQ)
    qdown = np.imag(IQ)
    v = idown + 1j*qdown
    return np.abs(v)


rf_channel = aeti_variables['rf_monitor_channel']
marker_channel = aeti_variables['marker_channel']
m_channel = aeti_variables['ae_channel'] 
pressure_signal_frequency = aeti_variables['pressure_frequency'] 
Fs       = aeti_variables['Fs'] 
duration = aeti_variables['duration'] 
savepath = aeti_variables['save_folder_path']
timestep = 1.0/Fs
N = int(Fs*duration)
# print("expected no. samples:",N)
t               = np.linspace(0, duration, N, endpoint=False)
start_pause     = int(aeti_variables['start_pause'] * N+1)
end_pause       = int(aeti_variables['end_pause'] *N-1)
# print ('start and end:',start_pause,end_pause)
resistor_current_mon    = 49.9  # 49.9 Ohms for current monitor, 


low                     = 2.5
high                    = 50
high                    = 300 
sos_band                = iirfilter(17, [low,high], rs=60, btype='bandpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')

cut     = 1000
clow    = 672000
chigh   = 674000
sos_carrier_band = iirfilter(17, [cut], rs=60, btype='highpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')


xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]


file_list = np.linspace(file_list_start,file_list_end,(file_list_end-file_list_start+1),dtype=np.int16)
print ('file list:', file_list)
for test_number in range(len(file_list)):
    aeti_variables['position']  = file_list[test_number]
    print ('test number:', file_list[test_number]) 
    result, data_out            = m.aeti_recording(**aeti_variables)
    data,idx_lag,original_data  = m.align_data_to_marker_channel(**aeti_variables)
    # extract data, so we can save out a sneak peak. 
    marker     = data[marker_channel]/np.max(data[marker_channel])
    rawdata    = 1e6*data[m_channel]/gain
    rfdata     = 10*data[rf_channel]

    fraw           = sosfiltfilt(sos_carrier_band, rawdata) 
    demod_data     = demodulate(fraw,pressure_signal_frequency,t)

    filtered_rawdata   = rawdata
    filtered_demoddata = demod_data
 
    filtered_rawdata    = sosfiltfilt(sos_band, filtered_rawdata)
    filtered_demoddata  = sosfiltfilt(sos_band, filtered_demoddata)

    fft_m = fft(rawdata[start_pause:end_pause])
    fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
    #
    fft_demod = fft(demod_data[start_pause:end_pause])
    fft_demod = np.abs(2.0/(end_pause-start_pause) * (fft_demod))[1:(end_pause-start_pause)//2]
    # 
    # Decimate arrays before plotting, so it is less memory intensive. 
    desired_plot_sample_rate = 1e5
    decimation_factor = int(Fs/desired_plot_sample_rate)
    marker = marker[::decimation_factor]    
    filtered_rawdata = filtered_rawdata[::decimation_factor]      
    filtered_demoddata = filtered_demoddata[::decimation_factor]  
    rfdata= rfdata[::decimation_factor]
    td = t[::decimation_factor]  
    #            
    # fig = plt.figure(figsize=(10,6))
    # ax = fig.add_subplot(311)

    # # plt.plot(td,filtered_demoddata,color='b')
    # plt.plot(td,filtered_rawdata,color='r')
    # plt.plot(td,50*marker,color='g')  
    # 
    # ax.set_ylim([-100,100])
    # ax.set_xlim([0,np.max(t)])
    # ax.set_ylabel('Volts ($\mu$V)')
    # ax2 = fig.add_subplot(312)
    # plt.plot(td,rfdata,color='k')
    # # plt.plot(t,data[m_channel],color='r')
    # ax2.set_xlim([0,np.max(t)])
    # ax2.set_ylabel('Volts ($\mu$V)')
    # 
    # ax3 = fig.add_subplot(313)
    # plt.plot(frequencies,fft_m,'r')
    # plt.plot(frequencies,fft_demod,'b')    
    # # ax3.set_ylim([0,10])
    # ax3.set_xlim([0,50])
    # ax3.set_ylabel('Volts ($\mu$V)')
    # ax3.set_xlabel('Frequency(Hz)')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    # ax3.spines['right'].set_visible(False)
    # ax3.spines['top'].set_visible(False)
    # plot_filename = savepath + '\\t'+str(file_list[test_number])+'_datacheck.png'
    # plt.savefig(plot_filename)
    # plt.show()


# plt.autoscale(enable=True, axis='x', tight=True)

