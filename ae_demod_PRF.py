#!/usr/bin/python
'''
 Author: Jean Rintoul Date: 13/08/2023

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
# 
file_list_start = 1
file_list_end   = 1
gain            = 1000
# 
aeti_variables = {
'type':'demodulation',       # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 1e4,                  # recording sampling frequency. 
#'Fs': 5e6, 
#'long_recording':1,        # if it is a long recording, loop the data stored in the function generator. Specify the number of seconds to loop. 
'duration': 12.0,            # overflows 
'position': 1,
'USMEP': 1,                  #  this allows the sample rate of recording to be different from the function generator output. 
'pressure_amplitude': 0.05,  #0.13
'pressure_frequency': 500000.0,
#'pressure_burst_length': 0.000004, # pressure burst length in milliseconds. (this should be 2 cycles). 
# 'pressure_burst_length': 0.004,
#'pressure_burst_length': 0.0004, # pressure burst length in milliseconds. (this should be more cycles). 
'pressure_burst_length': 0.0004, # pressure burst length in milliseconds. (this should be more cycles). 
'pressure_prf': 1020,         # pulse repetition frequency for the sine wave. Hz. 
#'pressure_prf':520,         # pulse repetition frequency for the sine wave. Hz. 
'pressure_ISI': 0,           # inter trial interval in seconds. 
# 'pi_frequency':500000 + 1020, 
'current_amplitude': 1.0,   # its actually a voltage .. Volts. 
'current_frequency': 100000,    # 
'current_ISI':0,
'current_burst_length':0.004,
'current_prf':1020,
# 'ti_frequency':23, 
# 'ti_frequency': 0,        # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 0,            # the channel of the measurement probe. 
'rf_monitor_channel': 4,    # this output of the rf amplifier.  
'e_channel': 6,             # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5, # this is the current measurement channel of the transformer. 
'marker_channel': 7,
'end_null': 0.00,           # start of end null. 
'end_pause': 1.0,           # start of end ramp
'start_null': 0.00,         # percent of file set to zero at the beginning. 
'start_pause': 0.00,        # percent of file in ramp mode or null at start. 
'no_ramp': 1.0,              # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation': 10,        # the current and voltage monitor both have attenuators on them 
'command_c':'code\\mouse_stream',
'save_folder_path':'D:\\ae_mouse\\e107_revision',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
# 

def demodulate(in_signal,carrier_f): 
    idown = in_signal*np.cos(2*np.pi*carrier_f*t)
    qdown = in_signal*np.sin(2*np.pi*carrier_f*t)   
    demodulated_signal = idown + 1j*qdown
    return np.abs(demodulated_signal)
# 

v_channel = aeti_variables['e_channel']
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
end_pause       = int(aeti_variables['end_pause'] * N-1)
# print ('start and end:',start_pause,end_pause)
resistor_current_mon    = 49.9  # 49.9 Ohms for current monitor, 
# window = np.hanning(end_pause-start_pause)


xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 
# signal_of_interest    = aeti_variables['current_frequency'] 
signal_of_interest      = 90 # 
prf                     = aeti_variables['pressure_prf'] 
# 
sos_finalpass = iirfilter(17, [signal_of_interest], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
low  = prf -(signal_of_interest)
high = prf +(signal_of_interest)
modulation_filter = iirfilter(17, [low,high], rs=60, btype='bandpass',
                   analog=False, ftype='cheby2', fs=Fs,
                   output='sos')
# 
file_list = np.linspace(file_list_start,file_list_end,(file_list_end-file_list_start+1),dtype=np.int16)
print ('file list:', file_list)
for test_number in range(len(file_list)):
    aeti_variables['position']  = file_list[test_number]
    print ('test number:', file_list[test_number]) 
    result, data_out            = m.aeti_recording(**aeti_variables)
    data                        = m.copy_to_folder_and_return_data(**aeti_variables)
    print ('shape:',data.shape,len(t))
    # Coment out here so I can skip the post-processing. 
    marker     = data[marker_channel]/np.max(data[marker_channel])
    rawdata    = 1e6*data[m_channel]/gain
    rfdata     = 10*data[rf_channel]
    vdata      = data[v_channel]
    # 
    modulated_signal    = sosfiltfilt(modulation_filter, rawdata)
    demodulated_signal  = demodulate(modulated_signal,prf)
    # Now filter both data the same way. 
    filtered_rawdata        = sosfiltfilt(sos_finalpass, rawdata)
    filtered_demodulated    = sosfiltfilt(sos_finalpass, demodulated_signal)
    # 
    fft_demodulated = fft(filtered_demodulated[start_pause:end_pause])
    fft_demodulated = np.abs(2.0/(end_pause-start_pause) * (fft_demodulated))[1:(end_pause-start_pause)//2]
    # 
    fft_m = fft(filtered_rawdata[start_pause:end_pause])
    fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]

    fft_r = fft(rawdata[start_pause:end_pause])
    fft_r = np.abs(2.0/(end_pause-start_pause) * (fft_r))[1:(end_pause-start_pause)//2]
    # 
    fft_rf = fft(rfdata[start_pause:end_pause])
    fft_rf = np.abs(2.0/(end_pause-start_pause) * (fft_rf))[1:(end_pause-start_pause)//2]
    #
    fft_v = fft(vdata[start_pause:end_pause])
    fft_v = np.abs(2.0/(end_pause-start_pause) * (fft_v))[1:(end_pause-start_pause)//2]    
    # Decimate arrays before plotting, so it is less memory intensive. 
    desired_plot_sample_rate    = 500
    decimation_factor           = int(Fs/desired_plot_sample_rate) 
    dfiltered_rawdata           = filtered_rawdata[::decimation_factor]   
    dfiltered_demodulated       = filtered_demodulated[::decimation_factor]           
    td                          = t[::decimation_factor]  
    #  
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(311)
    plt.plot(t,data[m_channel],'k')
    ax2 = fig.add_subplot(312)
    plt.plot(t,vdata,'k')
    plt.plot(t,rfdata,'r')    
    ax3 = fig.add_subplot(313)
    # plt.plot(frequencies,fft_r,'k')
    plt.plot(frequencies,fft_v,'k')    
    plt.plot(frequencies,fft_rf,'r')       
    ax3.set_xlim([0,prf + 50])
    plot_filename = savepath + '\\t'+str(file_list[test_number])+'_raw_datacheck.png'
    plt.savefig(plot_filename)
    plt.show()

    # Plot the result.           
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(411)
    plt.plot(td,dfiltered_rawdata,color='k') 
    ax.set_xlim([0,np.max(t)]) 
    ax.set_ylabel('Volts ($\mu$V)')    
    plt.legend(['original'],loc='upper right')
    # 
    ax2 = fig.add_subplot(412)
    plt.plot(td,dfiltered_demodulated,color='r')    
    ax2.set_xlim([0,np.max(t)])
    ax2.set_ylabel('Volts ($\mu$V)')
    plt.legend(['demodulated'],loc='upper right')
    # 
    # ax3 = fig.add_subplot(413)
    # plt.plot(t,rfdata,color='r') 
    # ax3.set_xlim([0,np.max(t)])
    # ax3.set_ylabel('Volts ($\mu$V)')
    ax4 = fig.add_subplot(223)
    plt.plot(frequencies,fft_m,'k')    
    ax4.set_xlim([0,signal_of_interest+30])    
    ax4.set_ylabel('Volts ($\mu$V)')
    ax4.set_xlabel('Frequency(Hz)')
    plt.legend(['original signal'],loc='upper right')
    # 
    ax5 = fig.add_subplot(224)
    plt.plot(frequencies,fft_demodulated,'r')       
    ax5.set_xlim([0,signal_of_interest+30])    
    ax5.set_ylabel('Volts ($\mu$V)')
    ax5.set_xlabel('Frequency(Hz)')
    plt.legend(['demodulated'],loc='upper right')
    # 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    # ax3.spines['right'].set_visible(False)
    # ax3.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)  
    ax5.spines['right'].set_visible(False)
    ax5.spines['top'].set_visible(False) 
    # 
    plot_filename = savepath + '\\t'+str(file_list[test_number])+'_datacheck.png'
    plt.savefig(plot_filename)
    plt.show()


# plt.autoscale(enable=True, axis='x', tight=True)

