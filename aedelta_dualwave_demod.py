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
file_list_start = 21
file_list_end   = 22
gain            = 1000
# 
# 
dfx             = 10000  # this means we have
base_frequency  = 500000
# 
aeti_variables = {
'type':'demodulation',    # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 5e6,                # 
'duration': 8.0,          # 
'position': 1, 
'pressure_amplitude': 0.0,    # 0.1 is 1MPa. 
'pressure_frequency': 500000.0,
# 'pressure_prf':1020,        # pulse repetition frequency for the sine wave. Hz. 
# 'pressure_ISI':0,           # inter trial interval in seconds. 
# 'current_amplitude': 0.0,   # its actually a voltage .. Volts. 
# 'current_frequency': 70,    # 
'current_amplitude': 0.0,     # its actually a voltage .. Volts. 
'current_frequency': 500000,  # 
'ti_frequency':500000 + dfx,  # comment this out to get a continuous wave. 
# 'ti_frequency': 0,        # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 0,            # the channel of the measurement probe. 
'rf_monitor_channel': 4,    # this output of the rf amplifier.  
'e_channel': 6,             # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5, # this is the current measurement channel of the transformer. 
'marker_channel':7,
'end_null': 0.02,           # start of end null. 
'end_pause': 0.875,         # start of end ramp
'start_null': 0.125,        # percent of file set to zero at the beginning. 
'start_pause': 0.25,        # percent of file in ramp mode or null at start. 
'no_ramp':0.0,              # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':10,        # the current and voltage monitor both have attenuators on them 
'command_c':'code\\mouse_stream',
'save_folder_path':'D:\\mouse_aeti\\e100_neural_recording_pat_e_mouse',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
# 
# 
# 
def demodulate(measured_signal):
    offset              = np.min(measured_signal)
    offset_adjustment   = offset*np.cos(2 * np.pi * carrier_f  * t)
    IQ                  = (measured_signal+offset_adjustment)*np.exp(1j*(2*np.pi*carrier_f *t )) 
    idown               = np.real(IQ)
    qdown               = np.imag(IQ)
    idown               = sosfiltfilt(lp_filter, idown)
    qdown               = sosfiltfilt(lp_filter, qdown)  
    rsignal             = -(idown + qdown)
    rsignal             = rsignal - np.mean(rsignal) 
    return rsignal
# 
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
# define lots of filters. 
# 
# define the carrier frequency as the difference frequency. 
carrier_f = dfx 
# 
# 
filter_cutoff           = 200
lp_filter               = iirfilter(17, [filter_cutoff], rs=60, btype='lowpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')
low  = 1
high = 40
sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
sos_center_cut = iirfilter(17, [carrier_f-2,carrier_f+2], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
double_carrier_filter   = iirfilter(17, [carrier_f*2-2,carrier_f*2+2], rs=60, btype='bandpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')
l = carrier_f - high
h = carrier_f + high
sos_carrier_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 
# 
# 
file_list = np.linspace(file_list_start,file_list_end,(file_list_end-file_list_start+1),dtype=np.int16)
print ('file list:', file_list)
for test_number in range(len(file_list)):
    aeti_variables['position']  = file_list[test_number]
    print ('test number:', file_list[test_number]) 
    result, data_out            = m.aeti_recording(**aeti_variables)
    data                        = m.copy_to_folder_and_return_data(**aeti_variables)
    #  
    marker     = data[marker_channel]/np.max(data[marker_channel])
    rawdata    = 1e6*data[m_channel]/gain
    rfdata     = 10*data[rf_channel]
    vdata      = data[v_channel]
    # perform filtering and demodulation. 
    fraw                    = sosfiltfilt(sos_carrier_band, rawdata) 
    fraw                    = sosfiltfilt(sos_center_cut, fraw) 
    demod_data              = demodulate(fraw)
    demodulated             = sosfiltfilt(sos_low, demod_data)
    filtered_rawdata        = sosfiltfilt(sos_low, rawdata)
    double_carrier_signal   = sosfiltfilt(double_carrier_filter, rawdata)
    # 
    fft_m = fft(rawdata[start_pause:end_pause])
    fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
    # 
    fft_rf = fft(rfdata[start_pause:end_pause])
    fft_rf = np.abs(2.0/(end_pause-start_pause) * (fft_rf))[1:(end_pause-start_pause)//2]
    #
    fft_demod = fft(demodulated[start_pause:end_pause])
    fft_demod = np.abs(2.0/(end_pause-start_pause) * (fft_demod))[1:(end_pause-start_pause)//2]
    # 
    # Decimate arrays before plotting, so it is less memory intensive. 
    desired_plot_sample_rate    = 1e5
    decimation_factor           = int(Fs/desired_plot_sample_rate)
    marker                      = marker[::decimation_factor]    
    filtered_rawdata            = filtered_rawdata[::decimation_factor]      
    demodulated                 = demodulated[::decimation_factor]  
    dvdata                      = vdata[::decimation_factor]
    drfdata                     = rfdata[::decimation_factor]    
    td                          = t[::decimation_factor]  
    # 
    # look at the raw data to check preamp OVLD stuff. 
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    plt.plot(t,data[m_channel],'k')
    plot_filename = savepath + '\\t'+str(file_list[test_number])+'raw_datacheck.png'
    plt.savefig(plot_filename)
    plt.show()

    # Plot the result.           
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(311)
    # plt.plot(td,np.max(filtered_rawdata[int(2*Fs):int(duration*Fs) ])*marker,color='g') 
    plt.plot(td,filtered_rawdata,color='r')
    plt.plot(td,demodulated,color='k')
    # ax.set_ylim([-1000,1000])
    ax.set_xlim([0,duration])
    ax.set_ylabel('Volts ($\mu$V)')
    plt.legend(['neural','demodulated'],loc='upper right')

    ax2 = fig.add_subplot(312)
    plt.plot(t,rfdata,color='k') 
    plt.plot(t,double_carrier_signal,color='pink',alpha =0.8)     
    ax2.set_xlim([0,duration])
    ax2.set_ylabel('Volts ($\mu$V)')
    plt.legend(['raw rf mon','sum f on m chan'],loc='upper right')

    ax3 = fig.add_subplot(313)
    plt.plot(frequencies,fft_m,'r')
    plt.plot(frequencies,fft_demod,'b')        
    ax3.set_xlim([0,40])
    ax3.set_ylim([0,50])
    plt.legend(['fft m','fft demod'],loc='upper right')
    ax3.set_ylabel('Volts ($\mu$V)')
    ax3.set_xlabel('Frequency(Hz)')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    plot_filename = savepath + '\\t'+str(file_list[test_number])+'_datacheck.png'
    plt.savefig(plot_filename)
    plt.show()


# plt.autoscale(enable=True, axis='x', tight=True)

