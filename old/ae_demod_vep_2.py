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
# start to stop file labels. 
file_list_start = 1
file_list_end   = 1
gain            = 10
frequency_of_interest   = 70
# 
aeti_variables = {
'type':'demodulation',    # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 5e6,                # 
'duration': 8.0,          # overflows at 4seconds, 3.5 seconds works. 
'position': 1,
# 'pressure_amplitude': 0.11,  # 0.15 is about 400kPa. 
'pressure_amplitude': 0.1,  # 0.1 is 1MPa. 
# 'pressure_frequency': 672800.0,
'pressure_frequency': 500000.0,
'current_amplitude': 0.0,   # its actually a voltage .. Volts. 
'current_frequency': 70,  # 
# 'ti_frequency': 0,        # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 0,            # the channel of the measurement probe. 
'rf_monitor_channel': 2,    # this output of the rf amplifier.  
'e_channel': 6,             # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5, # this is the current measurement channel of the transformer. 
'marker_channel':7,
'end_null': 0.02,         # start of end null. 
'end_pause': 0.875,         # start of end ramp
'start_null': 0.125,        # percent of file set to zero at the beginning. 
'start_pause': 0.25,        # percent of file in ramp mode or null at start. 
'no_ramp':0.0,              # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':10,        # the current and voltage monitor both have attenuators on them 
'command_c':'code\\mouse_stream',
'save_folder_path':'D:\\mouse_aeti\\e90_demod',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
# 
# 
carrier_f               = 500000
filter_cutoff           = 20000
lp_filter               = iirfilter(17, [filter_cutoff], rs=60, btype='lowpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')
# input filters to band pass and cut the raw data, getting rid of any excess. 
lcut                    = carrier_f - 2*frequency_of_interest 
hcut                    = carrier_f + 2*frequency_of_interest 
input_band_filter       = iirfilter(17, [lcut,hcut], rs=60, btype='bandpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')
# 
input_cut_filter = iirfilter(17, [carrier_f-5,carrier_f+5], rs=60, btype='bandstop',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')

def demodulate(measured_signal):
    IQ = measured_signal*np.exp(1j*(2*np.pi*carrier_f*t )) 
    idown = np.real(IQ)
    qdown = np.imag(IQ)
    # 
    idown = sosfiltfilt(lp_filter,idown)
    qdown = sosfiltfilt(lp_filter,qdown)
    # 
    v = idown + 1j*qdown
    # calculate the phase in order to figure out the sign of the recovered envelope. 
    mag     = np.abs(v)             # magnitude
    phase = np.unwrap(np.angle(v))  # instantaneous phase
    inst_freq = np.diff(phase)/(2*np.pi)*Fs #instantaneous frequency
    # Regenerate the carrier from the instantaneous phase
    regenerated_carrier = np.cos(phase)
    #  
    regenerated_carrier[regenerated_carrier >= 0] = 1
    regenerated_carrier[regenerated_carrier < 0] = 0

    rsignal = mag
    for i in range(len(regenerated_carrier)):
        if regenerated_carrier[i]>0:
            rsignal[i] = mag[i]
        else:
            rsignal[i] = -mag[i]
    return rsignal

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

window = np.hanning(end_pause-start_pause)
low                     = 4
high                    = 300
sos_band                = iirfilter(17, [low,high], rs=60, btype='bandpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')
# define the carrier frequency. 
fc = pressure_signal_frequency
l = fc - 2*high
h = fc + 2*high
sos_carrier_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
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
    data                        = m.copy_to_folder_and_return_data(**aeti_variables)
    marker     = data[marker_channel]/np.max(data[marker_channel])
    rawdata    = 1e6*data[m_channel]/gain
    nothing_d  = rawdata
    rfdata     = 10*data[rf_channel]
    vdata      = 10*data[v_channel]
    # 
    # first tight bandpass around the carrier frequency to remove any possible causes of 'other stuff'
    fraw           = sosfiltfilt(input_band_filter, rawdata)
    fraw           = sosfiltfilt(input_cut_filter, fraw)
    demod_data     = demodulate(fraw)
    rawdata        = rawdata - np.mean(rawdata)
    # Now filter both data the same way. 
    filtered_rawdata    = sosfiltfilt(sos_band, rawdata)
    filtered_demoddata  = sosfiltfilt(sos_band, demod_data)
    # 
    fft_m = fft(rawdata[start_pause:end_pause]*window)
    fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
    # 
    fft_rf = fft(rfdata[start_pause:end_pause])
    fft_rf = np.abs(2.0/(end_pause-start_pause) * (fft_rf))[1:(end_pause-start_pause)//2]
    #
    fft_demod = fft(filtered_demoddata[start_pause:end_pause]*window)
    fft_demod = np.abs(2.0/(end_pause-start_pause) * (fft_demod))[1:(end_pause-start_pause)//2]
    # 
    # Decimate arrays before plotting, so it is less memory intensive. 
    desired_plot_sample_rate = 1e5
    decimation_factor = int(Fs/desired_plot_sample_rate)
    marker = marker[::decimation_factor]    
    filtered_rawdata = filtered_rawdata[::decimation_factor]      
    filtered_demoddata = filtered_demoddata[::decimation_factor]  
    dvdata = vdata[::decimation_factor]
    drfdata = rfdata[::decimation_factor]    
    td = t[::decimation_factor]  
    #     
    # Plot the result.           
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(311)
    # plt.plot(t,nothing_d,color='k')  
    plt.plot(td,500*marker,color='g') 
    plt.plot(td,filtered_rawdata,color='r')
    plt.plot(td,filtered_demoddata,color='b')

    # ax.set_ylim([-1000,1000])
    ax.set_xlim([0,np.max(t)])
    ax.set_ylabel('Volts ($\mu$V)')

    ax2 = fig.add_subplot(312)
    # plt.plot(td,rfdata,color='b')
    # plt.plot(t,data[m_channel],color='k')
    plt.plot(t,rfdata,color='r') 
    plt.plot(td,dvdata,color='b')

    ax2.set_xlim([0,np.max(t)])
    ax2.set_ylabel('Volts ($\mu$V)')

    ax3 = fig.add_subplot(313)
    plt.plot(frequencies,fft_m,'r')
    plt.plot(frequencies,fft_demod,'b')  
    # plt.plot(frequencies,fft_rf,'k')        
    ax3.set_ylim([0,200])
    # ax3.set_xlim([0,high])
    ax3.set_xlim([pressure_signal_frequency-100,pressure_signal_frequency+100])
    # ax3.set_xlim([0,1000000])
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

