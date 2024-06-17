#!/usr/bin/python
'''
Library to control settings for mouse AE recording. 

Author: Jean Rintoul
Date: 23/03/2022

'''

from __future__ import print_function
from array import array
from math import sin
import sys
import time
import os
import libtiepie
from printinfo import *
import numpy as np
import matplotlib.pyplot as plt
from libtiepie.api import api
import serial, time
from scipy.fft import fft, fftfreq
from scipy.signal import blackman
from subprocess import check_output
import subprocess
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from scipy import signal


# sensitivity = 0.052  #  hydrophone sensitivity is 52mV/MPa at 500kHz
sensitivity = 0.033    #  hydrophone sensitivity is 33mV/MPa at 500kHz

# # demodulate function. 
# def demodulate(signal,carrier_f,t):
#     IQ = signal*np.exp(1j*(2*np.pi*carrier_f*t ))
#     idown = np.real(IQ)
#     qdown = np.imag(IQ)
#     # idown = signal*np.cos(2*np.pi*carrier_f*t)
#     # qdown = -signal*np.sin(2*np.pi*carrier_f*t)
#     idown_lp = sosfiltfilt(sos_band, idown)
#     qdown_lp = sosfiltfilt(sos_band, qdown)
#     # Add them together.  
#     v = idown_lp + 1j*qdown_lp
#     return np.real(-1j*v)

def demodulate(signal,carrier_f,t):
    IQ = signal*np.exp(1j*(2*np.pi*carrier_f*t ))
    # idown = signal*np.cos(2*np.pi*carrier_f*t)
    # qdown = -signal*np.sin(2*np.pi*carrier_f*t)    
    idown = np.real(IQ)
    qdown = np.imag(IQ)
    v = idown + 1j*qdown
    v = sosfiltfilt(sos_band, v)
    return v    

def frange(start, stop, step):
        x = start
        while x < stop:
            yield x
            x += step

def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx

'''
    Flexible input function to start a recording with a dictionary input. 

'''
def aeti_recording(**aeti_variables):
    # print("\nData type of argument:",type(data))
    prefix = ''
    command = 'marker_stream'
    for key, value in aeti_variables.items():
        if key == 'command_c':
           command = value

    for key, value in aeti_variables.items():
        # print("{} is {}".format(key,value))
        if key == 'Fs':
            command = command + ' -s ' + str(value) 
        elif key =='jitter_range':      
            command = command + ' -l' + str(value)               
        elif key =='USMEP':      
            command = command + ' -c' + str(value)      
        elif key =='duration':
            command = command + ' -d ' + str(value)
        elif key == 'fileprefix' or key == 'position': 
            if key == 'position':
                prefix = str(np.round(value,2) )
                command = command + ' -p ' + prefix
            else: 
                prefix = value
                command = command + ' -p ' + prefix
        elif key == 'pressure_frequency':
            command = command + ' -g ' + str(value)
        elif key == 'pressure_amplitude':
            command = command + ' -a ' + str(value)
        elif key == 'pressure_prf':
            command = command + ' -h ' + str(value)
        elif key == 'pressure_ISI':
            command = command + ' -i ' + str(value)
        elif key == 'pressure_burst_length':
            command = command + ' -j ' + str(value) 
        elif key == 'pressure_fswitching':
            command = command + ' -r ' + str(value)
        elif key == 'pressure_fswitching2':
            command = command + ' -o ' + str(value)              
        elif key == 'current_frequency':
            command = command + ' -f ' + str(value)
        elif key == 'current_fswitching':
            command = command + ' -v ' + str(value)            
        elif key == 'current_amplitude':
            command = command + ' -b ' + str(value)
        elif key == 'current_prf':
            command = command + ' -u ' + str(value)              
        elif key == 'current_burst_length':
            command = command + ' -w ' + str(value)
        elif key == 'current_ISI':
            command = command + ' -y ' + str(value)
        elif key == 'start_null':
            command = command + ' -k ' + str(value)   
        elif key == 'end_null':
            command = command + ' -n ' + str(value)                        
        elif key == 'start_pause':
            command = command + ' -t ' + str(value)
        elif key == 'end_pause':
            command = command + ' -e ' + str(value)     
        elif key == 'no_ramp':
            command = command + ' -z ' + str(value)  
        elif key == 'pi_frequency':
            command = command + ' -x ' + str(value)   
        elif key == 'ti_frequency':
            command = command + ' -m ' + str(value)  
        elif key == 'long_recording':
            command = command + ' -q ' + str(value) 
        elif key == 'difference_delta':
            command = command + ' -9 ' + str(value)             
    # print ('command: ',command)
    # print ('prefix: ',prefix)
    execution_result = None
    while execution_result is None:
        try:
            # print ('command',command)
            foo = check_output(command, shell=True)
            execution_result = str(foo, 'utf-8')
            # print ('execution_result',execution_result)
            filename = prefix+'_stream.npy'
            print ('filename is',filename)
        except Exception as e: print(e, execution_result)
    # 
    data_out = []
    if aeti_variables['type'] == 'pressure':
        max_pressure = AE_pressure(**aeti_variables)
        data_out.append(max_pressure)
    if aeti_variables['type'] == 'impedance':
        data,idx_lag,original_data  = align_data_to_marker_channel(**aeti_variables)
        impedance_variables = impedance(**aeti_variables)
        data_out.append(impedance_variables)   
        # data_out = [0]     
    elif aeti_variables['type'] == 'demodulation':
        # data,idx_lag,original_data  = align_data_to_marker_channel(**aeti_variables)
        data_out = [0] 
    else: 
        e_field = 0  
        ae_diff = 0 
        ae_sum  = 0 
        try:
            e_field,ae_diff,ae_sum = AE_phi(**aeti_variables)
            # data,idx_lag,original_data  = align_data_to_marker_channel(**aeti_variables)
            data_out.append(e_field)
            data_out.append(ae_diff)
            data_out.append(ae_sum)
        except Exception as e: print(e)        

    return execution_result,data_out
# 


def impedance(**aeti_variables):
    impedance_variables = 0
    prefix = ''
    for key, value in aeti_variables.items():
        if key == 'Fs':
            Fs = value
        elif key =='duration':
            duration = value
        elif key == 'fileprefix' or key == 'position': 
            if key == 'position':
                prefix = str(np.round(value,2) )
            else: 
                prefix = value
        elif key == 'pressure_frequency':
            transducer_frequency = value
        elif key == 'ae_channel':
            measure_channel = value
        elif key == 'current_frequency':
            current_signal_frequency = value
        elif key == 'current_monitor_channel':
            i_channel = value
        elif key == 'IV_attenuation':
            attenuation = value 
        elif key == 'e_channel':
            v_channel = value       
        elif key == 'save_folder_path':
            save_path = value   
        elif key == 'experiment_configuration':
            experiment_configuration = value 
    # First, align data based on time sync marker. 
    # print ('Duration/Fs:',duration,Fs)
    N             = int(Fs*duration)
    timestep      = 1.0/Fs
    xf = np.fft.fftfreq(N, d=timestep)[:N//2]
    frequencies   = xf[1:N//2]
    # Load the file. 
    filename = save_path+'\\t'+prefix+'_stream.npy'  # the t indicates that it is time synced. 
    # print ('filename is',filename)
    data = np.load(filename,allow_pickle = True)
    a,b = data.shape    
    # print ('time synced file shape:',a,b)    
    # Get some info out of the aeti variables. 
    m_channel = aeti_variables['ae_channel'] 
    v_channel = aeti_variables['e_channel'] 
    i_channel = aeti_variables['current_monitor_channel'] 
    current_signal_frequency = aeti_variables['current_frequency'] 
    Fs       = aeti_variables['Fs'] 
    duration = aeti_variables['duration'] 
    timestep = 1.0/Fs
    N = int(Fs*duration)
    predicted_peak_dist = int((Fs/current_signal_frequency) - 1)
    # print ('predicted peak dist: ',predicted_peak_dist)
    # print("expected no. samples:",N)
    start_pause = int(aeti_variables['start_pause'] * N) - predicted_peak_dist
    end_pause = int(aeti_variables['end_pause'] *N) + predicted_peak_dist

    t = np.linspace(0, duration, N, endpoint=False)[start_pause:end_pause]
    omega = 2*np.pi*current_signal_frequency
    resistor_current_mon    = 49.9  # 49.9 Ohms for current monitor, 
    # Now calculate the resistance and the reactance.   
    i_data         = -5*data[i_channel]/resistor_current_mon
    i_data         = i_data[start_pause:end_pause] # convert to 
    v_data         = -10*data[v_channel][start_pause:end_pause]
    if experiment_configuration == 'monopolar':
        # print ('experiment configuration is monopolar')
        resistor_current_mon = 47
        i_data         = -data[i_channel]/resistor_current_mon        
        i_data         = i_data[start_pause:end_pause] # convert to 
        v_data         = attenuation*data[v_channel][start_pause:end_pause]


    fft_m = fft(data[m_channel][start_pause:end_pause])
    fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
    xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
    frequencies = xf[1:(end_pause-start_pause)//2]    
    m_idx   = find_nearest(frequencies,current_signal_frequency)
    measurement_pp = fft_m[m_idx]*2
    print ('measurement pp', measurement_pp)
    # print ('measurement pp',measurement_pp)
    predicted_peak_dist = int((Fs/current_signal_frequency) - 1)
    print ('predicted peak dist',predicted_peak_dist, current_signal_frequency, Fs)
    if predicted_peak_dist < 1: 
       print ('your sample rate and frequency are too similar... not enough points per wavelength')
    # 
    fft_v = fft(v_data)
    fft_v = np.abs(2.0/(end_pause-start_pause) * (fft_v))[1:(end_pause-start_pause)//2]
    fft_i = fft(i_data)
    fft_i = np.abs(2.0/(end_pause-start_pause) * (fft_i))[1:(end_pause-start_pause)//2]
    i_idx = find_nearest(frequencies,current_signal_frequency)
    v_idx = find_nearest(frequencies,current_signal_frequency)
    V_pp  = fft_v[v_idx]*2
    I_pp  = fft_i[i_idx]*2    
    #     
    # For low frequencies, it is hard to get an accurate fft amplitude without a long time duration. 
    # if current_signal_frequency < 100:
        # V_pp = np.max(v_data) - np.min(v_data)
        # I_pp = np.max(i_data) - np.min(i_data)
    # 
    # if current_signal_frequency < 100000:    # stand in for transfer function calc. 
    #     V_pp = np.max(v_data) - np.min(v_data)
    #     I_pp = np.max(i_data) - np.min(i_data)
    #     measurement_pp = np.max(data[m_channel][start_pause:end_pause]) - np.min(data[m_channel][start_pause:end_pause])
    #     
    Z = np.abs(V_pp /I_pp)
    print ('max I(mA),V(V),Z(Ohms)', 1000*I_pp,V_pp,Z)
    amplitude = aeti_variables['current_amplitude']
    # 
    # Phase calculation
    # Center both I and V around zero, normalize and filter so sine wave is easier to align and work with. 
    tt = np.linspace(0, duration, N, endpoint=False)
    x = -data[i_channel]/resistor_current_mon 
    y = attenuation*data[v_channel]
    x = x - np.mean(x)
    y = y - np.mean(y)
    # Normalize before correlation. 
    x = (x-np.min(x))/ (np.max(x)-np.min(x))
    y = (y-np.min(y))/ (np.max(y)-np.min(y))
    x = x-0.5 
    y = y-0.5
    # band filter the signal to remove noise, otherwise peak finding is hard. 
    low_cut  = current_signal_frequency - 100  
    if low_cut <= 0:
       low_cut = 0.05
    high_cut = current_signal_frequency + 100
    sos = signal.iirfilter(17, [low_cut, high_cut], rs=60, btype='band',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
    xfilt = signal.sosfiltfilt(sos, x)
    yfilt= signal.sosfiltfilt(sos, y)
    #  Find peaks, only when amplitude is max. 
    vpeaks,properties = find_peaks(xfilt[start_pause:end_pause],distance=predicted_peak_dist*0.9 )
    ipeaks,properties = find_peaks(yfilt[start_pause:end_pause],distance=predicted_peak_dist*0.9 ) 
    # here I subtract voltage from current as this is the traditional way phase is reported.  
    # print( len(vpeaks), vpeaks,vpeaks[0])
    lead_or_lag = t[vpeaks[0]] -t[ipeaks[0]] 
    # print ('cap if negative',lead_or_lag)
    wave_period = 1/current_signal_frequency
    phase_shift = 360*lead_or_lag/wave_period 
    phase_shift = phase_shift%90
    print ('manual phase shift 1:',phase_shift)
    phase_rad =  phase_shift * (np.pi/180)

    correlation = signal.correlate(xfilt,yfilt, mode="full")
    lags = signal.correlation_lags(len(xfilt), len(yfilt), mode="full")
    idx_lag = lags[np.argmax(correlation*correlation)]
    # print ('idx lag', idx_lag)
    wave_period = 1/current_signal_frequency
    phase_shift2 = (360*idx_lag*timestep/wave_period)%90 -90
    # phase_shift2 = phase_shift2
    print ('phase shift2:',phase_shift2)
    phase_rad2 =  phase_shift2 * (np.pi/180)
    # print ('phase rad2: ', phase_rad)
    # It is well known that, to get the maximum power transfer from a source to a load, the source impedance must equal the complex conjugate of the load impedance, or:
    resistance = Z*np.cos(phase_rad)
    reactance  = Z*np.sin(phase_rad)

    # fig = plt.figure(figsize=(10,6))
    # ax = fig.add_subplot(111)
    # plt.plot(xfilt[vpeaks], color = 'purple')
    # plt.show()

    print ('R + Xj:',resistance, reactance)
    # Calculate inductive and capacitive reactance. 
    # X_l - X_C = reactance
    X_c = reactance
    X_l = reactance
    L = X_l/omega
    C = np.divide(1,(X_c* omega),where=X_c!=0)
    print ('Capacitive Load: Series Capacitance(uF)',np.round(np.abs(C)/1e-6,2 ) )
    if lead_or_lag <= 0: #  current leads voltage.  
        print ('Capacitive Load: Series Capacitance(uF)',np.round(np.abs(C)/1e-6,2 ) )
    else: #  voltage lead current
        print ('Inductive Load: Series Inductance(uH)',np.round(np.abs(L)/1e-6,2) )        

    # Let's also return the amplitude pp of the measurement electrode at the frequency of interest. 
    return [V_pp,I_pp,Z,L,C,phase_shift,resistance,reactance,measurement_pp,phase_shift2]

def copy_to_folder_and_return_data(**aeti_variables):
    impedance_variables = 0
    prefix = ''
    for key, value in aeti_variables.items():
        if key == 'Fs':
            Fs = value
        elif key =='duration':
            duration = value
        elif key == 'fileprefix' or key == 'position': 
            if key == 'position':
                prefix = str(np.round(value,2) )
            else: 
                prefix = value
        elif key == 'pressure_frequency':
            transducer_frequency = value
        elif key == 'ae_channel':
            measure_channel = value
        elif key == 'current_frequency':
            current_signal_frequency = value
        elif key == 'current_monitor_channel':
            i_channel = value
        elif key == 'e_channel':
            v_channel = value  
        elif key == 'marker_channel':
            marker_channel = value     
        elif key == 'save_folder_path':
            save_path = value      
    # Load the file. 
    filename = prefix+'_stream.npy'
    # print ('filename is',filename)
    d = np.load(filename,allow_pickle = True)
    a,b,c = d.shape
    # print ('data shape',a,b,c)
    data = d.transpose(1,0,2).reshape(b,-1) 
    a,b = data.shape
    # print ('data shape',a,b)
    outfile = save_path+'\\t'+filename  # the t indicates that it is time synced. 
    np.save(outfile,data)

    return data

def time_align_copy_to_folder_and_return_data(**aeti_variables):
    impedance_variables = 0
    prefix = ''
    for key, value in aeti_variables.items():
        if key == 'Fs':
            Fs = value
        elif key =='duration':
            duration = value
        elif key == 'fileprefix' or key == 'position': 
            if key == 'position':
                prefix = str(np.round(value,2) )
            else: 
                prefix = value
        elif key == 'pressure_frequency':
            transducer_frequency = value
        elif key == 'ae_channel':
            measure_channel = value
        elif key == 'current_frequency':
            current_signal_frequency = value
        elif key == 'current_monitor_channel':
            i_channel = value
        elif key == 'e_channel':
            v_channel = value  
        elif key == 'rf_monitor_channel':
            rfchannel = value 
        elif key == 'marker_channel':
            marker_channel = value     
        elif key == 'save_folder_path':
            save_path = value      
    # Load the file. 
    filename = prefix+'_stream.npy'
    # print ('filename is',filename)
    d = np.load(filename,allow_pickle = True)
    a,b,c = d.shape
    # print ('data shape',a,b,c)
    data = d.transpose(1,0,2).reshape(b,-1) 
    a,b = data.shape
    # print ('data shape',a,b)
    # pressure monitor is: ch 4 
    x = data[rfchannel]

    # fig = plt.figure(figsize=(5,5))
    # ax = fig.add_subplot(111)
    # plt.plot(data[rfchannel],'m')
    # plt.show()

    # peaks, properties = find_peaks(x,height = (0.02,0.09) )
    # with the rf amplifier    
    peaks, properties = find_peaks(x,height = (0.3,2.0) )  
    # with the bono tx. 
    # peaks, properties = find_peaks(x,height = (3,8.0) )    
    # 
    if len(peaks) == 0:
        print ('WARNING WARNING! : peaks not found, RF amplifier not on')  
        newdata = data 
    else:
        first_marker = peaks[0]
        
        print ('start peak is: ',first_marker,len(peaks) )
        # fig = plt.figure(figsize=(5,5))
        # ax = fig.add_subplot(111)
        # plt.plot(data[rfchannel],'b')
        # plt.axvline(x = first_marker,color='r'  )
        # plt.show()

        if first_marker < 1500000: 
            # Now, how to reshape the data so it all turns out the same shape? 
            # The easiest way is to add zeros the length of the file until the start peak, to the end. 
            # Then it is the same shape as it was previously. 
            a,b = data.shape
            # print ('data shape: ', data.shape,first_marker,a,b)
            newdata = np.zeros((a,b))
            endpoint = b - first_marker
            newdata[:,0:endpoint] = data[:,first_marker:] 
            # print ('newdata shape',newdata.shape)
            # 
            # Use this plot to check if time alignment is working. 
            # fig = plt.figure(figsize=(5,5))
            # ax = fig.add_subplot(111)
            # plt.plot(data[rfchannel],'b')
            # plt.plot(newdata[rfchannel],'k')
            # plt.axvline(x = first_marker,color='r'  )
            # plt.show()

        else: 
            print ('marker error occurred',first_marker)
            # Use this plot to check if time alignment is working. 
            # fig = plt.figure(figsize=(5,5))
            # ax = fig.add_subplot(111)
            # plt.plot(data[rfchannel],'b')
            # plt.plot(newdata[rfchannel],'k')
            # plt.axvline(x = first_marker,color='r'  )
            # plt.show()
            # for now, just store out the non-time aligned file. 
            newdata = data

    outfile = save_path+'\\t'+filename  # the t indicates that it is time synced. 
    np.save(outfile,newdata)

    return newdata

# 
# use the marker channel to align the data.  
def align_data_to_marker_channel(**aeti_variables):
    impedance_variables = 0
    prefix = ''
    for key, value in aeti_variables.items():
        if key == 'Fs':
            Fs = value
        elif key =='duration':
            duration = value
        elif key == 'fileprefix' or key == 'position': 
            if key == 'position':
                prefix = str(np.round(value,2) )
            else: 
                prefix = value
        elif key == 'pressure_frequency':
            transducer_frequency = value
        elif key == 'ae_channel':
            measure_channel = value
        elif key == 'current_frequency':
            current_signal_frequency = value
        elif key == 'current_monitor_channel':
            i_channel = value
        elif key == 'e_channel':
            v_channel = value  
        elif key == 'marker_channel':
            marker_channel = value     
        elif key == 'save_folder_path':
            save_path = value      
                  
    # First, align data based on time sync marker. 
    print ('Duration/Fs:',duration,Fs)
    N             = int(Fs*duration)
    timestep      = 1.0/Fs
    xf = np.fft.fftfreq(N, d=timestep)[:N//2]
    frequencies   = xf[1:N//2]

    # Load the file. 
    filename = prefix+'_stream.npy'
    # print ('filename is',filename)
    d = np.load(filename,allow_pickle = True)
    a,b,c = d.shape
    # print ('data shape',a,b,c)
    data = d.transpose(1,0,2).reshape(b,-1) 
    a,b = data.shape
    # print ('data shape',a,b)

    marker_length = 0.01     # this is manually set in thonny. 
    diffs  = np.diff(data[marker_channel])
    t      = np.linspace(0, duration, N, endpoint=False)
    zarray = t*[0]
    indexes = np.argwhere(diffs > 0.3)[:,0]
    dindexes = np.argwhere(diffs < -0.3)[:,0]
    # print ('len indexes', len(indexes),len(dindexes))
    zarray[indexes] = 1.0

    if (len(dindexes) == len(indexes)):
        ind_list = dindexes - indexes
    elif (len(dindexes)>len(indexes)):
        ind_list = dindexes[0:len(indexes)]- indexes
        # print ('indexes length issue 2')        
    else: 
        ind_list = dindexes - indexes[0:len(dindexes)]
        # print ('indexes length issue')
    # 
    # 
    #  
    # print ('lengths list', ind_list,np.argwhere(ind_list < 1010)[0][0] )
    # print (dindexes)
    # print (indexes)
    # print (ind_list)
    # print (np.min(indexes))
    # this is wrong. 
    # start_index = indexes[np.argwhere(ind_list < 1010)[0][0]]
    start_index = np.min(indexes)
    # print ('start index: ',start_index)
    original_data = data
    idx_lag = start_index
    data = np.concatenate([ data[:,idx_lag:],data[:,0:idx_lag] ],axis=1)

    # Plot the voltage and current so we can more easily see the phase lag. 
    # fig = plt.figure(figsize=(5,4))
    # ax1 = fig.add_subplot(111)
    # plt.plot(t,original_data[marker_channel],'b')
    # plt.plot(t,data[marker_channel],'r')
    # # plt.plot(t,v_data,'b')
    # # plt.plot(t,i_data,'r')
    # # plt.plot(t[vpeaks], v_data[vpeaks], "xr")
    # # plt.plot(t[ipeaks], i_data[ipeaks], "xg")
    # # # ax1.set_xlim([t[t_idx_start],t[t_idx_end]])
    # # ax1.set_xlabel('time(s)')
    # # plt.legend(['voltage(V)','current(A)'])
    # plt.show()
    outfile = save_path+'\\t'+filename  # the t indicates that it is time synced. 
    np.save(outfile,data)

    return data,idx_lag, original_data


def AE_pressure(**aeti_variables):
    prefix = ''
    for key, value in aeti_variables.items():
        if key == 'Fs':
            Fs = value
        elif key =='duration':
            duration = value
        elif key == 'fileprefix' or key == 'position': 
            if key == 'position':
                prefix = str(np.round(value,2) )
            else: 
                prefix = value
        elif key == 'pressure_frequency':
            transducer_frequency = value
        elif key == 'ae_channel':
            ae_channel = value
        elif key == 'current_frequency':
            f = value
        elif key == 'hydrophone_channel':
            hydrophone_channel = value
        elif key == 'e_channel':
            e_channel = value

    print ('Duration/Fs:',duration,Fs)
    N             = int(Fs*duration)
    timestep      = 1.0/Fs
    xf = np.fft.fftfreq(N, d=timestep)[:N//2]
    frequencies   = xf[1:N//2]

    # Load the file. 
    filename = prefix+'_stream.npy'
    print ('filename is',filename)
    d = np.load(filename,allow_pickle = True)
    a,b,c = d.shape
    # print ('shape',a,b,c)
    data = d.transpose(1,0,2).reshape(b,-1) 

    pressure        = np.array(data[hydrophone_channel])*1000/sensitivity  # KPa
    # pressure_pp     = np.pi*sum(np.fabs(pressure))/len(pressure)
    # max_pressure    = pressure_pp/2 
    max_pressure = np.max(pressure)    
    print ('max_pressure: ',max_pressure)

    return max_pressure

def AE_phi(**aeti_variables):
    prefix = ''
    for key, value in aeti_variables.items():
        if key == 'Fs':
            Fs = value
        elif key =='duration':
            duration = value
        elif key == 'fileprefix' or key == 'position': 
            if key == 'position':
                prefix = str(np.round(value,2) )
            else: 
                prefix = value
        elif key == 'pressure_frequency':
            transducer_frequency = value
        elif key == 'ae_channel':
            ae_channel = value
        elif key == 'current_frequency':
            f = value
        elif key == 'e_channel':
            e_channel = value

    print ('Duration/Fs:',duration,Fs)
    N             = int(Fs*duration)
    timestep      = 1.0/Fs
    xf = np.fft.fftfreq(N, d=timestep)[:N//2]
    frequencies   = xf[1:N//2]
    ae_diff_idx   = find_nearest(frequencies,transducer_frequency-f)
    ae_sum_idx    = find_nearest(frequencies,transducer_frequency+f)
    e_idx         = find_nearest(frequencies,f)
    # Load the file. 
    filename = prefix+'_stream.npy'
    # print ('filename is',filename)
    d = np.load(filename,allow_pickle = True)
    a,b,c = d.shape
    data = d.transpose(1,0,2).reshape(b,-1) 

    fft_efield = fft(10*data[e_channel])
    fft_e = np.abs(2.0/N * (fft_efield))[1:N//2]
    fft_aefield = fft(data[ae_channel])
    fft_ae = np.abs(2.0/N * (fft_aefield))[1:N//2]

    AE_sum_volts = 2*fft_ae[ae_sum_idx]
    AE_diff_volts = 2*fft_ae[ae_diff_idx]
    E_volts = 2*fft_e[e_idx]

    return E_volts,AE_diff_volts,AE_sum_volts

