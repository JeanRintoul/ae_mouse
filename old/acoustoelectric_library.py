#!/usr/bin/python
'''
Library to control relay, 3D printer and take some measurements. 

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



# sensitivity = 0.052  #  hydrophone sensitivity is 52mV/MPa at 500kHz
sensitivity = 0.033    #  hydrophone sensitivity is 33mV/MPa at 500kHz


def deltahome(axis):
    ser = serial.Serial('COM6', 115200) 
    time.sleep(2)
    if axis is 'x':
        ser.write(b"G28 X\r\n") # Return to machine zero point. 
    elif axis is 'y':
        ser.write(b"G28 Y\r\n")
    elif axis is 'z':
        ser.write(b"G28 Z\r\n")
    else: 
        ser.write(b"G28\r\n")
    ser.close()

def get_position():
    ser = serial.Serial('COM6', 115200) 
    time.sleep(2)
    ser.write(b"M114\r\n")
    time.sleep(0.5)
    response = ser.readline()
    numOfLines = 0
    while True:
       response = ser.readline()
       posstring = str(response, 'utf-8')
       position = 'NA'
       if posstring[0] is 'X':
            position = posstring
            break
       numOfLines = numOfLines + 1
       if (numOfLines >= 5):
           print (numOfLines)
           raise ValueError('reached max lines')
    ser.close()
    return position

def move(increment,axis='x'):
    # G1 X10  # G1 is linear move, G0 is rapid linear move. 
    # G4 P100 # pause for 100 ms. 
    # G28 or G28 X Z move to origin, or origin along a certain axis. 
    # G90: all coords after are relative to the origin of the machine. 
    # G91: all coords from now on are relative to the last position. 
    # G1 X0 Y0 F2400 ; move to the X=0 Y=0 position on the bed at a speed of 2400 mm/min
    ser = serial.Serial('COM6', 115200) 
    time.sleep(2)
    ser.flushInput()
    if axis is 'x':
        gcode = 'G1 X'+str(increment)+' F1000\r\n'
        #print (gcode)
        ser.write(gcode.encode()) 
    elif axis is 'y':
        gcode = 'G1 Y'+str(increment)+' F1000\r\n'
        #print (gcode)
        ser.write(gcode.encode())
    elif axis is 'z':
        gcode = 'G1 Z'+str(increment)+' F1000\r\n'
        #print (gcode)
        ser.write(gcode.encode())
    time.sleep(0.5)
    ser.close()
    return True

def goto_pressure_focus(x_start,y_start,z_start):
  move(0.0,axis='y')
  deltahome(axis='x')
  move(x_start,axis='x')
  move(z_start,axis='z')
  move(y_start,axis='y')

def frange(start, stop, step):
        x = start
        while x < stop:
            yield x
            x += step

def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx

def turn_printer_on(verbose=1):
    device_name = 'HURTM'
    command_open = 'CommandApp_USBRelay '+device_name+' open 01'
    result = subprocess.getoutput(command_open)
    if verbose == 1:
        print ('on')
    if result == 1: 
        print ('success')
    # print (result)
    return result

def turn_printer_off(verbose =1):
    device_name = 'HURTM'
    command_close = 'CommandApp_USBRelay '+device_name+' close 01'  
    foo = check_output(command_close, shell=False)
    # print (foo)
    if verbose ==1:
        print ('off')
    return foo


'''
  Given an array of increments to iterate through, find the focus in the x direction. 

'''
def find_focus(increment,startmm,endmm,axis_choice='x',**aeti_variables): # 
    print ('start range/increment: ',startmm,endmm,increment)
    scan_outputs     = []
    positions        = []
    for position in frange(startmm, endmm, increment): 
        # move the printer to the correct position.  
        bool_result = move(position,axis=axis_choice)   
        print ('position:',position)
        # make the AE recording and get an AE estimate. 
        result = None
        aeti_variables['position'] = position        
        dout = []
        # while result is None:
        #     try:
        result,dout = aeti_recording(**aeti_variables)
        # except:
        #     pass

        if aeti_variables['type'] == 'ae':
            print ('position/e/aemag: %4.6f, %4.6f, %4.6f, %4.6f' %(position,dout[0],dout[1],dout[2]))
            scan_outputs.append([position,dout[0],dout[1],dout[2]])
        else:  # it's pressure data .
            print ('position/pressure: %4.6f, %4.6f' %(position,dout[0]))
            scan_outputs.append([position,dout[0]])

    # print ('scan_outputs',scan_outputs)
    out_array = np.array(scan_outputs)

    positions = out_array[:,0].tolist()
    values    = out_array[:,1].tolist()
    if aeti_variables['type'] == 'ae':       
        values    = out_array[:,2].tolist()        
    maxindex = values.index(max(values))
    maxval = values[maxindex]
    position = positions[maxindex]
    print ('max/pos: ',maxval,position)
    return position,values,positions,out_array

'''
    Flexible input function to start a recording with a dictionary input. 

'''
def aeti_recording(**aeti_variables):
    # print("\nData type of argument:",type(data))
    prefix = ''
    command = 'marker_stream'
    for key, value in aeti_variables.items():
        # print("{} is {}".format(key,value))
        if key == 'Fs':
            command = command + ' -s ' + str(value) 
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
        elif key == 'current_frequency':
            command = command + ' -f ' + str(value)
        elif key == 'current_amplitude':
            command = command + ' -b ' + str(value)
        elif key == 'start_pause':
            command = command + ' -t ' + str(value)
        elif key == 'end_pause':
            command = command + ' -e ' + str(value)               

    print ('command: ',command)
    # print ('prefix: ',prefix)
    execution_result = None
    while execution_result is None:
        try:
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
    else: 
        e_field = 0  
        ae_diff = 0 
        ae_sum  = 0 
        try:
            e_field,ae_diff,ae_sum = AE_phi(**aeti_variables)
            data_out.append(e_field)
            data_out.append(ae_diff)
            data_out.append(ae_sum)
        except Exception as e: print(e)        

    return execution_result,data_out

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
    print ('shape',a,b,c)
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

# def time_sync(**aeti_variables):

#     success = 0
#     prefix  = ''
#     marker_channel = 0 
#     prefix = ''
#     Fs = 5e4
#     for key, value in aeti_variables.items():
#         if key == 'Fs':
#             Fs = value
#             print ('Fs: ',Fs)
#         elif key =='duration':
#             duration = value
#         elif key == 'fileprefix' or key == 'position': 
#             if key == 'position':
#                 prefix = str(np.round(value,2) )
#             else: 
#                 prefix = value
#         elif key == 'pressure_frequency':
#             transducer_frequency = value
#         elif key == 'ae_channel':
#             ae_channel = value
#         elif key == 'current_frequency':
#             f = value
#         elif key == 'e_channel':
#             e_channel = value
  

#     # Load the file. 
#     filename    = prefix+'_stream.npy'
#     d           = np.load(filename,allow_pickle = True)
#     a,b,c       = d.shape
#     data        = d.transpose(1,0,2).reshape(b,-1) 
#     # 
#     p = data[marker_channel]
#     distance_in_samples = Fs/(2*np.pi*transducer_frequency)
#     print ('distance in samples:',distance_in_samples)
#     ppeaks, pproperties = find_peaks(data[marker_channel], distance = distance_in_samples,prominence=(None, 0.6),width=distance_in_samples)
#     # this isolates where the peaks start until the end. 
#     t2 = t[ppeaks[0]:]
#     p2 = p[ppeaks[0]:]
#     # 
#     # Find the first peak, after the gap. 
#     xpeaks, xproperties = find_peaks(p2, distance = distance_in_samples,prominence=0.3,width=distance_in_samples)
#     # First marker. This should be the new zeropoint. 
#     first_marker = ppeaks[0] + xpeaks[1] 
#     a,b = data.shape
#     print ( data.shape,first_marker,a,b)
#     newdata = np.zeros( (a,b) )
#     front   = b-first_marker
#     newdata[:,0:front]  = data[:,first_marker:] 
#     newdata[:,front:]   = data[:,0:first_marker]
#     print ('newdata shape',newdata.shape)

#     # 
#     fig = plt.figure(figsize=(8,5))
#     ax = fig.add_subplot(211)
#     plt.plot(t, newdata[marker_channel], color = 'b',linewidth=2)
#     plt.axvline(x=t[front], linestyle=':', color='k')
#     ax2 = fig.add_subplot(212)
#     plt.plot(t, newdata[marker_channel], color = 'b',linewidth=2)
#     ax2.set_xlim([0,0.00006])
#     plt.show()


#     file = str(np.round(ae_z,2))+'_'+str(np.round(ae_x,2))+'_stream.npy'
#     # save out the data. 
#     outfile=out_folder+file   # save out the data. 
#     np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
#     np.save(outfile,newdata)

#     return success