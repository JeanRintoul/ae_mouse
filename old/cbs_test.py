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
# 
# 
# Define some constants
aeti_variables = {
'type':'impedance',         # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 1e8,                  # 
'duration': 0.08, 
# 'position': 50,
'pressure_amplitude': 0.0,
'pressure_frequency': 10000.0,
'current_amplitude': 6.0,   #  its actually a voltage .. Volts. 
'current_frequency': 10000,
'ti_frequency': 0,          # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 7,            # the channel of the measurement probe. 
'e_channel': 5,             # this is the voltage measured between the stimulator probes. 
'rf_monitor_channel': 1,    # this output of the rf amplifier. 
'current_monitor_channel': 3,  # this is the current measurement channel of the transformer. 
'start_pause': 0.4,         # percent of file in ramp mode. 
'end_pause': 0.6,           # 
'no_ramp':1.0,              # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':1,
'marker_channel':0,
'command_c':'mouse_stream'
}
#  
# result, data_out            = m.aeti_recording(**aeti_variables)
# print ('impedance:',data_out[0])
# data,idx_lag,original_data  = m.align_data(**aeti_variables)

# m_channel = aeti_variables['ae_channel'] 
# v_channel = aeti_variables['e_channel'] 
# i_channel = aeti_variables['current_monitor_channel'] 
# current_signal_frequency = aeti_variables['current_frequency'] 
# Fs       = aeti_variables['Fs'] 
# duration = aeti_variables['duration'] 

# timestep = 1.0/Fs
# N = int(Fs*duration)
# # print("expected no. samples:",N)
# t = np.linspace(0, duration, N, endpoint=False)
# start_pause = int(aeti_variables['start_pause'] * N+1)
# end_pause = int(aeti_variables['end_pause'] *N-1)
# # print ('start and end:',start_pause,end_pause)
# resistor_current_mon    = 49.9  # 49.9 Ohms for current monitor, 
# i_data         = -5*data[i_channel]/resistor_current_mon
# i_data          = 1000*i_data # convert to mA. 
# v_data         = -10*data[v_channel]

# fft_m = fft(data[m_channel][start_pause:end_pause])
# fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]

# xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# frequencies = xf[1:(end_pause-start_pause)//2]

# print('len freqs',len(frequencies),len(fft_m))

# fig = plt.figure(figsize=(8,5))
# ax = fig.add_subplot(211)
# plt.plot(t, v_data, color = 'b',linewidth=2)
# plt.plot(t, i_data, color = 'g')
# plt.plot(t, data[m_channel], color = 'r')
# # plt.plot(t[idx_lag],10,'x')

# ax2=fig.add_subplot(212)
# plt.plot(frequencies,fft_m)
# plt.xlim([0,20000])
# plt.show()

frequencies = np.linspace(5000,500000,num=20)
print ('current_range: ',frequencies)

data       = []
for i in range(len(frequencies)): 
  f =  round(frequencies[i],2)
  aeti_variables['current_frequency'] = f
  print('current current_frequency(Hz): ',f)  
  aeti_variables['position']  = f   
  # if f > 250000:
  #   aeti_variables['Fs']        = 1e8
  #   aeti_variables['duration']  = 0.005
  result, data_out            = m.aeti_recording(**aeti_variables)
  # print ('impedance:',data_out[0])
  data.append(data_out[0])
result = np.array(data)
print ('result: ',result.shape)

Is          = result[:,1]
phase       = result[:,5]
impedances  = result[:,6]
measure_pp  = result[:,7]

outfile="impedances.npz"  
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,impedances = data, frequencies=frequencies)

size_font = 12

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(311)
plt.plot(frequencies,np.real(impedances),color='r',marker='o')
# plt.plot(frequencies,np.imag(impedances),'g')
ax.set_ylabel('Z(Ohms)', color='r', fontsize=size_font)
ax2 = ax.twinx()
#add second line to plot
ax2.plot(frequencies,phase, color='b',marker='x')
ax2.set_ylabel('Phase(Degrees)', color='b', fontsize=size_font)
# plt.legend(['resistance','reactance'],loc="upper left")
ax3 = fig.add_subplot(312)
# plt.plot(frequencies,phase)
plt.plot(frequencies,measure_pp,color='b')
ax3.set_ylabel('measurement V', color='b', fontsize=size_font)
# plt.legend(['phase'],loc="upper left")
ax4 = fig.add_subplot(313)
plt.plot(frequencies,Is,color='r')
# plt.legend(['current'],loc="upper left")
ax4.set_ylabel('current(mA)', color='r', fontsize=size_font)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
plot_filename = 'impedance.png'
plt.savefig(plot_filename)
plt.show()

print ('Done!')
