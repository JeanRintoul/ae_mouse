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
test_no = '\\1'

# Define some constants
aeti_variables = {
'type':'impedance',         # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 5e7,                  # 
'duration': 1.0, 
# 'position': 50,
'pressure_amplitude': 0.0,
'pressure_frequency': 10000.0,
'current_amplitude': 3.0,   #  its actually a voltage .. Volts. 
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
'command_c':'mouse_stream',
'save_folder_path':'D:\mouse_aeti\e62_mouse_1\q5\\impedance'+test_no,
}
#  
savepath = aeti_variables['save_folder_path']

# If I remove the transformer box I can replace with the impedance adapter and do the two below...
# frequencies = np.linspace(0,100,num=100)
# 
# frequencies = np.linspace(0,5000,num=100)

frequencies = np.linspace(5000,500000,num=5)
# print ('current_range: ',frequencies)

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
# print ('result: ',result.shape)

Is          = result[:,1]
Zs          = result[:,2]
Cs          = np.abs(result[:,4])/1e-6
phase       = result[:,5]
resistance  = result[:,6]
reactance   = result[:,7]
measure_pp  = result[:,8]
phase2      = result[:,9]

print ('Zs: ',Zs)

outfile = savepath + '\\impedances.npz'

# outfile="impedances.npz"  
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,impedances = data, frequencies=frequencies)


# f1 = 't5000.0_stream.npy'
# f4 = 't500000.0_stream.npy'
# data = np.load(f1)
# print ('data shape:',data.shape)

# fig = plt.figure(figsize=(8,5))
# ax = fig.add_subplot(111)
# plt.plot(data[aeti_variables['e_channel']],'b')
# plt.plot(data[aeti_variables['current_monitor_channel']],'r')
# plt.show()

size_font = 12

# First plot is a bode plot, if I took the log of both axes. 
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(211)
plt.plot(frequencies,Zs,color='r',marker='o')
ax.set_yscale('log')
ax.set_xscale('log')
# plt.plot(frequencies,resistance,color='g',marker='o')
# plt.plot(frequencies,np.imag(impedances),'g')
ax.set_ylabel('Z(Ohms)', color='r', fontsize=size_font)
ax2 = ax.twinx()
#add second line to plot
ax2.plot(frequencies,-phase, color='b',marker='x')
ax2.plot(frequencies,-phase2, color='orange',marker='x')
# ax2.set_ylim([0,90])
ax2.set_ylabel('Phase(Degrees)', color='b', fontsize=size_font)
# plt.legend(['resistance','reactance'],loc="upper left")
ax2.set_title('Bode Plot')
ax2.set_xlabel('Frequency(Hz)')
ax3 = fig.add_subplot(212)
plt.plot(resistance,-reactance)
ax3.set_ylabel('-Imag(Z) ($\Omega$)')
ax3.set_xlabel('Real(Z) ($\Omega$)')
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.set_title('Cole-Cole Plot')



# ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
# ax4.spines['right'].set_visible(False)
# ax4.spines['top'].set_visible(False)
# plt.xlabel('frequencies(Hz)')
fig.tight_layout() 
plot_filename = 'BodeCole.png'
plot_filename = savepath + '\\BodeCole.png'
plt.savefig(plot_filename)
plt.show()

# Cole-Cole plot, Re(Z) vs -Im(Z)
# Nyquist plot. 
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(211)
# plt.plot(frequencies,phase)
plt.plot(frequencies,measure_pp,color='b')
ax.set_ylabel('measurement electrode (V)', color='b', fontsize=size_font)
# plt.legend(['phase'],loc="upper left")
ax2 = fig.add_subplot(212)
plt.plot(frequencies,1000*Is,color='r')
# plt.legend(['current'],loc="upper left")
ax2.set_ylabel('current(mA)', color='r', fontsize=size_font)

# plt.plot(resistance,-reactance)
# ax.set_ylabel('-Imag(Z) ($\Omega$)')
# ax.set_xlabel('Real(Z) ($\Omega$)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
# plt.title('Cole-Cole Plot')
# plt.plot(frequencies,reactance,'orange')
# ax2 = fig.add_subplot(212)
# plt.plot(frequencies,Cs,'r')
# plot_filename = 'currentandmeasuredvoltage.png'
plot_filename = savepath + '\\currentandmeasuredvoltage.png'

plt.savefig(plot_filename)
plt.show()
print ('Done!')
