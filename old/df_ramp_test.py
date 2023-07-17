#!/usr/bin/python
'''
 Author: Jean Rintoul Date: 23/11/2022

Carrier ramp test, ramps the carrier frequency for TI, while keeping the difference frequency the same. 

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
import skrf as rf
from skrf import Network
from skrf.calibration import OnePort

# increment this for each test. 
test_no = 1

aeti_variables = {
'type':'impedance',         # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs':1e5,                  # 
'duration': 15.0,            # 
'position': test_no,
'pressure_amplitude': 0.0,
'pressure_frequency': 10000.0,
'current_amplitude': 0.5,   #  its actually a voltage .. Volts. 0.0002
'current_frequency': 2000,
# 'ti_frequency': 2002,     # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ti_frequency': 0,          # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 0,            # the channel of the measurement probe. 
'e_channel': 4,             # this is the voltage measured between the stimulator probes. 
'rf_monitor_channel': 1,    # this output of the rf amplifier. 
'current_monitor_channel': 5,  # this is the current measurement channel of the transformer. 
'start_pause': 0.1,         # percent of file in ramp mode. 
'end_pause': 0.9,           # 
'no_ramp':0.0,              # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':1,                   # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':10,        # the current and voltage monitor both have attenuators on them 
'marker_channel':2,
'command_c':'mouse_stream',
'save_folder_path':'D:\\mouse_aeti\\e74_nr_ketxyl_screw_mouse\\transfer_function\\df_ramp',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
#  
#  
#  
savepath = aeti_variables['save_folder_path']
#  These are the frequencies we wish to ramp over. 
difference_frequencies = [0.5,1,3,5,10,20,40,60,100,130,200,400,800,1600]
#  
#  
data       = []
for i in range(len(difference_frequencies)): 
  f =  round(difference_frequencies[i],2)
  aeti_variables['ti_frequency'] = aeti_variables['current_frequency']+ f
  print('current df frequency(Hz): ',aeti_variables['current_frequency']+ f)  
  aeti_variables['position']  = f   # this is to label the file differently. 
  result, data_out            = m.aeti_recording(**aeti_variables)
  # print ('impedance:',data_out[0])
  data.append(data_out[0])
result = np.array(data)
# end array building. 
# 
# save the impedances out. 
V_pp        = result[:,0]
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

# # save off an .s1p file. 
impedances = []
for i in range(len(Zs)):
    impedances.append(complex(resistance[i],reactance[i]))
# print(impedances)
port_impedance = 50 
S11 = np.divide(impedances,port_impedance,where=impedances!=0)
print ('S11',S11)
freq = rf.Frequency.from_f(frequencies, unit='Hz')
ntwk = rf.Network(name='network',frequency=freq, z=impedances)
# extract 1-port objects from multiport objects
s11 = ntwk.s11
ntwk.write_touchstone(dir=savepath)

# # # # # 

size_font = 12

# Plot impedance vs frequency and phase vs frequency on separate subplots. The Cole-Cole plot is not as useful. 
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(211)
plt.plot(frequencies,Zs,color='r',marker='o')
ax.set_ylim([0.1,np.max(Zs)])
ax.set_ylabel('Z(Ohms)', fontsize=size_font)

ax2 = fig.add_subplot(212)
ax2.plot(frequencies,phase, color='b',marker='.')
ax2.plot(frequencies,phase2, color='orange',marker='.')
ax2.set_ylim([-90,90])
ax2.set_ylabel('Phase(Degrees)', fontsize=size_font)
ax2.set_xlabel('Frequency(Hz)')
plt.legend(['phase calc 1','phase calc 2'])
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
fig.tight_layout() 
plot_filename = 'Bode.png'
plot_filename = savepath + '\\Bode.png'
plt.savefig(plot_filename)
plt.show()


# First plot is a bode plot, if I took the log of both axes. 
# fig = plt.figure(figsize=(8,5))
# ax = fig.add_subplot(211)
# plt.plot(frequencies,Zs,color='r',marker='o')
# # ax.set_yscale('log')
# # ax.set_xscale('log')
# # ax.set_xlim([0.0001,np.max(frequencies)])
# ax.set_ylim([0.1,np.max(Zs)])
# # plt.plot(frequencies,resistance,color='g',marker='o')
# # plt.plot(frequencies,np.imag(impedances),'g')
# ax.set_ylabel('Z(Ohms)', color='r', fontsize=size_font)
# ax2 = ax.twinx()
# #add second line to plot
# ax2.plot(frequencies,-phase, color='b',marker='.')
# # ax2.plot(frequencies,-phase2, color='orange',marker='.')
# ax2.set_ylim([0.0001,90])
# ax2.set_ylabel('Phase(Degrees)', color='b', fontsize=size_font)
# # plt.legend(['resistance','reactance'],loc="upper left")
# ax2.set_title('Bode Plot')
# ax2.set_xlabel('Frequency(Hz)')
# ax2.set_xlim([0,np.max(frequencies)])
# ax3 = fig.add_subplot(212)
# plt.plot(resistance,-reactance,color='b',marker='.')
# ax3.set_ylabel('-Imag(Z) ($\Omega$)')
# ax3.set_xlabel('Real(Z) ($\Omega$)')
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# ax3.set_title('Cole-Cole Plot')

# # ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# # ax4.spines['right'].set_visible(False)
# # ax4.spines['top'].set_visible(False)
# # plt.xlabel('frequencies(Hz)')
# fig.tight_layout() 
# plot_filename = 'BodeCole.png'
# plot_filename = savepath + '\\BodeCole.png'
# plt.savefig(plot_filename)
# plt.show()

# Cole-Cole plot, Re(Z) vs -Im(Z)
# Nyquist plot. 
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(211)
# plt.plot(frequencies,phase)
plt.plot(frequencies,measure_pp,color='b',marker='.')
ax.set_ylabel('measurement electrode pp(V)', color='b',fontsize=size_font)
# plt.legend(['phase'],loc="upper left")
# ax.set_xlim([0.0001,np.max(frequencies)])
ax2 = fig.add_subplot(212)
plt.plot(frequencies,1000*Is,color='r',marker='.')
# ax2.set_xlim([0.0001,np.max(frequencies)])
# plt.legend(['current'],loc="upper left")
ax2.set_ylabel('current pp (mA)', color='r', fontsize=size_font)

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

# 
# Transfer function. Log of measure over the input? 
tx_log = 20*np.log(measure_pp/V_pp)
tx = measure_pp/V_pp  # v out, over v in. 

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
plt.plot(frequencies,tx,color='r',marker='.')
ax.set_ylabel('Vout/Vin', color='r', fontsize=size_font)
ax.set_xlabel('Frequency(Hz)')
ax.set_xlim([0,np.max(frequencies)])
ax.set_ylim([0,np.max(tx)+1])
plt.title('Transfer Function')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = savepath + '\\transfer_fcn.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
plt.plot(frequencies,tx_log,color='r',marker='.')
ax.set_ylabel('log Vout/Vin (dB)', color='k', fontsize=size_font)
ax.set_xlabel('Frequency(Hz)')
ax.set_xlim([0.001,np.max(frequencies)])
# ax.set_ylim([0.01,np.max(tx_log)+1])
plt.title('Transfer Function')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = savepath + '\\transfer_fcn_log.png'
plt.savefig(plot_filename)
plt.show()


print ('Done!')