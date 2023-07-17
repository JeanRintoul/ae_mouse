'''

Title: field_view.py
Function: View a stream code data file. 

Author: Jean Rintoul
Date: 18.03.2022

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import acoustoelectric_library as aelib
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from scipy import signal
import pandas as pd
# 
# Define some constants
# aeti_variables = {
# 'type':'ae', # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
# 'Fs': 5e4, 
# 'duration': 4.0, 
# 'position': 50,
# 'pressure_amplitude': 0.0,
# 'pressure_frequency': 0.0,
# 'current_amplitude': 1.0, 
# 'current_frequency': 50,
# 'ae_channel': 6,
# 'e_channel': 4,
# 'hydrophone_channel': 1,
# 'rf_monitor_channel': 0,
# 'start_pause': 0.1,
# 'end_pause': 0.9,
# 'gain':1,
# 'marker_channel':0,
# 'command_c':'mouse_stream'
# }

# 
# The marker needs to be on the same channel that the signal is on. 
# 
# success = aelib.time_sync(**aeti_variables)
# print (success)

current_signal_frequency = 5000
Fs       = 5e6
duration = 2.0
timestep = 1.0/Fs
N = int(Fs*duration)
print("expected no. samples:",N)
t = np.linspace(0, duration, N, endpoint=False)
xf = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies = xf[1:N//2]

fg_mon          = 2
i1_channel      = 3
v1_channel   	= 5     
m_channel  	    = 7     
event_channel   = 0 
filename = '_stream.npy'
filename = '500000.0_stream.npy'
filename = '5000.0_stream.npy'
# filename = '128750.0_stream.npy'

d = np.load(filename)
a,b,c = d.shape
data = d.transpose(1,0,2).reshape(b,-1) 
a,b = data.shape
# print (data.shape)
original = data 
# Compute the lag between the two signals. Create a marker signal to cross-correlate with the recorded signal.  

current_pulse_length     = 3*int(Fs/current_signal_frequency);
marker_base              = np.zeros(b)
# 
marker_base[1:(current_pulse_length+1)] = 1.0
marker_base[(current_pulse_length+1):(current_pulse_length*2+1)] = 0.0
marker_base[(current_pulse_length*2+1):(current_pulse_length*4+1)] = 4.0
marker_base[(current_pulse_length*4+1):(current_pulse_length*5+1)] = 2.0
marker_base[10000] = np.max(-10*data[v1_channel])


x = -10*data[v1_channel]
y = marker_base
correlation = signal.correlate(x-np.mean(x), y - np.mean(y), mode="full")

lags = signal.correlation_lags(len(x), len(y), mode="full")
print ('correlation',len(correlation),len(lags))
idx_lag = lags[np.argmax(correlation*correlation)]
print ('lag is',idx_lag)

# Now re-arrange to recreate time synced signal. 
new_data = np.concatenate([ data[:,idx_lag:],data[:,0:idx_lag] ],axis=1)
#print ('data',data.shape,new_data.shape)
data = new_data

resistor_current_mon    = 49.9  # 49.9 Ohms for current monitor, 

i_data         = -5*data[i1_channel]/resistor_current_mon
i_data          = 1000*i_data # convert to mA. 
v_data         = -10*data[v1_channel]
m_data         = -10*data[m_channel]

predicted_peak_dist = int((Fs/current_signal_frequency) - 1)
# print ('predicted peak dist',predicted_peak_dist)
vpeaks,properties = find_peaks(v_data,distance=predicted_peak_dist )
ipeaks,properties = find_peaks(i_data,distance=predicted_peak_dist )
V_pp = np.median(v_data[vpeaks])*2
I_pp = np.median(i_data[ipeaks])*2
Z    = V_pp /I_pp
print ('median I(mA),V(V),Z(Ohms)', np.round(I_pp,2),np.round(V_pp,2),np.round(Z,2) )


V1_pp = np.max(v_data)-np.min(v_data)
I1_pp = (np.max(i_data)-np.min(i_data))
Z1 = V1_pp /I1_pp
print ('max I(mA),V(V),Z(Ohms)', I1_pp,V1_pp,Z1)


fft_fg = fft(data[fg_mon])
fft_fg = np.abs(2.0/N * (fft_fg))[1:N//2]

fft_v1 = fft(v_data)
fft_v1 = np.abs(2.0/N * (fft_v1))[1:N//2]

fft_m = fft(data[m_channel])
fft_m = np.abs(2.0/N * (fft_m))[1:N//2]

fft_i1 = fft(i_data)
fft_i1 = np.abs(2.0/N * (fft_i1))[1:N//2]


print ('pre filter,post filter',np.mean(data[v1_channel]),np.mean(data[m_channel]) )

# These are the "Tableau Color Blind 10" colors as RGB.    
tableaucolorblind10 = [
(0, 107, 164), 
(255, 128, 14), 
(171, 171, 171), 
(89, 89, 89),    
(95, 158, 209), 
(200, 82, 0), 
(137, 137, 137), 
(162, 200, 236),  
(255, 188, 121), 
(207, 207, 207)]    
  
# Scale the RGB values to the [0, 1] ra                                                                                                                                                                                                                     nge, which is the format matplotlib accepts.    
for i in range(len(tableaucolorblind10)):    
    r, g, b = tableaucolorblind10[i]    
    tableaucolorblind10[i] = (r / 255., g / 255., b / 255.) 

# Change global font settings to help stylize the graph. 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2


fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(311)
# plt.plot(frequencies, fft_fg, color = 'orange',linewidth=2)
plt.plot(frequencies, fft_v1, color = 'b',linewidth=2)
# plt.plot(frequencies, fft_v2, color = 'g',linewidth=2)
plt.plot(frequencies, fft_i1, color = 'g',linewidth=2)
plt.plot(frequencies, fft_m, color = 'r',linewidth=2)
# plt.legend(['post filter','pre filter'])
plt.xlim([0,550000])
# plt.xlim([0,100])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2 = fig.add_subplot(312)
plt.plot(t, v_data, color = 'b',linewidth=2)
plt.plot(t, i_data, color = 'g',linewidth=2)
plt.plot(t, m_data, color = 'r',linewidth=2)
# plt.plot(t, -10*original[5], color = 'r',linewidth=2)
plt.plot(t[idx_lag],10,'x')
# plt.plot(t, marker_base, color = 'b',linewidth=2)
ax3 = fig.add_subplot(313)
plt.plot(lags[500000:],0.01*correlation[500000:], color = 'g',linewidth=2)

# plt.plot(t, data[event_channel], color = 'r')
# plt.plot(t, data[7], color = 'r')
# plt.plot(t, i_data, color = 'g')


# plt.plot(t,marker_base,'g',linewidth=2)
# plt.axvline(x=0.2)
# plt.axvline(x=0.8)
# plt.legend(['post filter','pre filter'])
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
save_title = '_signals'
plt.savefig(save_title+".png")
plt.show()

