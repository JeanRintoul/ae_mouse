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
# 
# Define some constants
aeti_variables = {
'type':'ae', # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 5e4, 
'duration': 4.0, 
'position': 50,
'pressure_amplitude': 0.0,
'pressure_frequency': 0.0,
'current_amplitude': 1.0, 
'current_frequency': 50,
'ae_channel': 6,
'e_channel': 4,
'hydrophone_channel': 1,
'rf_monitor_channel': 0,
'start_pause': 0.1,
'end_pause': 0.9,
'gain':1,
'marker_channel':0,
'command_c':'mouse_stream'
}

# 
# The marker needs to be on the same channel that the signal is on. 
# 
# success = aelib.time_sync(**aeti_variables)
# print (success)


Fs = 5e4
duration = 4.0

timestep = 1.0/Fs
N = int(Fs*duration)
print("expected no. samples:",N)
t = np.linspace(0, duration, N, endpoint=False)
xf = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies = xf[1:N//2]

rf_channel  	= 0    
v_channel   	= 2
ae_channel  	= 7
marker_channel 	= 7

# filename = '50_stream_dc.npy'
# filename = '50_stream_nodc.npy'

filename = '50_stream.npy'


d = np.load(filename)
a,b,c = d.shape
data = d.transpose(1,0,2).reshape(b,-1) 
print (data.shape)



p = data[marker_channel]
distance_in_samples = Fs/(2*np.pi*aeti_variables['current_frequency'])
peaks, properties = find_peaks(data[7], width=distance_in_samples)

print (len(peaks))
# this isolates where the peaks start until the end. 
# fig = plt.figure(figsize=(8,5))
# ax = fig.add_subplot(111)
# plt.plot(t, data[marker_channel], color = 'b',linewidth=2)
# plt.plot(t[peaks], data[marker_channel,peaks], 'xr')
# # plt.axvline(x=t[front], linestyle=':', color='k')
# plt.show()

total_gap = len(t)-peaks[-1] +peaks[0]
print ('total gap', total_gap)

if total_gap > int(distance_in_samples*2):
	print ('end_ramp')

start_offset = len(t) - (int(total_gap/2) + peaks[-1])
print ('start_offset',start_offset)
startpt = - start_offset
if start_offset > 0:
	startpt = peaks[-1] + start_offset
print ('startpt',startpt)



newdata = np.zeros((8,200000) )
print ('new data shape',newdata.shape)
newdata[:,0:(len(t)-startpt)]   = data[:,startpt:]
newdata[:,(len(t)-startpt):]  = data[:,0:startpt] 


# this isolates where the peaks start until the end. 
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(211)
plt.plot(t, data[marker_channel], color = 'b',linewidth=2)
plt.plot(t[peaks], data[marker_channel,peaks], 'xr')
plt.axvline(x=t[startpt], linestyle=':', color='k')

ax2 = fig.add_subplot(212)
plt.plot(t, newdata[marker_channel], color = 'b',linewidth=2)
plt.show()


# end_space = len(t)-peaks[-1]
# if end_space > distance_in_samples*2:
# 	print ('dist in samples',distance_in_samples*2)
# print (peaks)
# print (len(t))

# I need to wrap it around
# print (np.diff(peaks))

# np.diff(peaks)

# if any(y > x for y in lst):


# print ('distance in samples:',distance_in_samples)
# ppeaks, pproperties = find_peaks(data[marker_channel], distance = distance_in_samples,prominence=(None, 0.6),width=distance_in_samples)
# # this isolates where the peaks start until the end. 
# t2 = t[ppeaks[0]:]
# p2 = p[ppeaks[0]:]
# # 
# # Find the first peak, after the gap. 
# xpeaks, xproperties = find_peaks(p2, distance = distance_in_samples,prominence=0.3,width=distance_in_samples)
# # First marker. This should be the new zeropoint. 
# first_marker = ppeaks[0] + xpeaks[1] 
# a,b = data.shape
# print ( data.shape,first_marker,a,b)
# newdata = np.zeros( (a,b) )
# front   = b-first_marker
# newdata[:,0:front]  = data[:,first_marker:] 
# newdata[:,front:]   = data[:,0:first_marker]
# print ('newdata shape',newdata.shape)

# fig = plt.figure(figsize=(8,5))
# ax = fig.add_subplot(211)
# plt.plot(t, newdata[marker_channel], color = 'b',linewidth=2)
# plt.axvline(x=t[front], linestyle=':', color='k')


# ax2 = fig.add_subplot(212)
# plt.plot(t, data[marker_channel], color = 'b',linewidth=2)
# # ax2.set_xlim([0,0.00006])
# plt.show()

# fft_efield = fft(data[v_channel])
# fft_e = np.abs(2.0/N * (fft_efield))[1:N//2]

# fft_rf = fft(data[rf_channel])
# fft_rf = np.abs(2.0/N * (fft_rf))[1:N//2]

# fft_ae = fft(data[ae_channel])
# fft_ae = np.abs(2.0/N * (fft_ae))[1:N//2]

# print ('pre filter,post filter',np.mean(data[v_channel]),np.mean(data[ae_channel]) )

# # These are the "Tableau Color Blind 10" colors as RGB.    
# tableaucolorblind10 = [
# (0, 107, 164), 
# (255, 128, 14), 
# (171, 171, 171), 
# (89, 89, 89),    
# (95, 158, 209), 
# (200, 82, 0), 
# (137, 137, 137), 
# (162, 200, 236),  
# (255, 188, 121), 
# (207, 207, 207)]    
  
# # Scale the RGB values to the [0, 1] ra                                                                                                                                                                                                                     nge, which is the format matplotlib accepts.    
# for i in range(len(tableaucolorblind10)):    
#     r, g, b = tableaucolorblind10[i]    
#     tableaucolorblind10[i] = (r / 255., g / 255., b / 255.) 

# # Change global font settings to help stylize the graph. 
# plt.rc('font', family='serif')
# plt.rc('font', serif='Arial')
# plt.rcParams['axes.linewidth'] = 2

# fig = plt.figure(figsize=(8,5))
# ax = fig.add_subplot(211)
# plt.plot(frequencies, fft_ae, color = 'r',linewidth=2)
# plt.plot(frequencies, fft_e, color = 'b',linewidth=2)
# plt.legend(['post filter','pre filter'])
# plt.xlim([0,55])
# # plt.xlim([0,100])
# plt.yticks(fontsize=14)
# plt.xticks(fontsize=14)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2 = fig.add_subplot(212)
# # plt.plot(t, data[rf_channel], color = tableaucolorblind10[0],linewidth=2)
# plt.plot(t, data[ae_channel], color = 'r',linewidth=2)
# plt.plot(t, data[v_channel], color = 'b',linewidth=2)
# # plt.axvline(x=0.2)
# # plt.axvline(x=0.8)
# plt.legend(['post filter','pre filter'])
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# save_title = '_signals'
# # plt.savefig(save_title+".svg", format="svg") 
# plt.savefig(save_title+".png")
# plt.show()

