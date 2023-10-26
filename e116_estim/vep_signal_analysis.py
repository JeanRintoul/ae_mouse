'''

Title: vep signal inspection
Function: look at raw, lfp and spike filtered data with the time aligned marker channel of the LED. 

Author: Jean Rintoul
Date: 23.10.2022

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
# 
# 
# How do we assess the height of a VEP? I saw the min and max of the averaged signal. 
# The VEP height is the max-min of the averaged signal. 
# Save out a file that contains: 
# led_frequency, LFP height. 
# 
# 
file_number   = 21
led_frequency = 1 # in Hz. 



led_duration = 1/(2*led_frequency)
# when the led turned on and off. 
start_time = 0.0
end_time   = start_time+led_duration

start = led_duration   # this is where the trial starts seconds before marker. 
end   = led_duration*2   # this is where the trial ends

factor = 1
start = factor*led_duration   # this is where the trial starts seconds before marker. 
end   = factor*led_duration*2   # this is where the trial ends

Fs       = 1e5
duration = 60.0
timestep = 1.0/Fs
N = int(Fs*duration)
# print("expected no. samples:",N)
marker_channel = 2 
i_channel      = 5
v_channel      = 4   
m_channel  	   = 0     
gain           = 100
filename_t0    = 't'+str(file_number)+'_stream.npy'
t0data = np.load(filename_t0)
a,b = t0data.shape
# print (t0data.shape)
savepath = 'D:\\mouse_aeti\\e77_vep_test\\results' # change the savepath to where you want the outputs to go. 
# create time and frequencies arrays.
t = np.linspace(0, duration, N, endpoint=False)
xf = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies = xf[1:N//2]
#  
# convert it to microvolts by taking the gain into account. 
fsignal = 1e6*t0data[m_channel]/gain
# mains_harmonics = [50,100,150,158,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950]
mains_harmonics = [50,100,150,200,250,300]
for i in range(len(mains_harmonics)):
    mains_low  = mains_harmonics[i] -2
    mains_high = mains_harmonics[i] +2
    mains_sos = iirfilter(17, [mains_low,mains_high], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
    fsignal = sosfiltfilt(mains_sos, fsignal)
# 
low  = 0.1
high = 300 
sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
fsignal  = sosfiltfilt(sos_low, fsignal)


# 
# fft_data = fft(t0data[m_channel])
fft_data = fft(fsignal)

fft_data = np.abs(2.0/N * (fft_data))[1:N//2]
# 
# 
# 
# segment the select filtered data so I can see it overlayed on top of each other. 
# So VEP's do not always induce a spike... they do induce a frequency dependent swing, but only when the mouse is awake.
# create the marker channel. 
marker = t0data[marker_channel]/np.max(t0data[marker_channel])
diffs = np.diff(t0data[marker_channel] )
zarray = t*[0]
indexes = np.argwhere(diffs > 3.0)[:,0]
print ('len indexes', len(indexes),indexes)
zarray[indexes] = 1.0
# 
# 
pre_event_idxcount  = int(start*Fs)
post_event_idxcount = int(end*Fs)
data_to_segment = fsignal
filtered_segmented_data        = []
# It isn't segmenting correctly because the first index is the marker. 
for i in range(len(indexes)-1):  # the last index may get chopped short. 
  filtered_segmented_data.append(data_to_segment[(indexes[i+1]-pre_event_idxcount):(indexes[i+1]+post_event_idxcount)])
# 
# 
filtered_segmented_array = np.array(filtered_segmented_data)
a,b = filtered_segmented_array.shape
print ('shape', a,b)
time_segment = np.linspace(-start,end,num=b)
average_lfp = np.median(filtered_segmented_array,axis=0)
std_lfp     = np.std(filtered_segmented_array,axis=0)
print ('n_events: ', a)
# 
# 
lfp_height = np.max(average_lfp)- np.min(average_lfp)
print ('LFP size', lfp_height)




# 
# 
# 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(frequencies,fft_data,'b')
ax.set_xlim([0,100])
plot_filename = 'VEP_FFT.png'
plt.savefig(plot_filename)
plt.show()
# 
# 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(time_segment,average_lfp,color='r')
plt.fill_between(time_segment, average_lfp - std_lfp, average_lfp + std_lfp,
                  color='gray', alpha=0.2)
plt.axvline(x=start_time,color ='k')
plt.axvline(x=end_time,color ='k')
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Time(s)')
plt.autoscale(enable=True, axis='x', tight=True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.title('VEP')
plot_filename = 'VEP.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(311)
plt.plot(t,fsignal/np.max(fsignal),'g')
plt.plot(t,marker,'r')
ax.set_xlim([3,57])
ax.set_ylim([-2,2])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax2 = fig.add_subplot(312)
plt.plot(time_segment,filtered_segmented_array.T)
plt.axvline(x=start_time,color ='k')
plt.axvline(x=end_time,color ='k')
# ax2.text(0.5, 100, 'individual trials', color='black',fontsize = 6, ha='center')
plt.autoscale(enable=True, axis='x', tight=True)
ax2.set_ylabel('Individual trials ($\mu$V)')

ax3 = fig.add_subplot(313)
plt.plot(time_segment,average_lfp,color='r')
plt.fill_between(time_segment, average_lfp - std_lfp, average_lfp + std_lfp,
                  color='gray', alpha=0.2)
plt.axvline(x=start_time,color ='k')
plt.axvline(x=end_time,color ='k')
ax3.set_xlabel('time(s)')
ax3.set_ylabel('Volts ($\mu$V)')
plt.autoscale(enable=True, axis='x', tight=True)

# plot_filename = 'raw_VEP_data.png'
# plt.savefig(plot_filename)
plt.show()
