'''

Title: vep signal inspection
Function: takes a single file, and averages the veps to see them better. 

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
file_number    = 10
gain           = 1000
led_frequency  = 4 # in Hz. The number of times the LED is on per second. 
factor         = 1
savepath       = 'D:\\mouse_aeti\\e97_MEPS\\t18_mouse_eeg\\'
Fs              = 5e6
duration        = 8.0
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     

led_duration = 1/(2*led_frequency) # when the led turned on and off. 
start_time = 0.0
end_time   = start_time+led_duration
start = factor*led_duration   # this is where the trial starts seconds before marker. 
end   = factor*led_duration*2   # this is where the trial ends

filename    = savepath + 't'+str(file_number)+'_stream.npy'
data = np.load(filename)
a,b = data.shape
t = np.linspace(0, duration, N, endpoint=False)
# convert it to microvolts by taking the gain into account. 
fsignal = 1e6*data[m_channel]/gain
mains_harmonics = [50,100,150,200,250,300]
# mains_harmonics = [50,100]
for i in range(len(mains_harmonics)):
    mains_low  = mains_harmonics[i] -2
    mains_high = mains_harmonics[i] +2
    mains_sos = iirfilter(17, [mains_low,mains_high], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
    fsignal = sosfiltfilt(mains_sos, fsignal)
# 
low  = 0.5
high = 1000 
high = 50 
sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
fsignal  = sosfiltfilt(sos_low, fsignal)

# find the fft of the data. 
# TODO: this should be from the start to the end of the led flashing, and not including the start and end regions. 
start_pause     = int(0.25 * N+1)
end_pause       = int(0.875 * N-1)
window          = np.hanning(end_pause-start_pause)
fft_data        = fft(fsignal[start_pause:end_pause]*window)
fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]
xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies     = xf[1:(end_pause-start_pause)//2]

# create the marker channel. 
markerdata        = np.array(data[marker_channel])
marker            = markerdata/np.max(markerdata)
diffs             = np.diff( markerdata )
zarray            = t*[0]
indexes           = np.argwhere(diffs > 0.2)[:,0] # leading edge
marker_up_length  = int(led_duration*Fs)
for i in range(len(indexes)):
    if i > 0:
        zarray[indexes[i]:(indexes[i]+marker_up_length) ] = 1
markerdata = zarray
# 
pre_event_idxcount              = int(start*Fs)
post_event_idxcount             = int(end*Fs)
data_to_segment                 = fsignal
filtered_segmented_data         = []
for i in range(len(indexes)-1):  # the last index may get chopped short. 
  filtered_segmented_data.append(data_to_segment[(indexes[i+1]-pre_event_idxcount):(indexes[i+1]+post_event_idxcount)])
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
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(t,1e6*data[m_channel]/gain,'k')
ax.set_xlim([0,8])
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Time(s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = 'raw_data.png'
plt.savefig(plot_filename)
plt.show()
# 
# 
# 
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(frequencies,fft_data,'k')
ax.set_xlim([0,high])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
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
plt.plot(t,marker,'r')
plt.plot(t,fsignal/np.max(fsignal),'g')

ax.set_xlim([0.5,7.5])
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
