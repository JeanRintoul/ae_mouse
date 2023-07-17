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
import os
import time 
# 
# 2-25 is acoustic connection
# 26-40 is airgap. 
# 
# file_list_start = 2
# file_list_end   = 25
file_list_start = 6
file_list_end   = 18
file_list = np.linspace(file_list_start,file_list_end,(file_list_end-file_list_start+1),dtype=np.int16)
print ('file list:', file_list)


gain           = 1000
led_frequency  = 8 # in Hz. The number of times the LED is on per second. 
factor         = 1
savepath       = 'D:\\mouse_aeti\\e82_ae_demod\\t1\\'
Fs              = 5e6
duration        = 8.0
timestep        = 1.0/Fs
N               = int(Fs*duration)


# print("expected no. samples:",N)
marker_channel        = 2 
i_channel             = 5 
v_channel             = 4 
m_channel             = 7 
rf_channel            = 0

carrier_f       = 672800
led_duration    = 1/(2*led_frequency) # when the led turned on and off. 
start_time      = 0.0
end_time        = start_time+led_duration
start           = factor*led_duration   # this is where the trial starts seconds before marker. 
end             = factor*led_duration*2   # this is where the trial ends
start_pause     = int(0.25 * N+1)
end_pause       = int(0.875 * N-1)
window          = np.hanning(end_pause-start_pause)
window          = 1 
# 
# 
# my DC trend isn't great because I am AC Coupled everywhere to prevent OVLD.
# 
low     = 0.1
cut     = 5
sos_DC  = iirfilter(17, [low,cut], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx
#     
# Iterate through the files, and extract the frequency bands of interest, as well as the timestamp. 
# 
things = []
time_zero = 0 
for i in range(len(file_list) ):
    print ('file number:', file_list[i]) 
    filename    = savepath + 't'+str(file_list[i])+'_stream.npy'
    # time_created = creation_date(filename)
    # print("created: %s" % time.ctime(os.path.getctime(filename)),os.path.getctime(filename))
    # print ('time created:',time_created)
    time_passed = os.path.getctime(filename)
    if i == 0:
        time_zero = os.path.getctime(filename)
    time_created = int(time_passed - time_zero)
    print ('time created:',time_created)
    data = np.load(filename)
    a,b = data.shape
    t = np.linspace(0, duration, N, endpoint=False)
    # convert it to microvolts by taking the gain into account. 
    mdata = 1e6*data[m_channel]/gain
    # 
    # just use the 3 seconds onwards, as there is a weird filter effect at the start sometimes. 
    # really I need to adjust all my settings and do a more thorough DC study. 
    # 
    i1 = int(3*Fs)
    i2 = int(7.5*Fs)
    # filter the entirety of the data. 
    DC_filtered = sosfiltfilt(sos_DC, mdata) 
    DC_filtered = DC_filtered[i1:i2]
    DC_offset = np.max(DC_filtered)- np.min(DC_filtered)
    print('DC offset:',DC_offset)
    # look at the DC filtered data.  
    # fig = plt.figure(figsize=(10,6))
    # ax = fig.add_subplot(111)
    # plt.plot(t[i1:i2],data[rf_channel][i1:i2]/np.max(data[rf_channel]),color='b')
    # plt.plot(t[i1:i2],DC_filtered/np.max(DC_filtered),color='k')
    # plt.show()
    # 
    # 
    # 

    # 
    fft_data = fft(mdata[start_pause:end_pause]*window)
    fft_data = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]
    xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
    frequencies     = xf[1:(end_pause-start_pause)//2]
    
    # fig = plt.figure(figsize=(10,6))
    # ax = fig.add_subplot(211)
    # plt.plot(frequencies,fft_data,color='b')
    # ax = fig.add_subplot(212)    
    # plt.plot(t,data[m_channel],color='k')
    # plt.show()
    # 

    # print ('fft bin size ', frequencies[2]-frequencies[1])
    DC_start_idx = find_nearest(frequencies,0)
    DC_end_idx = find_nearest(frequencies,1)    
    US_idx = find_nearest(frequencies,carrier_f)

    VEP_idx = find_nearest(frequencies,led_frequency)
    Delta_start_idx = find_nearest(frequencies,1)
    Delta_end_idx = find_nearest(frequencies,4)
    Thirty_idx = find_nearest(frequencies,2*led_frequency)
    # Now what we need is the size FFT 
    modulated_5_idx  = find_nearest(frequencies,carrier_f+led_frequency)
    modulated_30_idx = find_nearest(frequencies,carrier_f+2*led_frequency)
    # 
    print ('US',fft_data[US_idx])
    things.append([time_created,np.median(fft_data[DC_start_idx:DC_end_idx]),fft_data[US_idx],fft_data[VEP_idx],np.median(fft_data[Delta_start_idx:Delta_end_idx]),fft_data[Thirty_idx],fft_data[modulated_5_idx] ,fft_data[modulated_30_idx], DC_offset ])


things = np.array(things).T
no_files,thing_items = things.shape
print ('things shape',things.shape)

outfile="fft_trends.npz"   # save out the data. 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile, things=things)
print ('saved out aggregated thing data!')


times  = things[0,:]
DC     = things[1,:]
US     = things[2,:]
VEP    = things[3,:]
Delta  = things[4,:]
Thirty = things[5,:]
modulated_Five   = things[6,:]
modulated_Thirty = things[7,:]
DC_offset        = things[8,:]

print ('times', times)
print ('DC', DC)
print ('US', US)
print ('VEP', VEP)
print ('Delta', Delta)
print ('Thirty', Thirty)
print ('DC offset', DC_offset)
#
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(221)
plt.plot(times,US,color='k')
plt.plot(times,DC_offset,color='r')
plt.plot(times,VEP,color='grey')
plt.plot(times,Delta,color='orange')
plt.plot(times,Thirty,color='cyan')

plt.plot(times,modulated_Five,color='purple')
plt.plot(times,modulated_Thirty,color='magenta')

ax.set_xlabel('Time(s)')
ax.set_ylabel('Volts ($\mu$V)')

ax2 = fig.add_subplot(222)
plt.plot(DC,US,'s',color='r')
ax2.set_xlabel('DC')
ax2.set_ylabel('US')


ax3 = fig.add_subplot(223)
plt.plot(VEP,Thirty,'s',color='r')
ax3.set_xlabel('VEP @ 5Hz')
ax3.set_ylabel('VEP @ 30Hz')


ax4 = fig.add_subplot(224)
plt.plot(DC,Delta,'s',color='r')
ax4.set_xlabel('Delta')
ax4.set_ylabel('US')


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)    

# ax.set_xlim([0,1e6])
# ax.set_ylim([0,10.0])
# plot_filename = 'VEP_FFT.png'
# plt.savefig(plot_filename)
plt.show()

