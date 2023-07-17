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
file_list_start = 26
file_list_end   = 40
file_list = np.linspace(file_list_start,file_list_end,(file_list_end-file_list_start+1),dtype=np.int16)
print ('file list:', file_list)


gain           = 1000
led_frequency  = 5 # in Hz. The number of times the LED is on per second. 
factor         = 1
savepath       = 'D:\\mouse_aeti\\e96_demod\\t2\\'
Fs              = 5e6
duration        = 8.0
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0  
rf_channel      = 2   
carrier_f       = 500000
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
sos_DC = iirfilter(17, [low,cut], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx
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
    fft_data = fft(mdata[start_pause:end_pause]*window)
    fft_data = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]
    xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
    frequencies     = xf[1:(end_pause-start_pause)//2]
    # print ('fft bin size ', frequencies[2]-frequencies[1])
    DC_start_idx = find_nearest(frequencies,0)
    DC_end_idx = find_nearest(frequencies,1)    
    US_idx = find_nearest(frequencies,500000)
    VEP_idx = find_nearest(frequencies,5)
    Delta_start_idx = find_nearest(frequencies,1)
    Delta_end_idx = find_nearest(frequencies,4)
    Thirty_idx = find_nearest(frequencies,30)
    # Now what we need is the size FFT 
    modulated_5_idx  = find_nearest(frequencies,500005)
    modulated_30_idx = find_nearest(frequencies,500030)
    # 
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


# def demodulate(measured_signal):
#     # That's interesting, if I add a DC offset here, I obtain the result correctly, otherwise the result is rectified into obscurity. 
#     offset = np.min(measured_signal)
#     offset_adjustment = offset*np.sin(2 * np.pi * carrier_f * t)
#     IQ = (measured_signal+offset_adjustment)*np.exp(1j*(2*np.pi*carrier_f*t )) 
#     # idown = measured_signal*np.cos(2*np.pi*carrier_f*t)
#     # qdown = -measured_signal*np.sin(2*np.pi*carrier_f*t)    
#     idown = np.real(IQ)
#     qdown = np.imag(IQ)
#     v = idown + 1j*qdown
#     mag = np.abs(v)
#     mag = mag - np.mean(mag)
#     sign_data = -(idown + qdown)
#     # get the magnitude, then correct it's sign as the magnitude is just the envelope.  
#     for i in range(len(mag)):
#         sign = sign_data[i]
#         if sign > 0:
#             mag[i] = mag[i]
#         else:
#             mag[i] -mag[i]

#     return mag

# def demodulate(measured_signal):
#     offset              = np.min(measured_signal)
#     offset_adjustment   = offset*np.cos(2 * np.pi * carrier_f * t)
#     IQ                  = (measured_signal+offset_adjustment)*np.exp(1j*(2*np.pi*carrier_f*t )) 
#     idown               = np.real(IQ)
#     qdown               = np.imag(IQ)
#     idown               = sosfiltfilt(lp_filter, idown)
#     qdown               = sosfiltfilt(lp_filter, qdown)  
#     rsignal             = -(idown + qdown)
#     rsignal             = rsignal - np.mean(rsignal) 
#     return rsignal

# carrier_f               = 500000
# filter_cutoff           = 2000


# low  = 4.5
# high = 5.5

# low  = 28
# high = 35

# # low  = 2
# # high = 35
# sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')


# sos_center_cut = iirfilter(17, [carrier_f-2,carrier_f+2], rs=60, btype='bandstop',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')

# # define the carrier frequency. 
# fc = carrier_f
# l = fc - high
# h = fc + high
# sos_carrier_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')

# lp_filter               = iirfilter(17, [filter_cutoff], rs=60, btype='lowpass',
#                             analog=False, ftype='cheby2', fs=Fs,
#                             output='sos')

# led_duration = 1/(2*led_frequency) # when the led turned on and off. 
# start_time = 0.0
# end_time   = start_time+led_duration
# start = factor*led_duration   # this is where the trial starts seconds before marker. 
# end   = factor*led_duration*2   # this is where the trial ends


# basefilename    = savepath + 't'+str(basefile_number)+'_stream.npy'
# bdata = np.load(basefilename)
# basesignal = 1e6*bdata[m_channel]/gain

# filename    = savepath + 't'+str(file_number)+'_stream.npy'
# data = np.load(filename)
# a,b = data.shape
# t = np.linspace(0, duration, N, endpoint=False)
# # convert it to microvolts by taking the gain into account. 
# fsignal = 1e6*data[m_channel]/gain
# unfiltered_data = fsignal
# # do the demodulation
# fraw           = sosfiltfilt(sos_carrier_band, fsignal) 
# fraw           = sosfiltfilt(sos_center_cut, fraw) 

# demod_data     = demodulate(fraw)
# # 
# demodulated     = sosfiltfilt(sos_low, demod_data)
# fsignal         = sosfiltfilt(sos_low, fsignal)

# # find the fft of the data. 
# # TODO: this should be from the start to the end of the led flashing, and not including the start and end regions. 
# start_pause     = int(0.25 * N+1)
# end_pause       = int(0.875 * N-1)
# window          = np.hanning(end_pause-start_pause)
# window          = 1 

# fft_unfiltered_data = fft(unfiltered_data[start_pause:end_pause]*window)
# fft_unfiltered_data = np.abs(2.0/(end_pause-start_pause) * (fft_unfiltered_data))[1:(end_pause-start_pause)//2]

# fft_base_data = fft(basesignal[start_pause:end_pause]*window)
# fft_base_data = np.abs(2.0/(end_pause-start_pause) * (fft_base_data))[1:(end_pause-start_pause)//2]


# fft_data        = fft(fsignal[start_pause:end_pause]*window)
# fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]

# fft_demod        = fft(demodulated[start_pause:end_pause]*window)
# fft_demod        = np.abs(2.0/(end_pause-start_pause) * (fft_demod))[1:(end_pause-start_pause)//2]


# xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# frequencies     = xf[1:(end_pause-start_pause)//2]

# print ('fft bin size ', frequencies[2]-frequencies[1])

# # create the marker channel. 
# markerdata        = np.array(data[marker_channel])
# marker            = markerdata/np.max(markerdata)
# diffs             = np.diff( markerdata )
# zarray            = t*[0]
# indexes           = np.argwhere(diffs > 0.2)[:,0] # leading edge
# marker_up_length  = int(led_duration*Fs)
# for i in range(len(indexes)):
#     if i > 0:
#         zarray[indexes[i]:(indexes[i]+marker_up_length) ] = 1
# markerdata = zarray
# # 
# pre_event_idxcount              = int(start*Fs)
# post_event_idxcount             = int(end*Fs)
# data_to_segment                 = fsignal
# filtered_segmented_data         = []
# for i in range(len(indexes)-1):  # the last index may get chopped short. 
#   filtered_segmented_data.append(data_to_segment[(indexes[i+1]-pre_event_idxcount):(indexes[i+1]+post_event_idxcount)])
# # 
# filtered_segmented_array = np.array(filtered_segmented_data)
# a,b = filtered_segmented_array.shape
# print ('shape', a,b)
# time_segment = np.linspace(-start,end,num=b)
# average_lfp = np.median(filtered_segmented_array,axis=0)
# std_lfp     = np.std(filtered_segmented_array,axis=0)
# print ('n_events: ', a)
# # 
# # 
# lfp_height = np.max(average_lfp)- np.min(average_lfp)
# print ('LFP size', lfp_height)
# # 
# # 
# # The FFT appears to be echoed at: 
# # 500kHz, 1MHz, 0.75MHz, 1.5MHz
# # 
# # there is a weird introduced phase shift during the demodulation. 
# # I need to understand this better. 
# # However, I think if I subtracted the value of a file with a smaller VEP... 
# # I might be able to remove that central 500khz spike. 
# # 
# fft_diff = fft_unfiltered_data - fft_base_data


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(121)
# # plt.plot(frequencies,fft_data,'r')

# plt.plot(frequencies,fft_unfiltered_data,'k')
# plt.plot(frequencies,fft_demod,'r')
# ax.set_xlim([0,50])
# ax.set_ylim([0,10.0])
# ax2 = fig.add_subplot(122)
# plt.plot(frequencies,fft_unfiltered_data,'k')
# plt.plot(frequencies,fft_base_data,'r')
# plt.plot(frequencies,fft_diff,'g')
# ax2.set_xlim([500000,500000+50])
# # ax2.set_ylim([0,2])
# plot_filename = 'VEP_FFT.png'
# plt.savefig(plot_filename)
# plt.show()
# # 
# # 
# # A big problem is how to deal with the center frequency. 
# # Ultrasound with no LED flash. 
# # 
# # 
# # fig = plt.figure(figsize=(10,6))
# # ax = fig.add_subplot(111)
# # plt.plot(time_segment,average_lfp,color='r')
# # plt.fill_between(time_segment, average_lfp - std_lfp, average_lfp + std_lfp,
# #                   color='gray', alpha=0.2)
# # plt.axvline(x=start_time,color ='k')
# # plt.axvline(x=end_time,color ='k')
# # ax.set_ylabel('Volts ($\mu$V)')
# # ax.set_xlabel('Time(s)')
# # plt.autoscale(enable=True, axis='x', tight=True)
# # ax.spines['right'].set_visible(False)
# # ax.spines['top'].set_visible(False)
# # plt.title('VEP')
# # plot_filename = 'VEP.png'
# # plt.savefig(plot_filename)
# # plt.show()

# # off = -(5.3115 - 5.2878)
# off = 0.0
# lim1 = int(0.5*Fs)
# lim2 = int(7.5*Fs)
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(t[lim1:lim2],0.5*marker[lim1:lim2],'r')
# plt.plot(t[lim1:lim2],fsignal[lim1:lim2]/np.max(fsignal[lim1:lim2]),'g')
# plt.plot(t[lim1:lim2]+off,demodulated[lim1:lim2]/np.max(demodulated[lim1:lim2]),'b')
# # ax.set_xlim([lim1,lim2])
# # ax.set_ylim([-2,2])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# ax2 = fig.add_subplot(312)
# plt.plot(time_segment,filtered_segmented_array.T)
# plt.axvline(x=start_time,color ='k')
# plt.axvline(x=end_time,color ='k')
# # ax2.text(0.5, 100, 'individual trials', color='black',fontsize = 6, ha='center')
# plt.autoscale(enable=True, axis='x', tight=True)
# ax2.set_ylabel('Individual trials ($\mu$V)')

# ax3 = fig.add_subplot(313)
# plt.plot(time_segment,average_lfp,color='r')
# plt.fill_between(time_segment, average_lfp - std_lfp, average_lfp + std_lfp,
#                   color='gray', alpha=0.2)
# plt.axvline(x=start_time,color ='k')
# plt.axvline(x=end_time,color ='k')
# ax3.set_xlabel('time(s)')
# ax3.set_ylabel('Volts ($\mu$V)')
# plt.autoscale(enable=True, axis='x', tight=True)

# # plot_filename = 'raw_VEP_data.png'
# # plt.savefig(plot_filename)
# plt.show()
