'''

Title: aggregate data after demodulation and sub-sampling. 
Author: Jean Rintoul
Date: 

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
# 
# 3 - 23
file_number = 4

start_files = 12
end_files   = 14
# end_files   = 5
files       = np.linspace(start_files,end_files,(end_files-start_files+1),dtype=np.int16)


gain           = 1000
led_frequency  = 4 # in Hz. The number of times the LED is on per second. 
factor         = 1
savepath       = 'D:\\mouse_aeti\\e100_neural_recording_pat_e_mouse\\t3_delta_wave_demodulation\\'
Fs              = 5e6
duration        = 8.0
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 4 
led_duration    = 1/(2*led_frequency) # when the led turned on and off.
t       = np.linspace(0, duration, N, endpoint=False)
# 
# 
# 
big_data = []
for i in range(len(files)):

    file_number = files[i]
    filename    = savepath + 't'+str(file_number)+'_stream.npy'
    # 
    data    = np.load(filename)
    a,b     = data.shape
    # convert it to microvolts by taking the gain into account. 
    fsignal = 1e6*data[m_channel]/gain
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
    # print ('indexes: ',f)
    print ('indexes: ',file_number,indexes[1],indexes[-1],len(indexes),len(t))
    start_time  = t[indexes[1]]
    end_time    = t[indexes[-1]]
    # Look at the raw VEPs, how are they? 
    # Ideally I would do nothing to the data, and then take the FFT of ALL of it. 
    fsignal_portion = fsignal[indexes[1]:indexes[-1]]
    t_portion       = t[indexes[1]:indexes[-1]]
    print ('t portion shape:',t_portion.shape)

    if i == 0:
        big_data = fsignal_portion
    else:
        big_data = np.concatenate((big_data,fsignal_portion))
    print ('big_data shape:',len(big_data))

    final_signal    = big_data
    N_len           = len(final_signal)
    fft_data        = rfft(np.float32(final_signal) )
    fft_data        = np.abs(2.0/(N_len) * (fft_data))[1:(N_len)//2]
    xf              = np.fft.fftfreq( (N_len), d=timestep)[:(N_len)//2]
    frequencies     = xf[1:(N_len)//2]

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(211)
    plt.plot(frequencies,fft_data,'k')
    # ax.set_xlim([0,1e6])
    ax.set_xlim([0,10])
    ax = fig.add_subplot(212)
    plt.plot(frequencies,fft_data,'k')
    ax.set_xlim([500000,500010])
    plt.show()

# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(t,fsignal/np.max(fsignal),'m')
# plt.plot(t,markerdata,'g')
# plt.axvline(x=start_time,color ='k')
# plt.axvline(x=end_time,color ='k')
# ax2 = fig.add_subplot(212)
# plt.plot(t_portion,fsignal_portion,'m')
# plt.show()
# 
# 
# 
# final_signal    = fsignal_portion
# N               = len(final_signal)
# fft_data        = fft(final_signal)
# fft_data        = np.abs(2.0/(N) * (fft_data))[1:(N)//2]
# xf              = np.fft.fftfreq( (N), d=timestep)[:(N)//2]
# frequencies     = xf[1:(N)//2]
# # 
# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_data,'k')
# plt.show()
# 
# h 
# fc                  = 500000  
# l                   = fc - 2
# h                   = fc + 2
# sos_carrier_stop    = iirfilter(17, [l,h], rs=60, btype='bandstop',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# # 
# remainder  = sosfiltfilt(sos_carrier_stop, hf_modulated_signal) 
# demodulated_signal   = demodulate(remainder,fc,t)

# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# fsignal = 1e6*data[m_channel]/gain
# plt.plot(remainder)
# # plt.plot(demodulated_signal)
# ax2 = fig.add_subplot(212)
# plt.plot(demodulated_signal)
# plt.show()
# 
# 


# # find the fft of the data. 
# # TODO: this should be from the start to the end of the led flashing, and not including the start and end regions. 
# start_pause     = int(0.25 * N+1)
# end_pause       = int(0.875 * N-1)
# window          = np.hanning(end_pause-start_pause)
# fft_data        = fft(fsignal[start_pause:end_pause]*window)
# fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]
# xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# frequencies     = xf[1:(end_pause-start_pause)//2]


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
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(t,1e6*data[m_channel]/gain,'k')
# ax.set_xlim([0,8])
# ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xlabel('Time(s)')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = 'raw_data.png'
# plt.savefig(plot_filename)
# plt.show()
# # 
# # 
# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_data,'k')
# ax.set_xlim([0,high])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = 'VEP_FFT.png'
# plt.savefig(plot_filename)
# plt.show()
# # 
# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(time_segment,average_lfp,color='r')
# plt.fill_between(time_segment, average_lfp - std_lfp, average_lfp + std_lfp,
#                   color='gray', alpha=0.2)
# plt.axvline(x=start_time,color ='k')
# plt.axvline(x=end_time,color ='k')
# ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xlabel('Time(s)')
# plt.autoscale(enable=True, axis='x', tight=True)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.title('VEP')
# plot_filename = 'VEP.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(t,marker,'r')
# plt.plot(t,fsignal/np.max(fsignal),'g')

# ax.set_xlim([0.5,7.5])
# ax.set_ylim([-2,2])
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
