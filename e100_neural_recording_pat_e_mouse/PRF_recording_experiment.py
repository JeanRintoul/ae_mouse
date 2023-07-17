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
# the 30 hz I see is coming from 1050Hz. 
c            = 2*500000
# in phase, out of phase, in phase, unclear - messy, out of phase. 1020*4 none
# carrier_list = [1020*1,1020*2,1020*3,1020*4,1020*5]
# in phase, in phase, in phase, out of phase but weak, in phase, out of phase- big, 1020*4 weak
# carrier_list = [c,c+1020*1,c+1020*2,c+1020*3,c+1020*4,c+1020*5]
# out of phase, out of phase, out of phase, out of phase, in phase but hard to tell/messy.
# carrier_list = [c-1020*1,c-1020*2,c-1020*3,c-1020*4,c-1020*5] #  

# 
# carrier_list = [1020*1,c-1020*1,c,c+1020*1,c+1020*2,c+1020*4]
# 
# in phase
carrier_list = [1020*1,c+1020*1]
# carrier_list = [c,c+1020*1]
# carrier_list = [c,c]
# out of phase. 
# carrier_list = [c-1020*1,c-1020*2,c-1020*3,c-1020*4,1020*2,1020*5,c+1020*5]
print ('carrier list',carrier_list)

file_number     = 18
gain            = 1000
# savepath      = 'D:\\mouse_aeti\\e100_neural_ recording_pat_e_mouse\\t6_mep_delta_wave_motorcortex\\'
savepath        = 'D:\\mouse_aeti\\e100_neural_recording_pat_e_mouse\\t6_mep_delta_wave_motorcortex\\'
# 
Fs              = 5e6
duration        = 8.0
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 4 
# 
# 
def demodulate(measured_signal,carrier_f):
    offset              = np.min(measured_signal)
    offset_adjustment   = offset*np.cos(2 * np.pi * carrier_f * t)
    IQ                  = (measured_signal+offset_adjustment)*np.exp(1j*(2*np.pi*carrier_f*t )) 
    idown               = np.real(IQ)
    qdown               = np.imag(IQ)
    idown               = sosfiltfilt(lp_filter, idown)
    qdown               = sosfiltfilt(lp_filter, qdown)  
    rsignal             = -(idown + qdown)
    rsignal             = rsignal - np.mean(rsignal) 
    return rsignal
# 
# 
low                     = 2
high                    = 300
# low pass filter. 
lp_filter               = iirfilter(17, [low,high], rs=60, btype='bandpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')
# 
sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

filename    = savepath + 't'+str(file_number)+'_stream.npy'
data = np.load(filename)
a,b = data.shape
t = np.linspace(0, duration, N, endpoint=False)
# convert it to microvolts by taking the gain into account. 
fsignal = 1e6*data[m_channel]/gain
unfiltered_data = fsignal
rf_data = 10*data[rf_channel]

fft_demod_list  = []
demod_list      = []
fsignal_list    = []
for i in range(len(carrier_list) ):
    carrier_f = carrier_list[i]
    print ('carrier_f',carrier_f)
    l = carrier_f - high
    h = carrier_f + high
    sos_carrier_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')

    # do the demodulation
    fraw            = sosfiltfilt(sos_carrier_band, fsignal) 
    demod_data      = demodulate(fraw,carrier_f)
    # final low pass filter. 
    demodulated     = sosfiltfilt(sos_low, demod_data)
    lfpsignal       = sosfiltfilt(sos_low, fsignal)

    # Now decimate the result to 100k
    desired_plot_sample_rate = 1e5
    decimation_factor        = int(Fs/desired_plot_sample_rate)  
    sig         = lfpsignal[::decimation_factor]      
    demod       = demodulated[::decimation_factor]    
    td          = t[::decimation_factor]   
    d_timestep  = 1.0/desired_plot_sample_rate 
    t1          = 2 
    t2          = 6
    start_pause = int(t1*desired_plot_sample_rate)  
    end_pause   = int(t2*desired_plot_sample_rate)
    # 
    fft_demod      = fft(demod[start_pause:end_pause])
    fft_demod      = np.abs(2.0/(end_pause-start_pause) * (fft_demod))[1:(end_pause-start_pause)//2]
    fft_sig        = fft(sig[start_pause:end_pause])
    fft_sig        = np.abs(2.0/(end_pause-start_pause) * (fft_sig))[1:(end_pause-start_pause)//2]
    # 
    xf               = np.fft.fftfreq( (end_pause-start_pause), d=d_timestep)[:(end_pause-start_pause)//2]
    frequencies      = xf[1:(end_pause-start_pause)//2]
    print ('fft bin size ', frequencies[2]-frequencies[1])
    # 
    fft_demod_list.append(fft_demod)
    demod_list.append(demod)
    fsignal_list.append(sig)
#
#
demods  = np.array(demod_list)
fsigs   = np.array(fsignal_list)
fftds   = np.array(fft_demod_list)
siggy   = fsigs[0,:]
a,b     = demods.shape
print ('demod shape:',demods.shape)
added = 0 
fft_added = 0 
for b in range(a):
    if b >0:
        added = added + demods[b,:]
        fft_added = fft_added + fftds[b,:]


#         
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(311)
plt.plot(td,demods.T )
# ax.set_xlim([0,duration])
plt.legend(['1','2','3','4','5'])
ax2 = fig.add_subplot(312)
plt.plot(td,siggy,'k')
# 
ax3 = fig.add_subplot(313)
plt.plot(td,(added - np.min(added)) /(np.max(added) - np.min(added)) ,'k' )
plt.plot(td,(siggy - np.min(siggy))/(np.max(siggy) - np.min(siggy)),'r' )
# plt.plot(td,demods[0,:]/np.max(demods[0,:]),'b' )
# plt.legend(['rf data','lp lfp','demod'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

plt.autoscale(enable=True, axis='x', tight=True)
plot_filename = 'raw_and_demod_data.png'
plt.savefig(plot_filename)
plt.show()


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(frequencies,fft_added/np.max(fft_added),'k')
plt.plot(frequencies,fft_sig/np.max(fft_sig),'r')
ax.set_xlim([0,high])
plt.legend(['demod','orig'],loc='upper right')
plt.show()
# find the fft of the data. 
# TODO: this should be from the start to the end of the led flashing, and not including the start and end regions. 
# start_pause     = int(2*Fs)
# end_pause       = int(6*Fs)
# window          = np.hanning(end_pause-start_pause)

# fft_unfiltered_data = fft(unfiltered_data[start_pause:end_pause]*window)
# fft_unfiltered_data = np.abs(2.0/(end_pause-start_pause) * (fft_unfiltered_data))[1:(end_pause-start_pause)//2]

# fft_rfdata        = fft(rf_data[start_pause:end_pause]*window)
# fft_rfdata        = np.abs(2.0/(end_pause-start_pause) * (fft_rfdata))[1:(end_pause-start_pause)//2]


# fft_data        = fft(fsignal[start_pause:end_pause]*window)
# fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]

# fft_demod        = fft(demodulated[start_pause:end_pause]*window)
# fft_demod        = np.abs(2.0/(end_pause-start_pause) * (fft_demod))[1:(end_pause-start_pause)//2]

# xf               = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# frequencies      = xf[1:(end_pause-start_pause)//2]
# print ('fft bin size ', frequencies[2]-frequencies[1])
# # 

# fig = plt.figure(figsize=(6,6))
# ax1 = fig.add_subplot(111)
# plt.plot(frequencies,fft_rfdata/np.max(fft_rfdata),color='r')
# plt.plot(frequencies,fft_unfiltered_data/np.max(fft_unfiltered_data),color='k')
# plt.legend(['rf signal','measured signal'],loc ='upper left')
# # ax2.set_ylim([0,2])
# ax1.set_xlim([400000,600000])
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# plot_filename = 'FFT_comparison.png'
# plt.savefig(plot_filename)
# plt.show()
# 
# fig = plt.figure(figsize=(6,6))
# ax1 = fig.add_subplot(211)
# plt.plot(t,unfiltered_data,color='k')
# ax2 = fig.add_subplot(212)
# plt.plot(frequencies,fft_rfdata/np.max(fft_rfdata),color='r')
# plt.plot(frequencies,fft_unfiltered_data/np.max(fft_unfiltered_data),color='k')
# plt.legend(['rf signal','measured signal'],loc ='upper left')
# # ax2.set_ylim([0,2])
# ax2.set_xlim([0,1010000])
# plt.show()
# # 
# the carrier frequencies. 
# 
# 
# 
# 
# fig = plt.figure(figsize=(10,6))

# ax1 = fig.add_subplot(311)
# plt.plot(frequencies,fft_data,'k')
# ax1.set_xlim([0,high])
# plt.legend(['raw'])
# # ax1.set_ylim([0,100])

# ax2 = fig.add_subplot(312)
# plt.plot(frequencies,fft_demod,'r')
# ax2.set_xlim([0,high])
# plt.legend(['demod'])

# ax3 = fig.add_subplot(313)
# plt.plot(frequencies,fft_unfiltered_data,'k')
# ax3.set_xlim([carrier_f,carrier_f + high])
# plt.legend(['at carrier'])
# # ax3.set_ylim([0,0.08])
# # 
# # ax4 = fig.add_subplot(414)
# # plt.plot(frequencies,fft_demod,'r')
# # # ax4.set_xlim([carrier_f - high,carrier_f + high ])
# # plt.legend(['demod'])
# # ax4.set_ylim([0,0.0001])
# # 
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# # ax4.spines['right'].set_visible(False)
# # ax4.spines['top'].set_visible(False)

# plot_filename = 'FFT.png'
# plt.savefig(plot_filename)
# plt.show()
# 
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(t,rf_data/np.max(rf_data),'grey')
# plt.plot(t,fsignal/np.max(fsignal),'r')
# plt.plot(t,demodulated/np.max(demodulated),'k')
# ax.set_xlim([0,duration])
# # ax.set_ylim([-2,2])
# plt.legend(['rf data','lp lfp','demod'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.autoscale(enable=True, axis='x', tight=True)
# plot_filename = 'raw_and_demod_data.png'
# plt.savefig(plot_filename)
# plt.show()

