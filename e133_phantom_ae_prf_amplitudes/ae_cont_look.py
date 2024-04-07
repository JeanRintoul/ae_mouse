'''

Title: compare the data going into generator with the data coming out of generator. 

Author: Jean Rintoul
Date:   26.10.2023

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
from math import floor
from scipy.signal import hilbert
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 
# 
carrier_frequency = 500000
# data = np.load('D:\\ae_mouse\\e133_phantom_ae_prf_amplitudes\\t2-vep_data.npz')
# savepath       = 'D:\\ae_mouse\\e133_phantom_ae_prf_amplitudes\\t1_phantom_prf\\continuous_demodulation\\'
# savepath       = 'D:\\ae_mouse\\e133_phantom_ae_prf_amplitudes\\t3_phantom\\'
# # 
# savepath     = 'D:\\ae_mouse\\e133_phantom_ae_prf_amplitudes\\t2_mouse\\5Hz_neuraldemod\\'
savepath       = 'D:\\ae_mouse\\e133_phantom_ae_prf_amplitudes\\t4_mouse\\'
# 
# 

duration        = 8
Fs              = 5e6
# Fs              = 2e6
timestep        = 1.0/Fs
N               = int(Fs*duration)
channel_of_interest = 0
# # 
filename    ='t1_stream.npy'
# filename    ='t26_stream_noF21.npy'
# # filename    ='t4_stream_noF21.npy'
# filename    ='t10_stream_noF21.npy'
# #
# filename    ='t1_stream.npy'

print ('filename: ',filename)

data = np.load(savepath+filename)
a,b = data.shape
print ('shape',a,b)
gain    = 1000 
fsignal = 1e6*data[channel_of_interest]/gain
t       = np.linspace(0, duration, N, endpoint=False)

current_signal_frequency = 4
print ('current_signal_frequency: ', current_signal_frequency)
start_time = np.round(0.8/duration ,2) 
end_time   = np.round((duration - 0.4)/duration,2)
print ('start and end',start_time,end_time)

# Play with windowing to decrese the spectral leakage of the big signal next to the small signal. 
start_pause     = int(start_time * N+1)
end_pause       = int(end_time *N-1)

print ('window length:', end_pause - start_pause)
multiple = floor((end_pause - start_pause)/carrier_frequency)
print ('sub-multiple:', floor((end_pause - start_pause)/carrier_frequency) )
print ('start time:', start_pause*timestep)
print ('end time:', end_pause*timestep)
# this makes it a perfect multiple, though it doesnt start on a zero point. 
# start_pause  = start_pause + carrier
# end_pause    = start_pause + int( multiple-1 )*carrier

window       = np.kaiser( (end_pause - start_pause), 20.0 )


newN = end_pause - start_pause

fft_m = fft(fsignal[start_pause:end_pause]*window)
fft_m = np.abs(2.0/(newN) * (fft_m))[1:(newN)//2]
xf = np.fft.fftfreq( (newN), d=timestep)[:(newN)//2]
frequencies = xf[1:(newN)//2]

bands = 40

fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_m,'k')
ax.set_xlim([carrier_frequency-bands,carrier_frequency+bands])
ax.set_ylim([0,0.4])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Frequency (Hz)',fontsize=16)
ax.set_ylabel('Normalized Volts ($\mu$V)',fontsize=16)
plt.title('Modulated LFP Signal',fontsize=16)
plt.tight_layout()
plot_filename = 'neural_demodulation.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_m,'k')
ax.set_xlim([0,bands])
ax.set_ylim([0,50])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Frequency (Hz)',fontsize=16)
ax.set_ylabel('Normalized Volts ($\mu$V)',fontsize=16)
plt.title('Simulated LFP Signal',fontsize=16)
plt.tight_layout()
plot_filename = 'base_signal.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()


l_cut  = 2
df_cut = 30
# 
# l_cut  = 9.2
# df_cut = 10.5
# new_Fs = 1e5 
# 
def demodulate(in_signal,carrier_f,dt): 

    # create a demodulated signal. 
    lowm  = carrier_f - df_cut
    highm = carrier_f + df_cut
    # print ('low/high',lowm,highm)
    modulation_filter = iirfilter(17, [lowm,highm], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
    modulated_signal  = sosfiltfilt(modulation_filter, in_signal)

    idown = modulated_signal*np.cos(2*np.pi*carrier_f*dt)
    qdown = modulated_signal*np.sin(2*np.pi*carrier_f*dt)   

    demodulated_signal = idown + 1j*qdown
    # demodulated_signal = np.real(idown)
    demodulated_signal = np.abs(demodulated_signal)
    demodulated_signal = demodulated_signal - np.mean(demodulated_signal)
    d_demod_data = sosfiltfilt(sos_df_band, demodulated_signal)     
    return d_demod_data
# 
sos_df_band = iirfilter(17, [l_cut,df_cut], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
sos_lp = iirfilter(17, [4000], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')


start_cut = carrier_frequency - 1
end_cut = carrier_frequency + 1
# start_cut = carrier_frequency - 2
# end_cut = carrier_frequency + 2
sos_bs = iirfilter(17, [start_cut,end_cut], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

sos_hp = iirfilter(17, [499000,501000], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

# 
# 
# downsampling_factor  = int(Fs/new_Fs)
# print('downsampling factor: ',downsampling_factor)
# # downsample for easier plotting. 
# downsampled_data1           = sosfiltfilt(sos_lp, fsignal)
# downsampled_lfp             = downsampled_data1[::downsampling_factor]
# td                          = t[::downsampling_factor]
# 
# Final filters. 
lfp_data                       = sosfiltfilt(sos_df_band, fsignal)   
# 
# fsignal                       = sosfiltfilt(sos_bs, fsignal)   
# bp_data                      = sosfiltfilt(sos_hp, bp_data)   
# 
lowm  = carrier_frequency - df_cut
highm = carrier_frequency + df_cut
# print ('low/high',lowm,highm)
modulation_filter = iirfilter(17, [lowm,highm], rs=60, btype='bandpass',
                   analog=False, ftype='cheby2', fs=Fs,
                   output='sos')
modulated_signal  = sosfiltfilt(modulation_filter, fsignal)

# ensure the entire signal is above zero. 
modulated_signal = modulated_signal + np.abs(np.min(modulated_signal))*1.5

analytical_signal = hilbert(modulated_signal)
h_signal = np.abs(analytical_signal)
h_signal = h_signal - np.mean(h_signal)
h_data = sosfiltfilt(sos_df_band, h_signal)  

# 
demodulated         = demodulate(fsignal,carrier_frequency,t)

fft_bp              = fft(modulated_signal[start_pause:end_pause]*window)
fft_bp              = np.abs(2.0/(newN) * (fft_bp))[1:(newN)//2]
#
# fft_bp              = fft(modulated_signal[start_pause:end_pause]*window)
# fft_bp              = np.abs(2.0/(newN) * (fft_bp))[1:(newN)//2]
#
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(311)
plt.plot(frequencies,fft_bp,'k')
ax.set_xlim([carrier_frequency-df_cut,carrier_frequency+df_cut])
# ax.set_xlim([carrier_frequency-100,carrier_frequency+100])
# ax.set_ylim([0,0.075])
ax2  = fig.add_subplot(312)
plt.plot(t,modulated_signal)
# ax.set_xlim([carrier_frequency-100,carrier_frequency+100])
ax3 = fig.add_subplot(313)
plt.plot(t,demodulated,'k')
plt.plot(t,h_data,'m')
plt.show()

# 
# demodulated = demodulate(fsignal[start_pause:end_pause],carrier_frequency,t[start_pause:end_pause])
# 
start_pause  = int(2*Fs ) 
end_pause    = int(6*Fs )

newN         = int(end_pause - start_pause)
window       = np.kaiser( (end_pause - start_pause), 14.0 )
# 
fft_dm              = fft(demodulated[start_pause:end_pause]*window)
fft_dm              = np.abs(2.0/(newN) * (fft_dm))[1:(newN)//2]

fft_h              = fft(h_data[start_pause:end_pause]*window)
fft_h              = np.abs(2.0/(newN) * (fft_h))[1:(newN)//2]

xf                  = np.fft.fftfreq( (newN), d=timestep)[:(newN)//2]
frequencies         = xf[1:(newN)//2]
#
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(211)
plt.plot(t,lfp_data/np.max(lfp_data),'r')
plt.plot(t,demodulated/np.max(demodulated),'k')
plt.plot(t,h_data/np.max(h_data),'m')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.set_xlabel('Time (s)',fontsize=16)
ax.set_ylabel('Normalized Volts ($\mu$V)',fontsize=16)
plt.title('Demodulated Signal',fontsize=16)
plt.tight_layout()
plt.legend(['LFP','Demodulated'],loc='upper right')

ax2  = fig.add_subplot(212)
plt.plot(frequencies,fft_dm,'k')
plt.plot(frequencies,fft_h,'m')
# plt.plot(frequencies,fft_m,'b')
ax2.set_xlim([0,bands])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = 'demodulated_signal.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()



