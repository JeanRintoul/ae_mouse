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
# savepath      = 'D:\\ae_mouse\\e135_ae_neural_decoding\\t5_phantom\\signal_to_noise_challenge\\'


# Continuous wave. 
savepath            = 'D:\\ae_mouse\\e135_ae_neural_decoding\\t5_phantom\\g1_continuous_wave\\'
filename            = 't9_stream.npy'


# PRF Wave. 
# savepath            = 'D:\\ae_mouse\\e135_ae_neural_decoding\\t5_phantom\\g1_prf_with_pressure\\'
# filename            = 't1_stream.npy'


# No Ultrasound
savepath_noUS            = 'D:\\ae_mouse\\e135_ae_neural_decoding\\t5_phantom\\g1_prf_without_pressure\\'
filename_noUS            = 't5_stream.npy'

saveprefix          = './/images//'

duration            = 12
Fs                  = 2e6
timestep            = 1.0/Fs
N                   = int(Fs*duration)
carrier_frequency   = 500000
channel_of_interest = 0
rfchannel           =  4
gain                = 100
t                   = np.linspace(0, duration, N, endpoint=False)
current_signal_frequency = 24
print ('filename: ',filename)
data = np.load(savepath+filename)
a,b = data.shape
print ('shape',a,b)
rfsignal = 10*data[rfchannel]
noUS_data = np.load(savepath_noUS+filename_noUS)


print ('current_signal_frequency: ', current_signal_frequency)
start_time      = np.round(0.8/duration ,2)
end_time        = np.round((duration - 0.2)/duration,2)
print ('start and end',start_time,end_time)
start_pause     = int(start_time * N+1)
end_pause       = int(end_time *N-1)

fsignal     = 1e6*data[channel_of_interest]/gain


fsignal_noUS = 1e6*noUS_data[channel_of_interest]/gain


beta        = 14
window      = np.kaiser( (end_pause-start_pause), beta )
print('window len:',len(window))

fft_m = fft(fsignal[start_pause:end_pause])
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
fft_noUS = fft(fsignal_noUS[start_pause:end_pause])
fft_noUS = np.abs(2.0/(end_pause-start_pause) * (fft_noUS))[1:(end_pause-start_pause)//2]

fft_rf = fft(rfsignal[start_pause:end_pause])
fft_rf = np.abs(2.0/(end_pause-start_pause) * (fft_rf))[1:(end_pause-start_pause)//2]

# 
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]

band_of_interest = 40

prf = 80

fig = plt.figure(figsize=(4,2))
ax  = fig.add_subplot(111)
plt.plot(t,rfsignal,'k')
# ax.set_xlim([3,3.08])
ax.set_xlim([3,3.0001])
plt.yticks([])
plt.xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plt.savefig(saveprefix +"PRF_rfwave_timeseries.png", bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(4,2))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_rf,'k')
ax.set_xlim([carrier_frequency- 1000,carrier_frequency+1000])
ax.set_ylim([0,10])
plt.xticks([499000,500000,501000])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(saveprefix +"PRF_rfwave_fft.png", bbox_inches="tight")
plt.show()


# # PRF 80 
# fig = plt.figure(figsize=(4,4))
# ax  = fig.add_subplot(111)
# # plt.axvline(x= (carrier_frequency+current_signal_frequency),color='g')
# # plt.axvline(x= (carrier_frequency-current_signal_frequency),color='g')
# # plt.plot(frequencies,fft_noUS,'g')
# plt.plot(frequencies,fft_m,'k')
# # ax.set_xlim([prf-band_of_interest,prf+band_of_interest])
# ax.set_xlim([0,120])
# # ax.set_ylim([0,10])
# ax.set_ylim([0,np.max(fft_m)])
# #ax.set_xticks([carrier_frequency-band_of_interest,carrier_frequency-current_signal_frequency,carrier_frequency,carrier_frequency+current_signal_frequency,carrier_frequency+band_of_interest]) 
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.ticklabel_format(useOffset=False)
# plt.tight_layout()
# plt.savefig(saveprefix +"PRF_wave_ASD.png", bbox_inches="tight")
# plt.show()

# Continuous wave. 
fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(111)
# plt.axvline(x= (carrier_frequency+current_signal_frequency),color='g')
# plt.axvline(x= (carrier_frequency-current_signal_frequency),color='g')
plt.plot(frequencies,fft_m,'k')
ax.set_xlim([carrier_frequency-band_of_interest,carrier_frequency+band_of_interest])
ax.set_ylim([0,35])
ax.set_xticks([carrier_frequency-current_signal_frequency,carrier_frequency,carrier_frequency+current_signal_frequency]) 
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.ticklabel_format(useOffset=False)
plt.tight_layout()
plt.savefig(saveprefix +"continuous_wave_ASD.png", bbox_inches="tight")
plt.show()

# 
# band_filter = iirfilter(17, [20,30], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# fsignal1x                       = sosfiltfilt(band_filter, fsignal)


# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(111)
# plt.plot(t,fsignal1,'k')
# plt.plot(t,fsignal2,'r')
# # plt.plot(t,fsignal3,'g')
# # plt.plot(t,fsignal4,'b')

# # plt.plot(t,data[7],'k')
# # plt.plot(t,data2[7],'r')
# # plt.plot(t,data3[7],'g')
# # plt.plot(t,data4[7],'b')
# plt.show()

# current_signal_frequency = 4
# print ('current_signal_frequency: ', current_signal_frequency)
# start_time = np.round(0.8/duration ,2) 
# end_time   = np.round((duration - 0.4)/duration,2)
# print ('start and end',start_time,end_time)

# # Play with windowing to decrese the spectral leakage of the big signal next to the small signal. 
# start_pause     = int(start_time * N+1)
# end_pause       = int(end_time * N-1)

# print ('window length:', end_pause - start_pause)
# multiple = floor((end_pause - start_pause)/carrier_frequency)
# print ('sub-multiple:', floor((end_pause - start_pause)/carrier_frequency) )
# print ('start time:', start_pause*timestep)
# print ('end time:', end_pause*timestep)

# window       = np.kaiser( (end_pause - start_pause), 20.0 )

# newN = end_pause - start_pause
# # 
# # a kaiser window means I have shitty frequency resolution at low frequencies. 
# # 
# fft_m = fft(fsignal[start_pause:end_pause])
# fft_m = np.abs(2.0/(newN) * (fft_m))[1:(newN)//2]

# xf = np.fft.fftfreq( (newN), d=timestep)[:(newN)//2]
# frequencies = xf[1:(newN)//2]

# bands   = 40
# vheight = 100
# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(211)
# plt.plot(frequencies,fft_m,'k')
# ax.set_xlim([carrier_frequency-bands,carrier_frequency+bands])
# ax.set_ylim([0,vheight])
# ax.set_xlabel('Frequency (Hz)',fontsize=16)
# ax.set_ylabel('Normalized Volts ($\mu$V)',fontsize=16)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)

# ax2  = fig.add_subplot(212)
# plt.plot(frequencies,fft_m,'k')
# ax2.set_xlim([0,bands])
# ax2.set_ylim([0,vheight])

# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax2.set_xlabel('Frequency (Hz)',fontsize=16)
# ax2.set_ylabel('Normalized Volts ($\mu$V)',fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# # plt.title('Modulated LFP Signal',fontsize=16)
# plt.tight_layout()

# plot_filename = 'neural_modulated_lfp_signal.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()


# l_cut  = 1.4
# df_cut = 5
# # 
# def demodulate(in_signal,carrier_f,dt): 

#     # create a demodulated signal. 
#     lowm  = carrier_f - df_cut
#     highm = carrier_f + df_cut
#     # print ('low/high',lowm,highm)
#     modulation_filter = iirfilter(17, [lowm,highm], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
#     modulated_signal  = sosfiltfilt(modulation_filter, in_signal)
#     # 
#     # at this point add the negative part of the signal to it so it is all above zero. 
#     # modulated_signal = modulated_signal + np.abs(np.min(modulated_signal)) + 10
#     # 
#     idown = modulated_signal*np.cos(2*np.pi*carrier_f*dt)
#     qdown = modulated_signal*np.sin(2*np.pi*carrier_f*dt)   

#     demodulated_signal = idown + 1j*qdown
#     # demodulated_signal = np.real(idown)
#     demodulated_signal = np.abs(demodulated_signal)
#     demodulated_signal = demodulated_signal - np.mean(demodulated_signal)
#     d_demod_data = sosfiltfilt(sos_df_band, demodulated_signal)     
#     return d_demod_data
# # 
# sos_df_band = iirfilter(17, [l_cut,df_cut], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# # 
# sos_lp = iirfilter(17, [4000], rs=60, btype='lowpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')


# start_cut = carrier_frequency - 1
# end_cut = carrier_frequency + 1

# sos_bs = iirfilter(17, [start_cut,end_cut], rs=60, btype='bandstop',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')

# sos_hp = iirfilter(17, [499000,501000], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')

# # 
# # 
# # Final filters. 
# lfp_data                       = sosfiltfilt(sos_df_band, fsignal)   
# # 
# # fsignal                       = sosfiltfilt(sos_bs, fsignal)   
# # bp_data                      = sosfiltfilt(sos_hp, bp_data)   
# # 
# lowm  = carrier_frequency - df_cut
# highm = carrier_frequency + df_cut
# # print ('low/high',lowm,highm)
# modulation_filter = iirfilter(17, [lowm,highm], rs=60, btype='bandpass',
#                    analog=False, ftype='cheby2', fs=Fs,
#                    output='sos')
# modulated_signal  = sosfiltfilt(modulation_filter, fsignal)

# # ensure the entire signal is above zero. 
# modulated_signal = modulated_signal + np.abs(np.min(modulated_signal))*1.5

# analytical_signal = hilbert(modulated_signal)
# h_signal = np.abs(analytical_signal)
# h_signal = h_signal - np.mean(h_signal)
# h_data = sosfiltfilt(sos_df_band, h_signal)  

# # 
# demodulated         = demodulate(fsignal,carrier_frequency,t)

# fft_bp              = fft(modulated_signal[start_pause:end_pause]*window)
# fft_bp              = np.abs(2.0/(newN) * (fft_bp))[1:(newN)//2]
# #

# # 
# # demodulated = demodulate(fsignal[start_pause:end_pause],carrier_frequency,t[start_pause:end_pause])
# # 
# start_pause  = int(start_time*Fs ) 
# end_pause    = int(end_time*Fs )

# newN         = int(end_pause - start_pause)
# window       = np.kaiser( (end_pause - start_pause), 14.0 )
# # 
# fft_dm              = fft(demodulated[start_pause:end_pause]*window)
# fft_dm              = np.abs(2.0/(newN) * (fft_dm))[1:(newN)//2]

# fft_h              = fft(h_data[start_pause:end_pause]*window)
# fft_h              = np.abs(2.0/(newN) * (fft_h))[1:(newN)//2]

# fft_lfp              = fft(lfp_data[start_pause:end_pause]*window)
# fft_lfp              = np.abs(2.0/(newN) * (fft_lfp))[1:(newN)//2]

# xf                  = np.fft.fftfreq( (newN), d=timestep)[:(newN)//2]
# frequencies         = xf[1:(newN)//2]
# #
# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(211)
# plt.plot(t,lfp_data/np.max(lfp_data),'r')
# plt.plot(t,demodulated/np.max(demodulated),'k')
# plt.plot(t,h_data/np.max(h_data),'m')
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.set_xlabel('Time (s)',fontsize=16)
# ax.set_ylabel('Normalized Volts ($\mu$V)',fontsize=16)
# plt.title('Demodulated Signal',fontsize=16)
# plt.tight_layout()
# plt.legend(['LFP','Demodulated','Hilbert'],loc='upper right')

# ax2  = fig.add_subplot(212)
# plt.plot(frequencies,fft_lfp,'r')
# plt.plot(frequencies,fft_dm,'k')
# plt.plot(frequencies,fft_h,'m')

# plt.legend(['LFP','Demodulated','Hilbert'],loc='upper right')

# # plt.plot(frequencies,fft_m,'b')
# ax2.set_xlim([0,bands])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = 'demodulated_signal.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()



