'''

Title: The recommendation for maximum drive level is 100Vpp. 
I am currently at only 40Vpp. This maximum drive level is for a CW operation immersion transducer. 
I am partially immersed with a cone, so I probably have a little heat dissipation. 
I DO have a lot of space to move here with increasing the pressure output. 

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
saveprefix      = './/images//' 
savepath       = 'D:\\ae_mouse\\e136_ae_neural_decoding\\t5-mouse\\twotone_test\\'

filelist = [1,2,4,5]

duration        = 8
Fs              = 5e6
filename        = 't1_stream.npy'
rf_channel      = 4 
m_channel       = 0
timestep        = 1.0/Fs
N               = int(Fs*duration)
carrier         = 500000
gain            = 500 
# 
print ('filename: ',filename)
start_time = np.round(0.8/duration ,2) 
end_time   = np.round((duration - 0.4)/duration,2)
print ('start and end',start_time,end_time)
# Play with windowing to decrese the spectral leakage of the big signal next to the small signal. 
start_pause     = int(start_time * N+1)
end_pause       = int(end_time * N-1)
newN = end_pause - start_pause
# 
t           = np.linspace(0, duration, N, endpoint=False)

summation = [0]*t
for d in range(len(filelist)):
    filename = 't'+str(filelist[d]) + '_stream.npy'
    data = np.load(savepath+filename)
    a,b = data.shape
    print ('shape',a,b)
    fsignal    = 1e6*data[m_channel]/gain
    # 
    # fig = plt.figure(figsize=(10,6))
    # ax  = fig.add_subplot(211)
    # plt.plot(t,fsignal)
    # ax2  = fig.add_subplot(212)
    # plt.plot(t,data[m_channel])
    # plt.show()
    # 
    if (d == 0):
        summation = fsignal 
    else:
        summation = summation + fsignal
print ('d',d+1)
average = summation/(d+1)
# 
# 
fft_m = fft(average[start_pause:end_pause])
fft_m = np.abs(2.0/(newN) * (fft_m))[1:(newN)//2]
# 
window       = np.kaiser( (end_pause - start_pause), 20.0 )
fft_k = fft(average[start_pause:end_pause]*window)
fft_k = np.abs(2.0/(newN) * (fft_k))[1:(newN)//2]

xf = np.fft.fftfreq( (newN), d=timestep)[:(newN)//2]
frequencies = xf[1:(newN)//2]

#
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(211)
plt.plot(frequencies,fft_m,'k')
plt.plot(frequencies,fft_k,'r')
ax.set_xlim([0,20])
ax.set_ylim([0,200])
ax2  = fig.add_subplot(212)
plt.plot(frequencies,fft_m,'k')
plt.plot(frequencies,fft_k,'r')
ax2.set_xlim([carrier-20,carrier+20])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = 'twotone.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

#

fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_m,'k')
# plt.plot(frequencies,fft_k,'r')
ax.set_xlim([0,20])
ax.set_ylim([0,125])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = saveprefix+'twotone_applied_low_frequencies.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_m,'k')
ax.set_xlim([carrier-20,carrier+20])
# 
ax.ticklabel_format(useOffset=False)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xticks([carrier-20,carrier,carrier+20])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = saveprefix+'twotone_applied_high_frequencies.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_m,'k')
plt.plot(frequencies,fft_k,'r')
ax.set_xlim([carrier-20,carrier+20])
plt.xticks([carrier-20,carrier,carrier+20])
ax.set_ylim([0,0.5])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.ticklabel_format(useOffset=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = saveprefix+'twotone_applied_high_frequencies_zoom.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

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



