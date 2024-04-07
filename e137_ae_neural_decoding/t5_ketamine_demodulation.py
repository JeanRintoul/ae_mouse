'''

Title: compare the data going into generator with the data coming out of generator. 
SNR analysis of t5. 
Author: Jean Rintoul
Date:   05.02.2024

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift,ifft
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
from scipy.signal import hilbert
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx

def demodulate(in_signal,carrier_f,t): 
    return np.abs(in_signal*np.exp(2*np.pi*1j*carrier_f*t))

plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 
saveprefix          = './/images//'
# Phantom
# phantom_savepath  = 'D:\\ae_mouse\\e137_ae_neural_decoding\\t4_phantom\\8Hz_15microvolts\\'
mouse_savepath    = 'D:\\ae_mouse\\e137_ae_neural_decoding\\t5_mouse_500kHz_noise\\NO_GND\\'
mouse_savepath    = 'D:\\ae_mouse\\e137_ae_neural_decoding\\t5_mouse_500kHz_noise\\GND\\'

mouse_savepath    = 'D:\\ae_mouse\\e137_ae_neural_decoding\\t5_mouse_500kHz_noise\\ae_4hz_vep\\'
# 
# 
factor      = 10
d           = 9
filelist    = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# filelist = [1]
# print ('filename: ',filename)
duration            = 12
Fs                  = 5e6
timestep            = 1.0/Fs
N                   = int(Fs*duration)
carrier             = 500000
channel_of_interest = 0
rfchannel           = 4
gain                = 500
t                   = np.linspace(0, duration, N, endpoint=False)
current_signal_frequency = 8
band_of_interest    = 40
# 
# print ('current_signal_frequency: ', current_signal_frequency)
# start_time      = np.round(0.8/duration ,2)
# end_time        = np.round((duration - 0.2)/duration,2)
# print ('start and end',start_time,end_time)
start_pause     = int(2*Fs)
end_pause       = int(10*Fs)
# 
# start_pause     = int(2*Fs)
# end_pause       = int(7*Fs)
xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies     = xf[1:(end_pause-start_pause)//2]
# 
mouse_summation  = [0]*t
mouse_filename   = mouse_savepath+'t'+str(filelist[d])+'_stream.npy'
data             = np.load(mouse_filename)
mouse_signal     = 1e6*data[channel_of_interest]/gain
# 
l_cut       = 0.5
h_cut       = 40
sos_low_band = iirfilter(17, [h_cut], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
sos_demodulate_band = iirfilter(17, [carrier-h_cut,carrier+h_cut], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
# First filter the data around the carrier. 
filtered_for_demodulation = sosfiltfilt(sos_demodulate_band, mouse_signal)
analytical_signal       = hilbert(filtered_for_demodulation) # Hilbert demodulate.  
h_signal                = -np.abs(analytical_signal)
demodulated_signal      = h_signal -np.mean(h_signal)
# Now filter the demodulated signal between l_cut and h_cut.  
final_demodulated       = factor*sosfiltfilt(sos_low_band, demodulated_signal)
# same filter on low frequency neural data. 
lfp_data                = sosfiltfilt(sos_low_band,mouse_signal)
# 
# 
# Check that the signal is not railing. 
# fig = plt.figure(figsize=(4,4))
# ax  = fig.add_subplot(111)
# plt.plot(t,fsignal,'k')
# plt.show()
# 
# 
beta      = 20
window    = np.kaiser( (end_pause-start_pause), beta)
fft_mouse = fft(mouse_signal[start_pause:end_pause]*window)
fft_mouse = np.abs(2.0/(end_pause-start_pause) * (fft_mouse))[1:(end_pause-start_pause)//2]

rfft_mouse = fft(mouse_signal[start_pause:end_pause]*window)
rfft_mouse = np.abs(2.0/(end_pause-start_pause) * (rfft_mouse))[1:(end_pause-start_pause)//2]

rfft_demod_mouse = fft(final_demodulated[start_pause:end_pause]*window)
rfft_demod_mouse = np.abs(2.0/(end_pause-start_pause) * (rfft_demod_mouse))[1:(end_pause-start_pause)//2]

bottom = np.min(final_demodulated)
# print('bottom',bottom)
# 
fig = plt.figure(figsize=(6,4))
ax  = fig.add_subplot(311)
plt.plot(frequencies,fft_mouse,'r')
plt.plot(frequencies,rfft_demod_mouse,'k')
plt.legend(['mouse','demod'],loc='upper right')
ax.set_xlim([0,20])
ax.set_ylim([0,30])
ax2  = fig.add_subplot(312)
plt.plot(t[start_pause:end_pause],lfp_data[start_pause:end_pause],'r' )
plt.plot(t[start_pause:end_pause],final_demodulated[start_pause:end_pause]-np.mean(final_demodulated[start_pause:end_pause]),'k' )
# ax2.set_ylim([-150,200])
# plt.yticks(fontsize=16)
# plt.xticks([carrier-10,carrier,carrier+10],fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax3  = fig.add_subplot(313)
plt.plot(frequencies,fft_mouse,'r')
ax3.set_xlim([carrier-30,carrier+30])
plt.axvline(x=carrier + 16,color='k')
plt.axvline(x=carrier - 16,color='k')
plt.axvline(x=carrier + 8,color='k')
plt.axvline(x=carrier - 8,color='k')
plt.axvline(x=carrier + 4,color='k')
plt.axvline(x=carrier - 4,color='k')
# plt.xticks([carrier-10,carrier,carrier+10],fontsize=16)
# # plt.legend(['stuff'],loc='upper right',framealpha=0.0,fontsize=16)
# # Save the figure and show
# # ax.set_xlim([0,30])
ax3.set_ylim([0,1])
# ax.ticklabel_format(useOffset=False)
plt.tight_layout()
plt.savefig(saveprefix+'ketamine_demodulation.png')
plt.show()
# 
# fig = plt.figure(figsize=(4,4))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_mouse,'r')
# ax.set_xlim([carrier-50,carrier+50])
# ax.set_ylim([0,1])
# plt.yticks(fontsize=16)
# plt.xticks([carrier-10,carrier,carrier+10],fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # plt.legend(['stuff'],loc='upper right',framealpha=0.0,fontsize=16)
# # Save the figure and show
# # ax.set_xlim([0,30])
# # ax.set_ylim([0,1600])
# ax.ticklabel_format(useOffset=False)
# plt.tight_layout()
# plt.savefig(saveprefix+'mouse_fft.png')
# plt.show()
# # 
# 
# fig = plt.figure(figsize=(4,4))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,rfft_mouse,'r')
# plt.plot(frequencies,rfft_phantom,'k')
# ax.set_xlim([0,20])
# # ax.set_ylim([0,1])
# plt.yticks(fontsize=16)
# plt.xticks([0,8,10,20],fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # plt.legend(['stuff'],loc='upper right',framealpha=0.0,fontsize=16)
# # Save the figure and show
# # ax.set_xlim([0,30])
# # ax.set_ylim([0,1600])
# ax.ticklabel_format(useOffset=False)
# plt.tight_layout()
# plt.savefig(saveprefix+'thermal_noise_analysis_original_signal.png')
# plt.show()
# #  
# 
# fig = plt.figure(figsize=(4,4))
# ax  = fig.add_subplot(311)
# plt.plot(t_bit,lfp_bit,'k')
# plt.plot(t_bit,d_h,'g')
# plt.plot(t_bit,d_iq,'r')
# ax2  = fig.add_subplot(312)
# plt.plot(ff,fft_h,'b')
# plt.plot(ff,fft_iq,'r')
# ax2.set_xlim([0,band_of_interest])
# ax3  = fig.add_subplot(313)
# plt.plot(ff,fft_lfp,'r')
# ax3.set_xlim([0,band_of_interest])
# plt.show()
# # 
# # Take the data in the kaiser window. 
# # remove the 500kHz peak, and then remove everything outside the band of interest. 
# # then inverse FFT it? 
# # 
# start_pause = int(start_time * N+1)
# end_pause   = int(end_time *N-1)
# xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# frequencies = xf[1:(end_pause-start_pause)//2]
# # 
# beta            = 20
# window          = np.kaiser( (end_pause-start_pause), beta)
# print('window len:',len(window))
# fft_m = fft(average[start_pause:end_pause])
# fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
# # 
# fft_k = fft(average[start_pause:end_pause]*window)
# fft_k = np.abs(2.0/(end_pause-start_pause) * (fft_k))[1:(end_pause-start_pause)//2]
# # 
# # Take the inverse Fourier Transform. 
# carrier_start_idx   = find_nearest(fft_k,carrier-3)
# carrier_end_idx     = find_nearest(fft_k,carrier+3)
# # fft_k[carrier_start_idx:carrier_end_idx]
# inverse_fft = ifft(fft_m)

# # fig = plt.figure(figsize=(4,4))
# # ax  = fig.add_subplot(111)
# # plt.plot(t,inverse_fft)
# # plt.show()

# # 
# # SNR calculation for both kaiser window 
# frequencies_of_interest = [carrier - current_signal_frequency,carrier+current_signal_frequency]
# # If it looks bad for some reason? For now, just note down the file number.     
# signal_totals                   = []
# kaiser_signal_totals            = []
# interest_frequencies            = []
# for n in range(len(frequencies_of_interest)): # sum all frequencies per unit time.
#   df_idx = find_nearest(frequencies,frequencies_of_interest[n])
#   # also eliminate on either side of bins of interest as I don't have great bin resolution.    
#   interest_frequencies.append(df_idx)    
#   signal_totals.append(fft_m[df_idx])
#   kaiser_signal_totals.append(fft_k[df_idx])
# # This is the start and end point to calculate the SNR from. 
# start_idx = find_nearest(frequencies,carrier + int(current_signal_frequency/2) )   # after the first 5Hz. 
# end_idx = find_nearest(frequencies,carrier + 40)
# # print ('end idx',end_idx)
# dis_signal_totals             = []  # 
# dis_kaiser_signal_totals      = []
# for n in range(start_idx, end_idx): # sum all frequencies per unit time.
#     if n not in interest_frequencies: 
#         dis_signal_totals.append(fft_m[n])
#         dis_kaiser_signal_totals.append(fft_k[n])
# #           
# signal_snr = 20*np.log(np.mean(signal_totals)/np.mean(dis_signal_totals))
# kaiser_snr = 20*np.log(np.mean(kaiser_signal_totals)/np.mean(dis_kaiser_signal_totals))
# print ('signal snr:',signal_snr,kaiser_snr)
# #  
# #  
# df_idx      = find_nearest(frequencies,carrier+current_signal_frequency)
# applied_idx = find_nearest(frequencies,current_signal_frequency)
# print ('ae ratio rectangular:',fft_m[df_idx],fft_m[applied_idx])
# print ('ae ratio kaiser:',fft_k[applied_idx]/fft_k[df_idx] )
# # 
# # Build the plot
# fig = plt.figure(figsize=(4,4))
# ax  = fig.add_subplot(111)
# # plt.axvline(x=carrier+current_signal_frequency,color='grey')
# # plt.axvline(x=carrier-current_signal_frequency,color='grey')
# plt.plot(frequencies,fft_m,'r')
# plt.plot(frequencies,fft_k,'k')
# ax.set_xlim([carrier-40,carrier+40])
# # ax.set_ylim([0,0.1])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.xticks([carrier-40,carrier,carrier+40])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # plt.legend(['stuff'],loc='upper right',framealpha=0.0,fontsize=16)
# # Save the figure and show
# # ax.set_xlim([0,30])
# # ax.set_ylim([0,1600])
# ax.ticklabel_format(useOffset=False)
# plt.tight_layout()
# plt.savefig(saveprefix+'modulated_signal.png')
# plt.show()


# fig = plt.figure(figsize=(4,4))
# ax  = fig.add_subplot(111)
# # plt.axvline(x=carrier+current_signal_frequency,color='grey')
# # plt.axvline(x=carrier-current_signal_frequency,color='grey')
# plt.plot(frequencies,fft_m,'r')
# plt.plot(frequencies,fft_k,'k')
# ax.set_xlim([carrier-40,carrier+40])
# ax.set_ylim([0,10])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.xticks([carrier-40,carrier,carrier+40])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # plt.legend(['stuff'],loc='upper right',framealpha=0.0,fontsize=16)
# # Save the figure and show
# # ax.set_xlim([0,30])
# # ax.set_ylim([0,1600])
# ax.ticklabel_format(useOffset=False)
# plt.tight_layout()
# plt.savefig(saveprefix+'modulated_semizoom_signal.png')
# plt.show()

# fig = plt.figure(figsize=(8,4))
# ax  = fig.add_subplot(111)
# # plt.axvline(x=carrier+current_signal_frequency,color='grey')
# # plt.axvline(x=carrier-current_signal_frequency,color='grey')
# # plt.plot(frequencies,fft_m,'k')
# plt.plot(frequencies,fft_k,'k')
# ax.set_xlim([carrier-40,carrier+40])
# ax.set_ylim([0,0.1])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.xticks([carrier-40,carrier,carrier+40])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # plt.legend(['stuff'],loc='upper right',framealpha=0.0,fontsize=16)
# # Save the figure and show
# # ax.set_xlim([0,30])
# # ax.set_ylim([0,1600])
# ax.ticklabel_format(useOffset=False)
# plt.tight_layout()
# plt.savefig(saveprefix+'modulated_zoom_signal.png')
# plt.show()
# # 
# # 
# fig = plt.figure(figsize=(4,4))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_m,'r')
# plt.plot(frequencies,fft_k,'k')
# ax.set_xlim([0,40])
# ax.set_ylim([0,80])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # plt.legend(['stuff'],loc='upper right',framealpha=0.0,fontsize=16)
# # Save the figure and show
# # ax.set_xlim([0,30])
# # ax.set_ylim([0,1600])
# ax.ticklabel_format(useOffset=False)
# plt.tight_layout()
# plt.savefig(saveprefix+'applied_signal.png')
# plt.show()
#  
# Build the plot
# fig = plt.figure(figsize=(4,4))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_c,'k')
# plt.plot(frequencies,fft_k,'r')
# ax.set_xlim([carrier - 30,carrier+30])
# ax.set_ylim([0,np.max(fft_c)+10])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # plt.legend(['rectangular','kaiser'],loc='upper right',framealpha=0.0,fontsize=16)
# # Save the figure and show
# plt.xticks([carrier-24,carrier,carrier+24])
# ax.ticklabel_format(useOffset=False)
# plt.tight_layout()
# plt.savefig(saveprefix+'kaiser_comparison.png')
# plt.show()


# fig = plt.figure(figsize=(4,4))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_c,'k')
# plt.plot(frequencies,fft_k,'r')
# ax.set_xlim([carrier - 30,carrier+30])
# ax.set_ylim([0,10])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # plt.legend(['rectangular','kaiser'],loc='upper right',framealpha=0.0,fontsize=16)
# # Save the figure and show
# plt.xticks([carrier-24,carrier,carrier+24])
# ax.ticklabel_format(useOffset=False)
# plt.tight_layout()
# plt.savefig(saveprefix+'kaiser_comparison_zoom.png')
# plt.show()