'''

FFT shift. 
Take the kaiser window. 
Then shift 500kHz down to 0Hz remove +-100Hz and take IFFT. 

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
# 
# Implement a matched filtering scheme. 
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 
saveprefix   = './/images//'
# 
savepath     = 'D:\\ae_mouse\\e137_ae_neural_decoding\\t2_mouse\\'
filename     = 't2-mouse_data.npz'
print ('filename: ', filename)
data                = np.load(savepath+filename)
averaged_raw_data   = data['averaged_raw_data']
averaged_marker     = data['averaged_marker']
giant_data_array    = data['giant_data_array']
number              = data['n']
# 
duration            = 8
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
print ('current_signal_frequency: ', current_signal_frequency)
# 
a,b = giant_data_array.shape
print ('a,b:',a,b)
summation   = np.sum(giant_data_array,axis=0)
average     = summation/a
print ('average len:',len(average))
# 
start_time      = np.round(0.8/duration ,2)
end_time        = np.round((duration - 0.2)/duration,2)
print ('start and end',start_time,end_time)
start_pause     = int(start_time * N+1)
end_pause       = int(end_time *N-1)
newN            = end_pause - start_pause
print ('new N: ', newN)
# 
xf          = np.fft.fftfreq( (newN), d=timestep)[:(newN)//2]
frequencies = xf[1:(newN)//2]
# 
# 
beta            = 20
window          = np.kaiser( (newN), beta)
print('window len:',len(window))
# 
fft_m2 = fft(average[start_pause:end_pause])
fft_m  = np.abs(2.0/(newN) * (fft_m2))[1:(newN)//2]
# 
fft_k2      = fft(average[start_pause:end_pause]*window)
fft_k       = np.abs(2.0/(end_pause-start_pause) * (fft_k2))[1:(end_pause-start_pause)//2]
xf          = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 
# 
carrier_idx     = find_nearest(frequencies,carrier)
end_band_idx    = find_nearest(frequencies,carrier+band_of_interest)
end2_band_idx   = find_nearest(frequencies,carrier-band_of_interest)
band_idx        = find_nearest(frequencies,band_of_interest)
# 
pt2             = fft_k[(end2_band_idx+1):(carrier_idx)][::-1]
pt1             = fft_k[(carrier_idx+1):(end_band_idx+1) ]
print ('stuff: ',len(pt2), len(fft_k[(carrier_idx):(end_band_idx) ]) )
result          = fft_k[carrier_idx:end_band_idx] + pt2
# 
ff = frequencies[0:band_idx]
print ('frequencies: ',len(frequencies[0:band_idx]))

offset_idx      = find_nearest(frequencies,5)
max_amp         = np.max(result[offset_idx:band_idx])
print ('max amp',max_amp)
maxamp_idx      = find_nearest(result,max_amp)
print ('maxamp_idx',maxamp_idx,ff[maxamp_idx])
# 
fig = plt.figure(figsize=(8,5))
ax  = fig.add_subplot(211)
plt.plot(frequencies,fft_k,'k')
ax.set_xlim([carrier-band_of_interest,carrier+band_of_interest])
# ax.set_xlim([0,band_of_interest])
ax2 = fig.add_subplot(212)
plt.plot(ff,result,'k')
plt.plot(ff,pt2,'r')
plt.plot(ff,pt1,'g')
plt.plot(ff[maxamp_idx],result[maxamp_idx],'xr')
ax2.set_xlim([0,20])
ax2.set_ylim([0,1.0])
plt.legend(['combined','pt2','pt1'],loc='upper right')
plt.show()
# 
# 
lowcut = 3
# Do demodulation. 
sos_demodulate_bandpass  = iirfilter(17, [carrier-band_of_interest,carrier+band_of_interest], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
#  
sos_df = iirfilter(17, [1,band_of_interest], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
filtered_data           = sosfiltfilt(sos_demodulate_bandpass, average) 
demodulated_signal      = -demodulate(filtered_data,carrier,t) # Demodulate
analytical_signal       = -hilbert(filtered_data) # Hilbert demodulate.  
h_signal                = np.abs(analytical_signal)
h_signal                = h_signal -np.mean(h_signal)
demodulated_signal_h    = sosfiltfilt(sos_df, h_signal) 
demodulated_signal_h    = demodulated_signal_h - np.mean(demodulated_signal_h)
demodulated_signal_iq   = sosfiltfilt(sos_df, demodulated_signal) 
demodulated_signal_iq   = demodulated_signal_iq - np.mean(demodulated_signal_iq)
lfp_data                = sosfiltfilt(sos_df, average)   # low frequency data. 
# 
x = demodulated_signal_h
y = lfp_data 
# 
# First, center about the mean.     
# Create a template from the kaiser windowed data. 
# 
# Try a template that is an 8 Hz sinusoid.     
template = [0]*int(1.25*Fs) # 1 second 
for i in range(len(template)):
    template[i] = np.cos( 2*np.pi*(current_signal_frequency)*i*timestep)  
#   
print ('sig len',len(template))
corr    = signal.correlate(y,template, mode='same')
corrd   = signal.correlate(x,template, mode='same')
# 
# 
start_index = int(2*Fs)
end_index   = int(6*Fs)
newN        = len(lfp_data[start_index:end_index])
# 
fft_mf      = fft(corrd[start_index:end_index])
fft_mf      = np.abs(2.0/(end_index-start_index) * (fft_mf))[1:(end_index-start_index)//2]
# 
fft_mflfp   = fft(corr[start_index:end_index])
fft_mflfp   = np.abs(2.0/(end_index-start_index) * (fft_mflfp))[1:(end_index-start_index)//2]
# 
xf          = np.fft.fftfreq( (newN), d=1/Fs)[:(newN)//2]
frequencies = xf[1:(newN)//2]
# 
# I can see it in the fft. But I cannot demodulate it. 
t_bit            = t[start_index:end_index]
y_bit            = y[start_index:end_index]
corr_bit         = corr[start_index:end_index]
corrd_bit        = corrd[start_index:end_index]
norm_lfp         = (y[start_index:end_index]/np.max(y[start_index:end_index]))
norm_demod       = (x[start_index:end_index]/np.max(x[start_index:end_index]))
norm_corr_lfp    = (corr[start_index:end_index]/np.max(corr[start_index:end_index]))
norm_corr_demod  = (corrd[start_index:end_index]/np.max(corrd[start_index:end_index]))
# 
# 
fig = plt.figure(figsize=(8,5))
ax  = fig.add_subplot(211)
plt.plot(t_bit,norm_lfp,'b')
plt.plot(t_bit,-norm_corr_demod,'r')
plt.legend(['lfp','corr demod'],loc='upper right')
# plt.legend(['lfp','demod','corr lfp','corr demod'],loc='upper right')
ax2 = fig.add_subplot(212)    
plt.plot(frequencies,fft_mf/np.max(fft_mf),'b')
plt.plot(frequencies,fft_mflfp/np.max(fft_mflfp),'purple')
ax2.set_xlim([0,band_of_interest])
plt.legend(['demod_correlation','lfp correlation'],loc='upper right')
plt.show()
# 
# 
# 
fig = plt.figure(figsize=(8,5))
ax  = fig.add_subplot(111)
plt.plot(t_bit,norm_lfp,color='k',linewidth=2.0)
plt.plot(t_bit,norm_corr_demod,color='r',linewidth=2.0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plot_filename = saveprefix+'final_waveform.png'
plt.savefig(plot_filename, transparent=True)
plt.show()
# 
# 
# # start_pause = 0 int(start_time * N+1)
# # end_pause   = int(end_time *N-1)
# start_pause = int(0)
# end_pause   = int(8*Fs)
# xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# frequencies = xf[1:(end_pause-start_pause)//2]
# # 
# beta            = 20
# window          = np.kaiser( (end_pause-start_pause), beta)
# print('window len:',len(window))
# fft_m2 = fft(average[start_pause:end_pause])
# fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m2))[1:(end_pause-start_pause)//2]
# # 
# fft_k2 = fft(average[start_pause:end_pause]*window)
# fft_k = np.abs(2.0/(end_pause-start_pause) * (fft_k2))[1:(end_pause-start_pause)//2]
# # 
# xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)]
# full_frequencies = xf[0:(end_pause-start_pause)]
# # 
# print ('lengths of bits: ',len(full_frequencies),len(fft_k2))
# # 
# # fig = plt.figure(figsize=(4,4))
# # ax  = fig.add_subplot(111)
# # plt.plot(full_frequencies,np.real(fft_k2),'k')
# # plt.show()

# # Take the inverse Fourier Transform. 
# carrier_start_idx   = find_nearest(full_frequencies,carrier-3)
# carrier_end_idx     = find_nearest(full_frequencies,carrier+3)
# carrier_lstart_idx  = find_nearest(full_frequencies,-carrier+3)
# carrier_lend_idx    = find_nearest(full_frequencies,-carrier-3)

# fft_k2[carrier_start_idx:carrier_end_idx]   = 0 
# fft_k2[carrier_lstart_idx:carrier_lend_idx] = 0 
# # fft_k[carrier_start_idx:carrier_end_idx]
# inverse_fft = np.real(ifft(fft_k2))

# print ('len stuff: ',len(inverse_fft))
# # 
# fig = plt.figure(figsize=(4,4))
# ax  = fig.add_subplot(111)
# plt.plot(t[start_pause:end_pause],average[start_pause:end_pause])
# plt.plot(t[start_pause:end_pause],inverse_fft)
# plt.show()
# # 
# average = inverse_fft
# #  
# lowcut = 3
# # Do demodulation. 
# sos_demodulate_bandpass  = iirfilter(17, [carrier-band_of_interest,carrier+band_of_interest], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# #  
# # sos_demodulate_bandstop  = iirfilter(17, [carrier-lowcut,carrier+lowcut], rs=60, btype='bandstop',
# #                        analog=False, ftype='cheby2', fs=Fs,
# #                        output='sos')
# #  
# sos_df = iirfilter(17, [lowcut,band_of_interest], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# # 
# filtered_data           = sosfiltfilt(sos_demodulate_bandpass, average) 
# # filtered_data           = sosfiltfilt(sos_demodulate_bandstop,filtered_data)
# #  
# demodulated_signal      = -demodulate(filtered_data,carrier,t) # Demodulate
# analytical_signal       = hilbert(filtered_data) # Hilbert demodulate.  
# h_signal                = -np.abs(analytical_signal)
# demodulated_signal_h    = sosfiltfilt(sos_df, h_signal) 
# demodulated_signal_iq   = sosfiltfilt(sos_df, demodulated_signal) 
# lfp_data                = sosfiltfilt(sos_df, average)   # low frequency data. 
# # 
# # 
# t_bit   = t[start_pause:end_pause]
# lfp_bit = lfp_data[start_pause:end_pause]
# d_h  = demodulated_signal_h[start_pause:end_pause]
# d_iq = demodulated_signal_iq[start_pause:end_pause]
# # 
# # # 
# fft_h = fft(demodulated_signal_h[start_pause:end_pause])
# fft_h = np.abs(2.0/(end_pause-start_pause) * (fft_h))[1:(end_pause-start_pause)//2]
# # 
# #  
# fft_iq = fft(demodulated_signal_iq[start_pause:end_pause])
# fft_iq = np.abs(2.0/(end_pause-start_pause) * (fft_iq))[1:(end_pause-start_pause)//2]
# # 
# #  
# fft_lfp = fft(lfp_data[start_pause:end_pause])
# fft_lfp = np.abs(2.0/(end_pause-start_pause) * (fft_lfp))[1:(end_pause-start_pause)//2]
# # 
# #  
# xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# ff = xf[1:(end_pause-start_pause)//2]
# # 
# #  
# fig = plt.figure(figsize=(4,4))
# ax  = fig.add_subplot(311)
# plt.plot(t_bit,lfp_bit/np.max(lfp_bit),'k')
# plt.plot(t_bit,d_h/np.max(d_h),'g')
# plt.plot(t_bit,d_iq/np.max(d_iq),'r')
# ax2  = fig.add_subplot(312)
# plt.plot(ff,fft_h,'b')
# plt.plot(ff,fft_iq,'r')
# ax2.set_xlim([0,band_of_interest])
# ax3  = fig.add_subplot(313)
# plt.plot(ff,fft_lfp,'r')
# ax3.set_xlim([0,band_of_interest])
# plt.show()
# # 

# 
# SNR calculation for both kaiser window 
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
# # plt.plot(frequencies,fft_m,'k')
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
# #  
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