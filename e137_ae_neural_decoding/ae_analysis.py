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
# from scipy.stats import ttest_ind
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
# 
# Continuous wave. 
savepath     = 'D:\\ae_mouse\\e137_ae_neural_decoding\\t1_phantom\\noF21_1MPa\\'
# 
filelist = [1,2,3,4]
# print ('filename: ',filename)
# 
duration            = 8
Fs                  = 5e6
timestep            = 1.0/Fs
N                   = int(Fs*duration)
carrier             = 500000
channel_of_interest = 0
rfchannel           = 4
gain                = 1000
t                   = np.linspace(0, duration, N, endpoint=False)
current_signal_frequency = 8
band_of_interest    = 40
# 
print ('current_signal_frequency: ', current_signal_frequency)

start_pause     = int(start_time * N+1)
end_pause       = int(end_time *N-1)

start_time      = np.round(0.8/duration ,2)
end_time        = np.round((duration - 0.2)/duration,2)
print ('start and end',start_time,end_time)
# xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# frequencies = xf[1:(end_pause-start_pause)//2]
# 
# 
summation = [0]*t
for d in range(len(filelist)):
    filename   = savepath+'t'+str(filelist[d])+'_stream.npy'
    data       = np.load(filename)
    fsignal    = 1e6*data[channel_of_interest]/gain
    # 
    # Check that the signal is not railing. 
    # fig = plt.figure(figsize=(4,4))
    # ax  = fig.add_subplot(111)
    # plt.plot(t,fsignal,'k')
    # plt.show()
    # 
    if d == 0: 
        summation = fsignal
    else: 
        summation = summation + fsignal

average = summation/len(filelist)
# 
lowcut = 3
# Do demodulation. 
sos_demodulate_bandpass  = iirfilter(17, [carrier-band_of_interest,carrier+band_of_interest], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

sos_demodulate_bandstop  = iirfilter(17, [carrier-lowcut,carrier+lowcut], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 

sos_df = iirfilter(17, [lowcut,band_of_interest], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
filtered_data           = sosfiltfilt(sos_demodulate_bandpass, average) 
filtered_data           = sosfiltfilt(sos_demodulate_bandstop,filtered_data)

demodulated_signal      = -demodulate(filtered_data,carrier,t) # Demodulate
analytical_signal       = hilbert(filtered_data) # Hilbert demodulate.  
h_signal                = -np.abs(analytical_signal)
demodulated_signal_h    = sosfiltfilt(sos_df, h_signal) 
demodulated_signal_iq   = sosfiltfilt(sos_df, demodulated_signal) 
lfp_data                = sosfiltfilt(sos_df, average)   # low frequency data. 
# 


# I have found the problem, or... one of them. 
# 
start_pause = int( 4 * Fs )
end_pause   = int( 6 * Fs) 
t_bit   = t[start_pause:end_pause]
lfp_bit = lfp_data[start_pause:end_pause]
d_h  = demodulated_signal_h[start_pause:end_pause]
d_iq = demodulated_signal_iq[start_pause:end_pause]
# 
# 
fft_h = fft(demodulated_signal_h[start_pause:end_pause])
fft_h = np.abs(2.0/(end_pause-start_pause) * (fft_h))[1:(end_pause-start_pause)//2]

fft_iq = fft(demodulated_signal_iq[start_pause:end_pause])
fft_iq = np.abs(2.0/(end_pause-start_pause) * (fft_iq))[1:(end_pause-start_pause)//2]

fft_lfp = fft(lfp_data[start_pause:end_pause])
fft_lfp = np.abs(2.0/(end_pause-start_pause) * (fft_lfp))[1:(end_pause-start_pause)//2]

xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
ff = xf[1:(end_pause-start_pause)//2]
# 
# 
fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(311)
plt.plot(t_bit,lfp_bit,'k')
plt.plot(t_bit,d_h,'g')
plt.plot(t_bit,d_iq,'r')
ax2  = fig.add_subplot(312)
plt.plot(ff,fft_h,'b')
plt.plot(ff,fft_iq,'r')
ax2.set_xlim([0,band_of_interest])
ax3  = fig.add_subplot(313)
plt.plot(ff,fft_lfp,'r')
ax3.set_xlim([0,band_of_interest])
plt.show()
# 
# Take the data in the kaiser window. 
# remove the 500kHz peak, and then remove everything outside the band of interest. 
# then inverse FFT it? 
# 
start_pause = int(start_time * N+1)
end_pause   = int(end_time *N-1)
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 
beta            = 20
window          = np.kaiser( (end_pause-start_pause), beta)
print('window len:',len(window))
fft_m = fft(average[start_pause:end_pause])
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
# 
fft_k = fft(average[start_pause:end_pause]*window)
fft_k = np.abs(2.0/(end_pause-start_pause) * (fft_k))[1:(end_pause-start_pause)//2]
# 
# Take the inverse Fourier Transform. 
carrier_start_idx   = find_nearest(fft_k,carrier-3)
carrier_end_idx     = find_nearest(fft_k,carrier+3)
# fft_k[carrier_start_idx:carrier_end_idx]
inverse_fft = ifft(fft_m)

# fig = plt.figure(figsize=(4,4))
# ax  = fig.add_subplot(111)
# plt.plot(t,inverse_fft)
# plt.show()

# 
# SNR calculation for both kaiser window 
frequencies_of_interest = [carrier - current_signal_frequency,carrier+current_signal_frequency]
# If it looks bad for some reason? For now, just note down the file number.     
signal_totals                   = []
kaiser_signal_totals            = []
interest_frequencies            = []
for n in range(len(frequencies_of_interest)): # sum all frequencies per unit time.
  df_idx = find_nearest(frequencies,frequencies_of_interest[n])
  # also eliminate on either side of bins of interest as I don't have great bin resolution.    
  interest_frequencies.append(df_idx)    
  signal_totals.append(fft_m[df_idx])
  kaiser_signal_totals.append(fft_k[df_idx])
# This is the start and end point to calculate the SNR from. 
start_idx = find_nearest(frequencies,carrier + int(current_signal_frequency/2) )   # after the first 5Hz. 
end_idx = find_nearest(frequencies,carrier + 40)
# print ('end idx',end_idx)
dis_signal_totals             = []  # 
dis_kaiser_signal_totals      = []
for n in range(start_idx, end_idx): # sum all frequencies per unit time.
    if n not in interest_frequencies: 
        dis_signal_totals.append(fft_m[n])
        dis_kaiser_signal_totals.append(fft_k[n])
#           
signal_snr = 20*np.log(np.mean(signal_totals)/np.mean(dis_signal_totals))
kaiser_snr = 20*np.log(np.mean(kaiser_signal_totals)/np.mean(dis_kaiser_signal_totals))
print ('signal snr:',signal_snr,kaiser_snr)
#  
#  
df_idx      = find_nearest(frequencies,carrier+current_signal_frequency)
applied_idx = find_nearest(frequencies,current_signal_frequency)
print ('ae ratio rectangular:',fft_m[df_idx],fft_m[applied_idx])
print ('ae ratio kaiser:',fft_k[applied_idx]/fft_k[df_idx] )
# 
# Build the plot
fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(111)
# plt.axvline(x=carrier+current_signal_frequency,color='grey')
# plt.axvline(x=carrier-current_signal_frequency,color='grey')
plt.plot(frequencies,fft_m,'r')
plt.plot(frequencies,fft_k,'k')
ax.set_xlim([carrier-40,carrier+40])
# ax.set_ylim([0,0.1])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xticks([carrier-40,carrier,carrier+40])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.legend(['stuff'],loc='upper right',framealpha=0.0,fontsize=16)
# Save the figure and show
# ax.set_xlim([0,30])
# ax.set_ylim([0,1600])
ax.ticklabel_format(useOffset=False)
plt.tight_layout()
plt.savefig(saveprefix+'modulated_signal.png')
plt.show()


fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(111)
# plt.axvline(x=carrier+current_signal_frequency,color='grey')
# plt.axvline(x=carrier-current_signal_frequency,color='grey')
plt.plot(frequencies,fft_m,'r')
plt.plot(frequencies,fft_k,'k')
ax.set_xlim([carrier-40,carrier+40])
ax.set_ylim([0,10])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xticks([carrier-40,carrier,carrier+40])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.legend(['stuff'],loc='upper right',framealpha=0.0,fontsize=16)
# Save the figure and show
# ax.set_xlim([0,30])
# ax.set_ylim([0,1600])
ax.ticklabel_format(useOffset=False)
plt.tight_layout()
plt.savefig(saveprefix+'modulated_semizoom_signal.png')
plt.show()

fig = plt.figure(figsize=(8,4))
ax  = fig.add_subplot(111)
# plt.axvline(x=carrier+current_signal_frequency,color='grey')
# plt.axvline(x=carrier-current_signal_frequency,color='grey')
# plt.plot(frequencies,fft_m,'k')
plt.plot(frequencies,fft_k,'k')
ax.set_xlim([carrier-40,carrier+40])
ax.set_ylim([0,0.1])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xticks([carrier-40,carrier,carrier+40])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.legend(['stuff'],loc='upper right',framealpha=0.0,fontsize=16)
# Save the figure and show
# ax.set_xlim([0,30])
# ax.set_ylim([0,1600])
ax.ticklabel_format(useOffset=False)
plt.tight_layout()
plt.savefig(saveprefix+'modulated_zoom_signal.png')
plt.show()
# 
# 
fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_m,'r')
plt.plot(frequencies,fft_k,'k')
ax.set_xlim([0,40])
ax.set_ylim([0,80])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.legend(['stuff'],loc='upper right',framealpha=0.0,fontsize=16)
# Save the figure and show
# ax.set_xlim([0,30])
# ax.set_ylim([0,1600])
ax.ticklabel_format(useOffset=False)
plt.tight_layout()
plt.savefig(saveprefix+'applied_signal.png')
plt.show()
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