'''

Title: read in the file. 

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
from scipy.fftpack import rfft, irfft, fftfreq
from scipy.stats import ttest_ind
from scipy.signal import find_peaks
import scipy.stats
import pandas as pd
from scipy.signal import spectrogram
from scipy.signal import hilbert
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   
# 
def demodulate(in_signal,carrier_f,t): 
    return np.abs(in_signal*np.exp(2*np.pi*1j*carrier_f*t))

# confidence interval. 
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m,h,se

#
saveprefix     = './/images//' 
savepath       = 'D:\\ae_mouse\\e136_ae_neural_decoding\\'
filename       = 't2-phantom_8Hz_100microv_data.npz'
print ('filename: ',filename)
# 
data                = np.load(savepath+filename)
averaged_raw_data   = data['averaged_raw_data']
averaged_marker     = data['averaged_marker']
giant_data_array    = data['giant_data_array']
n                   = data['n']

duration            = 12
Fs                  = 5e6
timestep            = 1.0/Fs
N                   = int(Fs*duration)
carrier             = 500000
t                   = np.linspace(0, duration, N, endpoint=False)
# Fs is at 20kHz. 
frequencies_of_interest = [8] 
lowcut                  = 7
bandwidth_of_interest   = 30 
# 
a,b = giant_data_array.shape
print ('giant data array shape',giant_data_array.shape,giant_data_array.dtype)
dd = giant_data_array.shape
d1 = giant_data_array[0,:]
d2 = giant_data_array[13,:]
# 
print ('d1: ',len(d1) )
# 
# fig = plt.figure(figsize=(8,3))
# ax = fig.add_subplot(211)
# plt.plot(t,d1)
# ax2 = fig.add_subplot(212)
# plt.plot(t,d2)
# plt.show()
# 
# Create filter bands, and add them together. 
sos_demodulate_bandpass  = iirfilter(17, [carrier-bandwidth_of_interest,carrier+bandwidth_of_interest], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
#
sos_df = iirfilter(17, [0.1,bandwidth_of_interest], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
# sos_df = iirfilter(17, [bandwidth_of_interest], rs=60, btype='lowpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# 
# start and end time of the ultrasound, so as to avoid the large
# transient bump in the demodulated data from the large electric signal at the carrier frequency. 
start_pause         = 0.0
end_pause           = 12.0
start_idx = int(start_pause*Fs)
end_idx   = int(end_pause*Fs)
# 
averaged_raw_data = averaged_raw_data[start_idx:end_idx]
t = t[start_idx:end_idx]
N = end_idx - start_idx
#  
# filter the data around the carrier.  
filtered_data           = sosfiltfilt(sos_demodulate_bandpass, averaged_raw_data) 
demodulated_signal      = -demodulate(filtered_data,carrier,t)      # Demodulate
demodulated_signal      = demodulated_signal - np.mean(demodulated_signal)
analytical_signal       = hilbert(filtered_data)                    # Hilbert demodulate.  
h_signal                = -np.abs(analytical_signal)
h_signal                = h_signal - np.mean(h_signal)
demodulated_signal_h    = sosfiltfilt(sos_df, h_signal) 
demodulated_signal_iq   = sosfiltfilt(sos_df, demodulated_signal) 
lfp_data                = sosfiltfilt(sos_df, averaged_raw_data)    # low frequency data. 
#  
# This data is NOT railing... 
# 
# fig = plt.figure(figsize=(8,3))
# ax = fig.add_subplot(211)
# plt.plot(t,h_signal,color='g')
# # plt.plot(t,demodulated_signal_iq,color='b')
# plt.plot(t,lfp_data,'r')
# plt.yticks(fontsize=16)
# # ax.set_ylim([-25,25])
# plt.xticks(fontsize=16)
# ax.set_xlabel('Time(s)',fontsize=16)
# ax.set_ylabel('Volts ($\mu$V)',fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2  = fig.add_subplot(212)
# plt.plot(t,averaged_raw_data,'k')
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plt.show()
#  
# 
# average_lfp      = lfp_data
# diq_average_lfp  = demodulated_signal_iq
# dh_average_lfp   = demodulated_signal_h
# # average_lfp = averaged_raw_data
# # print ('n_events: ', n_events)
# lfp_height   = np.max(lfp_data)- np.min(lfp_data)
# # print ('LFP size', lfp_height)
# dh_lfp_height   = np.max(dh_average_lfp)- np.min(dh_average_lfp)
# # print ('dh_LFP size', dh_lfp_height)
# diq_lfp_height   = np.max(diq_average_lfp)- np.min(diq_average_lfp)
# print ('diq_LFP size', diq_lfp_height)
# 
# print ('lfp/dlfp ratio',np.round(lfp_height/diq_lfp_height,2) )
# # height adjustment. 
# # diq_daverage_lfp = av_daverage_lfp*dlfp_height
#  
# Calculate the FFT. 
#  
start_pause = 1 
end_pause   = 10.5
start_index = int(Fs*start_pause)
end_index   = int(Fs*end_pause)
window      = np.kaiser( (end_index-start_index), 20.0 )
print('window len:',len(window))
newN        = len(lfp_data[start_index:end_index])
# # 
fft_lfp_spectrum = fft(lfp_data[start_index:end_index]*window )
fft_lfp = np.abs(2.0/(newN) * (fft_lfp_spectrum))[1:(newN)//2]

fft_h_spectrum = fft(demodulated_signal_h[start_index:end_index]*window)
fft_h = np.abs(2.0/(newN) * (fft_h_spectrum))[1:(newN)//2]

fft_iq_spectrum = fft(demodulated_signal_iq[start_index:end_index]*window)
fft_iq = np.abs(2.0/(newN) * (fft_iq_spectrum))[1:(newN)//2]
#  
xf  = np.fft.fftfreq( (newN), d=1/Fs)[:(newN)//2]
frequencies = xf[1:(newN)//2]
#  
#  
fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(211)
ax.set_xlim([0,bandwidth_of_interest])
plt.plot(frequencies,fft_lfp,color='k')
ax2  = fig.add_subplot(212)
plt.plot(frequencies,fft_h,color='k')
plt.plot(frequencies,fft_iq,color='g')
ax2.set_xlim([0,bandwidth_of_interest])
plt.show()
# # 
# # 
# # Really I need to recalculate the SNR of everything after the final averaging takes place. 
# # 
# # Calculate the SNR. 
# # 
# # print ('frequencies of interest:', frequencies_of_interest)
# demod_iq_totals              = []  # 
# demod_h_totals               = []  # 
# lfp_totals                   = []
# interest_frequencies         = []
# for i in range(len(frequencies_of_interest)): # sum all frequencies per unit time.
#   df_idx = find_nearest(frequencies,frequencies_of_interest[i])
#   # also eliminate on either side of bins of interest as I don't have great bin resolution.    
#   interest_frequencies.append(df_idx)    
#   demod_iq_totals.append(fft_iq_dm[df_idx])
#   demod_h_totals.append(fft_h_dm[df_idx])
#   lfp_totals.append(fft_m[df_idx])
# end_idx = find_nearest(frequencies,bandwidth_of_interest)
# # # print ('end idx',end_idx)
# dis_demod_iq_totals        = []  # 
# dis_demod_h_totals         = []  # 
# dis_lfp_totals             = []  # 
# for i in range(end_idx): # sum all frequencies per unit time.
#   if i not in interest_frequencies: 
#       dis_demod_iq_totals.append(fft_iq_dm[i])
#       dis_demod_h_totals.append(fft_h_dm[i])
#       dis_lfp_totals.append(fft_m[i])
# # signal to noise calculation. We take the total signal(), and divide it 
# lfp_snr         = 20*np.log(np.mean(lfp_totals)/np.mean(dis_lfp_totals))
# demod_iq_snr    = 20*np.log(np.mean(demod_iq_totals)/np.mean(dis_demod_iq_totals))
# demod_h_snr     = 20*np.log(np.mean(demod_h_totals)/np.mean(dis_demod_h_totals))
# # # print ('amplitude carrier: lfp /demod',carrier,lfp_amplitude,demod_amplitude)
# # # print ('amplitude carrier: lfp /demod',carrier,lfp_amplitude,demod_amplitude)
# print ('snr lfp /demod iq/demod h: ',np.round(lfp_snr,2),np.round(demod_iq_snr,2),np.round(demod_h_snr,2) )
# # 
# start_time  = 0
# start_times = []

# demod   = (diq_average_lfp[start_index:end_index] - np.min(diq_average_lfp[start_index:end_index]))/  (np.max(diq_average_lfp[start_index:end_index]) - np.min(diq_average_lfp[start_index:end_index])   )
# demod2   = (dh_average_lfp[start_index:end_index] - np.min(dh_average_lfp[start_index:end_index]))/  (np.max(dh_average_lfp[start_index:end_index]) - np.min(dh_average_lfp[start_index:end_index])   )

# avlfp   = (average_lfp[start_index:end_index] - np.min(average_lfp[start_index:end_index]))/ ( np.max(average_lfp[start_index:end_index]) - np.min(average_lfp[start_index:end_index]) )
# tt      = t[start_index:end_index]

# fig = plt.figure(figsize=(8,3))
# ax  = fig.add_subplot(111)
# # plt.plot(t,average_lfp/np.max(average_lfp),color='k')
# plt.plot(tt,demod,color='g')
# # plt.plot(tt,demod2,color='purple')
# plt.plot(tt,avlfp,color='k')
# plt.plot(tt,averaged_marker[start_index:end_index],'grey')
# # plt.plot(t,diq_average_lfp,'k')
# # plt.plot(t,average_lfp,'r')
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.set_xlabel('Time(s)',fontsize=16)
# ax.set_ylabel('Normalized Volts ($\mu$V)',fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# # ax2  = fig.add_subplot(212)
# # plt.plot(tt,avlfp,color='k')
# # plt.plot(tt,averaged_marker[start_index:end_index],'r')
# plt.legend(['Demodulated','LFP'],loc='upper left',frameon=False,framealpha=1.0,fontsize=16)
# plt.tight_layout()
# plot_filename = saveprefix+'demodulated_VEP.png'
# plt.savefig(plot_filename, transparent=True)
# plt.show()



# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(211)
# plt.plot(frequencies,fft_m,color='k')
# ax.set_xlim([0,bandwidth_of_interest])
# ax.set_ylabel('Volts ($\mu$V)',fontsize=16)
# ax.set_xlabel('Frequency(Hz)',fontsize=16)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.legend(['LFP'],loc='upper right',frameon=False,framealpha=1.0, fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()

# ax2  = fig.add_subplot(212)
# # plt.plot(frequencies,fft_h_dm,color='m')
# plt.plot(frequencies,fft_iq_dm,color='g')
# ax2.set_xlim([0,bandwidth_of_interest])
# ax2.set_ylabel('Volts ($\mu$V)',fontsize=16)
# ax2.set_xlabel('Frequency(Hz)',fontsize=16)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.legend(['IQ','Hilbert'],loc='upper right',frameon=False,framealpha=1.0, fontsize=16)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = saveprefix+'FFT.png'
# plt.savefig(plot_filename, transparent=True)
# plt.show()


# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(311)
# plt.plot(frequencies,fft_m,color='k')
# ax.set_xlim([0,bandwidth_of_interest])
# ax.set_ylabel('Volts ($\mu$V)',fontsize=16)
# ax.set_xlabel('Frequency(Hz)',fontsize=16)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# # plt.legend(['Demodulated \n FFT'],loc='upper right',frameon=False,framealpha=1.0, fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()

# ax2  = fig.add_subplot(312)
# plt.plot(frequencies,fft_iq_dm,color='g')
# plt.plot(frequencies,fft_h_dm,color='m')
# ax2.set_xlim([0,bandwidth_of_interest])
# ax2.set_ylabel('Volts ($\mu$V)',fontsize=16)
# ax2.set_xlabel('Frequency(Hz)',fontsize=16)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# # plt.legend(['Demodulated \n FFT'],loc='upper right',frameon=False,framealpha=1.0, fontsize=16)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)

# ax3  = fig.add_subplot(313)
# # plt.plot(frequencies,fft_m,color='k')
# plt.plot(frequencies,fft_k,color='k')
# ax3.set_xlim([carrier-bandwidth_of_interest,carrier+bandwidth_of_interest])
# ax3.set_ylim([0,0.25])
# ax3.set_ylabel('Volts ($\mu$V)',fontsize=16)
# ax3.set_xlabel('Frequency(Hz)',fontsize=16)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# # plt.legend(['IQ FFT','av FFT'],loc='upper right',frameon=False,framealpha=1.0, fontsize=16)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plt.tight_layout()
# # plot_filename = saveprefix+'FFT.png'
# # plt.savefig(plot_filename, transparent=True)
# plt.show()
# # 