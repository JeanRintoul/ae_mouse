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
# import colorednoise as cn
# 
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
saveprefix      = './/images//' 
savepath        = 'D:\\ae_mouse\\e136_ae_neural_decoding\\t3-mouse\\'
filename        = 'e136_t3_mouse.npz'
print ('filename: ',filename)
# 
data                = np.load(savepath+filename)
rf_summation        = data['rf_summation']
marker_summation    = data['marker_summation']
average_signal      = data['average_signal']
average_corrd       = data['average_corrd']
n                   = data['n']


print ('corrd',len(average_corrd))
averaged_raw_data   = average_signal
averaged_marker     = marker_summation/n
#  
duration            = 8
Fs                  = 5e6
timestep            = 1.0/Fs
N                   = int(Fs*duration)
carrier             = 500000
t                   = np.linspace(0, duration, N, endpoint=False)
# 
# Fs is at 20kHz. 
frequencies_of_interest = [8] 
lowcut                  = 6
bandwidth_of_interest   = 26
# 
# Create filter bands, and add them together. 
sos_demodulate_bandpass  = iirfilter(17, [carrier-bandwidth_of_interest,carrier+bandwidth_of_interest], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

stop_width = 5
sos_demodulate_bandstop  = iirfilter(17, [carrier-stop_width,carrier+stop_width], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
sos_df = iirfilter(17, [lowcut,bandwidth_of_interest], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
# start_pause         = 3.5
# end_pause           = 5.0
# 
start_pause         = 3
end_pause           = 7
# 
# filter the data around the carrier.  
filtered_data           = sosfiltfilt(sos_demodulate_bandstop, averaged_raw_data)
filtered_data           = sosfiltfilt(sos_demodulate_bandpass, filtered_data) 
 

demodulated_signal      = -demodulate(filtered_data,carrier,t)      # Demodulate
analytical_signal       = hilbert(filtered_data)                    # Hilbert demodulate.  
h_signal                = -np.abs(analytical_signal)
demodulated_signal_h    = sosfiltfilt(sos_df, h_signal) 
demodulated_signal_iq   = sosfiltfilt(sos_df, demodulated_signal) 
lfp_data                = sosfiltfilt(sos_df, averaged_raw_data)    # low frequency data. 
# 
# implement a matched filter between. 
# 
x = demodulated_signal_h
y = lfp_data 
# First, center about the mean.     
x = x - x.mean()
y = y - y.mean()
#     
# I need a template signal. 
# Try a template that is an 8 Hz sinusoid.     
template     = [0]*int(1.0*Fs)     # 1 second 
template2    = y[int(3*Fs):int(5*Fs) ]
for i in range(len(template)):
    template[i] = np.cos( 2*np.pi*(8)*i*timestep)  
#   
print ('sig len',len(template))
corr_lfp   = signal.correlate(y,template2, mode='same')
corr_demod = signal.correlate(x,template2, mode='same')
print ('corrs: ',len(corr_lfp),len(corr_demod))


# 
# lfp data and the demodulated data. ? 
# 
average_lfp      = lfp_data
diq_average_lfp  = demodulated_signal_iq
dh_average_lfp   = demodulated_signal_h
# 
# average_lfp = averaged_raw_data
# 
# print ('n_events: ', n_events)
lfp_height   = np.max(lfp_data)- np.min(lfp_data)
# print ('LFP size', lfp_height)
dh_lfp_height   = np.max(dh_average_lfp)- np.min(dh_average_lfp)
# print ('dh_LFP size', dh_lfp_height)
diq_lfp_height   = np.max(diq_average_lfp)- np.min(diq_average_lfp)
print ('diq_LFP size', diq_lfp_height)

print ('lfp/dlfp ratio',np.round(lfp_height/diq_lfp_height,2) )
# height adjustment. 
# diq_daverage_lfp = av_daverage_lfp*dlfp_height
# 
# Calculate the FFT. 
start_index = int(Fs*start_pause)
end_index   = int(Fs*end_pause)
window      = np.kaiser( (end_index-start_index), 20.0 )
print('window len:',len(window))
newN        = len(average_lfp[start_index:end_index])

fft_m_spectrum = fft(lfp_data[start_index:end_index]*window )
fft_m = np.abs(2.0/(newN) * (fft_m_spectrum))[1:(newN)//2]

fft_k_spectrum = fft(averaged_raw_data[start_index:end_index]*window)
fft_k = np.abs(2.0/(newN) * (fft_k_spectrum))[1:(newN)//2]

fft_h_dm_spectrum = fft(dh_average_lfp[start_index:end_index]*window)
fft_h_dm = np.abs(2.0/(newN) * (fft_h_dm_spectrum))[1:(newN)//2]

fft_iq_dm_spectrum = fft(diq_average_lfp[start_index:end_index]*window)
fft_iq_dm = np.abs(2.0/(newN) * (fft_iq_dm_spectrum))[1:(newN)//2]


fft_corr_lfp = fft(corr_lfp[start_index:end_index]*window )
fft_corr_lfp = np.abs(2.0/(newN) * (fft_corr_lfp))[1:(newN)//2]
fft_corr_demod = fft(corr_demod[start_index:end_index]*window )
fft_corr_demod = np.abs(2.0/(newN) * (fft_corr_demod))[1:(newN)//2]

xf  = np.fft.fftfreq( (newN), d=1/Fs)[:(newN)//2]
frequencies = xf[1:(newN)//2]
# 
# Really I need to recalculate the SNR of everything after the final averaging takes place. 
# 
# Calculate the SNR. 
# 
# print ('frequencies of interest:', frequencies_of_interest)
demod_iq_totals              = []  # 
demod_h_totals               = []  # 
lfp_totals                   = []
interest_frequencies         = []
for i in range(len(frequencies_of_interest)): # sum all frequencies per unit time.
  df_idx = find_nearest(frequencies,frequencies_of_interest[i])
  # also eliminate on either side of bins of interest as I don't have great bin resolution.    
  interest_frequencies.append(df_idx)    
  demod_iq_totals.append(fft_iq_dm[df_idx])
  demod_h_totals.append(fft_h_dm[df_idx])
  lfp_totals.append(fft_m[df_idx])
end_idx = find_nearest(frequencies,bandwidth_of_interest)
# # print ('end idx',end_idx)
dis_demod_iq_totals        = []  # 
dis_demod_h_totals         = []  # 
dis_lfp_totals             = []  # 
for i in range(end_idx): # sum all frequencies per unit time.
  if i not in interest_frequencies: 
      dis_demod_iq_totals.append(fft_iq_dm[i])
      dis_demod_h_totals.append(fft_h_dm[i])
      dis_lfp_totals.append(fft_m[i])
# signal to noise calculation. We take the total signal(), and divide it 
lfp_snr         = 20*np.log(np.mean(lfp_totals)/np.mean(dis_lfp_totals))
demod_iq_snr    = 20*np.log(np.mean(demod_iq_totals)/np.mean(dis_demod_iq_totals))
demod_h_snr     = 20*np.log(np.mean(demod_h_totals)/np.mean(dis_demod_h_totals))
# # print ('amplitude carrier: lfp /demod',carrier,lfp_amplitude,demod_amplitude)
# # print ('amplitude carrier: lfp /demod',carrier,lfp_amplitude,demod_amplitude)
print ('snr lfp /demod iq/demod h: ',np.round(lfp_snr,2),np.round(demod_iq_snr,2),np.round(demod_h_snr,2) )
# 
start_time  = 0
start_times = []

demod   = (diq_average_lfp[start_index:end_index] - np.min(diq_average_lfp[start_index:end_index]))/  (np.max(diq_average_lfp[start_index:end_index]) - np.min(diq_average_lfp[start_index:end_index])   )
demod2  = (dh_average_lfp[start_index:end_index] - np.min(dh_average_lfp[start_index:end_index]))/  (np.max(dh_average_lfp[start_index:end_index]) - np.min(dh_average_lfp[start_index:end_index])   )

avlfp   = (average_lfp[start_index:end_index] - np.min(average_lfp[start_index:end_index]))/ ( np.max(average_lfp[start_index:end_index]) - np.min(average_lfp[start_index:end_index]) )
tt      = t[start_index:end_index]
corrdn  = average_corrd[start_index:end_index]
# 
# there is ringingin the demod... which effects a large part of the signal.  
# 
fig = plt.figure(figsize=(8,5))
ax  = fig.add_subplot(311)
# plt.plot(t,average_lfp/np.max(average_lfp),color='k')
plt.plot(tt,demod,color='g')
# plt.plot(tt,corrdn,color='purple')
# plt.plot(tt,demod2,color='purple')
plt.plot(tt,avlfp,color='k')
plt.plot(tt,averaged_marker[start_index:end_index],'grey')
# plt.plot(t,diq_average_lfp,'k')
# plt.plot(t,average_lfp,'r')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.set_xlabel('Time(s)',fontsize=16)
ax.set_ylabel('Normalized Volts ($\mu$V)',fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(['Demodulated','LFP'],loc='upper left',frameon=False,framealpha=1.0,fontsize=16)

ax2  = fig.add_subplot(312)
# plt.plot(tt,corr_lfp[start_index:end_index],'k')
plt.plot(tt,avlfp,color='k')
plt.plot(tt,-corr_demod[start_index:end_index]/np.max(corr_demod[start_index:end_index]),'g')
plt.plot(tt,-1+averaged_marker[start_index:end_index],'grey')
# plt.plot(tt,avlfp,color='k')
# plt.plot(tt,averaged_marker[start_index:end_index],'r')
# corr_lfp   = signal.correlate(y,template2, mode='same')
# corr_demod = signal.correlate(x,template2, mode='same')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax3  = fig.add_subplot(313)
plt.plot(frequencies,fft_corr_demod,'g')
plt.plot(frequencies,fft_corr_lfp,'k')
ax3.set_xlim([0,bandwidth_of_interest])
plt.tight_layout()
plot_filename = saveprefix+'demodulated_VEP.png'
plt.savefig(plot_filename, transparent=True)
plt.show()


fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(211)
plt.plot(frequencies,fft_m,color='k')
ax.set_xlim([0,bandwidth_of_interest])
ax.set_ylabel('Volts ($\mu$V)',fontsize=16)
ax.set_xlabel('Frequency(Hz)',fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.legend(['LFP'],loc='upper right',frameon=False,framealpha=1.0, fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()

ax2  = fig.add_subplot(212)
# plt.plot(frequencies,fft_h_dm,color='m')
plt.plot(frequencies,fft_iq_dm,color='g')
ax2.set_xlim([0,bandwidth_of_interest])
ax2.set_ylabel('Volts ($\mu$V)',fontsize=16)
ax2.set_xlabel('Frequency(Hz)',fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.legend(['IQ','Hilbert'],loc='upper right',frameon=False,framealpha=1.0, fontsize=16)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = saveprefix+'FFT.png'
plt.savefig(plot_filename, transparent=True)
plt.show()


fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(311)
plt.plot(frequencies,fft_m,color='k')
ax.set_xlim([0,bandwidth_of_interest])
ax.set_ylabel('Volts ($\mu$V)',fontsize=16)
ax.set_xlabel('Frequency(Hz)',fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.legend(['Demodulated \n FFT'],loc='upper right',frameon=False,framealpha=1.0, fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()

ax2  = fig.add_subplot(312)
plt.plot(frequencies,fft_iq_dm,color='g')
plt.plot(frequencies,fft_h_dm,color='m')
ax2.set_xlim([0,bandwidth_of_interest])
ax2.set_ylabel('Volts ($\mu$V)',fontsize=16)
ax2.set_xlabel('Frequency(Hz)',fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.legend(['Demodulated \n FFT'],loc='upper right',frameon=False,framealpha=1.0, fontsize=16)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax3  = fig.add_subplot(313)
# plt.plot(frequencies,fft_m,color='k')
plt.plot(frequencies,fft_k,color='k')
ax3.set_xlim([carrier-bandwidth_of_interest,carrier+bandwidth_of_interest])
ax3.set_ylim([0,0.25])
ax3.set_ylabel('Volts ($\mu$V)',fontsize=16)
ax3.set_xlabel('Frequency(Hz)',fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.legend(['IQ FFT','av FFT'],loc='upper right',frameon=False,framealpha=1.0, fontsize=16)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
plt.tight_layout()
# plot_filename = saveprefix+'FFT.png'
# plt.savefig(plot_filename, transparent=True)
plt.show()
# 