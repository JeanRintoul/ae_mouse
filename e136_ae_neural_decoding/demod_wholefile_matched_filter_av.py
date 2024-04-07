'''

Title: Implement a matched filter between the original signal and demodulated signal. 
Function:
Author: Jean Rintoul
Date: 01.02.2024

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
from collections import deque 
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
start       = 1
stop        = 15
#
step        = 1 
file_list   = range(start,stop,step)
print ('file list',file_list)
#
t_series        = 't1'
outfile         = t_series + '-vep_data.npz'
Fs              = 5e6
new_Fs          = 1e4

timestep        = 1.0/Fs
duration        = 12
N               = int(Fs*duration)
t               = np.linspace(0, duration, N, endpoint=False)
carrier         = 500000 
gain            = 200
m_channel       = 0 
rf_channel      = 4
marker_channel  = 7 
# 
saveprefix      = './/images//'
# 2Hz files. 
savepath        = 'D:\\ae_mouse\\e136_ae_neural_decoding\\t2-phantom\\8hz_100microvolts\\'


# 



downsampling_factor  = int(Fs/new_Fs)
print('downsampling factor: ',downsampling_factor)
# 
# Fs is at 20kHz. 
frequencies_of_interest = [8] 
# frequencies_of_interest = [4,6,8]  
lowcut                  = 0.1
bandwidth_of_interest   = 30 

# Create filter bands, and add them together. 
sos_demodulate_bandpass  = iirfilter(17, [carrier-bandwidth_of_interest,carrier+bandwidth_of_interest], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

# sos_demodulate_bandstop  = iirfilter(17, [carrier-lowcut,carrier+lowcut], rs=60, btype='bandstop',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
sos_df = iirfilter(17, [lowcut,bandwidth_of_interest], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
sos_lowpass = iirfilter(17, [bandwidth_of_interest], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

start_pause         = 1.5
end_pause           = 10.5
ledtest             = []
lfp                 = []
demod_iq            = []
demod_h             = []

data_summation      = [0]*t
marker_summation    = [0]*t
rf_summation        = [0]*t
for n in range(len(file_list)):
    file_number = file_list[n] 
    # 
    print ('file_number',file_number)
    filename            = savepath + 't'+str(file_number)+'_stream.npy'
    data                = np.load(filename)
    fsignal             = 1e6*data[m_channel]/gain
    rfsignal            = 10*data[rf_channel]  
    marker              = data[marker_channel]
    marker_summation    = marker_summation + marker
    data_summation      = data_summation + fsignal 
    rf_summation        = rf_summation + rfsignal
    #  
    # 
    filtered_signal   = sosfiltfilt(sos_demodulate_bandpass, fsignal) 
    demodulate_signal = demodulate(filtered_signal,carrier,t)
    # These are the two signals to enter the matched filter.  
    demodulate_signal = sosfiltfilt(sos_lowpass,demodulate_signal)
    lfp_data = sosfiltfilt(sos_lowpass,fsignal)
    # 
    x = demodulate_signal
    y = lfp_data 
    # First, center about the mean.     
    x = x - x.mean()
    y = y - y.mean()
    #     
    # I need a template signal. 
    # Try a template that is an 8 Hz sinusoid.     
    template     = [0]*int(1.0*Fs)     # 1 second 
    template2    = lfp_data[int(7*Fs):int(8*Fs) ]
    for i in range(len(template)):
        template[i] = np.cos( 2*np.pi*(21)*i*timestep)  
    #   
    print ('sig len',len(template))
    corr = signal.correlate(y,template2, mode='same')
    corrd = signal.correlate(x,template2, mode='same')
    #   
    print ('corr len',len(corr),len(y) )
    start_index = int(Fs*start_pause)
    end_index   = int(Fs*end_pause)
    newN        = len(lfp_data[start_index:end_index])
    fft_mf      = fft(corrd[start_index:end_index])
    fft_mf      = np.abs(2.0/(end_index-start_index) * (fft_mf))[1:(end_index-start_index)//2]
    fft_mflfp   = fft(corr[start_index:end_index])
    fft_mflfp   = np.abs(2.0/(end_index-start_index) * (fft_mflfp))[1:(end_index-start_index)//2]
    xf          = np.fft.fftfreq( (newN), d=1/Fs)[:(newN)//2]
    frequencies = xf[1:(newN)//2]
    # 
    # 
    # 
    t_bit            = t[start_index:end_index]
    y_bit            = y[start_index:end_index]
    corr_bit         = corr[start_index:end_index]
    corrd_bit        = corrd[start_index:end_index]
    norm_lfp         = (y/np.max(y))[start_index:end_index]
    norm_demod       = (x/np.max(x))[start_index:end_index]
    norm_corr_lfp    = (corr/np.max(corr))[start_index:end_index]
    norm_corr_demod  = (corrd/np.max(corrd))[start_index:end_index]
    # 
    sos_corr_filt    = iirfilter(17, [5,bandwidth_of_interest], rs=60, btype='bandpass',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
    factor          = 10 
    norm_corr_demod = factor*sosfiltfilt(sos_corr_filt,norm_corr_demod)
    norm_demod      = factor*sosfiltfilt(sos_corr_filt,norm_demod)
    # 
    # The matched filter DOES have a lag. 
    # 
    # 
    fig = plt.figure(figsize=(8,5))
    ax  = fig.add_subplot(211)
    # plt.plot(t_bit,norm_lfp,'k')
    plt.plot(t_bit,norm_demod,'b')
    # plt.plot(t_bit,norm_corr_lfp,'g')
    plt.plot(t_bit,norm_corr_demod,'purple')
    plt.legend(['lfp','demod','corr lfp','corr demod'],loc='upper right')
    ax2 = fig.add_subplot(212)    
    plt.plot(frequencies,fft_mf/np.max(fft_mf),'b')
    plt.plot(frequencies,fft_mflfp/np.max(fft_mflfp),'purple')
    ax2.set_xlim([0,bandwidth_of_interest])
    plt.legend(['demod_correlation','lfp correlation'],loc='upper right')
    plt.show()
    # 
    # # fig = plt.figure(figsize=(8,5))
    # ax  = fig.add_subplot(211)
    # plt.plot(t,marker,'k')
    # plt.plot(t,marker_summation/file_number,'r')
    # ax  = fig.add_subplot(212)
    # plt.plot(t,rfsignal,'k')
    # plt.plot(t,rf_summation/file_number,'r')
    # plt.show()
# 
# 
# Work from this: 
averaged_raw_data = data_summation /len(file_list)
averaged_marker   = marker_summation /len(file_list)
# 

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,averaged_marker=averaged_marker,averaged_raw_data=averaged_raw_data,n=len(file_list) )
print ('saved out a data file!')
# 
# filter the data around the carrier.  
filtered_data           = sosfiltfilt(sos_demodulate_bandpass, averaged_raw_data) 
# filtered_data           = sosfiltfilt(sos_demodulate_bandstop, filtered_data) 

demodulated_signal      = -demodulate(filtered_data,carrier,t) # Demodulate
analytical_signal       = hilbert(filtered_data) # Hilbert demodulate.  
h_signal                = -np.abs(analytical_signal)
demodulated_signal_h    = sosfiltfilt(sos_df, h_signal) 
demodulated_signal_iq   = sosfiltfilt(sos_df, demodulated_signal) 
lfp_data                = sosfiltfilt(sos_df, averaged_raw_data)   # low frequency data. 
# 
# 
average_lfp      = lfp_data
diq_average_lfp  = demodulated_signal_iq
dh_average_lfp   = demodulated_signal_h

# print ('n_events: ', n_events)
lfp_height   = np.max(average_lfp)- np.min(average_lfp)
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

fft_m_spectrum = fft(average_lfp[start_index:end_index] )
fft_m = np.abs(2.0/(newN) * (fft_m_spectrum))[1:(newN)//2]

fft_k_spectrum = fft(averaged_raw_data[start_index:end_index]*window)
fft_k = np.abs(2.0/(newN) * (fft_k_spectrum))[1:(newN)//2]

fft_h_dm_spectrum = fft(dh_average_lfp[start_index:end_index])
fft_h_dm = np.abs(2.0/(newN) * (fft_h_dm_spectrum))[1:(newN)//2]

fft_iq_dm_spectrum = fft(diq_average_lfp[start_index:end_index])
fft_iq_dm = np.abs(2.0/(newN) * (fft_iq_dm_spectrum))[1:(newN)//2]

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
# for i in range(len(alignment_indices) ):
#     start_times.append(dt[alignment_indices[i]])
# 
fig = plt.figure(figsize=(8,3))
ax  = fig.add_subplot(211)
# plt.plot(t,average_lfp/np.max(average_lfp),color='k')
plt.plot(t,diq_average_lfp,color='m')
plt.plot(t,averaged_marker,'r')

plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.set_xlabel('Time(s)',fontsize=16)
ax.set_ylabel('Normalized Volts ($\mu$V)',fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax2  = fig.add_subplot(212)
plt.plot(t,average_lfp,color='k')
plt.plot(t,averaged_marker,'r')

# plt.legend(['LFP','Demodulated'],loc='lower right',frameon=False,framealpha=1.0,fontsize=16)
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
plt.plot(frequencies,fft_m,color='k')
plt.plot(frequencies,fft_k,color='r')
ax3.set_xlim([carrier-bandwidth_of_interest,carrier+bandwidth_of_interest])
# ax3.set_ylim([0,0.05])
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
# 
# fig = plt.figure(figsize=(3,3))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_m,color='k')
# ax.set_xlim([0,bandwidth_of_interest])
# 
# ax.set_ylabel('Volts ($\mu$V)',fontsize=16)
# ax.set_xlabel('Frequency(Hz)',fontsize=16)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# # plt.legend(['Demodulated \n FFT'],loc='upper right',frameon=False,framealpha=1.0, fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = saveprefix + 'FFT_LFP.png'
# plt.savefig(plot_filename, transparent=True)
# plt.show()

# fig = plt.figure(figsize=(3,3))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_av_dm,color='g')
# ax.set_xlim([0,bandwidth_of_interest])
# ax.set_ylabel('Volts ($\mu$V)',fontsize=16)
# ax.set_xlabel('Frequency(Hz)',fontsize=16)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# # plt.legend(['Demodulated \n FFT'],loc='upper right',frameon=False,framealpha=1.0, fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = saveprefix+'FFT_DM.png'
# plt.savefig(plot_filename, transparent=True)
# plt.show()



# fig = plt.figure(figsize=(8,3))
# ax  = fig.add_subplot(111)
# plt.plot(dt,daverage_lfp,color='g')
# plt.fill_between(dt, daverage_lfp - derror_lb, daverage_lfp + derror_ub,
#                   color='g', alpha=0.2)
# # for i in range(len(start_times) ):
# #     plt.axvline(x=start_times[i],color ='k')
# ax.set_ylim([-400,200])
# plt.yticks([])
# plt.xticks([])
# # ax.set_xlabel('Time(s)',fontsize=16)
# # ax.set_ylabel('Volts ($\mu$V)',fontsize=16)
# ax.set_xlim([xlim_start,xlim_end])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# plt.tight_layout()
# plot_filename = saveprefix+'wholefile_dm.png'
# plt.savefig(plot_filename, transparent=True)
# plt.show()






# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(511)
# plt.plot(dt,average_lfp/np.max(average_lfp),color='k')
# plt.plot(dt,daverage_lfp/np.max(daverage_lfp),color='purple')
# plt.plot(dt,1*d_marker-1.5,'grey')
# plt.plot(dt,1*ledtest_marker_data-0.8,'pink')

# # plot the start and end times. 
# for i in range(len(start_times) ):
#     plt.axvline(x=start_times[i],color ='k')
#     ax.set_ylim([-2.0,1])
# ax.set_ylabel('norm Volts ($\mu$V)')
# ax.set_xlabel('Time(s)')
# plt.legend(['LFP','Demodulated'],loc='lower right')
# plt.autoscale(enable=True, axis='x', tight=True)

# ax2  = fig.add_subplot(412)
# # plt.plot(dt,-100+50*d_rf_envelope/np.max(d_rf_envelope),'m')
# plt.plot(dt,average_lfp,color='k')
# plt.plot(dt,daverage_lfp,color='purple')
# plt.fill_between(dt, average_lfp - error_lb, average_lfp + error_ub,
#                   color='gray', alpha=0.2)
# plt.fill_between(dt, daverage_lfp - derror_lb, daverage_lfp + derror_ub,
#                   color='purple', alpha=0.2)

# # plot the led start times. 
# for i in range(len(start_times) ):
#     plt.axvline(x=start_times[i],color ='k')
# plt.legend(['rf envelope','scale lfp','scale demodulated'],loc='lower right')
# ax2.set_ylabel('Volts ($\mu$V)')
# ax2.set_xlabel('Time(s)')
# plt.autoscale(enable=True, axis='x', tight=True)

# ax3 = fig.add_subplot(413)
# plt.plot(frequencies,fft_m,color='k')
# plt.legend(['lfp'],loc='upper right')
# ax3.set_xlim([0,df_cut])
# ax3.set_ylabel('Volts ($\mu$V)')
# ax3.set_xlabel('Frequency(Hz)')
# ax4 = fig.add_subplot(414)
# plt.plot(frequencies,fft_dm,color='purple')
# ax4.set_xlim([0,df_cut])
# plt.legend(['demodulated'],loc='upper right')
# ax4.set_ylabel('Volts ($\mu$V)')
# ax4.set_xlabel('Frequency(Hz)')

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# ax4.spines['right'].set_visible(False)
# ax4.spines['top'].set_visible(False)
# # plot_filename = 'demodvepevents.png'
# # plt.savefig(plot_filename, transparent=True)
# plt.show()



# plot_filename = '_singletrials_'+str(prf)+'events_'+str(n_events)+'.png'
# plt.savefig(plot_filename, transparent=True)
# plt.show()
