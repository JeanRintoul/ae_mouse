'''

Title: spike demodulation
Function: takes a single file, and do spike demodulation. 
Author: Jean Rintoul
Date: 23.10.2022

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
import colorednoise as cn
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
# 
def demodulate(in_signal,carrier_f,dt): 

    return np.abs(in_signal*np.exp(2*np.pi*1j*carrier_f*dt))



start = 1
stop  = 11

step  = 1 
file_list = range(start,stop,step)
print ('file list',file_list)
#
t_series        = 't1'
outfile         = t_series + '-vep_data.npz'
Fs              = 5e6
new_Fs          = 1e4

timestep        = 1.0/Fs
duration        = 8
N               = int(Fs*duration)
t               = np.linspace(0, duration, N, endpoint=False)
# carrier         = 20 # 
gain            = 500
m_channel       = 0 
rf_channel      = 4
marker_channel  = 7 
# pulse_length    = 0.0  # pulse length in seconds. 
# 
# 
# sos_df_band     = iirfilter(17, [l_cut,df_cut], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=new_Fs,
#                        output='sos')
# 
# a higher sampling rate is better... 

# 
saveprefix      = './/images//'

# 4Hz files. 
savepath       = './/spike_demodulation_data//e133_t2_VEPS_shield//VEP_4Hz//' # this has higher snr. 
savepath       = './/spike_demodulation_data//e131_t3_VEPS//VEP_4Hz//'  # 37.44/35.2
# savepath       = './/spike_demodulation_data//e133_t4_VEPS//VEP_4Hz//'   # 22/24.
# savepath       = './/spike_demodulation_data//e133_t4_VEPS//VEP_4Hz//'   # 
# savepath       = './/spike_demodulation_data//e135_t2_VEPS//VEP_4Hz//'  # 23.3/15
# savepath       = './/spike_demodulation_data//e135_t2_VEPS//VEP_4Hz_2//'  # 7/10

# 1Hz files. 
# savepath       = './/spike_demodulation_data//e132_t1_VEPS//VEP_1Hz//'
# savepath       = './/spike_demodulation_data//e131_t4_VEPS//VEP_1Hz//'
savepath       = './/spike_demodulation_data//e132_t1_VEPS//VEP_1Hz//'
# savepath       = './/spike_demodulation_data//e133_t2_VEPS_shield//VEP_1Hz//'
# savepath       = './/spike_demodulation_data//e135_t2_VEPS//VEP_4Hz//'
# savepath       = 'D:\\ae_mouse\\e132_ae_neural_recording\\t1_mouse\\AE_PRF80_LED1Hz_g500\\'

# prf            = carrier
downsampling_factor  = int(Fs/new_Fs)
print('downsampling factor: ',downsampling_factor)
# 
# start_pause = 0.5  
# end_pause   = 7.6
# Fs is at 20kHz. 
# frequencies_of_interest     = [8,16]   
frequencies_of_interest     = [2,6,8]   
# f_band_test = np.linspace(0,419,420)
bandwidth_of_interest = 20 
min_df_cut = bandwidth_of_interest*2
df_cut  = min_df_cut   

f_band_test = np.linspace(0,1000,501)
downsampling_filter_lowcut = new_Fs/2 - 1 # i.e 10khz snr demod = 25.06
# 
max_f = np.max(f_band_test)
print (max)

start_pause = 0.1  
end_pause   = 7.9
ledtest            = []
total_lfp          = []
total_demodulation = []
lfp                = []
demod_lfp          = []
demod_lfp_iq       = []
demod_lfp_averaged = []
demod_average      = [] 
snr_lfp            = []
snr_demod          = []
snr_demod_iq       = []

for n in range(len(file_list)):
    file_number = file_list[n] 
    # 
    print ('file_number',file_number)
    filename    = savepath + 't'+str(file_number)+'_stream.npy'
    data        = np.load(filename)
    fsignal     = 1e6*data[m_channel]/gain
    # rfsignal    = 10*data[rf_channel]  
    marker      = data[marker_channel]

    # Calculate the FFT. 
    start_index = int(Fs*start_pause)
    end_index   = int(Fs*end_pause)
    midN = len(fsignal[start_index:end_index])
    fft_m = fft(fsignal[start_index:end_index] )
    fft_m = np.abs(2.0/(midN) * (fft_m))[1:(midN)//2]
    xf  = np.fft.fftfreq( (midN), d=1/Fs)[:(midN)//2]
    frequencies = xf[1:(midN)//2]

    # 
    sos_lp = iirfilter(17, [downsampling_filter_lowcut], rs=60, btype='lowpass',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')  
    # 
    # downsample for easier plotting. 
    downsampled_data1            = sosfiltfilt(sos_lp, fsignal)
    # Now downsample the data. 
    downsampled_data            = downsampled_data1[::downsampling_factor]
    dt                          = t[::downsampling_factor]
    d_marker                    = marker[::downsampling_factor]    
    # 
    # 
    marker  = d_marker/np.max(d_marker)
    diffs   = np.diff(marker)
    # print ('diffs max: ', np.max(diffs))
    # 1Hz
    # indexes = np.argwhere(diffs > 0.0025)[:,0]
    # diffs = np.abs(diffs)
    # 4Hz
    indexes = np.argwhere(diffs > 0.3)[:,0]
    X       = np.insert(indexes, 0, 0)
    m       = np.diff(X)
    # print ('m',m)
    m       = np.argwhere(m > 1000)[:,0]
    alignment_indices = indexes[m]
    # print ('the alignment_indices are:',alignment_indices)
    alignment_indices = alignment_indices[0::2]
    # print ('the alignment_indices are:',alignment_indices)   
    start_time  = 0
    start_times = []
    for i in range(len(alignment_indices) ):
        start_times.append(dt[alignment_indices[i]])
    # 
    # print (alignment_indices)
    # fig = plt.figure(figsize=(8,2))
    # ax = fig.add_subplot(111)
    # plt.plot(diffs,'k')
    # plt.show()
    # 
    # Create filter bands, and add them together. 
    df_high = df_cut + bandwidth_of_interest*2 
    sos_demodulate_bandpass  = iirfilter(17, [df_cut,df_high], rs=60, btype='bandpass',
                           analog=False, ftype='cheby2', fs=new_Fs,
                           output='sos')
    sos_df = iirfilter(17, [0.5,bandwidth_of_interest], rs=60, btype='bandpass',
                           analog=False, ftype='cheby2', fs=new_Fs,
                           output='sos')

    filtered_downsampled_data = sosfiltfilt(sos_demodulate_bandpass, downsampled_data) 
    

    # IQ demodulate.  
    carrier = int((df_high - df_cut)/2) + df_cut 
    print ('carrier',carrier)
    demodulated_signal = -demodulate(filtered_downsampled_data,carrier,dt)
    analytical_signal  = hilbert(filtered_downsampled_data)
    # Hilbert demodulate.  
    h_signal           = -np.abs(analytical_signal)
    demodulated_signal = sosfiltfilt(sos_df, h_signal) 
    demodulated_signal_iq = sosfiltfilt(sos_df, demodulated_signal) 
    # 
    lfp_data = sosfiltfilt(sos_df, downsampled_data)   

    #  Calculate the FFT of the resultant signal. 
    start_index = int(new_Fs*start_pause)
    end_index   = int(new_Fs*end_pause)
    newN = len(lfp_data[start_index:end_index])
    xf  = np.fft.fftfreq( (newN), d=1/new_Fs)[:(newN)//2]
    frequencies = xf[1:(newN)//2]
    fft_m_spectrum = fft(lfp_data[start_index:end_index] )
    fft_m = np.abs(2.0/(newN) * (fft_m_spectrum))[1:(newN)//2]

    # If we wish to leverage averaging, calculate the IQ demodulated signal (or Hilbert)
    # for a range of carriers(i.e. bandwidths). Add the demodulated results up, then average.   
    # This group result, should be better than a singular one? 
    # In simulation, they may all be the same size, in real life though they will get smaller in amplitude. 
    # 
    # 

    # normalize each result that is the best.     
    averaged_demodulated = demodulated_signal_iq
    # 
    # fig = plt.figure(figsize=(6,6))
    # ax  = fig.add_subplot(111)
    # plt.plot(f_band_test,av_snrs,'k')
    # plt.axhline(y=lfp_snr, color='r', linestyle='-')
    # plt.show()
    # # 
    # 
    # lfp_data = sosfiltfilt(sos_df, downsampled_data)   

    # #  Calculate the FFT of the resultant signal. 
    # start_index = int(new_Fs*start_pause)
    # end_index   = int(new_Fs*end_pause)
    # newN = len(lfp_data[start_index:end_index])
    # xf  = np.fft.fftfreq( (newN), d=1/new_Fs)[:(newN)//2]
    # frequencies = xf[1:(newN)//2]
    # fft_m_spectrum = fft(lfp_data[start_index:end_index] )
    # fft_m = np.abs(2.0/(newN) * (fft_m_spectrum))[1:(newN)//2]

    # fft_dm_spectrum = fft(demodulated_signal[start_index:end_index])
    # fft_dm = np.abs(2.0/(newN) * (fft_dm_spectrum))[1:(newN)//2]

    # fft_dm_spectrum = fft(demodulated_signal_iq[start_index:end_index])
    # fft_dm_iq = np.abs(2.0/(newN) * (fft_dm_spectrum))[1:(newN)//2]
    # 
    # Calculate the SNR. 
    # frequencies_of_interest     = [8,16]    
    # # print ('frequencies of interest:', frequencies_of_interest)
    # demod_totals                 = []  # sum all frequencies per unit time. 
    # demod_iq_totals              = []  # sum all frequencies per unit time. 
    # lfp_totals                   = []
    # interest_frequencies         = []
    # for i in range(len(frequencies_of_interest)): # sum all frequencies per unit time.
    #   df_idx = find_nearest(frequencies,frequencies_of_interest[i])
    #   # also eliminate on either side of bins of interest as I don't have great bin resolution.    
    #   interest_frequencies.append(df_idx)    
    #   demod_totals.append(fft_dm[df_idx])
    #   demod_iq_totals.append(fft_dm_iq[df_idx])
    #   lfp_totals.append(fft_m[df_idx])
    # # # print ('demod/lfp totals',lfp_amplitude, demod_amplitude)
    # # df_cut = np.max(frequencies_of_interest)
    # end_idx = find_nearest(frequencies,df_cut)
    # # # print ('end idx',end_idx)
    # dis_demod_totals            = []  # sum all frequencies per unit time. 
    # dis_demod_iq_totals            = []  # 
    # dis_lfp_totals              = []
    # for i in range(end_idx): # sum all frequencies per unit time.
    #   if i not in interest_frequencies: 
    #       dis_demod_totals.append(fft_dm[i])
    #       dis_demod_iq_totals.append(fft_dm_iq[i])
    #       dis_lfp_totals.append(fft_m[i])
    # # # signal to noise calculation. We take the total signal(), and divide it 
    # lfp_snr   = 20*np.log(np.mean(lfp_totals)/np.mean(dis_lfp_totals))
    # demod_snr = 20*np.log(np.mean(demod_totals)/np.mean(dis_demod_totals))
    # demod_iq_snr = 20*np.log(np.mean(demod_iq_totals)/np.mean(dis_demod_iq_totals))
    # # print ('amplitude carrier: lfp /demod',carrier,lfp_amplitude,demod_amplitude)
    # print ('snr lfp /demod: ',lfp_snr,demod_snr,demod_iq_snr)
    # 
    # snr_lfp.append(lfp_snr)
    # snr_demod.append(demod_snr)
    # snr_demod_iq.append(demod_iq_snr)
    ledtest.append(d_marker)
    lfp.append(lfp_data)
    demod_lfp.append(demodulated_signal)   
    demod_lfp_iq.append(demodulated_signal_iq)   
    demod_average.append(averaged_demodulated)   


# snr_lfp   =  np.array(snr_lfp)
# snr_demod = np.array(snr_demod)
# snr_demod_iq = np.array(snr_demod_iq)
# print ('snr lfp/demod: ', np.round(np.mean(snr_lfp),2)  , np.round(np.mean(snr_demod),2 ),np.round(np.mean(snr_demod_iq),2 ) )

lfp_array           = np.array(lfp)
dm_array            = np.array(demod_lfp)
dm_iq_array         = np.array(demod_lfp_iq)
dm_average_array    = np.array(demod_average)
# print ('shape',demod_segmented_array.shape)
# # 
# np.savez(outfile,demod_segmented_array=demod_segmented_array,nfiltered_segmented_array=nfiltered_segmented_array,start = start, end=end)
# print ('saved out a data file!')
# # 
average_ledtest = np.mean(np.array(ledtest),axis=0)
# 
n_events,b = lfp_array.shape
# time_segment = np.linspace(-start,end,num=b)
average_lfp  = np.mean(lfp_array,axis=0)
std_lfp      = np.std(lfp_array,axis=0)
sem_lfp      = np.std(lfp_array,axis=0)/np.sqrt(n_events)

dn_events,b = dm_array.shape
# time_segment = np.linspace(-start,end,num=b)
daverage_lfp  = np.mean(dm_array,axis=0)
dstd_lfp      = np.std(dm_array,axis=0)
dsem_lfp      = np.std(dm_array,axis=0)/np.sqrt(dn_events)

iq_daverage_lfp  = np.mean(dm_iq_array,axis=0)
iq_dstd_lfp      = np.std(dm_iq_array,axis=0)
iq_dsem_lfp      = np.std(dm_iq_array,axis=0)/np.sqrt(dn_events)


av_daverage_lfp  = np.mean(dm_average_array ,axis=0)
av_dstd_lfp      = np.std(dm_average_array ,axis=0)
av_dsem_lfp      = np.std(dm_average_array ,axis=0)/np.sqrt(dn_events)



# print ('n_events: ', n_events)
lfp_height   = np.max(average_lfp)- np.min(average_lfp)
# print ('LFP size', lfp_height)
dlfp_height   = np.max(daverage_lfp)- np.min(daverage_lfp)
# print ('dLFP size', dlfp_height)
print ('lfp/dlfp ratio',np.round(lfp_height/dlfp_height,2) )
# height adjustment. 
av_daverage_lfp = av_daverage_lfp*dlfp_height

# Calculate the FFT. 
start_index = int(new_Fs*start_pause)
end_index   = int(new_Fs*end_pause)

newN = len(average_lfp[start_index:end_index])
fft_m_spectrum = fft(average_lfp[start_index:end_index] )
fft_m = np.abs(2.0/(newN) * (fft_m_spectrum))[1:(newN)//2]
fft_dm_spectrum = fft(daverage_lfp[start_index:end_index])
fft_dm = np.abs(2.0/(newN) * (fft_dm_spectrum))[1:(newN)//2]

fft_iq_dm_spectrum = fft(iq_daverage_lfp[start_index:end_index])
fft_iq_dm = np.abs(2.0/(newN) * (fft_iq_dm_spectrum))[1:(newN)//2]

fft_av_dm_spectrum = fft(av_daverage_lfp[start_index:end_index])
fft_av_dm = np.abs(2.0/(newN) * (fft_av_dm_spectrum))[1:(newN)//2]

xf  = np.fft.fftfreq( (newN), d=1/new_Fs)[:(newN)//2]
frequencies = xf[1:(newN)//2]
# 
# Really I need to recalculate the SNR of everything after the final averaging takes place. 
# 
# Calculate the SNR. 

# print ('frequencies of interest:', frequencies_of_interest)
demod_totals                 = []  # sum all frequencies per unit time. 
demod_iq_totals              = []  # sum all frequencies per unit time. 
demod_av_totals              = []  # sum all frequencies per unit time. 
lfp_totals                   = []
interest_frequencies         = []
for i in range(len(frequencies_of_interest)): # sum all frequencies per unit time.
  df_idx = find_nearest(frequencies,frequencies_of_interest[i])
  # also eliminate on either side of bins of interest as I don't have great bin resolution.    
  interest_frequencies.append(df_idx)    
  demod_totals.append(fft_dm[df_idx])
  demod_iq_totals.append(fft_iq_dm[df_idx])
  demod_av_totals.append(fft_av_dm[df_idx])
  lfp_totals.append(fft_m[df_idx])
# # print ('demod/lfp totals',lfp_amplitude, demod_amplitude)
# df_cut = np.max(frequencies_of_interest)
end_idx = find_nearest(frequencies,bandwidth_of_interest)
# # print ('end idx',end_idx)
dis_demod_totals            = []  # sum all frequencies per unit time. 
dis_demod_iq_totals         = []  # 
dis_demod_av_totals         = []  # 
dis_lfp_totals              = []
for i in range(end_idx): # sum all frequencies per unit time.
  if i not in interest_frequencies: 
      dis_demod_totals.append(fft_dm[i])
      dis_demod_iq_totals.append(fft_iq_dm[i])
      dis_demod_av_totals.append(fft_av_dm[i])
      dis_lfp_totals.append(fft_m[i])
# # signal to noise calculation. We take the total signal(), and divide it 
lfp_snr   = 20*np.log(np.mean(lfp_totals)/np.mean(dis_lfp_totals))
demod_snr = 20*np.log(np.mean(demod_totals)/np.mean(dis_demod_totals))
demod_iq_snr = 20*np.log(np.mean(demod_iq_totals)/np.mean(dis_demod_iq_totals))
demod_av_snr = 20*np.log(np.mean(demod_av_totals)/np.mean(dis_demod_av_totals))
# # print ('amplitude carrier: lfp /demod',carrier,lfp_amplitude,demod_amplitude)
print ('snr lfp /demod: ',np.round(lfp_snr,2),np.round(demod_snr,2),np.round(demod_iq_snr,2),np.round(demod_av_snr,2) )
    # 
# 
# snr_lfp   =  np.array(snr_lfp)
# snr_demod = np.array(snr_demod)
# snr_demod_iq = np.array(snr_demod_iq)
# print ('snr lfp/demod: ', np.round(np.mean(snr_lfp),2)  , np.round(np.mean(snr_demod),2 ),np.round(np.mean(snr_demod_iq),2 ) )

# confidence interval. 
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m,h,se
# 
m,ci,se = mean_confidence_interval(lfp_array,confidence=0.95)
# print ('ci',len(ci))

dm,dci,dse = mean_confidence_interval(dm_array,confidence=0.95)
# print ('dci',len(dci))

# 
error_lb      = ci
error_ub      = ci
derror_lb     = dci
derror_ub     = dci
# 
start_time  = 0
start_times = []
for i in range(len(alignment_indices) ):
    start_times.append(dt[alignment_indices[i]])
# 
lfp_height = np.max(average_lfp) - np.min(average_lfp)
# print ('lfp height:',lfp_height)


fig = plt.figure(figsize=(8,3))
ax  = fig.add_subplot(111)
plt.plot(dt,average_lfp/np.max(average_lfp),color='k')
plt.plot(dt,daverage_lfp/np.max(daverage_lfp),color='green')
plt.plot(dt,iq_daverage_lfp/np.max(iq_daverage_lfp),color='m')
plt.plot(dt,1*d_marker-1.0,'grey')
# plt.plot(dt,1*average_ledtest -1.0,'pink')

# for i in range(len(start_times) ):
#     plt.axvline(x=start_times[i],color ='k')
# plt.plot(dt,1*ledtest_marker_data-0.8,'r')
# ax.set_xlim([xlim_start,xlim_end])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.set_xlabel('Time(s)',fontsize=16)
ax.set_ylabel('Normalized Volts ($\mu$V)',fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.legend(['LFP','Demodulated'],loc='lower right',frameon=False,framealpha=1.0,fontsize=16)
plt.tight_layout()
plot_filename = 'demodulated_VEP.png'
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
plt.plot(frequencies,fft_dm,color='g')
ax2.set_xlim([0,bandwidth_of_interest])
ax2.set_ylabel('Volts ($\mu$V)',fontsize=16)
ax2.set_xlabel('Frequency(Hz)',fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.legend(['Demodulated \n FFT'],loc='upper right',frameon=False,framealpha=1.0, fontsize=16)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)


ax3  = fig.add_subplot(313)
plt.plot(frequencies,fft_iq_dm,color='m')
plt.plot(frequencies,fft_av_dm,color='k')

ax3.set_xlim([0,bandwidth_of_interest])
ax3.set_ylabel('Volts ($\mu$V)',fontsize=16)
ax3.set_xlabel('Frequency(Hz)',fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.legend(['IQ FFT','av FFT'],loc='upper right',frameon=False,framealpha=1.0, fontsize=16)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

plt.tight_layout()
plot_filename = saveprefix+'FFT_HILBERT.png'
plt.savefig(plot_filename, transparent=True)
plt.show()


fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)
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
plot_filename = saveprefix + 'FFT_LFP.png'
plt.savefig(plot_filename, transparent=True)
plt.show()

fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_av_dm,color='g')
ax.set_xlim([0,bandwidth_of_interest])
ax.set_ylabel('Volts ($\mu$V)',fontsize=16)
ax.set_xlabel('Frequency(Hz)',fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.legend(['Demodulated \n FFT'],loc='upper right',frameon=False,framealpha=1.0, fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = saveprefix+'FFT_DM.png'
plt.savefig(plot_filename, transparent=True)
plt.show()



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
