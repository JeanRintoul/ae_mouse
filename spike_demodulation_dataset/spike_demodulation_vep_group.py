'''

Title: vep signal inspection
Function: takes a single file, and averages the veps to see them better. 

Author: Jean Rintoul
Date: 23.10.2022

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
from scipy.stats import ttest_ind
from scipy.signal import hilbert
from scipy.signal import find_peaks
import scipy.stats
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16
# 
# 
# 
start = 1
stop  = 11
step  = 1 
file_list = range(start,stop,step)
print ('file list',file_list)
# 
saveprefix      = './/images//'
# 
t_series        = '4hz'
outfile         = t_series + '-vep.npz'
# Fs = 100khz
#savepath       = 'D:\\ae_mouse\\spike_demodulation_dataset\\e1\\spike_demodulation_Fs_100kHz\\4Hz_VEP\\'
# Fs = 2Mhz
# savepath       = 'D:\\ae_mouse\\spike_demodulation_dataset\\e2\\VEP_4Hz\\'  # a bit wonky but there. 
# savepath       = 'D:\\ae_mouse\\spike_demodulation_dataset\\e3\\VEP_4Hz_noUS\\'  # looks beautiful
# Fs = 5MHz
savepath       = 'D:\\ae_mouse\\spike_demodulation_dataset\\e4\\VEP_4Hz\\' # clearly something there. 
# 
# savepath       = 'D:\\ae_mouse\\spike_demodulation_dataset\\e5\\4Hz_VEP\\' # clearly something there. 
savepath       = 'D:\\ae_mouse\\spike_demodulation_dataset\\e6\\4Hz_VEP\\' # clearly something there. 


# Fs = 100kHz
frequencies_of_interest       = [8,16,24]   
 
end_frequency   = 26
gain            = 500
m_channel       = 0 
rf_channel      = 4
marker_channel  = 7 
pulse_length    = 0.0 # pulse length in seconds. 
# 
Fs              = 1e5
# duration        = 12
# Fs              = 5e6
# Fs = 5e6
# duration        = 8
duration        = 12 
# 
new_Fs          = 1e4
timestep        = 1.0/Fs
N               = int(Fs*duration)
t               = np.linspace(0, duration, N, endpoint=False)
prf             = 4
# number of periods to look at? must be a sub-multiple of the number of repeats in the file.  
periods_of_interest  = 1
downsampling_factor  = int(Fs/new_Fs)
# 
start                           = 0.1*(1.0/prf)   # 0.2 seconds
end                             = (periods_of_interest-1)*(1.0/prf)+0.9*(1.0/prf)
pre_event_idxcount              = int(start*new_Fs)
post_event_idxcount             = int(end*new_Fs)
array_len                       = post_event_idxcount+pre_event_idxcount
nfiltered_segmented_array       = np.zeros((0,array_len))
ndemod_segmented_array          = np.zeros((0,array_len))
print ('array_len',array_len)
# 
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   
# 
def demodulate(in_signal,carrier_f,dt): 
    return np.abs(in_signal*np.exp(2*np.pi*1j*carrier_f*dt))

# confidence interval. 
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m,h,se

# 
start_pause                 = 1.5
end_pause                   = 10.5
bandwidth_of_interest       = 40
# bandwidth_of_interest       = 300

downsampling_filter_lowcut  = new_Fs/2 - 1 # i.e 10khz snr demod = 25.06
min_df_cut                  = bandwidth_of_interest*2
df_cut                      = min_df_cut   
f_band_test                 = np.linspace(0,2000,501)


# 
for n in range(len(file_list)):
    file_number = file_list[n] 
    # 
    print ('file_number',file_number)
    filename    = savepath + 't'+str(file_number)+'_stream.npy'
    data        = np.load(filename)
    fsignal     = 1e6*data[m_channel]/gain
    marker      = data[marker_channel]
    # 
    sos_lp = iirfilter(17, [downsampling_filter_lowcut], rs=60, btype='lowpass',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')  
    # 
    # downsample for easier plotting. 
    downsampled_data1           = sosfiltfilt(sos_lp, fsignal)
    # Now downsample the data. 
    downsampled_data            = downsampled_data1[::downsampling_factor]
    dt                          = t[::downsampling_factor]
    d_marker                    = marker[::downsampling_factor]    

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
    alignment_indices = alignment_indices[0::1]
    # print ('the alignment_indices are:',alignment_indices)   
    start_time  = 0
    start_times = []
    for i in range(len(alignment_indices) ):
        start_times.append(dt[alignment_indices[i]])
    # 
    # fig  = plt.figure(figsize=(10,6))
    # ax   = fig.add_subplot(311)
    # plt.plot(dt,downsampled_data,'k')
    # plt.plot(dt[alignment_indices],downsampled_data[alignment_indices],'.r')
    # ax2  = fig.add_subplot(312)
    # plt.plot(diffs,'k')
    # ax3  = fig.add_subplot(313)    
    # plt.plot(dt,d_marker,'k')
    # plt.plot(dt[alignment_indices],d_marker[alignment_indices],'.r')  
    # plt.show()
    # 
    # 
    sos_df   = iirfilter(17, [0.5,bandwidth_of_interest], rs=60, btype='bandpass',
                           analog=False, ftype='cheby2', fs=new_Fs,
                           output='sos')    

    lfp_data = sosfiltfilt(sos_df, downsampled_data)   
    # 
    # 
    start_index                 = int(new_Fs*start_pause)
    end_index                   = int(new_Fs*end_pause)
    newN                        = len(dt[start_index:end_index])
    fft_m_spectrum = fft(lfp_data[start_index:end_index] )
    fft_m = np.abs(2.0/(newN) * (fft_m_spectrum))[1:(newN)//2]
    xf  = np.fft.fftfreq( (newN), d=1/new_Fs)[:(newN)//2]
    frequencies = xf[1:(newN)//2]
    # 
    component_list = [] 
    biggest  = [0]*t
    prev_snr = 0
    av_snrs  = []
    winning_carrier = 0 
    # print ('f band test:',f_band_test)
    for j in range(len(f_band_test)):
        # no overlap parameters       
        x       = f_band_test[j] 
        df_cut  = min_df_cut + x
        df_high = df_cut + bandwidth_of_interest*2 
        carrier = int((df_high - df_cut)/2) + df_cut 
        # print ('carrier',carrier)
        # print ('df cut',df_cut,df_high)
        sos_demodulate_bandpass  = iirfilter(17, [df_cut,df_high], rs=60, btype='bandpass',
                               analog=False, ftype='cheby2', fs=new_Fs,
                               output='sos')
        filtered_downsampled_data       = sosfiltfilt(sos_demodulate_bandpass, downsampled_data) 
 
        demodulated_signal_component    = -demodulate(filtered_downsampled_data,carrier,dt)
        demodulated_signal_iq_component = sosfiltfilt(sos_df, demodulated_signal_component) 

        # This is a single band of the demodulated signal. 
        fft_dm_component_spectrum       = fft(demodulated_signal_iq_component[start_index:end_index])
        fft_dm_component                = np.abs(2.0/(newN) * (fft_dm_component_spectrum))[1:(newN)//2]
        fft_dm                          = fft_dm_component
        # SNR calculation per component. 
        # print ('frequencies of interest:', frequencies_of_interest)
        demod_totals                 = []  # sum all frequencies per unit time. 
        lfp_totals                   = []
        interest_frequencies         = []
        for i in range(len(frequencies_of_interest)): # sum all frequencies per unit time.
          df_idx = find_nearest(frequencies,frequencies_of_interest[i])
          # also eliminate on either side of bins of interest as I don't have great bin resolution.    
          # I think it'd be ideal here if each frequency bin were normalized.           
          interest_frequencies.append(df_idx)    
          demod_totals.append(fft_dm[df_idx])
          lfp_totals.append(fft_m[df_idx])
        # # print ('demod/lfp totals',lfp_amplitude, demod_amplitude)
        end_idx = find_nearest(frequencies,end_frequency)
        # # print ('end idx',end_idx)
        dis_demod_totals            = []  # sum all frequencies per unit time. 
        dis_lfp_totals              = []
        for i in range(end_idx): # sum all frequencies per unit time.
          if i not in interest_frequencies: 
              dis_demod_totals.append(fft_dm[i])
              dis_lfp_totals.append(fft_m[i])
        # signal to noise calculation. We take the total signal(), and divide it 
        lfp_snr   = 20*np.log(np.mean(lfp_totals)/np.mean(dis_lfp_totals))
        demod_snr = 20*np.log(np.mean(demod_totals)/np.mean(dis_demod_totals))
        # print ('snr lfp /demod: ',np.round(lfp_snr,2),np.round(demod_snr,2)  )  
        av_snrs.append(demod_snr)
        if demod_snr > prev_snr:
            # component_list.append(demodulated_signal_iq_component)
            biggest = demodulated_signal_iq_component
            prev_snr = demod_snr
            winning_carrier = carrier
            averaged_demodulated = biggest

    print ('carrier: snr lfp /demod: ',winning_carrier, np.round(lfp_snr,2),np.round(prev_snr,2)  )  
    #  
    # averaged_demodulated = np.mean(np.array(component_list),axis=0 )
    # normalize each result that is the best.     
    # averaged_demodulated = biggest
    # fig  = plt.figure(figsize=(10,6))
    # ax   = fig.add_subplot(111)
    # plt.plot(dt,averaged_demodulated ,'k')
    # plt.plot(dt[alignment_indices],averaged_demodulated[alignment_indices],'.r') 
    # plt.show()
    # 
    # 
    # print ('alignment_indices', alignment_indices,len(alignment_indices))
    data_to_segment                 = lfp_data
    demod_to_segment                = averaged_demodulated
    filtered_segmented_data         = []
    demod_segmented_data            = []
    for i in range(len(alignment_indices)):  #
        if i % periods_of_interest == 0:
            # print ('here',i,alignment_indices[i])
            baseline = np.mean(data_to_segment[(alignment_indices[i]-pre_event_idxcount):alignment_indices[i] ])
            # print ('a indices',alignment_indices[i],pre_event_idxcount,post_event_idxcount)
            segment = data_to_segment[(alignment_indices[i]-pre_event_idxcount):(alignment_indices[i]+post_event_idxcount)]-baseline            
            # print ('total height of segment:',(np.max(segment)-np.min(segment)))
            if (np.max(segment)-np.min(segment)) < 1000: # rudimentary filter. 
            # print ('length segment',len(segment))
                if len(segment) == array_len: # ensure only full sections are appended.
                    filtered_segmented_data.append(segment)
            # else: 
                # print ('skipped one too big')
            # do it now for the demodulated data. 
            demod_baseline = np.mean(demod_to_segment[(alignment_indices[i]-pre_event_idxcount):alignment_indices[i] ])
            # print ('a indices',alignment_indices[i],pre_event_idxcount,post_event_idxcount)
            demod_segment = demod_to_segment[(alignment_indices[i]-pre_event_idxcount):(alignment_indices[i]+post_event_idxcount)]-demod_baseline            
            if len(demod_segment) == array_len: # ensure only full sections are appended.
                demod_segmented_data.append(demod_segment )
    #     
    demod_segmented_array = np.array(demod_segmented_data)
    ndemod_segmented_array = np.concatenate((ndemod_segmented_array, demod_segmented_array), axis=0)

    # print ('len f seg data:', len(filtered_segmented_data))
    filtered_segmented_array = np.array(filtered_segmented_data)
    nfiltered_segmented_array = np.concatenate((nfiltered_segmented_array, filtered_segmented_array), axis=0)
    # print ('filtered segmented array shape', nfiltered_segmented_array.shape)
# 
# 
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,ndemod_segmented_array=ndemod_segmented_array,nfiltered_segmented_array=nfiltered_segmented_array,start = start, end=end)
print ('saved out a data file!')
# 
# 
print ('nfiltered shape: ',nfiltered_segmented_array.shape)
# 
n_events,b   = nfiltered_segmented_array.shape
time_segment = np.linspace(-start,end,num=b)
average_lfp  = np.mean(nfiltered_segmented_array,axis=0)
std_lfp      = np.std(nfiltered_segmented_array,axis=0)
sem_lfp      = np.std(nfiltered_segmented_array,axis=0)/np.sqrt(n_events)


ndemod_segmented_array = ndemod_segmented_array*1
# 

dn_events,b   = ndemod_segmented_array.shape
time_segment  = np.linspace(-start,end,num=b)
daverage_lfp  = np.mean(ndemod_segmented_array,axis=0)
dstd_lfp      = np.std(ndemod_segmented_array,axis=0)
dsem_lfp      = np.std(ndemod_segmented_array,axis=0)/np.sqrt(dn_events)

# 
m,ci,se = mean_confidence_interval(nfiltered_segmented_array,confidence=0.95)
print ('ci',len(ci))
print ('n_events: ', n_events)
lfp_height   = np.max(average_lfp)- np.min(average_lfp)
print ('LFP size', lfp_height)
error_lb     = ci
error_ub     = ci

m,dci,se = mean_confidence_interval(ndemod_segmented_array,confidence=0.95)
print ('dci',len(dci))
print ('dn_events: ', dn_events)
demod_height   = np.max(daverage_lfp)- np.min(daverage_lfp)
print ('demod size', demod_height)
derror_lb     = dci
derror_ub     = dci

# Height ratios. 
height_ratio = lfp_height/demod_height
print ('height ratio',height_ratio)
daverage_lfp = daverage_lfp *height_ratio
derror_lb = derror_lb *height_ratio
derror_ub = derror_ub *height_ratio

# Calculate the final SNR. 
# Calculate the FFT. 
start_index = int(new_Fs*start_pause)
end_index   = int(new_Fs*end_pause)
newN =  end_index - start_index
print ('lengths',newN,len(average_lfp[start_index:end_index]))

# fft_m_spectrum = fft(average_lfp[start_index:end_index] )
# fft_m = np.abs(2.0/(newN) * (fft_m_spectrum))[1:(newN)//2]
# fft_dm_spectrum = fft(daverage_lfp[start_index:end_index])
# fft_dm = np.abs(2.0/(newN) * (fft_dm_spectrum))[1:(newN)//2]
# xf  = np.fft.fftfreq( (newN), d=1/new_Fs)[:(newN)//2]
# frequencies = xf[1:(newN)//2]

# print ('frequencies of interest:', frequencies_of_interest)
demod_totals                 = []  # sum all frequencies per unit time. 
lfp_totals                   = []
interest_frequencies         = []
for i in range(len(frequencies_of_interest)): # sum all frequencies per unit time.
  df_idx = find_nearest(frequencies,frequencies_of_interest[i])
  # also eliminate on either side of bins of interest as I don't have great bin resolution.    
  interest_frequencies.append(df_idx)    
  demod_totals.append(fft_dm[df_idx])
  lfp_totals.append(fft_m[df_idx])
# # print ('demod/lfp totals',lfp_amplitude, demod_amplitude)
# df_cut = np.max(frequencies_of_interest)
end_idx = find_nearest(frequencies,end_frequency)
# end_idx = find_nearest(frequencies,bandwidth_of_interest)
# # print ('end idx',end_idx)
dis_demod_totals            = []  # sum all frequencies per unit time. 
dis_lfp_totals              = []
for i in range(end_idx): # sum all frequencies per unit time.
  if i not in interest_frequencies: 
      dis_demod_totals.append(fft_dm[i])
      dis_lfp_totals.append(fft_m[i])
# # signal to noise calculation. We take the total signal(), and divide it 
lfp_snr   = 20*np.log(np.mean(lfp_totals)/np.mean(dis_lfp_totals))
demod_snr = 20*np.log(np.mean(demod_totals)/np.mean(dis_demod_totals))
# # print ('amplitude carrier: lfp /demod',carrier,lfp_amplitude,demod_amplitude)
print ('final snr lfp /demod: ',np.round(lfp_snr,2),np.round(demod_snr,2) )
# 

start_time = 0
start_times = []
for i in range(periods_of_interest):
    start_times.append((i)*(1.0/prf))
# 
print ('pulse_length',pulse_length)

end_times = []
for i in range(periods_of_interest):
    end_times.append((i)*(1.0/prf)+pulse_length)
# 
# 
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(211)
plt.plot(time_segment,average_lfp,color='k')
plt.plot(time_segment,daverage_lfp,color='g')
plt.legend(['lfp','demod'],loc='lower right',framealpha=0.0)
plt.fill_between(time_segment, average_lfp - error_lb, average_lfp + error_ub,
                  color='gray', alpha=0.2)
plt.fill_between(time_segment, daverage_lfp - derror_lb, daverage_lfp + derror_ub,
                  color='green', alpha=0.2)

# plot the start and end times. 
for i in range(len(start_times) ):
    plt.axvline(x=start_times[i],color ='k')
for i in range(len(end_times) ):
    plt.axvline(x=end_times[i],color ='k')  

ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Time(s)')
plt.autoscale(enable=True, axis='x', tight=True)
ax2 = fig.add_subplot(212)
plt.plot(time_segment,nfiltered_segmented_array.T)
# plt.plot(time_segment,ndemod_segmented_array.T)
# plot the start and end times. 
for i in range(len(start_times) ):
    plt.axvline(x=start_times[i],color ='k')
for i in range(len(end_times) ):
    plt.axvline(x=end_times[i],color ='k')  

# ax2.text(0.5, 100, 'individual trials', color='black',fontsize = 6, ha='center')
plt.autoscale(enable=True, axis='x', tight=True)
ax2.set_ylabel('Individual trials ($\mu$V)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plot_filename = saveprefix +t_series+'_vep_'+str(prf)+'events_'+str(n_events)+'.png'
plt.savefig(plot_filename, transparent=True)
plt.show()
# 
# 
fig = plt.figure(figsize=(5,2))
ax  = fig.add_subplot(111)
plt.plot(time_segment,average_lfp,color='k')
plt.plot(time_segment,daverage_lfp,color='g')
plt.legend(['lfp','demod'],loc='lower right',framealpha=0.0)
plt.fill_between(time_segment, average_lfp - error_lb, average_lfp + error_ub,
                  color='gray', alpha=0.2)
plt.fill_between(time_segment, daverage_lfp - derror_lb, daverage_lfp + derror_ub,
                  color='green', alpha=0.2)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()

plot_filename = saveprefix +'vep_demod_image.png'
plt.savefig(plot_filename, transparent=True)
plt.show()


# sos_low  = iirfilter(17, [3], rs=60, btype='lowpass',
#                        analog=False, ftype='cheby2', fs=new_Fs,
#                        output='sos')
sos_mid_bandpass  = iirfilter(17, [3,26], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=new_Fs,
                       output='sos')
# # 
# # 
# lfp_low          = sosfiltfilt(sos_low , nfiltered_segmented_array)
# demod_low        = sosfiltfilt(sos_low , ndemod_segmented_array)
lfp_band         = sosfiltfilt(sos_mid_bandpass , nfiltered_segmented_array)
demod_band       = sosfiltfilt(sos_mid_bandpass , ndemod_segmented_array)
# # 
# # 
# lfp_low_mean    = np.mean(lfp_low,axis=0)
# m,lci,se         = mean_confidence_interval(lfp_low,confidence=0.95)
# # 
lfp_band_mean    = np.mean(lfp_band,axis=0)
# m,bci,se         = mean_confidence_interval(lfp_band,confidence=0.95)
# # 
# demod_low_mean    = np.mean(demod_low,axis=0)
# m,dlci,se         = mean_confidence_interval(demod_low,confidence=0.95)
# # 
demod_band_mean    = np.mean(demod_band,axis=0)
# m,dbci,se         = mean_confidence_interval(demod_band,confidence=0.95)
# # 
# fig = plt.figure(figsize=(5,2))
# ax  = fig.add_subplot(111)
# plt.plot(time_segment,lfp_low_mean ,color='k')
# plt.plot(time_segment,lfp_band_mean,color='g')
# # plt.legend(['lfp','demod'],loc='lower right',framealpha=0.0)
# plt.fill_between(time_segment, lfp_low_mean - lci, lfp_low_mean + lci,
#                   color='gray', alpha=0.2)
# plt.fill_between(time_segment, lfp_band_mean - bci, lfp_band_mean + bci,
#                   color='green', alpha=0.2)
# # plt.legend(['lfp','demod'],loc='lower right',framealpha=0.0)
# # plt.axvline(x=0,color ='k')
# # plt.axvline(x=0.125,color ='k')
# # plt.axvline(x=0.25,color ='k')
# # plt.axvline(x=0.375,color ='k')
# # plt.axvline(x=0.5,color ='k')
# # plt.axvline(x=0.625,color ='k')
# # plt.axvline(x=0.75,color ='k')
# # plt.axvline(x=0.875,color ='k')
# plt.yticks([])
# plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# plt.tight_layout()
# plot_filename = 'vep_filters.png'
# plt.savefig(plot_filename, transparent=True)
# plt.show()

# fig = plt.figure(figsize=(5,2))
# ax  = fig.add_subplot(111)
# plt.plot(time_segment,demod_low_mean,color='k')
# plt.plot(time_segment,demod_band_mean,color='g')
# # plt.legend(['lfp','demod'],loc='lower right',framealpha=0.0)
# plt.fill_between(time_segment, demod_low_mean - dlci, demod_low_mean + dlci,
#                   color='gray', alpha=0.2)
# plt.fill_between(time_segment, demod_band_mean - dbci, demod_band_mean + dbci,
#                   color='green', alpha=0.2)
# # plt.axvline(x=0,color ='k')
# # plt.axvline(x=0.125,color ='k')
# # plt.axvline(x=0.25,color ='k')
# # plt.axvline(x=0.375,color ='k')
# # plt.axvline(x=0.5,color ='k')
# # plt.axvline(x=0.625,color ='k')
# # plt.axvline(x=0.75,color ='k')
# # plt.axvline(x=0.875,color ='k')
# plt.yticks([])
# plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# plt.tight_layout()
# plot_filename = 'demod_filters.png'
# plt.savefig(plot_filename, transparent=True)
# plt.show()
# # 
# 
# 
from scipy.stats import pearsonr
import pandas as pd
from scipy import signal

x = demod_band_mean/np.max(demod_band_mean)
y = lfp_band_mean/np.max(lfp_band_mean)
print ('len x',len(x))
print ('start/end',start,end)

# remove the start and end of the filtered part to remove the filter only effects. 
# x = x[int(start*new_Fs): int((end - start)*new_Fs) ]
# y = y[int(start*new_Fs): int((end - start)*new_Fs) ]
# start                           = 0.1*(1.0/prf)   # 0.2 seconds
# end                             = (periods_of_interest-1)*(1.0/prf)+0.9*(1.0/prf)

df = pd.DataFrame({'x': x, 'y': y })
window          = len(x)
rolling_corr    = df['x'].rolling(window).corr(df['y'])

correlation = signal.correlate(x-np.mean(x), y - np.mean(y), mode="full")
print ('rolling_corr,correlation',np.mean(correlation) ,rolling_corr[len(x)-1] )

# Now do a correlation metric of the final VEP and demodulated signal. 
# 


