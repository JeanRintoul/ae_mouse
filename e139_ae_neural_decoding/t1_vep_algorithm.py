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
from scipy.stats import pearsonr
import pandas as pd
from scipy import signal
# 
# 
# Try with a differently generated VEP filter. 
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16
# 
saveprefix  = './/images//'
# 
template_savepath            = 'D:\\ae_mouse\\e139_ae_neural_decoding\\'
template_filename            = 'vep_template_t1.npz'
print ('filename: ', template_filename)
data                         = np.load(template_savepath+template_filename)
template                     = data['template']

l_cut       = 0.5
h_cut       = 40
# # # # # # # # # # # # #
t_series      = 'mouse'
# savepath    = 'D:\\ae_mouse\\e137_ae_neural_decoding\\t4_phantom\\8Hz_15microvolts\\'
# savepath    = 'D:\\ae_mouse\\e137_ae_neural_decoding\\t3_mouse\\ae4hzvep\\'
# savepath    = 'D:\\ae_mouse\\e137_ae_neural_decoding\\t6_mouse_isoflurane\\ae_4hzVEP\\'
# savepath    = 'D:\\ae_mouse\\e137_ae_neural_decoding\\t6_mouse_isoflurane\\ae_2hz_vep\\'
# savepath      = 'D:\\ae_mouse\\e137_ae_neural_decoding\\t6_mouse_isoflurane\\ae_5hzVEP\\'
# savepath      = 'D:\\ae_mouse\\e138_ae_neural_decoding\\t8_mouse\\4hz_aeneural_recording\\'
savepath      = 'D:\\ae_mouse\\e139_ae_neural_decoding\\t1_mouse\\3hz_aevep\\'
# savepath      = 'D:\\ae_mouse\\e139_ae_neural_decoding\\t1_mouse\\3hz_aevep2\\'
# savepath      = 'D:\\ae_mouse\\e139_ae_neural_decoding\\t1_mouse\\2hz_aevep2\\'
# 
match_frequency = 3
# 
start       = 2
stop        = 3
# 
# something wrong with file 11. 
# stop      = 10
# 
step        = 1 
file_list   = range(start,stop,step)
print ('file list',file_list)
outfile     = 'stuff.npz'
conversion_factor = 1
# 
Fs              = 2e6
new_Fs          = 1e4 
timestep        = 1.0/Fs
duration        = 30
gain            = 500
m_channel       = 0 
rf_channel      = 4
marker_channel  = 7 
pulse_length    = 0.0 # pulse length in seconds. 
N               = int(Fs*duration)
prf             = 3
carrier         = 500000
# 
# number of periods to look at? must be a sub-multiple of the number of repeats in the file.  
periods_of_interest             = 6
start_time                      = np.round(0.8/duration,2)
end_time                        = np.round((duration - 0.4)/duration,2)
# 
print ('start and end',start_time,end_time)
downsampling_factor             = int(Fs/new_Fs)
start                           = 0.1*(1.0/prf)   # 0.2 seconds
end                             = (periods_of_interest-1)*(1.0/prf)+0.9*(1.0/prf)
pre_event_idxcount              = int(start*new_Fs)
post_event_idxcount             = int(end*new_Fs)
# 
array_len                       = post_event_idxcount+pre_event_idxcount
nfiltered_segmented_array       = np.zeros((0,array_len))
ndemod_segmented_array          = np.zeros((0,array_len))
# 
match_lfp_segmented_array       = np.zeros((0,array_len))
match_demod_segmented_array     = np.zeros((0,array_len))
print ('array_len',array_len)
# 
# 
start_time  = 0
start_times = []
for i in range(periods_of_interest):
    start_times.append((i)*(1.0/prf))

# I need to remove the US onset and offset from the data. 
t  = np.linspace(0, duration, N, endpoint=False)
print ('initial N',N)
# 
# ss = int(start_time*2*duration*Fs )
# ee = int(end_time*duration*Fs-0.5*Fs)
start_seconds = 1
end_seconds   = duration
# 
ss  = int(start_seconds*Fs)
ee  = int(end_seconds*Fs)
# ss = 1 
tt = t 
# ee = 20000000
t  = t[ss:ee]
N  = ee-ss
print ('ss,ee',ss,ee,N)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   

# confidence interval. 
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m,h,se
# 
sos_low_band = iirfilter(17, [h_cut], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
sos_demodulate_band = iirfilter(17, [carrier-h_cut,carrier+h_cut], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
sos_hilbertlow_band = iirfilter(17, [h_cut], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

#  
for n in range(len(file_list)):

    file_number = file_list[n] 
    print ('file_number',file_number)
    filename    = savepath + 't'+str(file_number)+'_stream.npy'
    data        = np.load(filename)
    #
    # 
    fsignal     = (1e6*data[m_channel]/gain)[ss:ee]
    fsignal     = fsignal - np.mean(fsignal)
    rfsignal    = 10*data[rf_channel][ss:ee]
    marker      = data[marker_channel][ss:ee]
    # 
    # Calculate FFT of RF signal, and also the raw measured signal. 
    # 
    sos_bandstop = iirfilter(17, [carrier-5,carrier+5], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
    stopped_data     = sosfiltfilt(sos_bandstop,fsignal)
    # 
    #     
    rf_1           = int(0)
    rf_2           = int(len(rfsignal)-3*Fs)
    rf_timestep    = 1/Fs
    rfN            = len(rfsignal[rf_1:rf_2])
    # k_window     = np.kaiser( (rfN), rbeta )
    xf = np.fft.fftfreq( (rf_2-rf_1), d=rf_timestep)[:(rf_2-rf_1)//2]
    raw_frequencies = xf[1:(rf_2-rf_1)//2]
    # 
    fft_rf  = fft(rfsignal[rf_1:rf_2])
    fft_rf = np.abs(2.0/(rf_2-rf_1) * (fft_rf))[1:(rf_2-rf_1)//2]
    # 
    fft_raw  = fft(fsignal[rf_1:rf_2])
    fft_raw = np.abs(2.0/(rf_2-rf_1) * (fft_raw))[1:(rf_2-rf_1)//2]
    # 
    # 
    fig  = plt.figure(figsize=(10,6))
    ax   = fig.add_subplot(211)
    # plt.plot(fsignal,'k')  
    ax2  =  fig.add_subplot(212)
    plt.plot(raw_frequencies,fft_raw,'k')
    ax2.set_xlim([0,40])
    ax2.set_ylim([0,50])
    plt.show()
    # 
    # fig  = plt.figure(figsize=(10,6))
    # ax   = fig.add_subplot(211)
    # # plt.plot(tt[rf_1:rf_2],rfsignal[rf_1:rf_2],'k') 
    # # plt.plot(tt[rf_1:rf_2],fsignal[rf_1:rf_2],'k')     
    # # plt.plot(tt[rf_1:rf_2],stopped_data[rf_1:rf_2],'r')    
    # ax2 = fig.add_subplot(212)
    # plt.plot(raw_frequencies,fft_raw,'k')
    # # plt.plot(raw_frequencies,fft_rf,'r')
    # ax2.set_xlim([0,20])
    # ax2.set_ylim([0,50])
    # plt.show()
    # 
    # 
    # First filter the data around the carrier. 
    filtered_for_demodulation = sosfiltfilt(sos_demodulate_band, fsignal)
    analytical_signal       = hilbert(filtered_for_demodulation) # Hilbert demodulate.  
    h_signal                = -np.abs(analytical_signal)
    demodulated_signal      = h_signal -np.mean(h_signal)
    # Now filter the demodulated signal between l_cut and h_cut.  
    final_demodulated       = sosfiltfilt(sos_hilbertlow_band, demodulated_signal)
    # same filter on low frequency neural data. 
    lfp_data                = sosfiltfilt(sos_low_band,fsignal)
    # 
    # Now, do a match filter. 
    # match_frequency = prf*2
    # Try a template that is an 8 Hz sinusoid.     
    template = [0]*int(1.0*Fs) # 1 second 
    for i in range(len(template)):
        template[i] = np.cos( 2*np.pi*(match_frequency)*i*timestep) 
    # 
    # Now, normalize the output between 2:6 seconds. 
    # matching filter implementation
    # print ('sig len',len(template))
    corr_lfp     = signal.correlate(lfp_data/np.max(lfp_data),template, mode='same')
    corr_demod   = signal.correlate(final_demodulated/np.max(final_demodulated),template, mode='same')
    # 
    # normalize just by the middle stable part. 
    norm_start  = int(2*Fs) 
    norm_stop   = int(4*Fs)
    ctn         = corr_lfp[norm_start:norm_stop]
    bigun       = np.max(ctn)
    litun       = np.min(ctn)
    divider     = bigun -litun 
    corr_lfp    = corr_lfp/divider
    # normalize corr demod too. 
    ctn2        = corr_demod[norm_start:norm_stop]
    bigun       = np.max(ctn2)
    litun       = np.min(ctn2)
    divider2    = bigun -litun 
    # print ('bigun',bigun,litun,divider2)
    corr_demod  = corr_demod/divider2
    corr_demod  = conversion_factor*corr_demod
    # 
    # fig  = plt.figure(figsize=(10,6))
    # ax   = fig.add_subplot(411)
    # plt.plot(t,lfp_data,'k')  
    # ax.set_ylim([-100,100])
    # plt.legend([' LFP'],loc='upper right')
    # ax2  = fig.add_subplot(412)
    # plt.plot(t,lfp_data/np.max(lfp_data),'r') 
    # # plt.plot(t,corr_lfp,'g') 
    # plt.plot(t,corr_demod,'k')
    # # ax2.set_ylim([-1,1])
    # plt.legend(['LFP','match demod'],loc='upper right')
    # ax3  = fig.add_subplot(413)    
    # plt.plot(t,corr_demod,'k')
    # # plt.plot(t,corr_demod,'r')
    # # ax3.set_ylim([-1,1])
    # plt.legend(['Match Filter DEMOD'],loc='upper right')
    # ax4  = fig.add_subplot(414)    
    # plt.plot(t,demodulated_signal,'r')
    # # ax4.set_ylim([-20,20])
    # plt.legend(['Hilbert demod'],loc='upper right')
    # plt.show()
    # 
    # 
    # Hilbert transform marker, so I can downsample without losing it. 
    e_marker    = hilbert(marker)
    envelope    = np.abs(e_marker)
    sos_lp      = iirfilter(17, [new_Fs/2-1], rs=60, btype='lowpass',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
    c_hilberted = sosfiltfilt(sos_lp, envelope)  
    # 
    # downsample everything.     
    d_basic_marker              = marker[::downsampling_factor]  
    d_hilbert_marker            = c_hilberted[::downsampling_factor]  
    downsampled_lfp             = lfp_data[::downsampling_factor]
    dt                          = t[::downsampling_factor]
    drf                         = rfsignal[::downsampling_factor]
    downsampled_demodulated     = final_demodulated[::downsampling_factor] 
    corr_lfp                    = corr_lfp[::downsampling_factor]
    corr_demod                  = corr_demod[::downsampling_factor]
    # Find the alignment indices. 
    dmarker  = d_hilbert_marker/np.max(d_hilbert_marker)
    diffs   = np.diff(dmarker)
    # print ('diffs max: ', np.max(diffs),diffs)
    # indexes = np.argwhere(diffs > 0.1)[:,0]
    indexes = np.argwhere(dmarker > 0.6)[:,0]
    X       = np.insert(indexes, 0, 0)
    m       = np.diff(X)
    # print ('m',m)
    m                 = np.argwhere(m > 500)[:,0]
    alignment_indices = indexes[m]
    alignment_indices = alignment_indices[0::2]
    alignment_indices = alignment_indices[2:len(alignment_indices)-2]
    # print ('the alignment_indices are:',alignment_indices)   
    # 
    # at 5MHz we have. 
    # fsignal, and the alignment_indices* downsampling_factor
    #     
    fig  = plt.figure(figsize=(10,6))
    ax   = fig.add_subplot(311)
    plt.plot(dt,200*d_basic_marker,'g') 
    plt.plot(dt,downsampled_lfp,'k')
    # plt.plot(dt,corr_demod,'b')
    plt.plot(dt[alignment_indices],downsampled_lfp[alignment_indices],'.r')
    ax2  = fig.add_subplot(312)
    plt.plot(t,rfsignal,'k')
    ax3  = fig.add_subplot(313)    
    plt.plot(dt,dmarker,'k')    
    # plt.plot(dt,d_basic_marker,'r') 
    # plt.plot(dt,diffs,'b')
    plt.show()
    # # 
    # print ('alignment_indices', alignment_indices,len(alignment_indices))
    data_to_segment                 = downsampled_lfp
    demod_to_segment                = downsampled_demodulated
    # corr_lfp    
    # corr_demod 
    filtered_segmented_data         = []
    demod_segmented_data            = []
    corr_lfp_segmented_data         = []
    corr_demod_segmented_data       = []    
    for i in range(len(alignment_indices)):  #
        if i % periods_of_interest == 0:
            
            # first add each segment together for later averaging to achieving better noise removal levels. 
            rawdata_segment = fsignal[(downsampling_factor*alignment_indices[i]-pre_event_idxcount*downsampling_factor):(downsampling_factor*alignment_indices[i]+post_event_idxcount*downsampling_factor)]         
            if len(rawdata_segment) == array_len*downsampling_factor:
                if n == 0:
                    rawdata_summation = rawdata_segment
                else: 
                    rawdata_summation = rawdata_summation + rawdata_segment
                print ('len raw data segment: ',len(rawdata_segment))
            # 
            # 
            # print ('here',i,alignment_indices[i])
            baseline = np.mean(data_to_segment[(alignment_indices[i]-pre_event_idxcount):alignment_indices[i] ])
            # print ('a indices',alignment_indices[i],pre_event_idxcount,post_event_idxcount)
            segment = data_to_segment[(alignment_indices[i]-pre_event_idxcount):(alignment_indices[i]+post_event_idxcount)] -baseline            
            print ('total height of segment:',(np.max(segment)-np.min(segment)))
            # if (np.max(segment)-np.min(segment)) < 1000: # rudimentary filter. 
            # print ('length segment',len(segment))
            if len(segment) == array_len: # ensure only full sections are appended.
                filtered_segmented_data.append(segment)
            # else: 
            #     print ('skipped one too big, or segment wrong size')
            # 
            # do it now for the demodulated data. 
            # demod_baseline = np.mean(demod_to_segment[(alignment_indices[i]-pre_event_idxcount):alignment_indices[i] ])
            # print ('a indices',alignment_indices[i],pre_event_idxcount,post_event_idxcount)
            demod_segment = demod_to_segment[(alignment_indices[i]-pre_event_idxcount):(alignment_indices[i]+post_event_idxcount)]# -demod_baseline            
            if len(demod_segment) == array_len: # ensure only full sections are appended.
                demod_segmented_data.append(demod_segment )
            # do it now for corr_lfp
            # corr_lfp_baseline = np.mean(corr_lfp[(alignment_indices[i]-pre_event_idxcount):alignment_indices[i] ])
            # print ('a indices',alignment_indices[i],pre_event_idxcount,post_event_idxcount)
            corr_lfp_segment = corr_lfp[(alignment_indices[i]-pre_event_idxcount):(alignment_indices[i]+post_event_idxcount)]# -corr_lfp_baseline            
            if len(corr_lfp_segment) == array_len: # ensure only full sections are appended.
                corr_lfp_segmented_data.append(corr_lfp_segment )
            # do it now for corr_demod
            # corr_demod_baseline = np.mean(corr_demod[(alignment_indices[i]-pre_event_idxcount):alignment_indices[i] ])
            # print ('a indices',alignment_indices[i],pre_event_idxcount,post_event_idxcount)
            corr_demod_segment = corr_demod[(alignment_indices[i]-pre_event_idxcount):(alignment_indices[i]+post_event_idxcount)]# -corr_demod_baseline            
            if len(corr_demod_segment) == array_len: # ensure only full sections are appended.
                corr_demod_segmented_data.append(corr_demod_segment)
    # 
    a = len(corr_demod_segmented_data)
    print ('shape of array',a)  
    if a > 0:             # only calculate when a > 0 .            
        demod_segmented_array = np.array(demod_segmented_data)
        ndemod_segmented_array = np.concatenate((ndemod_segmented_array, demod_segmented_array), axis=0)
        # print ('len f seg data:', len(filtered_segmented_data))
        filtered_segmented_array = np.array(filtered_segmented_data)
        nfiltered_segmented_array = np.concatenate((nfiltered_segmented_array, filtered_segmented_array), axis=0)
        # corr_demod_segmented_data
        # corr_lfp_segmented_data
        match_lfp_segmented_array       = np.concatenate((match_lfp_segmented_array,-np.array(corr_lfp_segmented_data)),axis=0)
        match_demod_segmented_array     = np.concatenate((match_demod_segmented_array,-np.array(corr_demod_segmented_data)),axis=0)
        print ('filtered segmented array shape', nfiltered_segmented_array.shape)
# 
# 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,match_lfp_segmented_array=match_lfp_segmented_array,match_demod_segmented_array=match_demod_segmented_array,ndemod_segmented_array=ndemod_segmented_array,nfiltered_segmented_array=nfiltered_segmented_array,start = start, end=end)
print ('saved out a data file!')
# 
# FFT of raw data calculation, to look at the peaks around 500kHz. 
# start_seconds = 1
# end_seconds   = 10
rbeta          = 20
raw_timestep   = 1/Fs 
# 
s_a           = int(0)
e_a           = int(len(rawdata_summation))
rawN        = len(rawdata_summation)
raw_window      = np.kaiser( (rawN), rbeta )
xf = np.fft.fftfreq( (e_a-s_a), d=raw_timestep)[:(e_a-s_a)//2]
raw_frequencies = xf[1:(e_a-s_a)//2]
fft_raw  = fft(rawdata_summation[s_a:e_a]*raw_window)
fft_raw  = np.abs(2.0/(e_a-s_a) * (fft_raw))[1:(e_a-s_a)//2]
# 
# 
fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(111)
plt.plot(raw_frequencies,fft_raw,'k')
ax.set_xlim([carrier-h_cut,carrier+h_cut])
ax.set_ylim([0,5])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()
# 
#       
# 
n_events,b      = nfiltered_segmented_array.shape
time_segment    = np.linspace(-start,end,num=b)
average_lfp     = -np.mean(nfiltered_segmented_array,axis=0)
std_lfp         = np.std(nfiltered_segmented_array,axis=0)
sem_lfp         = np.std(nfiltered_segmented_array,axis=0)/np.sqrt(n_events)
m,ci,se         = mean_confidence_interval(nfiltered_segmented_array,confidence=0.95)
lfp_height      = np.max(average_lfp)- np.min(average_lfp)
error_lb        = ci
error_ub        = ci
error_lb        = 4
error_ub        = 5
print ('n_events/ LFP size', n_events, lfp_height)
# 
# 
dn_events,b             = ndemod_segmented_array.shape
time_segment            = np.linspace(-start,end,num=b)
demod_average_lfp       = np.mean(ndemod_segmented_array,axis=0)
demod_std_lfp           = np.std(ndemod_segmented_array,axis=0)
demod_sem_lfp           = np.std(ndemod_segmented_array,axis=0)/np.sqrt(n_events)
m,dci,se                = mean_confidence_interval(ndemod_segmented_array,confidence=0.95)
demod_height            = np.max(demod_average_lfp)- np.min(demod_average_lfp)
derror_lb                = dci
derror_ub                = dci
print ('d n_events/ demod LFP size', dn_events, demod_height)
# 
# match filter outputs of the lfp. 
match_lfp_n_events,b    = match_lfp_segmented_array.shape
time_segment            = np.linspace(-start,end,num=b)
match_average_lfp       = np.mean(match_lfp_segmented_array,axis=0)
match_std_lfp           = np.std(match_lfp_segmented_array,axis=0)
match_sem_lfp           = np.std(match_lfp_segmented_array,axis=0)/np.sqrt(match_lfp_n_events)
m,match_lfp_ci,se       = mean_confidence_interval(match_lfp_segmented_array,confidence=0.95)
match_lfp_height        = np.max(match_average_lfp)- np.min(match_average_lfp)
match_lfp_error_lb      = match_lfp_ci
match_lfp_error_ub      = match_lfp_ci
print ('match n_events: ', match_lfp_n_events)
# 
# match_demod_segmented_array
match_demod_n_events,b    = match_demod_segmented_array.shape
time_segment              = np.linspace(-start,end,num=b)
match_average_demod       = np.mean(match_demod_segmented_array,axis=0)
match_std_demod           = np.std(match_demod_segmented_array,axis=0)
match_sem_demod           = np.std(match_demod_segmented_array,axis=0)/np.sqrt(match_demod_n_events)
m,match_demod_ci,se       = mean_confidence_interval(match_demod_segmented_array,confidence=0.95)
match_demod_height        = np.max(match_average_demod)- np.min(match_average_demod)
match_demod_error_lb      = match_demod_ci
match_demod_error_ub      = match_demod_ci
print ('match demod n_events', match_demod_n_events)
# 
# 
start_time = 0
start_times = []
for i in range(periods_of_interest):
    start_times.append((i)*(1.0/prf))
# 
end_times = []
for i in range(periods_of_interest):
    end_times.append((i)*(1.0/prf)+pulse_length)
# 
# 
fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(211)
plt.plot(time_segment,average_lfp,color='r')
plt.fill_between(time_segment, average_lfp - error_lb, average_lfp + error_ub,
                  color='gray', alpha=0.2)
# plot the start and end times. 
for i in range(len(start_times) ):
    plt.axvline(x=start_times[i],color ='k')
# for i in range(len(end_times) ):
#     plt.axvline(x=end_times[i],color ='k')  
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Time(s)')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(['lfp'],loc='upper right')

# ax2 = fig.add_subplot(412)
# # plot the start and end times. 
# for i in range(len(start_times) ):
#     plt.axvline(x=start_times[i],color ='k')
# plt.plot(time_segment,match_average_lfp,color='r')
# plt.fill_between(time_segment, match_average_lfp - match_lfp_ci, match_average_lfp + match_lfp_ci,
#                   color='gray', alpha=0.2)
# plt.autoscale(enable=True, axis='x', tight=True)
# ax2.set_ylabel('Volts ($\mu$V)')
# ax2.set_xlabel('Time(s)')
# plt.legend(['match lfp'],loc='upper right')

# ax3 = fig.add_subplot(413)
# plt.plot(time_segment,demod_average_lfp,color='r')
# plt.fill_between(time_segment, demod_average_lfp - derror_lb, demod_average_lfp + derror_ub,
#                   color='gray', alpha=0.2)
# # plot the start and end times. 
# for i in range(len(start_times) ):
#     plt.axvline(x=start_times[i],color ='k')
# # for i in range(len(end_times) ):
# #     plt.axvline(x=end_times[i],color ='k')  
# ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xlabel('Time(s)')
# plt.autoscale(enable=True, axis='x', tight=True)
# plt.legend(['demod'],loc='upper right')

ax2 = fig.add_subplot(212)
plt.plot(time_segment,match_average_demod)
plt.fill_between(time_segment, match_average_demod - match_demod_ci, match_average_demod + match_demod_ci,
                  color='gray', alpha=0.2)
# 
# plot the start and end times. 
for i in range(len(start_times) ):
    plt.axvline(x=start_times[i],color ='k')
# for i in range(len(end_times) ):
#     plt.axvline(x=end_times[i],color ='k')  
# ax2.text(0.5, 100, 'individual trials', color='black',fontsize = 6, ha='center')
plt.autoscale(enable=True, axis='x', tight=True)
ax2.set_ylabel('Volts ($\mu$V)')
ax2.set_xlabel('Time(s)')
plt.legend(['match demod'],loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# ax4.spines['right'].set_visible(False)
# ax4.spines['top'].set_visible(False)
plot_filename = t_series+'_vep_events_'+str(n_events)+'.png'
plt.savefig(plot_filename, transparent=True)
plt.show()
# 
# 
start_pause = int(0)
end_pause   = int(len(average_lfp))
timestep    = 1/new_Fs
xf          = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
beta        = 20
window      = np.kaiser( (end_pause-start_pause), beta)
# 
fft_avlfp   = fft(average_lfp[start_pause:end_pause])
fft_avlfp   = np.abs(2.0/(end_pause-start_pause) * (fft_avlfp))[1:(end_pause-start_pause)//2]
# 
fft_davlfp   = fft(demod_average_lfp[start_pause:end_pause]*window)
fft_davlfp   = np.abs(2.0/(end_pause-start_pause) * (fft_davlfp))[1:(end_pause-start_pause)//2]
# 
fft_mdavlfp   = fft(match_average_demod[start_pause:end_pause]*window)
fft_mdavlfp   = np.abs(2.0/(end_pause-start_pause) * (fft_mdavlfp))[1:(end_pause-start_pause)//2]
# 
# 
# Correlations between final results. 
# Correctly compute the lag between the two signals. 
x = match_average_demod/np.max(match_average_demod)
y = average_lfp/np.max(average_lfp)
print ('len x',len(x))
print ('start/end',start,end)
x = x - np.mean(x)
x = x/np.max(x)
y = y - np.mean(y)
y = y/np.max(y)

correlation = signal.correlate(x, y, mode="full")
lags = signal.correlation_lags(len(x), len(y), mode="full")
idx_lag = -lags[np.argmax(correlation)]
# print ('lag is',idx_lag)
# print ('match demod: ',match_average_demod.shape)
#
if idx_lag > 0:
    matched_subsection = match_demod_segmented_array[:,:(len(x)-idx_lag)]
    x = x[:(len(x)-idx_lag)]
    y = y[idx_lag:]
    t_cut1 = time_segment[idx_lag:]

else: 
    x = x[-idx_lag:]
    matched_subsection = match_demod_segmented_array[:,-idx_lag:]
    y = y[:(len(y)+idx_lag)]
    t_cut1 = time_segment[-idx_lag:]
#  
# print ('x,y, lengths: ',len(x),len(y))

df = pd.DataFrame({'x': x, 'y': y })
window          = len(x)
rolling_corr    = df['x'].rolling(window).corr(df['y'])
print ('match filter rolling corr max:',np.max(rolling_corr))  # 
result = np.nanmedian(rolling_corr)
print ('match filter median corr: ',result)
max_index = np.argmax(rolling_corr) 
#  
# 
# print ('lens ',len(t_cut1),len(x),len(y)) 
# calculate the error. 
m,match_ci,se       = mean_confidence_interval(matched_subsection,confidence=0.95)

fig = plt.figure(figsize=(4,2))
ax  = fig.add_subplot(111)
plt.fill_between(t_cut1, x - match_ci, x + match_ci,
                  color='r', alpha=0.2)
plt.plot(t_cut1,y,color='k')
plt.plot(t_cut1,x,color='r')
# for i in range(len(start_times) ):
#     plt.axvline(x=start_times[i],color ='k')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title('match demod vs lfp result')
plt.tight_layout()
plot_filename = saveprefix + 'joint_match_plot.png'
plt.savefig(plot_filename)
plt.show()

# 
# 
x   = demod_average_lfp/np.max(demod_average_lfp)
y   = average_lfp/np.max(average_lfp)
x   = x - np.mean(x)
x   = x/np.max(x)
y   = y-np.mean(y)
y   = y/np.max(y)
# 
# len_x = len(x)
correlation = signal.correlate(x, y, mode="full")
lags = signal.correlation_lags(len(x), len(y), mode="full")
idx_lag = -lags[np.argmax(correlation)]
print ('lag is',idx_lag)
demodx = demod_average_lfp
# 
# 
# print ('demod array: ',ndemod_segmented_array.shape)
# 
if idx_lag > 0:
    demodded = ndemod_segmented_array[:,:(len(x)-idx_lag)]
    x        = x[:(len(x)-idx_lag)]
    y        = y[idx_lag:]
    t_cut    = time_segment[idx_lag:]
else: 
    x = x[-idx_lag:]
    demodded = ndemod_segmented_array[:,-idx_lag:]
    y = y[:(len(y)+idx_lag)]
    t_cut = time_segment[-idx_lag:]
#  
# print ('x,y,t_cut lengths: ',len(x),len(y),len(t_cut))

df = pd.DataFrame({'x': x, 'y': y })
window          = len(x)
rolling_corr    = df['x'].rolling(window).corr(df['y'])
# print ('rolling corr max:',np.max(rolling_corr))  # 
result = np.nanmedian(rolling_corr)
# print ('filter corr: ',result)
max_index = np.argmax(rolling_corr) 

# calculate the error. 
m,demodded_ci,se       = mean_confidence_interval(demodded,confidence=0.95)
# print ('demodded ci:',len(demodded_ci) )

fig = plt.figure(figsize=(4,2))
ax  = fig.add_subplot(111)
plt.fill_between(t_cut, x - demodded_ci, x + demodded_ci,
                  color='r', alpha=0.2)
plt.plot(t_cut,y,color='k')
plt.plot(t_cut,x,color='r')
# plt.fill_between(t_cut, x - demodded_ci, x + demodded_ci,
#                   color='r', alpha=0.2)
# for i in range(len(start_times) ):
#     plt.axvline(x=start_times[i],color ='k')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title('demod(red) vs LFP')
plot_filename = saveprefix+ 'joint_demod_plot.png'
plt.savefig(plot_filename)
plt.show()

# fig = plt.figure(figsize=(4,2))
# ax  = fig.add_subplot(111)
# plt.fill_between(t_cut, x - demodded_ci, x + demodded_ci,
#                   color='r', alpha=0.2)
# plt.plot(t_cut,y,color='k')
# plt.plot(t_cut,x,color='r')
# # plt.fill_between(t_cut, x - demodded_ci, x + demodded_ci,
# #                   color='r', alpha=0.2)

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.xticks([])
# plt.yticks([])
# plot_filename = saveprefix+ 'joint_demod_plot.png'
# plt.savefig(plot_filename)
# plt.show()
# 
# 
# fig = plt.figure(figsize=(4,2))
# ax  = fig.add_subplot(111)
# plt.plot(x,'k')
# plt.plot(y,'r')
# plot_filename = saveprefix+ 'signals_going_into_correlation.png'
# plt.savefig(plot_filename)
# plt.show()
# 
# 
df  = pd.DataFrame({'x': x, 'y': y })
window          = len(x)
rolling_corr    = df['x'].rolling(window).corr(df['y'])
correlation     = signal.correlate(x-np.mean(x), y - np.mean(y), mode="full")
print ('H demod correlation',rolling_corr[len(x)-1] )


# Now determine the signal to noise ratio of each of the FFTs. 
# SNR calculation for both kaiser window 
frequencies_of_interest         = [prf]
# If it looks bad for some reason? For now, just note down the file number.     
signal_totals                   = []
hilbert_signal_totals           = []
match_signal_totals             = []
interest_frequencies            = []
for n in range(len(frequencies_of_interest)): # sum all frequencies per unit time.
  df_idx = find_nearest(frequencies,frequencies_of_interest[n])
  # also eliminate on either side of bins of interest as I don't have great bin resolution.    
  interest_frequencies.append(df_idx)    
  signal_totals.append(fft_avlfp[df_idx])
  hilbert_signal_totals.append(fft_davlfp[df_idx])
  match_signal_totals.append(fft_mdavlfp[df_idx])
# 
# This is the start and end point to calculate the SNR from. 
start_idx = find_nearest(frequencies,3 )   # after the first 5Hz. 
end_idx = find_nearest(frequencies,h_cut)
# print ('end idx',end_idx)
dis_signal_totals              = []  # 
dis_hilbert_signal_totals      = []
dis_match_signal_totals        = []
for n in range(start_idx, end_idx):  # sum all frequencies per unit time.
    if n not in interest_frequencies: 
        dis_signal_totals.append(fft_avlfp[n])
        dis_hilbert_signal_totals.append(fft_davlfp[n])
        dis_match_signal_totals.append(fft_mdavlfp[n])
#           
signal_snr  = 20*np.log(np.mean(signal_totals)/np.mean(dis_signal_totals))
hilbert_snr = 20*np.log(np.mean(hilbert_signal_totals)/np.mean(dis_hilbert_signal_totals))
match_snr   = 20*np.log(np.mean(match_signal_totals)/np.mean(dis_match_signal_totals))
print ('signal snr:',signal_snr,hilbert_snr,match_snr)




# Compute the cross-correlation between LFP and segment from x->x 
# 
# corr_demod   = signal.correlate(final_demodulated/np.max(final_demodulated),template, mode='same')
# Calculate the FFT between 2-6 seconds. 
# 
fig = plt.figure(figsize=(4,3))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_avlfp,'k')
ax.set_xlim([0,h_cut])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.title('lfp fft')
plt.tight_layout()
plot_filename = saveprefix+ 'average_lfp_fft.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(4,3))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_davlfp,'k')
ax.set_xlim([0,h_cut])
ax.set_ylim([0,0.2])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.title('demod fft')
plt.tight_layout()
plot_filename = saveprefix+ 'daverage_lfp_fft.png'
plt.savefig(plot_filename)
plt.show()


fig = plt.figure(figsize=(4,3))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_mdavlfp,'k')
ax.set_xlim([0,h_cut])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.title('match demod fft')
plt.tight_layout()

plot_filename = saveprefix+ 'mdaverage_lfp_fft.png'
plt.savefig(plot_filename)
plt.show()
# 
# 
# fig = plt.figure(figsize=(4,2))
# ax  = fig.add_subplot(111)
# plt.plot(time_segment,average_lfp,color='k')
# plt.fill_between(time_segment, average_lfp - error_lb, average_lfp + error_ub,
#                   color='gray', alpha=0.2)
# # ax.set_xlim([0.5,1.5])
# # ax.set_ylim([-200,200])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.xticks([])
# plt.yticks([])
# plot_filename = saveprefix+ 'average_lfp.png'
# plt.savefig(plot_filename)
# plt.show()
# # 
fig = plt.figure(figsize=(4,2))
ax  = fig.add_subplot(111)
plt.fill_between(time_segment, demod_average_lfp - derror_lb, demod_average_lfp + derror_ub,
                  color='r', alpha=0.2)
plt.plot(time_segment,average_lfp/34 - 7, color='k')
plt.plot(time_segment,demod_average_lfp, color='r')
# ax.set_xlim([0.5,1.5])
# ax.set_ylim([-8.5,-4.8])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks([])
plot_filename = saveprefix+ 'average_demod.png'
plt.savefig(plot_filename)
plt.show()
# # 
# fig = plt.figure(figsize=(4,2))
# ax  = fig.add_subplot(111)
# plt.plot(time_segment,match_average_demod,color='r')
# plt.fill_between(time_segment, match_average_demod - match_demod_ci, match_average_demod + match_demod_ci,
#                   color='r', alpha=0.2)
# # ax.set_xlim([0.5,1.5])
# # ax.set_ylim([-550,550])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.xticks([])
# plt.yticks([])
# plot_filename = saveprefix+ 'match_average_demod.png'
# plt.savefig(plot_filename)
# plt.show()
# 
# 
# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(211)
# plt.plot(time_segment,demod_average_lfp,color='r')
# plt.fill_between(time_segment, demod_average_lfp - derror_lb, demod_average_lfp + derror_ub,
#                   color='gray', alpha=0.2)
# # plot the start and end times. 
# for i in range(len(start_times) ):
#     plt.axvline(x=start_times[i],color ='k')
# # for i in range(len(end_times) ):
# #     plt.axvline(x=end_times[i],color ='k')  
# ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xlabel('Time(s)')
# plt.autoscale(enable=True, axis='x', tight=True)
# ax2 = fig.add_subplot(212)
# plt.plot(time_segment,match_average_demod)
# plt.fill_between(time_segment, match_average_demod - match_demod_ci, match_average_demod + match_demod_ci,
#                   color='gray', alpha=0.2)
# # 
# # plot the start and end times. 
# for i in range(len(start_times) ):
#     plt.axvline(x=start_times[i],color ='k')
# # for i in range(len(end_times) ):
# #     plt.axvline(x=end_times[i],color ='k')  
# # ax2.text(0.5, 100, 'individual trials', color='black',fontsize = 6, ha='center')
# plt.autoscale(enable=True, axis='x', tight=True)
# ax2.set_ylabel('Volts ($\mu$V)')
# ax2.set_xlabel('Time(s)')
# # 
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = t_series+'_demod_vep_'+str(n_events)+'.png'
# plt.savefig(plot_filename, transparent=True)
# plt.show()

# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(211)
# plt.plot(time_segment,average_lfp,color='r')
# plt.fill_between(time_segment, average_lfp - error_lb, average_lfp + error_ub,
#                   color='gray', alpha=0.2)
# # plot the start and end times. 
# for i in range(len(start_times) ):
#     plt.axvline(x=start_times[i],color ='k')
# for i in range(len(end_times) ):
#     plt.axvline(x=end_times[i],color ='k')  
# ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xlabel('Time(s)')
# plt.autoscale(enable=True, axis='x', tight=True)
# ax2 = fig.add_subplot(212)
# plt.plot(time_segment,nfiltered_segmented_array.T)
# # plot the start and end times. 
# for i in range(len(start_times) ):
#     plt.axvline(x=start_times[i],color ='k')
# for i in range(len(end_times) ):
#     plt.axvline(x=end_times[i],color ='k')  
# # ax2.text(0.5, 100, 'individual trials', color='black',fontsize = 6, ha='center')
# plt.autoscale(enable=True, axis='x', tight=True)
# ax2.set_ylabel('Individual trials ($\mu$V)')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = t_series+'_vep_events_'+str(n_events)+'.png'
# plt.savefig(plot_filename, transparent=True)
# plt.show()
# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(211)
# plt.plot(time_segment,demod_average_lfp,color='r')
# plt.fill_between(time_segment, demod_average_lfp - derror_lb, demod_average_lfp + derror_ub,
#                   color='gray', alpha=0.2)
# # plot the start and end times. 
# for i in range(len(start_times) ):
#     plt.axvline(x=start_times[i],color ='k')
# # for i in range(len(end_times) ):
# #     plt.axvline(x=end_times[i],color ='k')  
# ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xlabel('Time(s)')
# plt.autoscale(enable=True, axis='x', tight=True)
# ax2 = fig.add_subplot(212)
# plt.plot(time_segment,ndemod_segmented_array.T)
# # plot the start and end times. 
# # for i in range(len(start_times) ):
# #     plt.axvline(x=start_times[i],color ='k')
# for i in range(len(end_times) ):
#     plt.axvline(x=end_times[i],color ='k')  
# # ax2.text(0.5, 100, 'individual trials', color='black',fontsize = 6, ha='center')
# plt.autoscale(enable=True, axis='x', tight=True)
# ax2.set_ylabel('Individual trials ($\mu$V)')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = t_series+'_demod_vep_'+str(n_events)+'.png'
# plt.savefig(plot_filename, transparent=True)
# plt.show()



