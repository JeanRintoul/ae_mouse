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
c1 ='darkseagreen'

l_cut  = 1 
h_cut  = 20
# 
# 
start = 1
stop  = 11
# stop = 2 
step  = 1 
file_list = range(start,stop,step)
print ('file list',file_list)
saveprefix      = './/images//'
# 
# 
t_series = 't1'
outfile = t_series + '-aevep_data.npz'
# savepath       = 'D:\\ae_mouse\\e135_ae_neural_decoding\\t3_new_mouse_2\\ae_4Hz\\'
# savepath       = 'D:\\ae_mouse\\e135_ae_neural_decoding\\t3_new_mouse_2\\ae_2Hz\\'
# savepath       = 'D:\\ae_mouse\\e135_ae_neural_decoding\\t3_new_mouse_2\\VEP_4Hz_noUS\\'
savepath       = 'D:\\ae_mouse\\e132_ae_neural_recording\\t1_mouse\\AE_PRF80_LED1Hz_g500\\'
# savepath       = 'D:\\ae_mouse\\e132_ae_neural_recording\\t1_mouse\\VEP_1Hz\\'
# savepath       = 'D:\\ae_mouse\\e132_ae_neural_recording\\t1_mouse\\AE_PRF40_LED1Hz_g500\\'
# savepath       = 'D:\\ae_mouse\\e132_ae_neural_recording\\t1_mouse\\AE_PRF40_LED10Hz_g500\\'


Fs              = 5e6
new_Fs          = 1e4
timestep        = 1.0/Fs
duration        = 8
gain            = 500
m_channel       = 0 
rf_channel      = 4
marker_channel  = 7 
pulse_length    = 0.0 # pulse length in seconds. 
N               = int(Fs*duration)

prf             = 0.5
# carrier         = 500000
carrier = 980
# number of periods to look at? must be a sub-multiple of the number of repeats in the file.  
periods_of_interest             = 1
start_time                      = np.round(0.8/duration ,2)
end_time                        = np.round((duration - 0.4)/duration,2)
print ('start and end',start_time,end_time)
downsampling_factor             = int(Fs/new_Fs)
start                           = 0.1*(1.0/prf)   # 0.2 seconds
end                             = (periods_of_interest-1)*(1.0/prf)+0.9*(1.0/prf)
pre_event_idxcount              = int(start*new_Fs)
post_event_idxcount             = int(end*new_Fs)

array_len                       = post_event_idxcount+pre_event_idxcount
nfiltered_segmented_array       = np.zeros((0,array_len))
ndemod_segmented_array          = np.zeros((0,array_len))
print ('array_len',array_len)

# 
# I need to remove the US onset and offset from the data. 
t  = np.linspace(0, duration, N, endpoint=False)
print ('initial N',N)
ss = int(start_time*2*duration*Fs )
ee = int(end_time*duration*Fs-0.5*Fs)
# ss = 1 
# ee = 20000000
t  = t[ss:ee]
N  = ee-ss

print ('ss,ee',ss,ee,N)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   

def demodulate(in_signal,carrier_f,t): 
    return np.abs(in_signal*np.exp(2*np.pi*1j*carrier_f*t))

# confidence interval. 
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m,h,se

sos_low_band = iirfilter(17, [l_cut,h_cut], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
sos_demodulate_band = iirfilter(17, [carrier-h_cut,carrier+h_cut], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
sos_demodulate_stop = iirfilter(17, [carrier-2,carrier+2], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
for n in range(len(file_list)):
    file_number = file_list[n] 
    # 
    print ('file_number',file_number)
    filename    = savepath + 't'+str(file_number)+'_stream.npy'
    data        = np.load(filename)
    fsignal     = (1e6*data[m_channel]/gain)[ss:ee]
    rfsignal    = 10*data[rf_channel][ss:ee]
    marker      = data[marker_channel][ss:ee]
    # 
    # fig  = plt.figure(figsize=(5,2))
    # ax   = fig.add_subplot(111)
    # plt.plot(t,rfsignal,'k') 

    # plt.show()
    # 
    # First filter the data around the carrier. 
    filtered_for_demodulation   = sosfiltfilt(sos_demodulate_band, fsignal)
    # filtered_for_demodulation   = sosfiltfilt(sos_demodulate_stop, filtered_for_demodulation)    
    demodulated_signal          = -demodulate(filtered_for_demodulation,carrier,t)
    # Now filter the demodulated signal between l_cut and h_cut.  
    final_demodulated           = sosfiltfilt(sos_low_band, demodulated_signal)
    # same filter on low frequency neural data. 
    lfp_data                    = sosfiltfilt(sos_low_band,fsignal)

    # Hilbert transform marker, so I can downsample without losing it. 
    e_marker    = hilbert(marker)
    envelope    = np.abs(e_marker)
    sos_lp      = iirfilter(17, [new_Fs/2-1], rs=60, btype='lowpass',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
    c_hilberted             = sosfiltfilt(sos_lp, envelope)  

    # downsample everything.     
    d_basic_marker              = marker[::downsampling_factor]  
    d_hilbert_marker            = c_hilberted[::downsampling_factor]  
    downsampled_lfp             = lfp_data[::downsampling_factor]
    dt                          = t[::downsampling_factor]
    downsampled_demodulated     = final_demodulated[::downsampling_factor] 
    #  Find the alignment indices. 
    dmarker  = d_hilbert_marker/np.max(d_hilbert_marker)
    diffs   = np.diff(dmarker)
    # print ('diffs max: ', np.max(diffs))
    indexes = np.argwhere(diffs > 0.05)[:,0]
    X       = np.insert(indexes, 0, 0)
    m       = np.diff(X)
    # print ('m',m)
    m       = np.argwhere(m > 1000)[:,0]
    alignment_indices = indexes[m]
    alignment_indices = alignment_indices[0::2]
    alignment_indices = alignment_indices[2:len(alignment_indices)-1]
    print ('the alignment_indices are:',alignment_indices)   
    # 
    # fig  = plt.figure(figsize=(10,6))
    # ax   = fig.add_subplot(311)
    # plt.plot(dt,200*d_basic_marker,'g') 
    # plt.plot(dt,downsampled_lfp,'k')
    # plt.plot(dt[alignment_indices],downsampled_lfp[alignment_indices],'.r')
    # ax2  = fig.add_subplot(312)
    # plt.plot(diffs,'k')
    # ax3  = fig.add_subplot(313)    
    # plt.plot(dt,dmarker,'k')    
    # plt.plot(dt,d_basic_marker,'r') 
    # plt.show()
    # 
    print ('alignment_indices', alignment_indices,len(alignment_indices))
    data_to_segment                 = downsampled_lfp
    demod_to_segment                = downsampled_demodulated
    filtered_segmented_data         = []
    demod_segmented_data            = []
    for i in range(len(alignment_indices)):  #
        if i % periods_of_interest == 0:
            # print ('here',i,alignment_indices[i])
            baseline = np.mean(data_to_segment[(alignment_indices[i]-pre_event_idxcount):alignment_indices[i] ])
            # print ('a indices',alignment_indices[i],pre_event_idxcount,post_event_idxcount)
            segment = data_to_segment[(alignment_indices[i]-pre_event_idxcount):(alignment_indices[i]+post_event_idxcount)]-baseline            
            print ('total height of segment:',(np.max(segment)-np.min(segment)))
            if (np.max(segment)-np.min(segment)) < 1000: # rudimentary filter. 
            # print ('length segment',len(segment))
                if len(segment) == array_len: # ensure only full sections are appended.
                    filtered_segmented_data.append(segment)
            else: 
                print ('skipped one too big')
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
    print ('filtered segmented array shape', nfiltered_segmented_array.shape)
# 
# 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,ndemod_segmented_array=ndemod_segmented_array,nfiltered_segmented_array=nfiltered_segmented_array,start = start, end=end)
print ('saved out a data file!')
# 
# 
# 
n_events,b      = nfiltered_segmented_array.shape
time_segment    = np.linspace(-start,end,num=b)
average_lfp     = np.mean(nfiltered_segmented_array,axis=0)
std_lfp         = np.std(nfiltered_segmented_array,axis=0)
sem_lfp         = np.std(nfiltered_segmented_array,axis=0)/np.sqrt(n_events)
m,ci,se         = mean_confidence_interval(nfiltered_segmented_array,confidence=0.95)
lfp_height      = np.max(average_lfp)- np.min(average_lfp)
error_lb        = ci
error_ub        = ci
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
print ('d n_events/ demod LFP size', dn_events,demod_height)

# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(211)
# plt.plot(time_segment,average_lfp,color='r')
# ax2  = fig.add_subplot(212)
# plt.plot(time_segment,demod_average_lfp,color='r')
# plt.show()
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
norm_demod = ndemod_segmented_array/np.max(ndemod_segmented_array)

dn_events,b             = ndemod_segmented_array.shape
norm_demod_average_lfp       = np.mean(norm_demod,axis=0)
norm_demod_std_lfp           = np.std(norm_demod,axis=0)
norm_demod_sem_lfp           = np.std(norm_demod,axis=0)/np.sqrt(dn_events)
m,ndci,se                = mean_confidence_interval(norm_demod,confidence=0.95)
ndemod_height            = np.max(norm_demod_average_lfp)- np.min(norm_demod_average_lfp)
# 
# 
# 
norm_lfp = nfiltered_segmented_array/np.max(nfiltered_segmented_array)
nn_events,b             = norm_lfp.shape
norm_average_lfp       = np.mean(norm_lfp,axis=0)
norm_std_lfp           = np.std(norm_lfp,axis=0)
norm_sem_lfp           = np.std(norm_lfp,axis=0)/np.sqrt(n_events)
m,nnci,se                = mean_confidence_interval(norm_lfp,confidence=0.95)
ndemod_height            = np.max(norm_average_lfp)- np.min(norm_average_lfp)

# fig  = plt.figure(figsize=(5,2))
# ax   = fig.add_subplot(111)
# plt.plot(t,rfsignal,'k') 
# ax.set_xlim([4.033,4.1195])
# plt.xticks([])
# plt.yticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plot_filename = 'rfsignal.png'
# plt.savefig(plot_filename, transparent=True)
# plt.show()

fig = plt.figure(figsize=(5,2))
ax  = fig.add_subplot(111)
plt.fill_between(time_segment, norm_demod_average_lfp - ndci, norm_demod_average_lfp + ndci,
                  color='g', alpha=0.2)
plt.fill_between(time_segment, norm_average_lfp - nnci, norm_average_lfp + nnci,
                  color='grey', alpha=0.2)
plt.plot(time_segment,norm_average_lfp,color='k')
plt.plot(time_segment,norm_demod_average_lfp,color='g')
# plt.plot(time_segment,average_lfp/np.max(average_lfp),color='k')
# plt.plot(time_segment,demod_average_lfp/np.max(demod_average_lfp),color='g')
# plt.legend(['average lfp','demod lfp'],loc='lower left',framealpha=0.0,fontsize=16)
# for i in range(len(start_times) ):
#     plt.axvline(x=start_times[i],color ='k')
# plt.axvline(x=0,color ='k')
# plt.axvline(x=1,color ='k')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks([])
plot_filename = saveprefix+'signal_comparisons.png'
plt.savefig(plot_filename, transparent=True)
plt.show()

# fig = plt.figure(figsize=(5,2))
# ax  = fig.add_subplot(111)
# plt.plot(time_segment,average_lfp/np.max(average_lfp),color='r')
# plt.plot(time_segment,demod_average_lfp/np.max(demod_average_lfp),color='k')
# # plt.legend(['average lfp','demod lfp'],loc='upper right',framealpha=0.0,fontsize=16)
# # for i in range(len(start_times) ):
# #     plt.axvline(x=start_times[i],color ='k')
# plt.axvline(x=0,color ='k')
# plt.axvline(x=1,color ='k')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# # plt.xticks([])
# # plt.yticks([])
# plot_filename = 'signal_comparisons.png'
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



