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

l_cut  = 1 
df_cut = 20
# 
start = 1
stop  = 10

step  = 1 
file_list = range(start,stop,step)

# file_list = np.concatenate( (file_list1,file_list2) )
print ('file list',file_list)
#

t_series = 't2'

outfile = t_series + '-vep_4Hz_shielded.npz'
# 
# savepath      = 'D:\\ae_mouse\\e131_postDAQdisaster\\t4_mouse\\8Hz_VEPs\\'
# savepath       = 'D:\\ae_mouse\\e133_phantom_ae_prf_amplitudes\\t2_mouse\\VEP_1Hz\\'
# savepath     = 'D:\\ae_mouse\\e132_ae_neural_recording\\t1_mouse\\VEP_10Hz\\'
# savepath       = 'D:\\ae_mouse\\e133_phantom_ae_prf_amplitudes\\t2_mouse\\VEP_1Hz_shielded\\'
# 
# savepath       = 'D:\\ae_mouse\\e135_ae_neural_decoding\\t2_new_mouse\\VEP_1Hz\\'
# savepath       = 'D:\\ae_mouse\\e131_postDAQdisaster\\t2_mouse\\VEPS\\'
# savepath       = 'D:\\ae_mouse\\e131_postDAQdisaster\\t2_mouse\\VEPS\\'

savepath       = 'D:\\ae_mouse\\e133_phantom_ae_prf_amplitudes\\t2_mouse\\VEP_4Hz\\'
savepath       = 'D:\\ae_mouse\\e133_phantom_ae_prf_amplitudes\\t2_mouse\\VEP_4Hz_shielded\\'

gain            = 500
m_channel       = 0 
rf_channel      = 4
marker_channel  = 7 
pulse_length    = 0.0 #  pulse length in seconds. 

Fs          = 5e6
timestep    = 1.0/Fs
duration    = 8
N           = int(Fs*duration)
t = np.linspace(0, duration, N, endpoint=False)
              
prf                  = 4
# number of periods to look at? must be a sub-multiple of the number of repeats in the file.  
periods_of_interest  = 1
new_Fs               = 1e5
downsampling_factor  = int(Fs/new_Fs)
# 
# start                         = periods_of_interest*0.2*(1.0/prf)  # 0.2 seconds
# end                           = periods_of_interest*0.8*(1.0/prf)
start                           = 0.1*(1.0/prf)   # 0.2 seconds
end                             = (periods_of_interest-1)*(1.0/prf)+0.9*(1.0/prf)
pre_event_idxcount              = int(start*new_Fs)
post_event_idxcount             = int(end*new_Fs)

array_len = post_event_idxcount+pre_event_idxcount
nfiltered_segmented_array = np.zeros((0,array_len))
print ('array_len',array_len)

for n in range(len(file_list)):
    file_number = file_list[n] 
    # 
    print ('file_number',file_number)
    filename    = savepath + 't'+str(file_number)+'_stream.npy'
    data        = np.load(filename)
    fsignal     = 1e6*data[m_channel]/gain
    rfsignal    = 10*data[rf_channel]  
    marker     = data[marker_channel]
    # 
    # sos_c_band = iirfilter(17, [1e6+500,1e6+5000], rs=60, btype='bandpass',
    #                        analog=False, ftype='cheby2', fs=Fs,
    #                        output='sos')
    # c_data              = sosfiltfilt(sos_c_band, fsignal)
    # 
    # Hilbert transform c_data, so I can later downsample 
    c_signal   = hilbert(marker)
    c_envelope = np.abs(c_signal)
    sos_lp = iirfilter(17, [new_Fs/2], rs=60, btype='lowpass',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
    c_hilberted             = sosfiltfilt(sos_lp, c_envelope)     
    # 
    sos_df_band = iirfilter(17, [l_cut,df_cut], rs=60, btype='bandpass',
                           analog=False, ftype='cheby2', fs=new_Fs,
                           output='sos')
    # 
    # 
    sos_downsampling_band = iirfilter(17, [l_cut,new_Fs/2], rs=60, btype='bandpass',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
    # 
    # #
    # downsample for easier plotting. 
    downsampled_data1            = sosfiltfilt(sos_downsampling_band, fsignal)
    # Now downsample the data. 
    downsampled_data            = downsampled_data1[::downsampling_factor]
    dt                          = t[::downsampling_factor]
    d_c_hilberted               = c_hilberted[::downsampling_factor]
    d_marker               = marker[::downsampling_factor]    
    # 
    marker  = d_c_hilberted/np.max(d_c_hilberted)
    diffs   = np.diff(marker)
    # print ('diffs max: ', np.max(diffs))
    indexes = np.argwhere(diffs > 0.05)[:,0]
    X       = np.insert(indexes, 0, 0)
    m       = np.diff(X)
    # print ('m',m)
    m       = np.argwhere(m > 1000)[:,0]
    alignment_indices = indexes[m]
    print ('the alignment_indices are:',alignment_indices)
    alignment_indices = alignment_indices[1::2]
    # print ('the alignment_indices are:',alignment_indices)   
    # 
    # fig  = plt.figure(figsize=(10,6))
    # ax   = fig.add_subplot(311)
    # plt.plot(dt,downsampled_data,'k')
    # plt.plot(dt[alignment_indices],downsampled_data[alignment_indices],'.r')
    # ax2  = fig.add_subplot(312)
    # plt.plot(diffs,'k')
    # ax3  = fig.add_subplot(313)    
    # plt.plot(dt,d_marker,'k')    
    # plt.show()
    # # 
    #
    # zeroed_fsignal = downsampled_data 
    # for j in range(len(alignment_indices)):
    #     background_mean = np.mean(downsampled_data[(alignment_indices[j]-pre_event_idxcount):alignment_indices[j] ])
    #     # zeroed_fsignal[ (alignment_indices[j]):(alignment_indices[j]+int(1.0*pulse_length*new_Fs) )] = background_mean
    d_df_data = sosfiltfilt(sos_df_band, downsampled_data)    
    # 
    mains_list = [50,100,150]
    for i in range(len(mains_list)):
        sos_mains_stop = iirfilter(17, [mains_list[i]-2 , mains_list[i]+2 ], rs=60, btype='bandstop',
                               analog=False, ftype='cheby2', fs=new_Fs,
                               output='sos')
        d_df_data = sosfiltfilt(sos_mains_stop , d_df_data)

    print ('alignment_indices', alignment_indices,len(alignment_indices))
    # # just take the middle ones as the edges are messed up by the ramp. 
    # alignment_indices = alignment_indices[10:50]
    # print ('subset:',alignment_indices)
    # 
    # 
    # fig  = plt.figure(figsize=(10,6))
    # ax   = fig.add_subplot(311)
    # plt.plot(dt,d_df_data,'k')
    # plt.plot(dt[alignment_indices],d_df_data[alignment_indices],'.r')
    # ax2  = fig.add_subplot(312)
    # plt.plot(diffs,'k')
    # ax3  = fig.add_subplot(313)
    # plt.plot(dt,d_marker,'k') 
    # plt.show()
    
    # # add selection where I select a subsection of these indices.
    # n = int(input("Enter number of elements in indices: "))
    # if n != 0:
    #     subset_indices = list(map(int, input("\nEnter correct alignment indices : ").strip().split()))[:n]
    #     print("\nalignment indices - ", subset_indices,alignment_indices[subset_indices])
    #     alignment_indices = alignment_indices[subset_indices]
    # 
    data_to_segment                 = d_df_data
    filtered_segmented_data         = []
    for i in range(len(alignment_indices)):  #
        # if i >=  1 and i < 4 :    # skip the first and last entry in each file. 
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
    #     
    # print ('len f seg data:', len(filtered_segmented_data))
    if len(filtered_segmented_data)> 0:
        filtered_segmented_array = np.array(filtered_segmented_data)
        nfiltered_segmented_array = np.concatenate((nfiltered_segmented_array, filtered_segmented_array), axis=0)
        print ('filtered segmented array shape', nfiltered_segmented_array.shape)
# 
# 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,nfiltered_segmented_array=nfiltered_segmented_array,start = start, end=end)
print ('saved out a data file!')
# 
# 
# 
n_events,b = nfiltered_segmented_array.shape
time_segment = np.linspace(-start,end,num=b)
average_lfp  = np.mean(nfiltered_segmented_array,axis=0)
std_lfp      = np.std(nfiltered_segmented_array,axis=0)
sem_lfp      = np.std(nfiltered_segmented_array,axis=0)/np.sqrt(n_events)
# confidence interval. 
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m,h,se
# 
m,ci,se = mean_confidence_interval(nfiltered_segmented_array,confidence=0.95)
print ('ci',len(ci))
print ('n_events: ', n_events)
lfp_height   = np.max(average_lfp)- np.min(average_lfp)
print ('LFP size', lfp_height)
# 
error_lb     = ci
error_ub     = ci
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
lfp_height = np.max(average_lfp) - np.min(average_lfp)
print ('lfp height:',lfp_height)
# 
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(211)
plt.plot(time_segment,average_lfp,color='r')
plt.fill_between(time_segment, average_lfp - error_lb, average_lfp + error_ub,
                  color='gray', alpha=0.2)
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
plot_filename = t_series+'_vep_'+str(prf)+'events_'+str(n_events)+'.png'
plt.savefig(plot_filename, transparent=True)
plt.show()




