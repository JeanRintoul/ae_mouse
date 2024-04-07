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

l_cut  = 0.5
df_cut = 100
# 
start = 1
stop  = 10

step  = 1 
file_list = range(start,stop,step)

# file_list = np.concatenate( (file_list1,file_list2) )
print ('file list',file_list)
#

t_series = 't1'

outfile = t_series + '-vep_data.npz'
# 
# savepath    = 'D:\\ae_mouse\\e131_postDAQdisaster\\t2_mouse\\VEPS\\'
# gain        = 500
def demodulate(in_signal,carrier_f,dt): 
    idown = in_signal*np.cos(2*np.pi*carrier_f*dt)
    qdown = in_signal*np.sin(2*np.pi*carrier_f*dt)   
    demodulated_signal = idown + 1j*qdown
    return -np.abs(demodulated_signal)

# savepath    = 'D:\\ae_mouse\\e131_postDAQdisaster\\t2_mouse\\VEP_mix_500Hz_1HzLED\\'
# savepath    = 'D:\\ae_mouse\\e131_postDAQdisaster\\t3_mouse\\vep_mix_500Hz_8HzLED\\'
savepath    = 'D:\\ae_mouse\\e131_postDAQdisaster\\t4_mouse\\ae_r_prf520_led4hz\\'
savepath    = 'D:\\ae_mouse\\e131_postDAQdisaster\\t4_mouse\\ae_r_prf120_led4hz\\'
# savepath    = 'D:\\ae_mouse\\e131_postDAQdisaster\\t4_mouse\\ae_r_prf1020_led4hz\\'
# savepath    = 'D:\\ae_mouse\\e131_postDAQdisaster\\t4_mouse\\ae_r_cont_led4hz\\'



gain            = 500
m_channel       = 0 
rf_channel      = 4
marker_channel  = 7 
carrier         = 120
pulse_length    = 0.0 #  pulse length in seconds. 

Fs          = 5e6
timestep    = 1.0/Fs
duration    = 6
N           = int(Fs*duration)
t = np.linspace(0, duration, N, endpoint=False)
              
prf                  = 1
# number of periods to look at? must be a sub-multiple of the number of repeats in the file.  
periods_of_interest  = 1
new_Fs               = 1e5

downsampling_factor  = int(Fs/new_Fs)
print('downsampling factor: ',downsampling_factor)
# 
# start                         = periods_of_interest*0.2*(1.0/prf)  # 0.2 seconds
# end                           = periods_of_interest*0.8*(1.0/prf)
start                           = 0.25*(1.0/prf)   # 0.2 seconds
end                             = (periods_of_interest-1)*(1.0/prf)+0.75*(1.0/prf)
pre_event_idxcount              = int(start*new_Fs)
post_event_idxcount             = int(end*new_Fs)

array_len = post_event_idxcount+pre_event_idxcount
nfiltered_segmented_array = np.zeros((0,array_len))
demod_segmented_array = np.zeros((0,array_len))
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
    # downsampled_data            = fsignal[::downsampling_factor]
    dt                          = t[::downsampling_factor]
    d_c_hilberted               = c_hilberted[::downsampling_factor]
    d_marker                    = marker[::downsampling_factor]    
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

    # create a demodulated signal. 
    lowm  = carrier - df_cut
    highm = carrier + df_cut
    modulation_filter = iirfilter(17, [lowm,highm], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=new_Fs,
                       output='sos')
    # modulation_filter2 = iirfilter(17, [200,10000], rs=60, btype='bandpass',
    #                    analog=False, ftype='cheby2', fs=new_Fs,
    #                    output='sos')    

    # modulation_notch_filter = iirfilter(17, [carrier-2,carrier+2], rs=60, btype='bandstop',
    #                    analog=False, ftype='cheby2', fs=new_Fs,
    #                    output='sos')    
    # modulated_signal    = sosfiltfilt(modulation_notch_filter , downsampled_data)    
    modulated_signal    = sosfiltfilt(modulation_filter, downsampled_data)
    # # 
    # modulated_signal    = sosfiltfilt(modulation_filter2, modulated_signal)
    print ('lengths',len(dt),len(modulated_signal))
    # #  
    demodulated_signal  = demodulate(modulated_signal,carrier,dt)
    demodulated_signal2  = demodulate(modulated_signal,carrier*3,dt)   
    demodulated_signal = demodulated_signal+demodulated_signal2 
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
    d_demod_data = sosfiltfilt(sos_df_band, demodulated_signal) 
    # 
    # 
    mains_list = [50,100,150]
    for i in range(len(mains_list)):
        sos_mains_stop = iirfilter(17, [mains_list[i]-2 , mains_list[i]+2 ], rs=60, btype='bandstop',
                               analog=False, ftype='cheby2', fs=new_Fs,
                               output='sos')
        d_df_data = sosfiltfilt(sos_mains_stop , d_df_data)
        d_demod_data = sosfiltfilt(sos_mains_stop , d_demod_data)

    print ('alignment_indices', alignment_indices)
    # # just take the middle ones as the edges are messed up by the ramp. 
    alignment_indices = alignment_indices[4:54]
    # print ('subset:',alignment_indices)
    # 
    # 
    # fig  = plt.figure(figsize=(10,6))
    # ax   = fig.add_subplot(311)
    # plt.plot(dt,d_df_data,'k')
    # plt.plot(dt[alignment_indices],d_df_data[alignment_indices],'.r')
    # ax2  = fig.add_subplot(312)
    # # plt.plot(diffs,'k')
    # # plt.plot(dt,modulated_signal,'k')        
    # #plt.plot(dt,d_demod_data,'k')    
    # plt.plot(fsignal,'k')  
    # # plt.plot(dt[alignment_indices],d_demod_data[alignment_indices],'.r')    
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
    demod_data_to_segment                   = d_demod_data
    data_to_segment                         = d_df_data
    filtered_segmented_data                 = []
    demod_filtered_segmented_data           = []
    for i in range(len(alignment_indices)):  #
        # if i >=  1 and i < 4 :    # skip the first and last entry in each file. 
        if i % periods_of_interest == 0:
            # print ('here',i,alignment_indices[i])
            baseline = np.mean(data_to_segment[(alignment_indices[i]-pre_event_idxcount):alignment_indices[i] ])
            # print ('a indices',alignment_indices[i],pre_event_idxcount,post_event_idxcount)
            segment = data_to_segment[(alignment_indices[i]-pre_event_idxcount):(alignment_indices[i]+post_event_idxcount)]-baseline            
            
            # 
            dbaseline = np.mean(demod_data_to_segment[(alignment_indices[i]-pre_event_idxcount):alignment_indices[i] ])
            demod_segment = demod_data_to_segment[(alignment_indices[i]-pre_event_idxcount):(alignment_indices[i]+post_event_idxcount)]-dbaseline            
            

            print ('total height of segment:',(np.max(segment)-np.min(segment)))
            if (np.max(segment)-np.min(segment)) < 1000: # rudimentary filter. 
            # print ('length segment',len(segment))
                if len(segment) == array_len: # ensure only full sections are appended.
                    filtered_segmented_data.append(segment)
                    demod_filtered_segmented_data.append(demod_segment)                    
            else: 
                print ('skipped one too big')
    #     
    # print ('len f seg data:', len(filtered_segmented_data))
    filtered_segmented_array = np.array(filtered_segmented_data)
    demod_filtered_segmented_array = np.array(demod_filtered_segmented_data)    
    nfiltered_segmented_array = np.concatenate((nfiltered_segmented_array, filtered_segmented_array), axis=0)
    print ('filtered segmented array shape', nfiltered_segmented_array.shape)
    demod_segmented_array = np.concatenate((demod_segmented_array,demod_filtered_segmented_array),axis=0)
 


# 
# 

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,demod_segmented_array=demod_segmented_array,nfiltered_segmented_array=nfiltered_segmented_array,start = start, end=end)
print ('saved out a data file!')
# 
# 
# 
n_events,b = nfiltered_segmented_array.shape
time_segment = np.linspace(-start,end,num=b)
average_lfp  = np.mean(nfiltered_segmented_array,axis=0)
std_lfp      = np.std(nfiltered_segmented_array,axis=0)
sem_lfp      = np.std(nfiltered_segmented_array,axis=0)/np.sqrt(n_events)

dn_eventsd,b = demod_segmented_array.shape
time_segment = np.linspace(-start,end,num=b)
daverage_lfp  = np.mean(demod_segmented_array,axis=0)
dstd_lfp      = np.std(demod_segmented_array,axis=0)
dsem_lfp      = np.std(demod_segmented_array,axis=0)/np.sqrt(n_events)



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

dm,dci,dse = mean_confidence_interval(demod_segmented_array,confidence=0.95)
print ('ci',len(dci))

print ('n_events: ', n_events)
lfp_height   = np.max(average_lfp)- np.min(average_lfp)
print ('LFP size', lfp_height)
dlfp_height   = np.max(daverage_lfp)- np.min(daverage_lfp)
print ('dLFP size', dlfp_height)

print ('ratio',lfp_height/dlfp_height)
# 
error_lb     = ci
error_ub     = ci

derror_lb     = dci
derror_ub     = dci
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
ax  = fig.add_subplot(311)
plt.plot(time_segment,average_lfp/np.max(average_lfp),color='r')
plt.plot(time_segment,daverage_lfp/np.max(daverage_lfp),color='purple')
# plot the start and end times. 
for i in range(len(start_times) ):
    plt.axvline(x=start_times[i],color ='k')
for i in range(len(end_times) ):
    plt.axvline(x=end_times[i],color ='k')  
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Time(s)')

ax2  = fig.add_subplot(312)
plt.plot(time_segment,average_lfp,color='r')
plt.fill_between(time_segment, average_lfp - error_lb, average_lfp + error_ub,
                  color='gray', alpha=0.2)

plt.plot(time_segment,daverage_lfp,color='purple')
plt.fill_between(time_segment, daverage_lfp - derror_lb, daverage_lfp + derror_ub,
                  color='purple', alpha=0.2)
# plot the start and end times. 
for i in range(len(start_times) ):
    plt.axvline(x=start_times[i],color ='k')
for i in range(len(end_times) ):
    plt.axvline(x=end_times[i],color ='k')  

ax2.set_ylabel('Volts ($\mu$V)')
ax2.set_xlabel('Time(s)')
plt.autoscale(enable=True, axis='x', tight=True)
ax3 = fig.add_subplot(313)
plt.plot(time_segment,nfiltered_segmented_array.T)

# plot the start and end times. 
for i in range(len(start_times) ):
    plt.axvline(x=start_times[i],color ='k')
for i in range(len(end_times) ):
    plt.axvline(x=end_times[i],color ='k')  

# ax2.text(0.5, 100, 'individual trials', color='black',fontsize = 6, ha='center')
plt.autoscale(enable=True, axis='x', tight=True)
ax3.set_ylabel('Individual trials ($\mu$V)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

plot_filename = t_series+'_demodvep_'+str(prf)+'events_'+str(n_events)+'.png'
plt.savefig(plot_filename, transparent=True)
plt.show()




