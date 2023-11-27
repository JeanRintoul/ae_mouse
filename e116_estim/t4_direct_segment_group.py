'''

Title: vep signal inspection
Function: takes a single file, and averages the veps to see them better. 

Author: Jean Rintoul
Date:   23.10.2023

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
import scipy.stats
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 
# 
savepath       = 'D:\\ae_mouse\\e116_estim\\t4_epulse_mouse\\'

# prf = 0.5
# pulse_length   = 0.01
# file_list      = [26,27,28,29,30,31,32,33,34,35]  # 100ms
# # 'current_burst_length':0.0002,  # 200 microseconds.
# 'current_burst_length':0.001,   # 1ms 
# 'current_burst_length':0.050,   # 50ms 
# 'current_burst_length':0.1,     # 100ms 
# 'current_burst_length':0.01,     # 10ms 

prf            = 2
# pulse_length   = 0.0002
# file_list      = [1,2,3,4,5]       # 200 microsecond
# pulse_length   = 0.001
# file_list      = [6,7,8,9,10]      # 1ms
# pulse_length   = 0.01
# file_list      = [21,22,23,24,25]  # 10ms
# pulse_length   = 0.05
# file_list      = [11,12,13,14,15]  # 50ms
pulse_length   = 0.1
file_list      = [16,17,18,19,20]  # 100ms

# 
# 
gain            = 100
duration        = 6.0	
# 
Fs              = 5e6
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 4
start                           = 0.2*(1.0/prf)  # 0.2 seconds
end                             = 0.8*(1.0/prf)
pre_event_idxcount              = int(start*Fs)
post_event_idxcount             = int(end*Fs)
t = np.linspace(0, duration, N, endpoint=False)
print ('len t',len(t))
df_l = 0.1 # dfx-2
df_l = 2 # dfx-2
df_h = 300 
# df_h = 1500
# df_h = 40
if df_l <= 0: 
    df_l = 0.05 
sos_lfp_band = iirfilter(17, [df_l,df_h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
pulse_indices = int(pulse_length*Fs)
print ('pulse indices',pulse_indices)

for n in range(len(file_list)):
    file_number = file_list[n] 
    filename    = savepath + 't'+str(file_number)+'_stream.npy'
    data = np.load(filename)
    a,b = data.shape    
    print ('shape',a,b)
    # convert it to microvolts by taking the gain into account. 
    fsignal     = 1e6*data[m_channel]/gain
    vsignal     = data[6] 
    mains_harmonics = [50,100,150,200,250,300]
    for i in range(len(mains_harmonics)):
        mains_low  = mains_harmonics[i] -2
        mains_high = mains_harmonics[i] +2
        mains_sos = iirfilter(17, [mains_low,mains_high], rs=60, btype='bandstop',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
        fsignal = sosfiltfilt(mains_sos, fsignal)
    # 
    marker  = vsignal/np.max(vsignal)
    diffs   = np.diff(marker)
    print ('diffs max: ', np.max(diffs))
    zarray  = t*[0]
    indexes = np.argwhere(diffs > 0.2)[:,0]
    X       = np.insert(indexes, 0, 0)
    m       = np.diff(X)
    print ('m',m)
    m       = np.argwhere(m > 1000)[:,0]
    alignment_indices = indexes[m]
    print ('the alignment_indices are:',alignment_indices)
    zarray[alignment_indices] = -200 

    # set the pulse data to zero. 
    for i in range(len(alignment_indices)):  # 
        fsignal[alignment_indices[i]:(alignment_indices[i]+pulse_indices+400)] = 0 
        # fsignal[alignment_indices[i]:(alignment_indices[i]+pulse_indices+400)] = fsignal[(alignment_indices[i]-(pulse_indices+400) ):alignment_indices[i]]
    # filter the data. 
    lfp_data                        = sosfiltfilt(sos_lfp_band, fsignal)
    data_to_segment                 = lfp_data
    filtered_segmented_data         = []
    for i in range(len(alignment_indices)):  # 
        # 
        baseline = np.mean(data_to_segment[(alignment_indices[i]-pre_event_idxcount):alignment_indices[i] ])
        # print ('baseline',baseline)
        segment = data_to_segment[(alignment_indices[i]-pre_event_idxcount):(alignment_indices[i]+post_event_idxcount)]-baseline
        filtered_segmented_data.append(segment)
    filtered_segmented_array = np.array(filtered_segmented_data)
    a,b = filtered_segmented_array.shape
# 
    # fig = plt.figure(figsize=(10,6))
    # ax  = fig.add_subplot(111)
    # plt.plot(t,fsignal,'k')
    # plt.plot(t[alignment_indices],fsignal[alignment_indices],'.r')
    # plt.plot(t,-250+100*vsignal/np.max(vsignal),'r')
    # plt.show()  
    # print ('filtered segmented array shape', a,b)
    if n == 0: 
        nfiltered_segmented_array = filtered_segmented_array
    else:
        nfiltered_segmented_array = np.concatenate((nfiltered_segmented_array, filtered_segmented_array), axis=0)
    a,b = nfiltered_segmented_array.shape
    print ('filtered segmented array shape', a,b)


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
print ('n_events: ', n_events)
lfp_height   = np.max(average_lfp)- np.min(average_lfp)
print ('LFP size', lfp_height)

m,ci,se = mean_confidence_interval(filtered_segmented_array,confidence=0.95)
print ('ci',len(ci))
error_lb     = ci
error_ub     = ci

start_time = 0
stop_time = 0 + pulse_length
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(time_segment,average_lfp,color='r')
plt.fill_between(time_segment, average_lfp - error_lb, average_lfp + error_ub,
                  color='gray', alpha=0.2)
plt.axvline(x=start_time,color ='k')
plt.axvline(x=stop_time,color ='k')
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Time(s)')
plt.autoscale(enable=True, axis='x', tight=True)
# ax2 = fig.add_subplot(212)
# plt.plot(time_segment,nfiltered_segmented_array.T)
# plt.axvline(x=start_time,color ='k')
# # ax2.text(0.5, 100, 'individual trials', color='black',fontsize = 6, ha='center')
# plt.autoscale(enable=True, axis='x', tight=True)
# ax2.set_ylabel('Individual trials ($\mu$V)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = 'direct_stim_prf_'+str(prf)+'_erp_'+str(n_events)+'_'+str(pulse_length)+'.png'
plt.savefig(plot_filename)
plt.show()

# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(111)
# plt.plot(t,lfp_data,'k')
# plt.plot(t,-250+100*vsignal/np.max(vsignal),'r')
# plt.plot(t,zarray,'.b')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # plot_filename = 'alignment_indices.png'
# # plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()
# # 
# # 
# start_time = 0
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(time_segment,average_lfp,color='r')
# plt.fill_between(time_segment, average_lfp - error_lb, average_lfp + error_ub,
#                   color='gray', alpha=0.2)
# plt.axvline(x=start_time,color ='k')
# ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xlabel('Time(s)')
# plt.autoscale(enable=True, axis='x', tight=True)

# ax2 = fig.add_subplot(212)
# plt.plot(time_segment,filtered_segmented_array.T)
# plt.axvline(x=start_time,color ='k')
# # ax2.text(0.5, 100, 'individual trials', color='black',fontsize = 6, ha='center')
# plt.autoscale(enable=True, axis='x', tight=True)
# ax2.set_ylabel('Individual trials ($\mu$V)')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# # plt.title('VEP')
# plot_filename = 'direct_prf_'+str(prf)+'_erp'+str(n_events)+'.png'
# plt.savefig(plot_filename)
# plt.show()

# 
# 

# #
# #
# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_data,'k')
# # ax.set_xlim([0,110])
# ax.set_xlim([0,110])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = 'rfti_fft.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()

# # Plots
# # start = 1.5 
# # stop  = 4.5

# start = 0
# stop  = duration
# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(t,lfp_data,'k')
# # plt.plot(t,10*spike_data,'r')
# plt.plot(t,-250+100*rfsignal/np.max(rfsignal),'r')
# plt.legend(['lfp','applied'],loc='upper right')
# ax.set_xlim([start,stop])
# ax2 = fig.add_subplot(312)
# plt.plot(t,spike_data,'k')
# plt.plot(t,-100+10*rfsignal/np.max(rfsignal),'r')
# # 
# ax2.set_xlim([start,stop])
# plt.legend(['spikes','applied'],loc='upper right')
# ax3 = fig.add_subplot(313)
# plt.plot(t,fsignal/np.max(fsignal),'k')
# # plt.plot(t,vsignal/np.max(vsignal),'r')
# plt.legend(['raw measured signal'],loc='upper right')
# ax3.set_xlim([start,stop])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plot_filename = 'rfti_debugging.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()
# 
# 
# fig = plt.figure(figsize=(5,7.5))
# ax = fig.add_subplot(311)
# plt.plot(frequencies,fft_data,'k')
# # plt.plot(frequencies,fft_data,'k')
# # ax.set_xlim([0,110])
# ax.set_xlim([499950,500050])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax2 = fig.add_subplot(312)
# plt.plot(frequencies,fft_data,'k')
# ax2.set_xlim([0,50])
# ax2.set_ylim([0,750])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax3 = fig.add_subplot(313)
# plt.plot(t,lfp_data,'k')
# ax3.set_ylim([-2500,4500])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = 'fft_mdata.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()

# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# plt.plot(t,lfp_data,'k')
# plt.tight_layout()
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = 'lfp.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()

# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# plt.plot(t,spike_data,'k')
# plt.tight_layout()
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = 'spikes.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()



# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# # plt.plot(t,fsignal,'k')
# plt.plot(t,lfp_data,'m')
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.set_xlim([0,duration])
# ax2 = fig.add_subplot(312)
# #plt.plot(t,vsignal,'k')
# plt.plot(t,lfp_v_data,'m')
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax2.set_xlim([0,duration])
# ax3 = fig.add_subplot(313)
# #plt.plot(t,rfsignal,'k')
# plt.plot(t,lfp_rf_data,'m')
# plot_filename = 'mep_artefact_data.png'
# ax3.set_xlim([0,duration])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plt.tight_layout()
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()

# fig_shape_1 = 6 
# fig_shape_2 = 2 

# fig = plt.figure(figsize=(fig_shape_1,fig_shape_2))
# ax = fig.add_subplot(111)
# plt.plot(t,vsignal,'k')
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = 'voltage_data.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()

# fig = plt.figure(figsize=(fig_shape_1,fig_shape_2))
# ax = fig.add_subplot(111)
# plt.plot(t,rfsignal,'k')
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = 'rf_data.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()


# fig = plt.figure(figsize=(fig_shape_1,fig_shape_2))
# ax = fig.add_subplot(111)
# plt.plot(t,spike_data,'k')
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = 'spike_data.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()


# fig = plt.figure(figsize=(fig_shape_1,fig_shape_2))
# ax = fig.add_subplot(111)
# plt.plot(t,lfp_data,'m')
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = 'lfp_data.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()


# ax3 = fig.add_subplot(313)
# plt.plot(frequencies,fft_data,'k')

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(221)
# plt.plot(frequencies,fft_data,'k')
# ax.set_xlim([0,max_lim])
# plt.legend(['ae signal'],loc='upper right')
# ax2 = fig.add_subplot(222)
# plt.plot(frequencies,fft_rfdata,'k')
# ax2.set_xlim([0,max_lim])
# plt.legend(['rf signal'],loc='upper right')



# ax3 = fig.add_subplot(223)
# plt.plot(frequencies2,fft_data2,'k')
# ax3.set_xlim([0,max_lim])
# plt.legend(['ae signal2'],loc='upper right')
# ax4 = fig.add_subplot(224)
# plt.plot(frequencies2,fft_rfdata2,'k')
# ax4.set_xlim([0,max_lim])
# plt.legend(['rf signa2l'],loc='upper right')

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# ax4.spines['right'].set_visible(False)
# ax4.spines['top'].set_visible(False)


# plot_filename = 'mep_artefact_data.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(t,spike_data/np.max(spike_data),'k')
# plt.plot(t,df_data/np.max(df_data),'r')
# ax.set_xlim([0,duration])
# ax2 = fig.add_subplot(312)
# plt.plot(frequencies,fft_data,'k')

# ax3 = fig.add_subplot(313)
# plt.plot(t,df_data,'k')
# ax3.set_xlim([0,duration])

# plot_filename = 'spike_data.png'
# plt.savefig(plot_filename)
# plt.show()
# 
# 400 p-p was giving a good amplitude signal in dual acoustic meps. 
# 
# demodulated_signal = rfsignal * vsignal
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(t,1+rfsignal/np.max(rfsignal),'r')
# plt.plot(t,vsignal/np.max(vsignal),'m')
# plt.plot(t,-1-fsignal/np.max(fsignal),'k')
# plot_filename = 'draw_data.png'
# plt.savefig(plot_filename)
# plt.show()
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(t,1+rfsignal/np.max(rfsignal),'r')
# plt.plot(t,vsignal/np.max(vsignal),'m')
# plt.plot(t,-1-fsignal/np.max(fsignal),'k')
# ax2 = fig.add_subplot(312)
# plt.plot(t,demodulated_signal,'k')
# ax3 = fig.add_subplot(313)
# plt.plot(frequencies,fft_data,'k')
# plt.plot(frequencies,fft_rfdata,'r')
# # ax2.set_xlim([0,1100])
# # ax2.set_xlim([0,5])
# # ax2.set_ylim([0,20])
# # plot_filename = 'draw_data.png'
# # plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(frequencies,fft_data,'k')
# # plt.plot(frequencies,fft_rfdata,'r')
# ax.set_xlim([1000000-200,1000000+200])
# ax2 = fig.add_subplot(212)
# plt.plot(frequencies,fft_data,'k')
# # plt.plot(frequencies,fft_rfdata,'r')
# ax2.set_xlim([500000-200,500000+200])
# # ax2.set_xlim([0,5])
# # ax2.set_ylim([0,20])
# plt.show()

# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(t,1e6*data[m_channel]/gain,'k')
# ax.set_xlim([0,duration])
# # ax.set_xlim([1.755,1.756])
# ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xlabel('Time(s)')
# plt.legend(['measurement channel at focus'],loc='upper right')
# ax2 = fig.add_subplot(212)
# plt.plot(t,10*data[rf_channel],'r')
# ax2.set_xlim([0,duration])
# ax2.set_ylabel('Volts (V)')
# plt.legend(['rf monitor channel'],loc='upper right')
# # ax2.set_xlim([1.755,1.756])
# ax.set_xlabel('time (s)')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = 'draw_data.png'
# plt.savefig(plot_filename)
# plt.show()
# 

# 
# low  = 1000-5
# high = 1000 +5

# sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# f1signal  = sosfiltfilt(sos_low, fsignal)


# low  = 3000-5
# high = 3000 +5

# sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# f2signal  = sosfiltfilt(sos_low, fsignal)


# low  = 1001000 -5
# high = 1001000 +5

# sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# f3signal  = sosfiltfilt(sos_low, fsignal)


# low  = 500000 -5000
# high = 500000 +5000
# high = 10
# sos_low = iirfilter(17, [high], rs=60, btype='lowpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# f4signal  = sosfiltfilt(sos_low, fsignal)

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(t,f4signal,'k')
# # ax.set_xlim([1.755,1.756])
# plot_filename = 'dc_components.png'
# plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(t,f1signal,'k')
# plt.legend(['1kHz filtered'],loc='upper right')
# ax.set_xlim([1.755,1.756])
# ax2 = fig.add_subplot(212)
# plt.plot(t,fsignal,'m')
# ax2.set_xlim([1.755,1.756])
# plt.legend(['raw single pulse'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = 'dcomponents.png'
# plt.savefig(plot_filename)
# plt.show()
# # 
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_data,'k')
# # ax.set_xlim([0,1000000])
# ax.set_xlim([0,5000])
# ax.set_ylim([0,2.5])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = 'dFFT.png'
# plt.savefig(plot_filename)
# plt.show()
# # 
# 

