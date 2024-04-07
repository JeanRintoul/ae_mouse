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
Fs              = 1e5
# Fs              = 2e6
#Fs              = 5e6
duration        = 12
new_Fs          = 1e4
# start_pause = 0.5  
# end_pause   = 7.6
start_pause = 1.5
end_pause   = 10.5

timestep        = 1.0/Fs
N               = int(Fs*duration)
t               = np.linspace(0, duration, N, endpoint=False)

gain            = 500
m_channel       = 0 
rf_channel      = 4
marker_channel  = 7 

#
# 
saveprefix      = './/images//'

# 4Hz files. 
# savepath       = 'D:\\ae_mouse\\spike_demodulation_dataset\\e1\\spike_demodulation_Fs_100kHz\\4Hz_VEP\\'
# savepath       = 'D:\\ae_mouse\\spike_demodulation_dataset\\e2\\VEP_4Hz\\'
#savepath       = 'D:\\ae_mouse\\spike_demodulation_dataset\\e3\\VEP_4Hz_noUS\\'
# Fs = 5MHz
#savepath       = 'D:\\ae_mouse\\spike_demodulation_dataset\\e4\\VEP_4Hz\\' # clearly something there. 
# # Fs = 100kHz
# savepath       = 'D:\\ae_mouse\\spike_demodulation_dataset\\e5\\4Hz_VEP\\' # clearly something there. 

# savepath       = 'D:\\ae_mouse\\spike_demodulation_dataset\\e6\\4Hz_VEP\\' # clearly something there. 
savepath       = 'D:\\ae_mouse\\spike_demodulation_dataset\\e6\\4Hz_VEP\\' # clearly something there. 


downsampling_factor  = int(Fs/new_Fs)
print('downsampling factor: ',downsampling_factor)
# 

# Fs is at 20kHz. 
frequencies_of_interest     = [8,16,24]   
# frequencies_of_interest   = [8,12,16,20,24]   

# f_band_test = np.linspace(0,419,420)
bandwidth_of_interest = 30
bandwidth_of_interest = 200
min_df_cut = bandwidth_of_interest*2
df_cut  = min_df_cut   
end_frequency = 26
f_band_test = np.linspace(0,2500,101)

downsampling_filter_lowcut = new_Fs/2 - 1 # i.e 10khz snr demod = 25.06
# 
# Im not sure if I am optimizing for the time synced 8Hz...
# I am optimizing for having 8,12,16 Hz in there. 

max_f = np.max(f_band_test)
print ('max f:',max_f)


ledtest            = []
lfp                = []
demod_average      = [] 
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
    downsampled_data1            = sosfiltfilt(sos_lp, fsignal)
    # Now downsample the data. 
    downsampled_data            = downsampled_data1[::downsampling_factor]
    dt                          = t[::downsampling_factor]
    d_marker                    = marker[::downsampling_factor]    

    sos_df = iirfilter(17, [0.5,bandwidth_of_interest], rs=60, btype='bandpass',
                           analog=False, ftype='cheby2', fs=new_Fs,
                           output='sos')

    lfp_data = sosfiltfilt(sos_df, downsampled_data)   

    #  Calculate the FFT of the resultant signal. 
    start_index = int(new_Fs*start_pause)
    end_index   = int(new_Fs*end_pause)
    newN = len(lfp_data[start_index:end_index])
    xf  = np.fft.fftfreq( (newN), d=1/new_Fs)[:(newN)//2]
    frequencies = xf[1:(newN)//2]
    fft_m_spectrum = fft(lfp_data[start_index:end_index] )
    fft_m = np.abs(2.0/(newN) * (fft_m_spectrum))[1:(newN)//2]

    component_list = [] 
    biggest  = [0]*dt
    prev_snr = 0
    av_snrs  = []
    winning_carrier = 0 
    # print ('f band test:',f_band_test)
    for j in range(len(f_band_test)):

        # no overlap paramters       
        x       = f_band_test[j] 
        df_cut  = min_df_cut + x
        df_high = df_cut + bandwidth_of_interest*2 
        carrier = int((df_high - df_cut)/2) + df_cut 
        # print ('carrier',carrier)
        # print ('df cut',df_cut,df_high)
        sos_demodulate_bandpass  = iirfilter(17, [df_cut,df_high], rs=60, btype='bandpass',
                               analog=False, ftype='cheby2', fs=new_Fs,
                               output='sos')
        filtered_downsampled_data = sosfiltfilt(sos_demodulate_bandpass, downsampled_data) 
 
        demodulated_signal_component    = -demodulate(filtered_downsampled_data,carrier,dt)
        demodulated_signal_iq_component = sosfiltfilt(sos_df, demodulated_signal_component) 

        # This is a single band of the demodulated signal. 
        fft_dm_component_spectrum   = fft(demodulated_signal_iq_component[start_index:end_index])
        fft_dm_component            = np.abs(2.0/(newN) * (fft_dm_component_spectrum))[1:(newN)//2]
        # 
        fft_dm = fft_dm_component
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
    # 
    print ('carrier: snr lfp /demod: ',winning_carrier, np.round(lfp_snr,2),np.round(prev_snr,2)  )  
    # averaged_demodulated = np.mean(np.array(component_list),axis=0 )
    # normalize each result that is the best.     
    averaged_demodulated = biggest
    print ('len av demodded: ',len(averaged_demodulated),len(biggest))
    # 
    # fig = plt.figure(figsize=(6,6))
    # ax  = fig.add_subplot(111)
    # plt.plot(f_band_test,av_snrs,'k')
    # plt.axhline(y=lfp_snr, color='r', linestyle='-')
    # plt.show()
    # # 
    ledtest.append(d_marker)
    lfp.append(lfp_data) 
    demod_average.append(biggest)   

lfp_array           = np.array(lfp)
dm_average_array    = np.array(demod_average)

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

# 
av_daverage_lfp  = np.mean(dm_average_array, axis=0)
av_dstd_lfp      = np.std(dm_average_array, axis=0)
av_dsem_lfp      = np.std(dm_average_array, axis=0)/np.sqrt(n_events)

# print ('n_events: ', n_events)
lfp_height   = np.max(average_lfp)- np.min(average_lfp)


# Calculate the FFT. 
start_index = int(new_Fs*start_pause)
end_index   = int(new_Fs*end_pause)

# newN = len(average_lfp[start_index:end_index])
fft_m_spectrum = fft(average_lfp[start_index:end_index] )
fft_m = np.abs(2.0/(newN) * (fft_m_spectrum))[1:(newN)//2]

fft_av_dm_spectrum = fft(av_daverage_lfp[start_index:end_index])
fft_av_dm = np.abs(2.0/(newN) * (fft_av_dm_spectrum))[1:(newN)//2]

xf  = np.fft.fftfreq( (newN), d=1/new_Fs)[:(newN)//2]
frequencies = xf[1:(newN)//2]

# Calculate the SNR. 
# print ('frequencies of interest:', frequencies_of_interest)
demod_av_totals              = []  # sum all frequencies per unit time. 
lfp_totals                   = []
interest_frequencies         = []
for i in range(len(frequencies_of_interest)): # sum all frequencies per unit time.
  df_idx = find_nearest(frequencies,frequencies_of_interest[i])
  # also eliminate on either side of bins of interest as I don't have great bin resolution.    
  interest_frequencies.append(df_idx)    
  demod_av_totals.append(fft_av_dm[df_idx])
  lfp_totals.append(fft_m[df_idx])
# # print ('demod/lfp totals',lfp_amplitude, demod_amplitude)
# df_cut = np.max(frequencies_of_interest)
end_idx = find_nearest(frequencies,end_frequency)
# # print ('end idx',end_idx)
dis_demod_av_totals         = []  # 
dis_lfp_totals              = []
for i in range(end_idx): # sum all frequencies per unit time.
  if i not in interest_frequencies: 
      dis_demod_av_totals.append(fft_av_dm[i])
      dis_lfp_totals.append(fft_m[i])
# # signal to noise calculation. We take the total signal(), and divide it 
lfp_snr      = 20*np.log(np.mean(lfp_totals)/np.mean(dis_lfp_totals))
demod_av_snr = 20*np.log(np.mean(demod_av_totals)/np.mean(dis_demod_av_totals))
# # print ('amplitude carrier: lfp /demod',carrier,lfp_amplitude,demod_amplitude)
print ('snr lfp /demod: ',np.round(lfp_snr,2),np.round(demod_av_snr,2) )
# 

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

# dm,dci,dse = mean_confidence_interval(dm_array,confidence=0.95)
# print ('dci',len(dci))
# 
error_lb      = ci
error_ub      = ci

lfp_height = np.max(average_lfp) - np.min(average_lfp)
# print ('lfp height:',lfp_height)


fig = plt.figure(figsize=(8,3))
ax  = fig.add_subplot(111)
plt.plot(dt,average_lfp/np.max(average_lfp),color='k')
plt.plot(dt,av_daverage_lfp/np.max(av_daverage_lfp),color='green')
plt.plot(dt,1*d_marker-1.0,'grey')
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
ax  = fig.add_subplot(211)
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


ax2  = fig.add_subplot(212)
# plt.plot(frequencies,fft_iq_dm,color='m')
plt.plot(frequencies,fft_av_dm,color='k')
ax2.set_xlim([0,bandwidth_of_interest])
ax2.set_ylabel('Volts ($\mu$V)',fontsize=16)
ax2.set_xlabel('Frequency(Hz)',fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.legend(['IQ FFT','av FFT'],loc='upper right',frameon=False,framealpha=1.0, fontsize=16)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout()
plot_filename = saveprefix+'FFT_Comparisons.png'
plt.savefig(plot_filename, transparent=True)
plt.show()

bandwidth_of_interest = 30 

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
