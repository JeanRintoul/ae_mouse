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
template_filename            = '4hz.npz'
print ('filename: ', template_filename)
data                    = np.load(template_savepath+template_filename)
raw                     = data['rawdata_summation']
lfp                     = data['lfp_summation']




# l_cut       = 0.5
# h_cut       = 40
# # # # # # # # # # # # #
t_series      = 'mouse'
savepath      = 'D:\\ae_mouse\\e139_ae_neural_decoding\\t2_mouse\\4hz_aevep\\'
# savepath      = 'D:\\ae_mouse\\e139_ae_neural_decoding\\t2_mouse\\3Hz_aevep\\'
prf                         = 4
match_frequency             = 4
# 
# 
Fs              = 2e6
timestep        = 1.0/Fs
duration        = 30
gain            = 500
m_channel       = 0 
rf_channel      = 4
marker_channel  = 7 
pulse_length    = 0.0 # pulse length in seconds. 
N               = int(Fs*duration)

carrier         = 500000
t               = np.linspace(0, duration, N, endpoint=False)
# 
# number of periods to look at? must be a sub-multiple of the number of repeats in the file.  
periods_of_interest           = prf*14 # whole file. 
# periods_of_interest           = prf*9 # whole file. 
# 
start_time                      = np.round(0.8/duration,2)
end_time                        = np.round((duration - 0.4)/duration,2)
# 
print ('start and end',start_time,end_time)
# downsampling_factor             = int(Fs/new_Fs)
start                           = 0.1*(1.0/prf)   # 0.2 seconds
end                             = (periods_of_interest-1)*(1.0/prf)+0.9*(1.0/prf)
pre_event_idxcount              = int(start*Fs)
post_event_idxcount             = int(end*Fs)
array_len                       = post_event_idxcount+pre_event_idxcount
time_segment    = np.linspace(-start,end,num=array_len)

sos_synaptic = iirfilter(17, [2.5,300], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
sos_spike = iirfilter(17, [300,2000], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')


mains_list = [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]

N       = len(raw)
fsignal = raw 
t       = time_segment
for i in range(len(mains_list)):
    sos_mains_stop = iirfilter(17, [mains_list[i]-2 , mains_list[i]+2 ], rs=60, btype='bandstop',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
    fsignal = sosfiltfilt(sos_mains_stop , fsignal)
#  
synaptic_signal    = sosfiltfilt(sos_synaptic, fsignal)
spike_signal       = sosfiltfilt(sos_spike, fsignal)
#  
analytical_signal  = hilbert(spike_signal)
h_signal                = -np.abs(analytical_signal)
demodulated_signal      = h_signal - np.mean(h_signal)
#  
fft_synaptic = fft(synaptic_signal)
fft_synaptic = np.abs(2.0/(N) * (fft_synaptic))[1:(N)//2]

fft_spike = fft(spike_signal)
fft_spike = np.abs(2.0/(N) * (fft_spike))[1:(N)//2]

fft_h = fft(demodulated_signal)
fft_h = np.abs(2.0/(N) * (fft_h))[1:(N)//2]
# 
xf = np.fft.fftfreq( (N), d=1/Fs)[:(N)//2]
frequencies = xf[1:(N)//2]
# 
# 
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(211)
plt.plot(frequencies,fft_synaptic,'k')
ax.set_xlim([0,2000])

ax2  = fig.add_subplot(212)
plt.plot(frequencies,fft_h,'k')
ax2.set_xlim([0,2000])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.show()

fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(211)
plt.plot(t,spike_signal,'r')
plt.plot(t,synaptic_signal,'k')
ax.set_xlim([0,8])
ax.set_ylim([-50,50])
ax2  = fig.add_subplot(212)
plt.plot(t,spike_signal,'k')
ax2.set_xlim([0,8])
ax2.set_ylim([-15,15])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.show()

# sos_low_band    = iirfilter(17, [h_cut], rs=60, btype='lowpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# # 
# sos_demodulate_band = iirfilter(17, [carrier-h_cut,carrier+h_cut], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')



# 
# summed_array     = np.zeros((0,array_len))
# repeats = 0 
# for n in range(len(file_list)):
#     # 
#     file_number = file_list[n] 
#     print ('file_number',file_number)
#     filename    = savepath + 't'+str(file_number)+'_stream.npy'
#     data        = np.load(filename)
#     fsignal     = (1e6*data[m_channel]/gain)
#     fsignal     = fsignal - np.mean(fsignal)
#     rfsignal    = 10*data[rf_channel]
#     marker      = data[marker_channel]
#     # 
#     # 
#     carrier_band    = sosfiltfilt(sos_demodulate_band, fsignal)
#     lfp_low         = sosfiltfilt(sos_low_band, fsignal)
#     #         
#     # find the onset markers. 
#     diffs   = np.diff(marker)
#     indexes = np.argwhere(diffs > 0.2)[:,0]
#     indexes = indexes[1:] # skip the first one. 
#     print ('total marker indexes: ',len(indexes))
#     # Now,find the right segmentation indices
#     start_indexes = []
#     for i in range(len(indexes)):  #
#         if i % periods_of_interest == 0:
#             print ('i',i,indexes[i])
#             preseg = fsignal[(indexes[i]-pre_event_idxcount):(indexes[i]+post_event_idxcount)]         
#             peaks, properties = find_peaks(preseg,height = (100,100000) ) 
#             # print ('len peaks: ',len(peaks))
#             if len(peaks) > 0:
#                 offset = peaks[0]
#                 rawdata_segment = fsignal[(indexes[i]-pre_event_idxcount+offset):(indexes[i]+post_event_idxcount+offset)]
#                 lfp_segment = lfp_low[(indexes[i]-pre_event_idxcount+offset):(indexes[i]+post_event_idxcount+offset)]
#                 carrier_band_segment = carrier_band[(indexes[i]-pre_event_idxcount+offset):(indexes[i]+post_event_idxcount+offset)]
#                 start_indexes.append(indexes[i]-pre_event_idxcount+offset)
#             # 
#             if len(peaks) > 0 and len(rawdata_segment) == array_len:
#                 repeats = repeats + 1
#                 print ('repeats',repeats)
#                 if n == 0 and i == 0:     # do this only for the first one. 
#                     rawdata_summation = rawdata_segment
#                     lfp_summation = lfp_segment
#                     carrier_band_summation = carrier_band_segment 
#                     stuff = rawdata_segment
#                 else: # for all other files. 
#                     rawdata_summation = rawdata_summation + rawdata_segment
#                     lfp_summation = lfp_summation + lfp_segment
#                     carrier_band_summation = carrier_band_summation + carrier_band_segment
#                     stuff = np.vstack((stuff,rawdata_segment))
# #                     
# # Average. 
# rawdata_summation       = rawdata_summation/repeats
# lfp_summation           = lfp_summation/repeats
# carrier_band_summation  = carrier_band_summation/repeats

# # Save out all the averaged data. 
# outfile = 'new.npz'
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
# np.savez(outfile,time_segment=time_segment,rawdata_summation=rawdata_summation,lfp_summation=lfp_summation,carrier_band_summation=carrier_band_summation)
# print ('saved out a data file!')
# #   
# rbeta           = 14
# raw_timestep    = 1/Fs 
# rawN            = len(rawdata_summation)
# raw_window      = np.kaiser( (rawN), rbeta )
# xf              = np.fft.fftfreq( (rawN), d=raw_timestep)[:(rawN)//2]
# raw_frequencies = xf[1:(rawN)//2]
# fft_raw         = fft(rawdata_summation)
# fft_raw         = np.abs(2.0/(rawN) * (fft_raw))[1:(rawN)//2]
# fft_rawk         = fft(rawdata_summation*raw_window)
# fft_rawk         = np.abs(2.0/(rawN) * (fft_raw))[1:(rawN)//2]

# fft_lfp         = fft(lfp_summation)
# fft_lfp         = np.abs(2.0/(rawN) * (fft_lfp))[1:(rawN)//2]
# fft_carrier_band         = fft(carrier_band_summation*raw_window)
# fft_carrier_band         = np.abs(2.0/(rawN) * (fft_carrier_band))[1:(rawN)//2]
# print ('stuff shape: ',start_indexes)
# # demodulate the signal.
# analytical_signal       = hilbert(carrier_band_summation) # Hilbert demodulate.  
# h_signal                = -np.abs(analytical_signal)
# demodulated_signal      = h_signal - np.mean(h_signal)
# # temp adjust. 
# demodulated_signal  = np.real(analytical_signal)
# # 
# # 
# def demodulate(in_signal,carrier_f,tt): 
#     # return np.abs(in_signal*np.exp(2*np.pi*1j*carrier_f*tt))
#     return np.imag(in_signal*np.exp(2*np.pi*1j*carrier_f*tt))    
# demodulated_signal2       = demodulate(carrier_band_summation,carrier,time_segment) 
# # 
# # calculate the fft of the recovered signal.
# fft_demod         = fft(demodulated_signal)
# fft_demod         = np.abs(2.0/(rawN) * (fft_demod))[1:(rawN)//2]
# # 
# fft_demod2        = fft(demodulated_signal2)
# fft_demod2        = np.abs(2.0/(rawN) * (fft_demod2))[1:(rawN)//2]   
# print ('N lengths',rawN,N)
# # 
# # 
# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(411)
# plt.plot(time_segment,demodulated_signal,'k')
# plt.plot(time_segment,demodulated_signal2,'r')
# ax2 = fig.add_subplot(412)
# plt.plot(time_segment,lfp_summation,'k')
# ax3 = fig.add_subplot(413)
# plt.plot(time_segment,rawdata_summation,'k')
# ax4 = fig.add_subplot(414)
# plt.plot(time_segment,carrier_band_summation,'k')
# plt.show()
# # 
# # NB: It only makes sense to demodulate after much averaging. 
# # Plot the results. 
# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(311)
# plt.plot(raw_frequencies,fft_carrier_band,'k')    
# ax.set_xlim([carrier-h_cut,carrier+h_cut])
# ax.set_ylim([0,20])
# # 
# ax2  = fig.add_subplot(312)
# plt.plot(raw_frequencies,fft_raw,'k')    
# ax2.set_xlim([0,h_cut])
# ax2.set_ylim([0,10])
# # 
# ax3  = fig.add_subplot(313)
# plt.plot(raw_frequencies,fft_demod,'k')   
# plt.plot(raw_frequencies,fft_demod2,'r')   
# ax3.set_xlim([0,20])
# # ax3.set_ylim([0,10])
# plt.show()
# # 
