'''

Title: demodulation and a few metrics. Shows FFT comparison of Demodded and original, and time series. 

Author: Jean Rintoul
Date: 02.02.2023

'''
# 
# Should I get a baseline of something as well? 
# 
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
import pandas as pd
import sys
sys.path.append('D:\\mouse_aeti')  #  so that we can import from the parent folder. 
import mouse_library as m
# 
# This is the chosen VEP.  
idx         = 7
# 
file_number = 20 # thinks there are only 2 events in total?
gain        = 1 
data_filepath       = 'D:\\mouse_aeti\\e86_demod\\t10_mouse\\'
factor              = 1  # total cycles. 
# 
carrier_f               = 500000
led_frequency           = 14 # in Hz. 
duration                = 8.0
print ('duration',duration)
led_duration = 1/(2*led_frequency)
Fs                      = 5e6
timestep                = 1.0/Fs
N                       = int(Fs*duration)
# print("expected no. samples:",N)
marker_channel          = 7 
m_channel               = 4 
rf_channel              = 2
# create time and frequencies arrays.
t  = np.linspace(0, duration, N, endpoint=False)
 

low = 4.5 
high = 100 
# high = 50 
sos_band                = iirfilter(17, [low,high], rs=60, btype='bandpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')


# 
filename    = data_filepath + 't'+str(file_number)+'_stream.npy'
data = np.load(filename)
a,b = data.shape
print ('data shape',a,b)
markerdata = data[marker_channel]
marker     = markerdata/np.max(markerdata)
rawdata    = 1e6*data[m_channel]/gain
rfdata     = 10*data[rf_channel]
rawdata    = rawdata-np.mean(rawdata)

filtered_rawdata    = sosfiltfilt(sos_band, rawdata)
print ('data shape: ',filtered_rawdata.shape,len(filtered_rawdata))
# # # # # # # # # # # # # # # # # # # # # # # # # 

# # # # # # # # # # # # # # # # # # # # # # # # # 
marker  = markerdata/np.max(markerdata)
diffs   = np.diff( markerdata )
indexes = np.argwhere(diffs > 0.2)[:,0] # leading edge, note:this doesnt work because there are 'blips'
# indexes = np.argwhere(diffs < -2.0)[:,0]  # falling edge
print ('len indexes', len(indexes),indexes)
zarray          = t*[0]
marker_up_length = int(led_duration*Fs)
for i in range(len(indexes)):
    if i > 0:
        zarray[indexes[i]:(indexes[i]+marker_up_length) ] = 1

marker = zarray
# 
# factor_duration = int(2*led_duration * Fs *factor)
# print ('factor duration',factor_duration)
# 
# factor                        = 1  #  this is how many repeats we want in the data. 
factor_duration                 = int(led_duration*Fs*factor)
time_segment                    = np.linspace(0, timestep*factor_duration*2, factor_duration*2, endpoint=False) 
time_segment                    = time_segment - led_duration
# time_segment        = np.linspace(0, timestep*factor_duration, factor_duration, endpoint=False)

chosen_VEP_idx           = indexes[idx]
# 
startid = chosen_VEP_idx - int(factor_duration/(2*factor) ) 
stopid  = chosen_VEP_idx + int(2*factor_duration - int(factor_duration/(2*factor)  ))
# 
prestimulus_period          = rawdata[startid:indexes[idx]]
baseline                    = np.mean(prestimulus_period)
# choose whether or not to subtract the baseline. 


chosen_VEP              = filtered_rawdata[startid:stopid] 
chosen_marker           = marker[startid:stopid] 
print ('chosen VEP length:',len(chosen_VEP),len(time_segment))

outfile="chosen_vep.npz"   # save out the data. 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile, chosen_VEP=chosen_VEP, chosen_marker=chosen_marker)
print ('saved out a chosen vep!')

fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(211)
plt.plot(t,10*markerdata,'g')
plt.plot(t,100*zarray,'b')
plt.plot(t,filtered_rawdata,'k')
for i in range(len(indexes)):
    plt.axvline(x=t[indexes[i]],color='r')
plt.axvspan(t[startid], t[stopid], color='green', alpha=0.5)
# plt.axvspan(t[indexes[idx-1]], t[indexes[idx]], color='gray', alpha=0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.set_ylim([-160,100])
ax.set_ylim([-350,200])
plt.legend(['LED marker','raw data($\mu$V)'],loc='lower right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xlabel('Time (s)')

ax2  = fig.add_subplot(212)
plt.plot(time_segment[0:len(chosen_VEP)],chosen_VEP,'k')
plt.plot(time_segment[0:len(chosen_VEP)],20*chosen_marker,'g')
# plt.plot(time_segment[0:len(chosen_VEP)],20*za,'g')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_ylabel('Volts ($\mu$V)')
ax2.set_xlabel('Time (s)')
plt.legend(['chosen VEP'],loc='lower left')
plot_filename = 'epoch_selector.png'
plt.savefig(plot_filename)
plt.show()



# subtract the mean first. 
# demodulated_signal = demod_average_lfp-np.mean(demod_average_lfp)
# raw_signal = average_lfp-np.mean(average_lfp)
# df = pd.DataFrame({'x': np.real(raw_signal), 'y': np.real(demodulated_signal) })
# window = len(raw_signal)
# # window          = 100000 # this is the number of samples to be used in the rolling cross-correlation. 
# rolling_corr    = df['x'].rolling(window).corr(df['y'])
# print ('rolling corr max:',np.max(rolling_corr))  # 
# result = np.nanmedian(rolling_corr)
# print ('median corr: ',result)
# max_index = np.argmax(rolling_corr) 
# print ('len rolling corr:', len(rolling_corr))
# print ('shape', a,b)
# 
# correlation_list.append(np.max(rolling_corr))
# median_correlation_list.append(result)

# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(111)
# plt.plot(t,rfdata,'k')
# plt.plot(t,filtered_demoddata,'purple')
# # ax.set_xlim([1,4.6])
# plt.legend(['rf monitor(V)','demodulated data(($\mu$V))'])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = 'transients.png'
# plt.savefig(plot_filename)
# plt.show()



# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(211)
# plt.plot(t,rfdata,'k')
# plt.plot(t,filtered_demoddata,'purple')
# plt.legend(['rf monitor(V)','demodulated data(($\mu$V))'],loc='upper right')
# # ax.set_xlim([1,4.6])
# # ax.set_ylim([-160,150])
# ax2  = fig.add_subplot(212)
# plt.plot(t,20*markerdata,'g')
# plt.plot(t,filtered_rawdata,'r')
# plt.plot(t,filtered_demoddata,'purple')
# # ax2.set_ylim([-100,100])
# # ax2.set_xlim([1,4.6])
# plt.legend(['LED marker','raw data($\mu$V)','demodulated data($\mu$V)'],loc='lower right')
# # ax2.set_xlim([1,7])
# ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xlabel('Time (s)')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = 'transient_rawvsdemod.png'
# plt.savefig(plot_filename)
# plt.show()

# f_idx = m.find_nearest(frequencies, 100)
# fs_idx = m.find_nearest(frequencies, 0.0)

# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(311)
# plt.plot(frequencies[fs_idx:f_idx],fft_rawdata[fs_idx:f_idx]/np.max(fft_rawdata[fs_idx:f_idx]),'red')
# plt.plot(frequencies[fs_idx:f_idx],fft_demodrawdata[fs_idx:f_idx]/np.max(fft_demodrawdata[fs_idx:f_idx]),'purple')
# ax.set_xlim([0,100])
# plt.legend(['raw','demod'])

# ax.set_ylabel('Volts ($\mu$V)')

# ax2  = fig.add_subplot(312)
# plt.plot(frequencies,fft_baseline,'orange')
# plt.plot(frequencies,fft_rawdata,'red')

# plt.legend(['baseline','raw'])
# ax2.set_xlim([0,800000])
# # ax2.set_xlim([672600,672900])
# ax2.set_ylabel('Volts ($\mu$V)')

# ax3  = fig.add_subplot(313)
# plt.plot(frequencies,-fft_diff,'blue')
# # ax3.set_xlim([0,800000])
# ax3.set_xlim([0,800000])
# # ax3.set_xlim([672600,672900])
# plt.legend(['diff'])

# ax3.set_ylabel('Volts ($\mu$V)')
# ax3.set_ylabel('Norm Volts (V)')
# ax3.set_xlabel('Frequency(Hz)')
# plt.suptitle('Frequency analysis')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plt.show()


# ax3  = fig.add_subplot(313)
# plt.plot(time_segment, demod_average_lfp ,'purple')
# ax3.set_ylabel('Volts ($\mu$V)')
# for i in range(factor):
#   ax3.axvspan(led_duration*(2*i+1), led_duration*(2*i+2), alpha=0.5, color='gray')
# ax3.set_xlim([0.0,timestep*len(average_lfp)])


# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# # # 
# plt.suptitle('VEP@'+str(led_frequency)+'Hz LED flashing frequency n='+str(n_events))
# plot_filename = savepath+'\\isitreal.png'
# plt.savefig(plot_filename)
# # # 
# plt.show()
# # 
