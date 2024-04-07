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
# savepath      = 'D:\\ae_mouse\\e139_ae_neural_decoding\\t1_mouse\\3hz_aevep\\'
# savepath      = 'D:\\ae_mouse\\e139_ae_neural_decoding\\t1_mouse\\3hz_aevep2\\'
savepath      = 'D:\\ae_mouse\\e139_ae_neural_decoding\\t2_mouse\\4hz_aevep\\'
# 
match_frequency = 4
# 
start       = 2
stop        = 4 
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
prf             = 4
carrier         = 500000
# 
# number of periods to look at? must be a sub-multiple of the number of repeats in the file.  
# periods_of_interest             = 64
periods_of_interest             = prf*28 # whole file. 
# 
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

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   



for n in range(len(file_list)):

    file_number = file_list[n] 
    print ('file_number',file_number)
    filename    = savepath + 't'+str(file_number)+'_stream.npy'
    data        = np.load(filename)
    #
    # 
    fsignal     = (1e6*data[m_channel]/gain)
    fsignal     = fsignal - np.mean(fsignal)
    rfsignal    = 10*data[rf_channel]
    marker      = data[marker_channel]
    # 
    if n == 0:
        rawdata_summation = fsignal
        stuff = rfsignal
    else: 
        rawdata_summation = rawdata_summation + fsignal
        stuff = np.vstack((stuff,rfsignal))
# 
# 
print ('stuff shape',stuff.shape)
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
# np.savez(outfile,match_lfp_segmented_array=match_lfp_segmented_array,match_demod_segmented_array=match_demod_segmented_array,ndemod_segmented_array=ndemod_segmented_array,nfiltered_segmented_array=nfiltered_segmented_array,start = start, end=end)
# print ('saved out a data file!')
# 
# FFT of raw data calculation, to look at the peaks around 500kHz. 
# start_seconds = 1
# end_seconds   = 10
rbeta           = 20
raw_timestep    = 1/Fs 
# 
s_a             = int(0)
e_a             = int(len(rawdata_summation))
rawN            = len(rawdata_summation)
raw_window      = np.kaiser( (rawN), rbeta )
xf = np.fft.fftfreq( (N), d=raw_timestep)[:(N)//2]
raw_frequencies = xf[1:(N)//2]
fft_raw  = fft(rawdata_summation*raw_window)
fft_raw  = np.abs(2.0/(N) * (fft_raw))[1:(N)//2]

fft_stuff  = fft(stuff*raw_window)
fft_stuff  = np.abs(2.0/(N) * (fft_stuff))[1:(N)//2]
# 
# 
fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(111)
plt.plot(t,stuff[0,:],'k')
plt.plot(t,stuff[1,:],'r')
# plt.plot(t,stuff[2,:],'g')
# ax.set_xlim([2,2.5])
plt.show()
# 
fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(111)
# plt.plot(raw_frequencies,fft_raw,'k')
plt.plot(raw_frequencies,fft_stuff,'r')
# ax.set_xlim([carrier-h_cut,carrier+h_cut])
# ax.set_ylim([0,5])
ax.set_xlim([0,50])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()
# 