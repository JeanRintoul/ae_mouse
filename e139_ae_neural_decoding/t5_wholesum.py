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
# template_filename            = 'vep_template_t2.npz'
# print ('filename: ', template_filename)
# data                         = np.load(template_savepath+template_filename)
# template                     = data['template']
# 
# 
l_cut       = 0.5
h_cut       = 50
# # # # # # # # # # # # #
t_series      = 'mouse'
savepath      = 'D:\\ae_mouse\\e139_ae_neural_decoding\\t5_mouse\\ae_8Hz_g2000\\'
outpath      = 'D:\\ae_mouse\\e139_ae_neural_decoding\\'
prf                         = 10
match_frequency             = 10
# 
# 
start       = 1
stop        = 11
step        = 1 
file_list   = range(start,stop,step)
print ('file list',file_list)
outfile     = 't5_data.npz'
conversion_factor = 1
# 
Fs              = 2e6
# new_Fs          = 1e4 
timestep        = 1.0/Fs
duration        = 30
gain            = 2000
m_channel       = 0 
rf_channel      = 4
marker_channel  = 7 
pulse_length    = 0.0 # pulse length in seconds. 
N               = int(Fs*duration)

carrier         = 500000
t               = np.linspace(0, duration, N, endpoint=False)
# 

sos_low_band    = iirfilter(17, [h_cut], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
sos_demodulate_band = iirfilter(17, [carrier-h_cut,carrier+h_cut], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
repeats = 0 
for n in range(len(file_list)):
    # 
    file_number = file_list[n] 
    print ('file_number',file_number)
    filename    = savepath + 't'+str(file_number)+'_stream.npy'
    data        = np.load(filename)
    fsignal     = (1e6*data[m_channel]/gain)
    fsignal     = fsignal - np.mean(fsignal)
    rfsignal    = 10*data[rf_channel]
    marker      = data[marker_channel]
    # 
    # 
    carrier_band    = sosfiltfilt(sos_demodulate_band, fsignal)
    lfp_low         = sosfiltfilt(sos_low_band, fsignal)
    #         
    # find the onset markers. 
    diffs   = np.diff(marker)
    indexes = np.argwhere(diffs > 0.2)[:,0]
    indexes = indexes[1:] # skip the first one. 
    print ('total marker indexes: ',len(indexes))
    # Look at the extracted marker channel.  
    # fig = plt.figure(figsize=(6,6))
    # ax  = fig.add_subplot(111)
    # # plt.plot(t,fsignal,'k')
    # plt.plot(t,marker,'b')
    # plt.plot(t[indexes],marker[indexes],'.r')
    # plt.show()
    repeats = repeats + 1 
    if n == 0:
        rawdata_summation       = fsignal 
        lfp_summation           =  lfp_low
        carrier_band_summation  =  carrier_band
    else: 
        rawdata_summation = rawdata_summation + fsignal 
        lfp_summation = lfp_summation + lfp_low
        carrier_band_summation = carrier_band_summation + carrier_band




#                     
# Average. 
rawdata_summation       = rawdata_summation/repeats
lfp_summation           = lfp_summation/repeats
carrier_band_summation  = carrier_band_summation/repeats

# Save out all the averaged data. 
# outfile = 'new.npz'
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,t = t,rawdata_summation=rawdata_summation,lfp_summation=lfp_summation,carrier_band_summation=carrier_band_summation)
print ('saved out a data file!')
#   
rbeta           = 14
raw_timestep    = 1/Fs 
rawN            = len(rawdata_summation)
raw_window      = np.kaiser( (rawN), rbeta )
xf              = np.fft.fftfreq( (rawN), d=raw_timestep)[:(rawN)//2]
raw_frequencies = xf[1:(rawN)//2]
fft_raw         = fft(rawdata_summation)
fft_raw         = np.abs(2.0/(rawN) * (fft_raw))[1:(rawN)//2]
fft_rawk         = fft(rawdata_summation*raw_window)
fft_rawk         = np.abs(2.0/(rawN) * (fft_rawk))[1:(rawN)//2]

fft_lfp         = fft(lfp_summation)
fft_lfp         = np.abs(2.0/(rawN) * (fft_lfp))[1:(rawN)//2]
fft_carrier_band         = fft(carrier_band_summation*raw_window)
fft_carrier_band         = np.abs(2.0/(rawN) * (fft_carrier_band))[1:(rawN)//2]
# print ('stuff shape: ',start_indexes)
# demodulate the signal.
analytical_signal       = hilbert(carrier_band_summation) # Hilbert demodulate.  
h_signal                = -np.abs(analytical_signal)
demodulated_signal      = h_signal - np.mean(h_signal)
# temp adjust. 
demodulated_signal  = np.real(analytical_signal)
# 
# 
def demodulate(in_signal,carrier_f,tt): 
    # return np.abs(in_signal*np.exp(2*np.pi*1j*carrier_f*tt))
    return np.imag(in_signal*np.exp(2*np.pi*1j*carrier_f*tt))    
demodulated_signal2       = demodulate(carrier_band_summation,carrier,t) 
# 
# calculate the fft of the recovered signal.
fft_demod         = fft(demodulated_signal)
fft_demod         = np.abs(2.0/(rawN) * (fft_demod))[1:(rawN)//2]
# 
fft_demod2        = fft(demodulated_signal2)
fft_demod2        = np.abs(2.0/(rawN) * (fft_demod2))[1:(rawN)//2]   
print ('N lengths',rawN,N)
# 
# 
time_segment = t 
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
# 
f = 18 
# NB: It only makes sense to demodulate after much averaging. 
# Plot the results. 
fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(211)
plt.plot(raw_frequencies,fft_carrier_band,'k')    
ax.set_xlim([carrier-h_cut,carrier+h_cut])
plt.xticks([carrier-20,carrier,carrier+20])
# ax.set_ylim([0,20])
ax.set_ylim([0,0.1])
ax.ticklabel_format(useOffset=False)
plt.xticks(fontsize=f)
plt.yticks(fontsize=f)
# 
ax2  = fig.add_subplot(212)
plt.plot(raw_frequencies,fft_raw,'k')    
plt.plot(raw_frequencies,fft_rawk,'r')    
ax2.set_xlim([0,h_cut])
ax2.set_ylim([0,10])
plt.xticks(fontsize=f)
plt.yticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
# 
plot_filename = outpath+'_wholefile_sum.png'
plt.savefig(plot_filename)


# ax3  = fig.add_subplot(313)
# plt.plot(raw_frequencies,fft_carrier_band,'k')    
# ax3.set_xlim([carrier-h_cut,carrier+h_cut])
# # plt.plot(raw_frequencies,fft_demod,'k')   
# # plt.plot(raw_frequencies,fft_demod2,'r')   
# # ax3.set_xlim([0,20])
# ax3.set_ylim([0,10])
plt.show()
# 
