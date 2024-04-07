'''

Title: compare the data going into generator with the data coming out of generator. 

SNR analysis of t5. 
Author: Jean Rintoul
Date:   05.02.2024

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.stats import ttest_ind
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 
saveprefix          = './/images//'
# 
# Continuous wave. 
savepath2     = 'D:\\ae_mouse\\e137_ae_neural_decoding\\t1_phantom\\noF21_1MPa\\'
savepath     = 'D:\\ae_mouse\\e137_ae_neural_decoding\\t1_phantom\\F21_1MPa\\'
# 
filelist = [1,2,3,4]


# F21_filename     = 't1_stream.npy'
# noF21_filename   = 't1_stream.npy'
# 
# print ('filename: ',filename)
# 
duration            = 8
Fs                  = 5e6
timestep            = 1.0/Fs
N                   = int(Fs*duration)
carrier             = 500000
channel_of_interest = 0
rfchannel           = 4
gain                = 1000
t                   = np.linspace(0, duration, N, endpoint=False)
current_signal_frequency = 10
band_of_interest    = 40
# 
print ('current_signal_frequency: ', current_signal_frequency)
start_time      = np.round(0.8/duration ,2)
end_time        = np.round((duration - 0.2)/duration,2)
print ('start and end',start_time,end_time)
start_pause     = int(start_time * N+1)
end_pause       = int(end_time *N-1)
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 
# 
F21_summation = [0]*t 
noF21_summation = [0]*t

for d in range(len(filelist)):
    F21_filename     = savepath+'t'+str(filelist[d])+'_stream.npy'
    noF21_filename   = savepath2+'t'+str(filelist[d])+'_stream.npy'
    # 
    print ('F21 filename:',F21_filename)
    F21_data        = np.load(F21_filename)
    F21_fsignal     = 1e6*F21_data[channel_of_interest]/gain
    # 
    noF21_data      = np.load(noF21_filename)
    noF21_fsignal   = 1e6*noF21_data[channel_of_interest]/gain
    # 
    if d == 0: 
        F21_summation   = F21_fsignal
        noF21_summation = noF21_fsignal
    else: 
        F21_summation = F21_summation + F21_fsignal
        noF21_summation = noF21_summation + noF21_fsignal
    # 
F21_average =   F21_summation/len(filelist)
noF21_average = noF21_summation/len(filelist)

beta            = 20
window          = np.kaiser( (end_pause-start_pause), beta )
print('window len:',len(window))

fft_noF21 = fft(noF21_average[start_pause:end_pause]*window)
fft_noF21 = np.abs(2.0/(end_pause-start_pause) * (fft_noF21))[1:(end_pause-start_pause)//2]

fft_F21 = fft(F21_average[start_pause:end_pause]*window)
fft_F21 = np.abs(2.0/(end_pause-start_pause) * (fft_F21))[1:(end_pause-start_pause)//2]
#
# 
# 
# SNR calculation for both kaiser window and 
frequencies_of_interest = [carrier - current_signal_frequency,carrier+current_signal_frequency]
# If it looks bad for some reason? For now, just note down the file number.     
signal_totals                   = []
kaiser_signal_totals            = []
interest_frequencies            = []
for n in range(len(frequencies_of_interest)): # sum all frequencies per unit time.
  df_idx = find_nearest(frequencies,frequencies_of_interest[n])
  # also eliminate on either side of bins of interest as I don't have great bin resolution.    
  interest_frequencies.append(df_idx)    
  kaiser_signal_totals.append(fft_F21[df_idx])
  signal_totals.append(fft_noF21[df_idx])
# This is the start and end point to calculate the SNR from. 
start_idx = find_nearest(frequencies,carrier + int(current_signal_frequency/2) )   # after the first 5Hz. 
end_idx = find_nearest(frequencies,carrier + 40)
# # print ('end idx',end_idx)
dis_kaiser_signal_totals      = []
dis_signal_totals             = []  # 
for n in range(start_idx, end_idx): # sum all frequencies per unit time.
    if n not in interest_frequencies: 
        dis_kaiser_signal_totals.append(fft_F21[n])
        dis_signal_totals.append(fft_noF21[n])
#           
signal_snr              = 20*np.log(np.mean(signal_totals)/np.mean(dis_signal_totals))
kaiser_signal_snr       = 20*np.log(np.mean(kaiser_signal_totals)/np.mean(dis_kaiser_signal_totals))
print ('signal and kaiser snr:',signal_snr,kaiser_signal_snr)


# Build the plot
fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_noF21,'r')
plt.plot(frequencies,fft_F21,'k')

plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(['noF21','F21'],loc='upper right',framealpha=0.0,fontsize=16)
# Save the figure and show
# ax.set_xlim([0,30])
# ax.set_ylim([0,1600])
ax.ticklabel_format(useOffset=False)
plt.tight_layout()
plt.savefig(saveprefix+'kaiser_applied_signal.png')
plt.show()

# # Build the plot
# fig = plt.figure(figsize=(4,4))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_c,'k')
# plt.plot(frequencies,fft_k,'r')
# ax.set_xlim([carrier - 30,carrier+30])
# ax.set_ylim([0,np.max(fft_c)+10])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # plt.legend(['rectangular','kaiser'],loc='upper right',framealpha=0.0,fontsize=16)
# # Save the figure and show
# plt.xticks([carrier-24,carrier,carrier+24])
# ax.ticklabel_format(useOffset=False)
# plt.tight_layout()
# plt.savefig(saveprefix+'kaiser_comparison.png')
# plt.show()


# fig = plt.figure(figsize=(4,4))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_c,'k')
# plt.plot(frequencies,fft_k,'r')
# ax.set_xlim([carrier - 30,carrier+30])
# ax.set_ylim([0,10])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # plt.legend(['rectangular','kaiser'],loc='upper right',framealpha=0.0,fontsize=16)
# # Save the figure and show
# plt.xticks([carrier-24,carrier,carrier+24])
# ax.ticklabel_format(useOffset=False)
# plt.tight_layout()
# plt.savefig(saveprefix+'kaiser_comparison_zoom.png')
# plt.show()