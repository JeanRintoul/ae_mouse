'''

Title: compare the data going into generator with the data coming out of generator. 

Author: Jean Rintoul
Date:   26.10.2023

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
import scipy.stats
# 
# def find_nearest(array, value):
#     idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
#     return idx
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 
saveprefix     = './/images//'
savepath       = './/processed//'
# 
data            = np.load(savepath + 't2-vep_4Hz.npz')
veps            = data['nfiltered_segmented_array']
# demod_veps      = data['ndemod_segmented_array']
# 
data            = np.load(savepath + 't2-vep_4Hz_shielded.npz')
noveps          = data['nfiltered_segmented_array']
# demod_noveps    = data['ndemod_segmented_array']
start           = data['start']
end             = data['end']
# 
print ('start end:',start,end)
frequency = 4 
on_marker = 0  
off_marker = 0.5*(1.0/frequency)
period     = 1.0/4

n_events,b = veps.shape
time_segment = np.linspace(-start,end,num=b)
average_lfp  = np.mean(veps,axis=0)
std_lfp      = np.std(veps,axis=0)
sem_lfp      = np.std(veps,axis=0)/np.sqrt(n_events)

non_events,b = noveps.shape
print ('n_events',n_events,non_events)
time_segment = np.linspace(-start,end,num=b)
no_average_lfp  = np.mean(noveps,axis=0)
no_std_lfp      = np.std(noveps,axis=0)
no_sem_lfp      = np.std(noveps,axis=0)/np.sqrt(non_events)


print (len(average_lfp))
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m,h,se
# 
m,ci,se = mean_confidence_interval(veps,confidence=0.95) 
error_lb     = ci
error_ub     = ci

m,nci,se = mean_confidence_interval(noveps,confidence=0.95) 
nerror_lb     = nci
nerror_ub     = nci

Fs = 1e5 
timestep = 1/Fs
N = len(average_lfp)
fft_m = fft(average_lfp)
fft_m = np.abs(2.0/(N) * (fft_m))[1:(N)//2]
# 
xf = np.fft.fftfreq( (N), d=timestep)[:(N)//2]
frequencies = xf[1:(N)//2]
# 
# 
fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(111)
plt.plot(frequencies,fft_m,'k')
ax.set_xlim([0,20])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = saveprefix+'vep_fft_3periods.png'
plt.savefig(plot_filename)
plt.show()
# 
# 
fig = plt.figure(figsize=(6,3))
ax  = fig.add_subplot(111)
plt.plot(time_segment,average_lfp,color='g')
plt.plot(time_segment,no_average_lfp,color='k')
plt.fill_between(time_segment, average_lfp - error_lb, average_lfp + error_ub,
                  color='g', alpha=0.2)
plt.fill_between(time_segment, no_average_lfp -nerror_lb, no_average_lfp + nerror_ub,
                  color='gray', alpha=0.2)
plt.axvline(x=on_marker,color ='k')
plt.axvline(x=off_marker,color ='k')
# ax.set_ylabel('Volts ($\mu$V)',fontsize=16)
# ax.set_xlabel('Time(s)',fontsize=16)
# plt.legend(['VEPS','Light Shield'],loc='lower right',fontsize=16,framealpha=0.0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
plt.yticks([])
plt.xticks([])
ax.set_xlim([-start,end])
plt.tight_layout()
plot_filename = saveprefix+'vep_shield_test_bunch.png'
plt.savefig(plot_filename)
plt.show()

# fig = plt.figure(figsize=(6,3))
# ax  = fig.add_subplot(111)
# plt.plot(time_segment,demod_average_lfp,color='g')
# plt.plot(time_segment,no_demod_average_lfp,color='k')
# plt.fill_between(time_segment, demod_average_lfp - derror_lb, demod_average_lfp + derror_ub,
#                   color='green', alpha=0.2)
# plt.fill_between(time_segment, no_demod_average_lfp -dnerror_lb, no_demod_average_lfp + dnerror_ub,
#                   color='gray', alpha=0.2)
# plt.axvline(x=on_marker,color ='k')
# plt.axvline(x=off_marker,color ='k')
# # ax.set_ylabel('Volts ($\mu$V)',fontsize=16)
# # ax.set_xlabel('Time(s)',fontsize=16)
# # plt.legend(['demodulated VEPS','demodulated Light Shield'],loc='lower right',fontsize=16,framealpha=0.0)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# # plt.yticks(fontsize=16)
# # plt.xticks(fontsize=16)
# plt.yticks([])
# plt.xticks([])
# ax.set_xlim([-start,end])
# plt.tight_layout()
# plot_filename = saveprefix+'demod_vep_shield_test.png'
# plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(t,data[channel_of_interest],'k')
# plt.legend(['v chan'],loc='upper right')
# ax2 = fig.add_subplot(212)
# plt.plot(frequencies,fft_m,'k')
# ax2.set_xlim([0,carrier_frequency*3])
# # ax2.set_xlim([0,carrier_frequency+carrier_frequency/2])
# # plt.legend(['from recording chan'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# plot_filename = 'carrier_data.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(t,from_generator,'k')
# ax.set_xlim([start,stop])
# plt.legend(['from generator'],loc='upper right')
# ax2 = fig.add_subplot(212)
# plt.plot(t,from_recording_chan,'k')
# ax2.set_xlim([start,stop])
# plt.legend(['from recording chan'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = 'zoom_file_comparison.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()

