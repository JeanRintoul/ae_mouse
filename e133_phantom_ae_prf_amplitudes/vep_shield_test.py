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
# 
data            = np.load('D:\\ae_mouse\\e133_phantom_ae_prf_amplitudes\\demod-vep_data.npz')
veps            = data['nfiltered_segmented_array']
demod_veps      = data['ndemod_segmented_array']
# 
data            = np.load('D:\\ae_mouse\\e133_phantom_ae_prf_amplitudes\\demod-shieldedvep_data.npz')
noveps          = data['nfiltered_segmented_array']
demod_noveps    = data['ndemod_segmented_array']
start           = data['start']
end             = data['end']


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

# 
# demods 
dn_events,b = veps.shape
demod_average_lfp  = np.mean(demod_veps,axis=0)
demod_std_lfp      = np.std(demod_veps,axis=0)
demod_sem_lfp      = np.std(demod_veps,axis=0)/np.sqrt(dn_events)

dnon_events,b   = noveps.shape
print ('dn_events',dn_events,dnon_events)
no_demod_average_lfp  = np.mean(demod_noveps,axis=0)
no_demod_std_lfp      = np.std(demod_noveps,axis=0)
no_demod_sem_lfp      = np.std(demod_noveps,axis=0)/np.sqrt(dnon_events)

# end demods

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

m,dnci,se = mean_confidence_interval(demod_noveps,confidence=0.95) 
dnerror_lb     = dnci
dnerror_ub     = dnci

m,dci,se = mean_confidence_interval(demod_veps,confidence=0.95) 
derror_lb     = dci
derror_ub     = dci


fig = plt.figure(figsize=(6,3))
ax  = fig.add_subplot(111)
plt.plot(time_segment,average_lfp,color='g')
plt.plot(time_segment,no_average_lfp,color='k')
plt.fill_between(time_segment, average_lfp - error_lb, average_lfp + error_ub,
                  color='g', alpha=0.2)
plt.fill_between(time_segment, no_average_lfp -nerror_lb, no_average_lfp + nerror_ub,
                  color='gray', alpha=0.2)
plt.axvline(x=0,color ='k')
plt.axvline(x=0.5,color ='k')
ax.set_ylabel('Volts ($\mu$V)',fontsize=16)
ax.set_xlabel('Time(s)',fontsize=16)
# plt.legend(['VEPS','Light Shield'],loc='lower right',fontsize=16,framealpha=0.0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.set_xlim([-0.1,0.9])
plt.tight_layout()
plot_filename = 'vep_shield_test.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(6,3))
ax  = fig.add_subplot(111)
plt.plot(time_segment,demod_average_lfp,color='r')
plt.plot(time_segment,no_demod_average_lfp,color='k')
plt.fill_between(time_segment, demod_average_lfp - derror_lb, demod_average_lfp + derror_ub,
                  color='red', alpha=0.2)
plt.fill_between(time_segment, no_demod_average_lfp -dnerror_lb, no_demod_average_lfp + dnerror_ub,
                  color='gray', alpha=0.2)
plt.axvline(x=0,color ='k')
plt.axvline(x=0.5,color ='k')
ax.set_ylabel('Volts ($\mu$V)',fontsize=16)
ax.set_xlabel('Time(s)',fontsize=16)
# plt.legend(['demodulated VEPS','demodulated Light Shield'],loc='lower right',fontsize=16,framealpha=0.0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.set_xlim([-0.1,0.9])
plt.tight_layout()
plot_filename = 'demod_vep_shield_test.png'
plt.savefig(plot_filename)
plt.show()


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

