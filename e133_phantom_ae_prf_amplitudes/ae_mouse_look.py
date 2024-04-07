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
carrier_frequency = 500000

# savepath       = 'D:\\ae_mouse\\e133_phantom_ae_prf_amplitudes\\t1_phantom_prf\\continuous_demodulation\\'
# savepath = 'D:\\ae_mouse\\e133_phantom_ae_prf_amplitudes\\t2_mouse\\5Hz_2_whp\\'
savepath = 'D:\\ae_mouse\\e133_phantom_ae_prf_amplitudes\\t2_mouse\\5Hz_neuraldemod\\'
basepath = 'D:\\ae_mouse\\e133_phantom_ae_prf_amplitudes\\t2_mouse\\noVEP_baseline_forneuraldemod\\'

# basepath = 'D:\\ae_mouse\\e133_phantom_ae_prf_amplitudes\\t2_mouse\\noVEP_baseline2_whp\\'

duration        = 8
Fs              = 5e6
timestep        = 1.0/Fs
N               = int(Fs*duration)
channel_of_interest = 0
# # 
start = 1
stop  = 6
step  = 1 
file_list = range(start,stop,step)
print ('file list',file_list)
gain = 500 
# 
new_Fs                = 1e6
downsampling_factor   = int(Fs/new_Fs)
sos_downsampling_band = iirfilter(17, [2e6], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
# 
#  
mixed_df    = 20

# t1_stream_small.npy

for n in range(len(file_list)):
    file_number = file_list[n] 
    print ('file_number',file_number)
    filename = 't'+str(file_number)+'_stream.npy'

    data = np.load(savepath+filename)
    a,b  = data.shape
    fsignal = 1e6*data[channel_of_interest]/gain

    basedata = np.load(basepath+filename)
    a,b      = basedata.shape
    basesignal = 1e6*basedata[channel_of_interest]/gain
    # 
    dsignal              = sosfiltfilt(sos_downsampling_band, fsignal)
    basesignal           = sosfiltfilt(sos_downsampling_band, basesignal)    
    # Now downsample the data. 
    # downsampled_data            = dsignal[::downsampling_factor]

    # start_idx   = int(1*Fs)
    # end_idx     = int(7*Fs)
    # basesignal  = basesignal[start_idx:end_idx]
    # fsignal     = fsignal[start_idx:end_idx]

    if n == 0 :
        base_total = basesignal
        vep_total  = dsignal
        print ('base total shape', base_total.shape)
    else:
        base_total = np.concatenate((base_total, basesignal ), axis=0)
        vep_total  = np.concatenate((vep_total, dsignal ), axis=0)
        print ('base total shape', base_total.shape)


print ('len base total:',len(base_total))
# t       = np.linspace(0, base_total*timestep, len(base_total), endpoint=False)
total_N = len(base_total)

fft_m = fft(vep_total)
fft_m = np.abs(2.0/(total_N) * (fft_m))[1:(total_N)//2]
fft_base = fft(base_total)
fft_base = np.abs(2.0/(total_N) * (fft_base))[1:(total_N)//2]
xf = np.fft.fftfreq( (total_N), d=timestep)[:(total_N)//2]
frequencies = xf[1:(total_N)//2]

bands = 40

fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(211)
plt.axvline(x=carrier_frequency+mixed_df,color='r')
plt.axvline(x=carrier_frequency-mixed_df,color='r')
plt.plot(frequencies,fft_base,'m')
plt.plot(frequencies,fft_m,'k')

ax.set_xlim([carrier_frequency-bands,carrier_frequency+bands])
# ax.set_ylim([0,0.2])
ax.set_ylim([0,5])

ax2  = fig.add_subplot(212)
plt.axvline(x=carrier_frequency+mixed_df,color='r')
plt.axvline(x=carrier_frequency-mixed_df,color='r')
plt.plot(frequencies,fft_m-fft_base,'k')
ax2.set_xlim([carrier_frequency-bands,carrier_frequency+bands])
# ax.set_ylim([0,0.2])
ax2.set_ylim([0,5])

# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
# ax.set_xlabel('Frequency (Hz)',fontsize=16)
# ax.set_ylabel('Normalized Volts ($\mu$V)',fontsize=16)
# plt.title('Modulated LFP Signal',fontsize=16)
plt.tight_layout()
plot_filename = 'neural_demodulation.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(211)
plt.plot(frequencies,fft_base,'m')
plt.plot(frequencies,fft_m,'k')
ax.set_xlim([0,bands])
ax.set_ylim([0,200])

ax2  = fig.add_subplot(212)
plt.plot(frequencies,fft_m-fft_base,'k')
plt.legend(['subtracted'],loc='upper right')
ax2.set_xlim([0,bands])
ax2.set_ylim([0,50])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
# ax.set_xlabel('Frequency (Hz)',fontsize=16)
# ax.set_ylabel('Normalized Volts ($\mu$V)',fontsize=16)
# plt.title('LFP Signal',fontsize=16)
plt.tight_layout()
plot_filename = 'base_signal.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()


# l_cut  = 20
# df_cut = 30
# new_Fs = 1e5 

# def demodulate(in_signal,carrier_f,dt): 
#     # create a demodulated signal. 
#     lowm  = carrier_f - df_cut
#     highm = carrier_f + df_cut
#     # print ('low/high',lowm,highm)
#     modulation_filter = iirfilter(17, [lowm,highm], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
#     modulated_signal  = sosfiltfilt(modulation_filter, in_signal)

#     idown = modulated_signal*np.cos(2*np.pi*carrier_f*dt)
#     qdown = modulated_signal*np.sin(2*np.pi*carrier_f*dt)   

#     demodulated_signal = idown + 1j*qdown
#     demodulated_signal = np.abs(demodulated_signal)
#     d_demod_data = sosfiltfilt(sos_df_band, demodulated_signal)     
#     return d_demod_data
# # 
# sos_df_band = iirfilter(17, [l_cut,df_cut], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# # 
# sos_lp = iirfilter(17, [4000], rs=60, btype='lowpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# 

# downsampling_factor  = int(Fs/new_Fs)
# print('downsampling factor: ',downsampling_factor)
# # downsample for easier plotting. 
# downsampled_data1           = sosfiltfilt(sos_lp, fsignal)
# downsampled_lfp             = downsampled_data1[::downsampling_factor]
# td                          = t[::downsampling_factor]
# 
# Final filters. 
# lfp_data                        = sosfiltfilt(sos_df_band, fsignal)   
# demodulated                     = demodulate(fsignal,carrier_frequency,t)


# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(111)
# plt.plot(t,lfp_data/np.max(lfp_data),'r')
# plt.plot(t,demodulated/np.max(demodulated),'k')
# # ax.set_xlim([0,bands])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Time (s)',fontsize=16)
# ax.set_ylabel('Normalized Volts ($\mu$V)',fontsize=16)
# plt.title('Demodulated Signal',fontsize=16)
# plt.tight_layout()
# plt.legend(['LFP','Demodulated'],loc='upper right')
# plot_filename = 'demodulated_signal.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()



