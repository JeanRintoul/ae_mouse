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
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16

savepath    = 'D:\\ae_mouse\\e121_stimulation\\t3_mouse\\'
file_number = 90
gain        = 100
m_channel   = 0 
rf_channel  = 2
filename    = savepath + 't'+str(file_number)+'_stream.npy'
data        = np.load(filename)
fsignal     = 1e6*data[m_channel]/gain
rfsignal    = 10*data[rf_channel]  
Fs          = 5e6
timestep    = 1.0/Fs
duration    = 6 
N           = int(Fs*duration)
t = np.linspace(0, duration, N, endpoint=False)

dfx = 1020                
# 
# 
sos_c_band = iirfilter(17, [1e6+500,1e6+5000], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
c_data              = sosfiltfilt(sos_c_band, fsignal)


# Hilbert transform c_data, so I can later downsample 
c_signal  = hilbert(c_data)
c_envelope = np.abs(c_signal)
sos_lp = iirfilter(17, [dfx*2+100], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
c_hilberted             = sosfiltfilt(sos_lp, c_envelope)
  

sos_df_band = iirfilter(17, [2000], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
df_data              = sosfiltfilt(sos_df_band, fsignal)
# 
# downsample for easier plotting. 
# 
new_Fs               = 1e4
downsampling_factor  = int(Fs/new_Fs)
dt                   = t[::downsampling_factor]
d_df_data            = df_data[::downsampling_factor]
d_c_hilberted        = c_hilberted[::downsampling_factor]
mains_list = [50,100,150,200,250,300,350,400,450,500,1000]
for i in range(len(mains_list)):
    sos_mains_stop = iirfilter(17, [mains_list[i]-2 , mains_list[i]+2 ], rs=60, btype='bandstop',
                           analog=False, ftype='cheby2', fs=new_Fs,
                           output='sos')
    d_df_data               = sosfiltfilt(sos_mains_stop , d_df_data)


s1 = 0 
s2 = duration 
fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111)
plt.plot(dt,d_df_data/np.max(df_data),'k')
plt.plot(dt,d_c_hilberted/np.max(d_c_hilberted),'r')
ax.set_xlim([s1,s2])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()
# 
# 
# element              = 2
# frequencies          = [1,2,5,10,40,100,300,1000,2000,5000]
# representative_files = [2,22,37,52,62,63,65,67,69,71]

# frequency = frequencies[element]
# Fs      = 1e4
# if frequency> 300:
#     Fs      = 1e6

# applied_signal_file = representative_files[element]
# print ('frequency is:', frequency)
# #
# data = np.load('f_'+str(frequency)+'Hz_mouse_data.npz',allow_pickle=True)
# m_lfp_heights = data['lfp_heights']
# m_peakds      = data['peakds']
# m_peakrs      = data['peakrs']
# m_raw_datas   = data['raw_datas']
# m_r2s         = data['r2s']
# m_d2s         = data['d2s']
# #
# pdata = np.load('f_'+str(frequency)+'Hz_phantom_data.npz',allow_pickle=True)
# p_lfp_heights = pdata['lfp_heights']
# p_peakds      = pdata['peakds']
# p_peakrs      = pdata['peakrs']
# p_raw_datas   = pdata['raw_datas']
# p_r2s         = pdata['r2s']
# p_d2s         = pdata['d2s']

# mean_p_lfp_heights = np.mean(p_lfp_heights,axis=0)
# std_p_lfp_heights = np.std(p_lfp_heights,axis=0)

# mean_m_lfp_heights = np.mean(m_lfp_heights,axis=0)
# std_m_lfp_heights = np.std(m_lfp_heights,axis=0)

# print ('mean lfp heights p/m: ',mean_p_lfp_heights,mean_m_lfp_heights)
# # T-test. 
# sample1 = m_lfp_heights
# sample2 = p_lfp_heights
# t_stat, p_value = ttest_ind(sample1, sample2) 
# # print('T-statistic value: ', t_stat) 
# print('P-Value: ', p_value)

# # Create lists for the plot
# materials = ['Mouse','Phantom']
# x_pos = np.arange(len(materials))
# CTEs = [mean_m_lfp_heights, mean_p_lfp_heights]
# error = [std_m_lfp_heights, std_p_lfp_heights]

# # Build the plot
# fig = plt.figure(figsize=(5,5))
# ax  = fig.add_subplot(111)
# ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,color='grey', ecolor='black', capsize=10)
# ax.set_xticks(x_pos)
# ax.set_xticklabels(materials,fontsize=16)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # Save the figure and show
# plt.tight_layout()
# plt.savefig(str(frequency)+'amplitude@1Hz_bar_plot.png')
# plt.show()



# m_raw_datas = np.array(m_raw_datas)
# p_raw_datas = np.array(p_raw_datas)
# lines,lengths = m_raw_datas.shape


# # Double check the data. 
# data_to_check = m_raw_datas[1,:]
# N = len(data_to_check)
# timestep        = 1.0/Fs
# start_section  = 1.7
# end_section    = 5.5
# fft_start       = int(start_section*Fs)
# fft_end         = int(end_section*Fs) 
# fft_data        = fft(data_to_check[fft_start:fft_end])
# N = fft_end - fft_start
# fft_data        = np.abs(2.0/(N-1) * (fft_data))[1:(N-1)//2]
# xf              = np.fft.fftfreq( (N-1), d=timestep)[:(N-1)//2]
# frequenciesx     = xf[1:(N-1)//2]

# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(111)
# plt.plot(frequenciesx,fft_data,'k')
# ax.set_xlim([0,frequency*2+50])
# plt.show()
# # Something has gone really wrong. 
# bl = 0.4  
# bh = 100
# if frequency >= 100:
#     bl = 60
#     bh = frequency + 1000

# sos_dfx_band = iirfilter(17, [bl,bh], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# sos_mains_cut = iirfilter(17, [48,52], rs=60, btype='bandstop',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# m_raw_datas   = sosfiltfilt(sos_mains_cut,m_raw_datas)
# p_raw_datas   = sosfiltfilt(sos_mains_cut,p_raw_datas)

# m_dfx_data   = sosfiltfilt(sos_dfx_band,m_raw_datas)
# p_dfx_data   = sosfiltfilt(sos_dfx_band,p_raw_datas)

# mean_m_raw_datas = np.mean(m_dfx_data,axis=0)
# # std_m_raw_datas = np.std(m_raw_datas,axis=0)
# mean_p_raw_datas = np.mean(p_dfx_data,axis=0)
# # std_p_raw_datas = np.std(p_raw_datas,axis=0)

# timestep        = 1.0/Fs
# N               = int(lengths)
# t               = np.linspace(0, lengths/Fs, N, endpoint=False)
# # savepath    = 'D:\\ae_mouse\\e121_stimulation\\t4_phantom\\'
# savepath    = 'D:\\ae_mouse\\e121_stimulation\\t3_mouse\\'
# file_number = applied_signal_file
# gain        = 500
# m_channel   = 0 
# rf_channel  = 2
# filename    = savepath + 't'+str(file_number)+'_stream.npy'
# data        = np.load(filename)
# fsignal     = 1e6*data[m_channel]/gain
# rfsignal    = 10*data[rf_channel]  
# # Fs          = 1e7
# # timestep    = 1.0/Fs
# # duration    = 6 
# # N           = int(Fs*duration)
# t2 = np.linspace(0, 6, int(1e7*6), endpoint=False)

# rf_cleaner = iirfilter(17, [1e6-10000,1e6 +10000], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=1e7,
#                        output='sos')
# clean_rf   = sosfiltfilt(rf_cleaner,rfsignal)
# #low frequencies. 

# # s1 = 0
# # s2 = 6
# s1 = 1.8
# s2 = 4.7


# if frequency >= 40 and frequency<300:
#     s1 = 3.0
#     s2 = 3.4
# if frequency >= 300:
#     s1 = 3.8
#     s2 = 3.805
# #
# s1 = 2
# s2 = 3
# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(211)
# plt.plot(t2,clean_rf,'k')
# ax.set_xlim([s1,s2])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax2  = fig.add_subplot(212)
# plt.plot(t,mean_m_raw_datas,'k')
# plt.plot(t,mean_p_raw_datas,'b')
# plt.plot(t,m_dfx_data.T,alpha=0.2)
# plt.plot(t,p_dfx_data.T,alpha=0.2)
# plt.plot(t,mean_m_raw_datas,'k')
# plt.plot(t,mean_p_raw_datas,'b')
# ax2.set_xlim([s1,s2])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.legend(['mouse','phantom'],loc='upper center',framealpha=0.0,fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax2.set_ylim([-200,200])
# plt.tight_layout()
# plt.savefig(str(frequency)+'representative_applied_vs_measured.png')
# plt.show()


