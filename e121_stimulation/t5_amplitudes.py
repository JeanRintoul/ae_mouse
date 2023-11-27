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
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16

element              = 0


frequency = 1
Fs      = 1e4
if frequency> 300:
    Fs      = 1e6

applied_signal_file = 33
print ('frequency is:', frequency)
#
data = np.load('f_1Hz_low_anesthesia_t5_data.npz',allow_pickle=True)
m_lfp_heights = data['lfp_heights']
m_peakds      = data['peakds']
m_peakrs      = data['peakrs']
m_raw_datas   = data['raw_datas']
m_r2s         = data['r2s']
m_d2s         = data['d2s']
#
pdata = np.load('f_1Hz_high_anesthesia_t5_data.npz',allow_pickle=True)
p_lfp_heights = pdata['lfp_heights']
p_peakds      = pdata['peakds']
p_peakrs      = pdata['peakrs']
p_raw_datas   = pdata['raw_datas']
p_r2s         = pdata['r2s']
p_d2s         = pdata['d2s']

mean_p_lfp_heights = np.mean(p_lfp_heights,axis=0)
std_p_lfp_heights = np.std(p_lfp_heights,axis=0)

mean_m_lfp_heights = np.mean(m_lfp_heights,axis=0)
std_m_lfp_heights = np.std(m_lfp_heights,axis=0)

print ('mean lfp heights p/m: ',mean_m_lfp_heights,mean_p_lfp_heights)
# T-test. 
sample1 = m_lfp_heights
sample2 = p_lfp_heights
t_stat, p_value = ttest_ind(sample1, sample2) 
# print('T-statistic value: ', t_stat) 
print('P-Value: ', p_value)

print ('lfp_heights numbers',len(m_lfp_heights),len(p_lfp_heights))
# Create lists for the plot
materials = ['Low Anesthesia','High Anesthesia']
x_pos = np.arange(len(materials))
CTEs = [mean_m_lfp_heights, mean_p_lfp_heights]
error = [std_m_lfp_heights, std_p_lfp_heights]

# Build the plot
fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111)
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,color='grey', ecolor='black', capsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(materials,fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Save the figure and show
plt.tight_layout()
plt.savefig(str(frequency)+'amplitude@1Hz_anesthesia_bar_plot.png')
plt.show()


# 
# 
toviolin = [m_lfp_heights, p_lfp_heights]

names = ['low anesthesia','high anesthesia']
fig = plt.figure(figsize=(5,3))
ax  = fig.add_subplot(111)
violin = ax.violinplot(toviolin,showmeans=True,showextrema=True)
for pc in violin["bodies"]:
    pc.set_facecolor("grey")
    # pc.set_edgecolor("black")
    pc.set_linewidth(1) 
    pc.set_alpha(0.5)
for partname in ('cbars','cmins','cmaxes','cmeans'):
        vp = violin[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1.6)
        vp.set_alpha(1) 
ax.set_xticks([1,2])  
ax.set_xticklabels(names)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.set_ylim([-70,120])
plt.tight_layout()
plt.savefig(str(frequency)+'Hz_phase_violin_anesthesia.png', bbox_inches='tight')
plt.show()



m_raw_datas = np.array(m_raw_datas)
p_raw_datas = np.array(p_raw_datas)
lines,lengths = m_raw_datas.shape
print ('low anesthesia lines',lines)
plines,plengths = p_raw_datas.shape
# Double check the data. 

timestep        = 1.0/Fs
# start_section  = 0.7
# end_section    = 4.8

# start_section  = 1.7
# end_section    = 5.5
start_section  = 0
end_section    = 6
fft_start       = int(start_section*Fs)
fft_end         = int(end_section*Fs) 
N = fft_end - fft_start
total_ffts_m = []
total_ffts_p = []
# for i in range(lines):
#     dtc_m = m_raw_datas[i,:]
dtc_m = m_raw_datas[lines-1,:]
fft_data        = fft(dtc_m[fft_start:fft_end])
fft_data        = np.abs(2.0/(N-1) * (fft_data))[1:(N-1)//2]
total_ffts_m.append(fft_data)

for i in range(plines):
    dtc_p = p_raw_datas[i,:]       
    fft_data        = fft(dtc_p[fft_start:fft_end])
    fft_data        = np.abs(2.0/(N-1) * (fft_data))[1:(N-1)//2]
    total_ffts_p.append(fft_data)

average_fft_m = np.mean(np.array(total_ffts_m),axis=0 )
average_fft_p = np.mean(np.array(total_ffts_p),axis=0 )

xf              = np.fft.fftfreq( (N-1), d=timestep)[:(N-1)//2]
frequenciesx     = xf[1:(N-1)//2]

fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111)
plt.plot(frequenciesx,average_fft_m,'k')
plt.plot(frequenciesx,average_fft_p,'r')
ax.set_xlim([0,10])
plt.legend(['low anesthesia','high anesthesia'],loc='upper right',framealpha=0.0,fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.tight_layout()
plt.savefig('anesthesia_fft.png', bbox_inches='tight')
plt.show()


# Something has gone really wrong. 
bl = 0.1  
bh = 40
# if frequency >= 100:
#     bl = 60
#     bh = frequency + 1000

sos_dfx_band = iirfilter(17, [bl,bh], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

sos_spike_band = iirfilter(17, [300,1500], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# sos_mains_cut = iirfilter(17, [48,52], rs=60, btype='bandstop',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# m_raw_datas   = sosfiltfilt(sos_mains_cut,m_raw_datas)
# p_raw_datas   = sosfiltfilt(sos_mains_cut,p_raw_datas)

m_dfx_data   = sosfiltfilt(sos_dfx_band,m_raw_datas)
p_dfx_data   = sosfiltfilt(sos_dfx_band,p_raw_datas)

m_spike_data   = sosfiltfilt(sos_spike_band,m_raw_datas)
p_spike_data   = sosfiltfilt(sos_spike_band,p_raw_datas)

mean_m_raw_datas = np.mean(m_dfx_data,axis=0)
# std_m_raw_datas = np.std(m_raw_datas,axis=0)
mean_p_raw_datas = np.mean(p_dfx_data,axis=0)
# std_p_raw_datas = np.std(p_raw_datas,axis=0)


timestep        = 1.0/Fs
N               = int(lengths)
t               = np.linspace(0, lengths/Fs, N, endpoint=False)

savepath    = 'D:\\ae_mouse\\e121_stimulation\\t5_anesthesia\\'
file_number = applied_signal_file
gain        = 500
m_channel   = 0 
rf_channel  = 2
filename    = savepath + 't'+str(file_number)+'_stream.npy'
data        = np.load(filename)
fsignal     = 1e6*data[m_channel]/gain
rfsignal    = 10*data[rf_channel]  
# Fs          = 1e7
# timestep    = 1.0/Fs
# duration    = 6 
# N           = int(Fs*duration)
t2 = np.linspace(0, 6, int(1e7*6), endpoint=False)

rf_cleaner = iirfilter(17, [1e6-10000,1e6 +10000], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=1e7,
                       output='sos')
clean_rf   = sosfiltfilt(rf_cleaner,rfsignal)
#low frequencies. 

# s1 = 0.7
# s2 = 4.6
# s1 = 1.8
# s2 = 4.7
s1 = 0.0
s2 = 6.0

fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(211)
plt.plot(t2,clean_rf,'k')
ax.set_xlim([s1,s2])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax2  = fig.add_subplot(212)
plt.plot(t,mean_m_raw_datas,'k')
plt.plot(t,mean_p_raw_datas,'b')
plt.plot(t,m_dfx_data.T,alpha=0.2)
plt.plot(t,p_dfx_data.T,alpha=0.2)
# plt.plot(t,mean_m_raw_datas,'k')
# plt.plot(t,m_raw_datas[lines-1,:],'k')
plt.plot(t,mean_p_raw_datas,'b')
ax2.set_xlim([s1,s2])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.legend(['mouse','phantom'],loc='upper center',framealpha=0.0,fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
# ax2.set_ylim([-300,300])
plt.tight_layout()
plt.savefig(str(frequency)+'representative_applied_vs_measured.png')
plt.show()

# # spikes
# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(311)
# plt.plot(t2,clean_rf,'k')
# ax.set_xlim([s1,s2])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax2  = fig.add_subplot(312)
# plt.plot(t,m_spike_data.T)
# ax2.set_xlim([s1,s2])
# # plt.plot(t,mean_p_raw_datas,'b')
# ax3  = fig.add_subplot(313)
# plt.plot(t,p_spike_data.T)
# ax3.set_xlim([s1,s2])
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# # plt.legend(['mouse','phantom'],loc='upper center',framealpha=0.0,fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plt.tight_layout()
# plt.savefig(str(frequency)+'spikes.png')
# plt.show()


# m_spike_data   = sosfiltfilt(sos_spike_band,m_raw_datas)
# p_spike_data