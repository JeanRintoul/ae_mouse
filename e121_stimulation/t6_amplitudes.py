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

element     = 0
frequency   = 1
Fs          = 1e4


applied_signal_file = 14
print ('frequency is:', frequency)
#
data = np.load('2low_anesthesia_t6_data.npz',allow_pickle=True)
m_raw_datas   = data['raw_datas']
m_r2s         = data['r2s']
m_d2s         = data['d2s']
m_carrier_datas = data['carrier_datas']
m_final_peak_heights = data['final_peak_heights']
#
pdata = np.load('2high_anesthesia_t6_data.npz',allow_pickle=True)
p_raw_datas   = pdata['raw_datas']
p_r2s         = pdata['r2s']
p_d2s         = pdata['d2s']
p_carrier_datas = pdata['carrier_datas']
p_final_peak_heights = pdata['final_peak_heights']


#  carrier amplitudes 
mean_p_carrier_heights = np.mean(p_carrier_datas,axis=0)
std_p_carrier_heights = np.std(p_carrier_datas,axis=0)
mean_m_carrier_heights = np.mean(m_carrier_datas,axis=0)
std_m_carrier_heights = np.std(m_carrier_datas,axis=0)

# 1Hz amplitudes. 
mean_p_lfp_heights = np.mean(p_final_peak_heights,axis=0)
std_p_lfp_heights = np.std(p_final_peak_heights,axis=0)
mean_m_lfp_heights = np.mean(m_final_peak_heights,axis=0)
std_m_lfp_heights = np.std(m_final_peak_heights,axis=0)

print ('mean lfp heights p/m: ',mean_m_lfp_heights,mean_p_lfp_heights)
# T-test. 
sample1 = m_final_peak_heights
sample2 = p_final_peak_heights
t_stat, p_value_1hz = ttest_ind(sample1, sample2) 
# print('T-statistic value: ', t_stat) 
print('P-Value 1hz: ', p_value_1hz)
t_stat, p_value_carrier = ttest_ind(m_carrier_datas, p_carrier_datas) 
print('P-Value carrier: ', p_value_carrier)

print ('lfp_heights numbers',len(m_final_peak_heights),len(p_final_peak_heights))

# Do a longitudinal plot of carrier amplitudes too. 
fig = plt.figure(figsize=(5,3))
ax  = fig.add_subplot(111)
plt.plot(p_carrier_datas,'.k')
plt.plot(m_carrier_datas,'.r')
plt.legend(['high anesthesia','low anesthesia'])
plt.tight_layout()
plt.savefig('carrier_mean_points.png', bbox_inches='tight')
plt.show()

names = ['low anesthesia','high anesthesia']

# Do a carrier amplitude plot. 
carrier_toviolin = [m_carrier_datas, p_carrier_datas]
fig = plt.figure(figsize=(5,3))
ax  = fig.add_subplot(111)
violin = ax.violinplot(carrier_toviolin,showmeans=True,showextrema=True)
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
# plt.legend(['Carrier Amplitudes:'],str(np.round(p_value_carrier,2)),loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.set_ylim([-70,120])
plt.tight_layout()
plt.savefig('anesthesia_carrier_violin.png', bbox_inches='tight')
plt.show()


# 
# 
toviolin = [m_final_peak_heights, p_final_peak_heights]

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
# plt.legend(['Carrier Amplitudes:'],str(np.round(p_value_1hz,2)),loc='upper right' )
# ax.set_ylim([-70,120])
plt.tight_layout()
plt.savefig('anesthesia_1Hz_violin.png', bbox_inches='tight')
plt.show()



m_raw_datas = np.array(m_raw_datas)
p_raw_datas = np.array(p_raw_datas)
lines,lengths = m_raw_datas.shape
print ('low anesthesia lines',lines)
plines,plengths = p_raw_datas.shape
# Double check the data. 
timestep        = 1.0/1e4
start_section  = 0
end_section    = 6
fft_start       = int(start_section*Fs)
fft_end         = int(end_section*Fs) 

total_ffts_m = []
total_ffts_p = []

for i in range(lines):
    if i > 0:
        dtc_m = np.concatenate( (dtc_m,m_raw_datas[i,:]) )
    else: 
        dtc_m = m_raw_datas[i,:]
    print (len(dtc_m))
N               = len(dtc_m)
fft_data        = fft(dtc_m)
mfft_data        = np.abs(2.0/(N-1) * (fft_data))[1:(N-1)//2]
xf              = np.fft.fftfreq( (N-1), d=timestep)[:(N-1)//2]
mfrequencies    = xf[1:(N-1)//2]

for i in range(plines):
    if i > 0:
        dtc_p = np.concatenate( (dtc_p,p_raw_datas[i,:]) )
    else: 
        dtc_p = p_raw_datas[i,:]
N               = len(dtc_p)
fft_data        = fft(dtc_p)
pfft_data       = np.abs(2.0/(N-1) * (fft_data))[1:(N-1)//2]
xf              = np.fft.fftfreq( (N-1), d=timestep)[:(N-1)//2]
pfrequencies    = xf[1:(N-1)//2]


# fig = plt.figure(figsize=(5,5))
# ax  = fig.add_subplot(111)
# plt.plot(mfrequencies,mfft_data,'k')
# plt.plot(pfrequencies,pfft_data,'r')
# ax.set_xlim([0,10])
# plt.legend(['low anesthesia','high anesthesia'],loc='upper right',framealpha=0.0,fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.tight_layout()
# plt.savefig('anesthesia_fft.png', bbox_inches='tight')
# plt.show()



bl = 0.1  
bh = 300
sos_dfx_band = iirfilter(17, [bl,bh], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

sos_mains_cut = iirfilter(17, [48,52], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
m_raw_datas   = sosfiltfilt(sos_mains_cut,m_raw_datas)
p_raw_datas   = sosfiltfilt(sos_mains_cut,p_raw_datas)

m_dfx_data   = sosfiltfilt(sos_dfx_band,m_raw_datas)
p_dfx_data   = sosfiltfilt(sos_dfx_band,p_raw_datas)



mean_m_raw_datas = np.mean(m_dfx_data,axis=0)
mean_p_raw_datas = np.mean(p_dfx_data,axis=0)

timestep        = 1.0/Fs
N               = int(lengths)
t               = np.linspace(0, lengths/Fs, N, endpoint=False)

savepath    = 'D:\\ae_mouse\\e121_stimulation\\t6_anesthesia\\'
file_number = 22
gain        = 500
m_channel   = 0 
rf_channel  = 2
filename    = savepath + 't'+str(file_number)+'_stream.npy'
data        = np.load(filename)
fsignal     = 1e6*data[m_channel]/gain
rfsignal    = 10*data[rf_channel]  
t2 = np.linspace(0, 6, int(1e7*6), endpoint=False)

rf_cleaner = iirfilter(17, [1e6-10000,1e6 +10000], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=1e7,
                       output='sos')
clean_rf   = sosfiltfilt(rf_cleaner,rfsignal)


pfile_number = 23
pfilename    = savepath + 't'+str(pfile_number)+'_stream.npy'
pdata        = np.load(pfilename)
pfsignal     = 1e6*pdata[m_channel]/gain
pfsignal_carrier   = sosfiltfilt(rf_cleaner,pfsignal)

# low frequencies. 
fsignal_carrier   = sosfiltfilt(rf_cleaner,fsignal)
s1 = 0.0
s2 = 6.0
# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(111)
# plt.plot(t2,fsignal_carrier,'k')
# plt.plot(t2,pfsignal_carrier,'r')
# plt.legend(['low anesthesia','high anesthesia'])
# # plt.tight_layout()
# plt.savefig('representative_carrier.png', bbox_inches='tight')
# plt.show()



y1 = -300 
y2 = 300
fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(411)
plt.plot(t2,clean_rf,'k')
ax.set_xlim([s1,s2])
ax.set_ylim([y1,y2])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax2  = fig.add_subplot(412)
plt.plot(t,p_dfx_data.T,alpha=0.4)
plt.plot(t,mean_p_raw_datas,'b')
ax2.set_xlim([s1,s2])
ax2.set_ylim([y1,y2])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax2.set_xlim([s1,s2])
ax3  = fig.add_subplot(413)
plt.plot(t,m_dfx_data.T,alpha=0.4)
plt.plot(t,mean_m_raw_datas,'k')
ax3.set_xlim([s1,s2])
ax3.set_ylim([y1,y2])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

ax4  = fig.add_subplot(414)
plt.plot(t,mean_p_raw_datas,'b')
plt.plot(t,mean_m_raw_datas,'k')
ax4.set_xlim([s1,s2])
ax4.set_ylim([y1,y2])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)

# ax2.set_ylim([-300,300])
plt.tight_layout()
plt.savefig(str(frequency)+'representative_applied_vs_measured.png')
plt.show()

