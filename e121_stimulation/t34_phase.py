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


Fs 		    = 1e4
frequency   = 1 # in Hz. 

print ('frequency is:', frequency)
#
data = np.load('f_'+str(frequency)+'Hz_mouse_data.npz',allow_pickle=True)
m_lfp_heights = data['lfp_heights']
m_peakds      = data['peakds']
m_peakrs      = data['peakrs']
m_raw_datas   = data['raw_datas']
m_r2s         = data['r2s']
m_d2s         = data['d2s']
#
pdata = np.load('f_'+str(frequency)+'Hz_phantom_data.npz',allow_pickle=True)
p_lfp_heights = pdata['lfp_heights']
p_peakds      = pdata['peakds']
p_peakrs      = pdata['peakrs']
p_raw_datas   = pdata['raw_datas']
p_r2s         = pdata['r2s']
p_d2s         = pdata['d2s']

p_raw_datas = np.array(p_raw_datas)
a,lengths = p_raw_datas.shape
print('shape is:',a,lengths)  # 14,60000

timestep        = 1.0/Fs
N               = int(lengths)
t               = np.linspace(0, lengths/Fs, lengths, endpoint=False)
timestep        = t[2]-t[1]
dfx             = frequency

p_phases = []
p_phases2 = []
for i in range(a):
    # print (i)
    peaksr  = p_peakrs
    peaksd  = p_peakds
    peak_offsets        = t[peaksr[i]]-t[peaksd[i]]
    peak_times          = peak_offsets
    # print ('peak times',peak_times)
    period              = 1/dfx
    peaks_degree_offset = 360*peak_times/period
    # print ('offsets',peaks_degree_offset)
    for j in range(len(peaks_degree_offset)):
        p_phases2.append(peaks_degree_offset[j])
        if peaks_degree_offset[j] < 0:
            peaks_degree_offset[j] = 360+peaks_degree_offset[j]
        p_phases.append(peaks_degree_offset[j])

m_phases = []
m_phases2 = []
for i in range(a):
    # print (i)
    peaksr  = m_peakrs
    peaksd  = m_peakds
    peak_offsets        = t[peaksr[i]]-t[peaksd[i]]
    peak_times          = peak_offsets
    # print ('peak times',peak_times)
    period              = 1/dfx
    peaks_degree_offset = 360*peak_times/period
    # print ('offsets',peaks_degree_offset)
    for j in range(len(peaks_degree_offset)):
        m_phases2.append(peaks_degree_offset[j])
        if peaks_degree_offset[j] < 0:
            peaks_degree_offset[j] = 360+peaks_degree_offset[j]
        m_phases.append(peaks_degree_offset[j])

# print ('p phases',p_phases)
# print ('m phases',m_phases)
p_mean_offset         = np.mean(p_phases)
m_mean_offset         = np.median(m_phases)
print ('median offset',m_mean_offset)

# T-test. 
sample1 = m_phases
sample2 = p_phases
t_stat, p_value = ttest_ind(sample1, sample2) 
# print('T-statistic value: ', t_stat) 

print('P-Value: ', p_value)

print ('n m and p phases',len(m_phases),len(p_phases))

toviolin = [m_phases2, p_phases2]

names = ['mouse','phantom']
fig = plt.figure(figsize=(5,3))
ax  = fig.add_subplot(111)
violin = ax.violinplot(toviolin,showmedians=True)
for pc in violin["bodies"]:
    pc.set_facecolor("grey")
    # pc.set_edgecolor("black")
    pc.set_linewidth(1) 
    pc.set_alpha(0.5)
for partname in ('cbars','cmins','cmaxes','cmedians'):
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
plt.savefig(str(frequency)+'Hz_phase_violin.png', bbox_inches='tight')
plt.show()





m_degrees = m_phases
m_radians = np.deg2rad(m_degrees)
p_degrees = p_phases
p_radians = np.deg2rad(p_degrees)




bin_size = 5
a , b=np.histogram(m_degrees, bins=np.arange(0, 360+bin_size, bin_size))
centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])
c , d=np.histogram(p_degrees, bins=np.arange(0, 360+bin_size, bin_size))
pcenters = np.deg2rad(np.ediff1d(d)//2 + d[:-1])

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='polar')
ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.8', edgecolor='k',label='mouse')
ax.bar(pcenters, c, width=np.deg2rad(bin_size), bottom=0.0, color='.3', edgecolor='k',label='phantom')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.set_theta_zero_location("E")
ax.set_theta_direction(1)
# plt.legend(['mouse','phantom'],loc='upper right',framealpha=0.0,fontsize=16)
plt.tight_layout()
plt.savefig(str(frequency)+'Hz_circular_histogram.png', bbox_inches='tight')
plt.show()

