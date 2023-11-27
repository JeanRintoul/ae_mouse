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


# frequencies          = [1,2,5,10,40,100,300,1000,2000,5000]
frequencies          = [1,2,5,10,40,100]
m_phase_totals = []
p_phase_totals = []
for i in range(len(frequencies)):
    frequency = frequencies[i]
    print ('frequency/i:',frequency,i)
    Fs              = 1e4 
    timestep        = 1.0/Fs
    #
    data = np.load('f_'+str(frequency)+'Hz_mouse_data.npz',allow_pickle=True)
    m_peakds      = np.array(data['peakds'])
    m_peakrs      = np.array(data['peakrs'])
    # print ('m_peaksds',lenm_peakds)
    m_phases = []
    for j in range(len(m_peakds)):
        peaksr  = m_peakrs[j]
        peaksd  = m_peakds[j]
        # print ('peaksr',peaksr)
        peak_offsets        = (peaksr-peaksd)
        # print ('peak offsets', peak_offsets)
        peak_times          = peak_offsets*timestep
        # print ('peak times',peak_times)
        period              = 1/frequency
        peaks_degree_offset = 360*peak_times/period
        # print ('offsets',peaks_degree_offset)
        for k in range(len(peaks_degree_offset)):
            m_phases.append(peaks_degree_offset[k])
    # print ('m_phases',m_phases)
    m_phase_totals.append(m_phases)
    # 
    pdata = np.load('f_'+str(frequency)+'Hz_phantom_data.npz',allow_pickle=True)
    p_peakds      = pdata['peakds']
    p_peakrs      = pdata['peakrs']
    p_phases = []
    for j in range(len(m_peakds)):
        peaksr  = p_peakrs[j]
        peaksd  = p_peakds[j]
        # print ('peaksr',peaksr)
        peak_offsets        = (peaksr-peaksd)
        # print ('peak offsets', peak_offsets)
        peak_times          = peak_offsets*timestep
        # print ('peak times',peak_times)
        period              = 1/frequency
        peaks_degree_offset = 360*peak_times/period
        # print ('offsets',peaks_degree_offset)
        for k in range(len(peaks_degree_offset)):
            p_phases.append(peaks_degree_offset[k])
    # print ('p_phases',p_phases)
    p_phase_totals.append(p_phases)


# print (p_phase_totals[0])
toviolin = p_phase_totals 
toviolin = m_phase_totals 
# names = ['mouse','phantom']
fig = plt.figure(figsize=(5,3))
ax  = fig.add_subplot(111)
violin = ax.violinplot(toviolin,showmeans=True)
violin = ax.violinplot(toviolin,showmedians=True,showmeans=True)
for pc in violin["bodies"]:
    pc.set_facecolor("grey")
    # pc.set_edgecolor("black")
    pc.set_linewidth(1) 
    pc.set_alpha(0.5)  
for partname in ('cbars','cmins','cmaxes','cmedians','cmeans'):
        vp = violin[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1.6)
        vp.set_alpha(1) 
ax.set_xticks([1,2,3,4,5,6])  
ax.set_xticklabels(frequencies)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.set_ylim([-70,120])
plt.tight_layout()
plt.savefig('phaseramp_violin.png', bbox_inches='tight')
plt.show()












