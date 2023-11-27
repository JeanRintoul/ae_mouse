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
import seaborn as sns 
import pandas as pd 
import scipy.stats as stats 
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16
# 
# 
# Load in the pre-aligned phase data files. 
m_data = np.load('mouse_phase_data.npz',allow_pickle=True)
m_r2s 		= m_data['r2s']
m_d2s 		= m_data['d2s']
m_peakrs 	= m_data['peakrs']
m_peakds 	= m_data['peakds']
m_t2 		= m_data['t2']
# 
p_data = np.load('phantom_phase_data.npz',allow_pickle=True)
p_r2s 		= p_data['r2s']
p_d2s 		= p_data['d2s']
p_peakrs 	= p_data['peakrs']
p_peakds 	= p_data['peakds']
p_t2 		= p_data['t2']
# 

frequencies = [1,2,5,10,40,100,300,1000]
t 			= p_t2
# 
# 
element = 0
# 
# 
m_peaks = [] 
p_peaks = []
m_peaks_means = [] 
p_peaks_means = []
m_peaks_stds  = [] 
p_peaks_stds  = []

for k in range(len(frequencies)):
	element = k
	print ('frequency',frequencies[k])
	print ('p peaks rs',np.array(p_peakrs).shape )
	a,b 					= np.array(p_peakrs).shape
	period            		= 1/frequencies[element]
	frequency_of_interest 	= frequencies[element]
	m_peaks_degrees 		= []
	p_peaks_degrees 		= []


	for i in range(a): # iterate through each file.
		m_internal_list 			= m_peakrs[i]
		m_internal_list_measured 	= m_peakds[i]
		p_internal_list 			= p_peakrs[i]
		p_internal_list_measured 	= p_peakds[i]
		# print ('file',i)
		if m_internal_list[1] == element: # frequency match. 
			m_peaks_applied  		= m_internal_list[2]
			m_peaks_measured 		= m_internal_list_measured[2]
			# print ('m_peaks',len(m_peaks_applied),len(m_peaks_measured))
			# dodge work around if I missed one. 
			# if (len(m_peaks_measured) - len(m_peaks_applied)) > 0:
			# 	m_peaks_measured = m_peaks_measured[0:len(m_peaks_applied)]

			m_peak_times        	= t[m_peaks_applied]-t[m_peaks_measured]
			
			m_peaks_degree_offset = 360*m_peak_times/period
			# print ('degree offset: ',m_peaks_degree_offset)
			for j in range(len(m_peaks_degree_offset)):
				m_peaks_degrees.append(m_peaks_degree_offset[j])

			p_peaks_applied  		= p_internal_list[2]
			p_peaks_measured 		= p_internal_list_measured[2]
			# print ('p_peaks',len(p_peaks_applied),len(p_peaks_measured))
			p_peak_times        	= t[p_peaks_applied]-t[p_peaks_measured]
			p_peaks_degree_offset = 360*p_peak_times/period
			for j in range(len(p_peaks_degree_offset)):
				p_peaks_degrees.append(p_peaks_degree_offset[j])
			# p_peaks_degrees.append(p_peaks_degree_offset)
	# print ('m_peaks_degrees',m_peaks_degrees)
	# print ('p_peaks_degrees',p_peaks_degrees)
	# print ('len m and p',len(m_peaks_degrees),len(p_peaks_degrees))
	m_peaks.append(m_peaks_degrees)
	p_peaks.append(p_peaks_degrees)
	# This breaks, when one of the values above is NOT aligned. 
	m_mean_offset        = np.mean(m_peaks_degrees)
	m_std_offset         = np.std(m_peaks_degrees)
	p_mean_offset        = np.mean(p_peaks_degrees)
	p_std_offset         = np.std(p_peaks_degrees)
	print ('m offsets',m_mean_offset,m_std_offset)
	print ('p offsets',p_mean_offset,p_std_offset)
	m_peaks_means.append(m_mean_offset)
	p_peaks_means.append(p_mean_offset)
	m_peaks_stds.append(m_std_offset)
	p_peaks_stds.append(p_std_offset)


print ('final set of m peak means',m_peaks_means)
print ('final set of p peak means',p_peaks_means)

element = 0 
for k in range(len(frequencies)):
	element = k 
	group1 = m_peaks[element]
	group2 = p_peaks[element]
	#perform two sample t-test with equal variances
	sta,p = stats.ttest_ind(a=group1, b=group2, equal_var=True)
	print ('freq,p-value',frequencies[k],p)


x = np.arange(len(frequencies)) 
width = 0.40
# Do a grouped bar plot. 
fig = plt.figure(figsize=(6,3))
ax  = fig.add_subplot(111)
# plot data in grouped manner of bar type 
plt.bar(x-0.2, m_peaks_means, width,color='grey') 
plt.bar(x+0.2, p_peaks_means, width,color='blue') 
plt.errorbar(x-0.2, m_peaks_means, yerr=m_peaks_stds, fmt=".",color="k")
plt.errorbar(x+0.2, p_peaks_means, yerr=p_peaks_stds, fmt=".",color="b")
plt.xticks(x, frequencies) 
plt.legend(["Mouse", "Phantom"],framealpha=0.0,fontsize=16,loc='upper right') 
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.tight_layout()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('phase_bar_plot.png', bbox_inches='tight')
plt.show()



# fig = plt.figure(figsize=(5,3))
# ax  = fig.add_subplot(111)
# violin = ax.violinplot(m_peaks,showmeans=True)
# for pc in violin["bodies"]:
#     pc.set_facecolor("grey")
#     # pc.set_edgecolor("black")
#     pc.set_linewidth(1) 
#     pc.set_alpha(0.5)
# for partname in ('cbars','cmins','cmaxes','cmeans'):
#         vp = violin[partname]
#         vp.set_edgecolor("black")
#         vp.set_linewidth(1.6)
#         vp.set_alpha(1) 
# ax.set_xticks([1,2,3,4,5,6,7,8])  
# ax.set_xticklabels(frequencies)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_ylim([-70,120])
# plt.tight_layout()
# plt.savefig('mouse_violin_peak_offsets.png', bbox_inches='tight')
# plt.show()

# fig = plt.figure(figsize=(5,3))
# ax  = fig.add_subplot(111)
# violin = ax.violinplot(p_peaks,showmeans=True)
# for pc in violin["bodies"]:
#     pc.set_facecolor("grey")
#     # pc.set_edgecolor("black")
#     pc.set_linewidth(1) 
#     pc.set_alpha(0.5)
# for partname in ('cbars','cmins','cmaxes','cmeans'):
#         vp = violin[partname]
#         vp.set_edgecolor("black")
#         vp.set_linewidth(1.6)
#         vp.set_alpha(1) 
# ax.set_xticks([1,2,3,4,5,6,7,8])  
# ax.set_xticklabels(frequencies)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.set_ylim([-70,120])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plt.savefig('phantom_violin_peak_offsets.png', bbox_inches='tight')
# plt.show()


# element = 3
# print ('r2 shape',m_r2s.shape)
# m_applied  = m_r2s[:,element,:].T
# m_response = m_d2s[:,element,:].T
# p_applied  = p_r2s[:,element,:].T
# p_response = p_d2s[:,element,:].T


# fig = plt.figure(figsize=(5,3))
# ax  = fig.add_subplot(211)
# plt.plot(t,m_applied,alpha=0.4)
# plt.plot(t,m_response)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax2  = fig.add_subplot(212)
# plt.plot(t,p_applied,alpha=0.4)
# plt.plot(t,p_response)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plt.tight_layout()
# plt.savefig('phase_offsets_frequency_'+str(frequencies[element])+'.png', bbox_inches='tight')

# plt.show()
# 
# 
# peak_offsets        = t2[peaksr]-t2[peaksd]
# timestep            = t2[2]-t2[1]
# peak_times          = peak_offsets
# print ('peak 0times',peak_times)
# period              = 1/dfx
# peaks_degree_offset = 360*peak_times/period
# mean_offset         = np.mean(peaks_degree_offset)
# print ('peak_times degree offset: ', peaks_degree_offset,mean_offset)
# print ('dfx: ', dfx)
# print ('mean degree offset: ', mean_offset)


# Fs 		= 1e4
# df_h 	= 1400
# sos_lfp_band = iirfilter(17, [df_h], rs=60, btype='lowpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')

# mains_cut_band = iirfilter(17, [48,52], rs=60, btype='bandstop',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')

# N 				= int(Fs*6)
# duration 		= 3 
# t               = np.linspace(0, duration, N, endpoint=False)

# experiment_type 	= 'mouse'





# frequency_element 	= 3
# frequency 			= df_ramp_frequencies[frequency_element]
# print ('frequency is:', frequency)
# print ('shape is: ',lfp_aggregates.shape)
# lfp_aggregatesm     = sosfiltfilt(mains_cut_band, lfp_aggregates[:,frequency_element,])
# plfp_aggregatesm   = sosfiltfilt(mains_cut_band, plfp_aggregates[:,frequency_element,])
# print ('shape is: ',lfp_aggregates.shape)

# mouse_lfp_data     = sosfiltfilt(sos_lfp_band, lfp_aggregatesm[:,]).T
# phantom_lfp_data   = sosfiltfilt(sos_lfp_band, plfp_aggregatesm[:,]).T


# print ('t',len(t))
# mouse_lfp_means = np.mean(mouse_lfp_data,axis=1)
# mouse_lfp_stds = np.std(mouse_lfp_data,axis=1)
# print (mouse_lfp_means.shape)
# print ('mouse lfp data shape',mouse_lfp_data.shape)

# phantom_lfp_means = np.mean(phantom_lfp_data,axis=1)
# phantom_lfp_stds = np.std(phantom_lfp_data,axis=1)
# print (phantom_lfp_means.shape)
# print ('phantom lfp data shape',phantom_lfp_data.shape)


# fig = plt.figure(figsize=(5,3))
# ax  = fig.add_subplot(111)
# plt.plot(t,mouse_lfp_data,alpha=0.2)
# plt.plot(t,mouse_lfp_means,'k')
# plt.fill_between(t,mouse_lfp_means-mouse_lfp_stds, mouse_lfp_means+mouse_lfp_stds,alpha=0.2,color='grey')
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.set_xlim([1,1.3])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plt.savefig('mouse_lfp_dfx_frequency_'+str(frequency)+'.png', bbox_inches='tight')
# plt.show()

# fig = plt.figure(figsize=(5,3))
# ax  = fig.add_subplot(111)
# plt.plot(t,phantom_lfp_data,alpha=0.2)
# plt.plot(t,phantom_lfp_means,'k')
# plt.fill_between(t,phantom_lfp_means-phantom_lfp_stds, phantom_lfp_means+phantom_lfp_stds,alpha=0.2,color='grey')
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.set_xlim([1,1.3])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plt.savefig('phantom_lfp_dfx_frequency_'+str(frequency)+'.png', bbox_inches='tight')
# plt.show()


# plt.fill_between(df_ramp_frequencies, lfp_height_means-lfp_height_stds, lfp_height_means+lfp_height_stds,alpha=0.2,color='grey')
# plt.plot(df_ramp_frequencies,ppp3.T,'.b')

# plt.fill_between(df_ramp_frequencies, plfp_height_means-plfp_height_stds, plfp_height_means+plfp_height_stds,alpha=0.2,color='b')
# ax.set_xlim([5,1050])
# ax.set_ylim([0,np.max(pp3)])
# # plt.legend(['mouse','phantom'],loc='lower right')
# plt.legend(['mouse','phantom'],loc='lower right',framealpha=0.0,fontsize=16)

# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plt.savefig('dfx_filter.png', bbox_inches='tight')
# plt.show()

# fig = plt.figure(figsize=(5,5))
# ax  = fig.add_subplot(111)
# plt.plot(df_ramp_frequencies,lfp_height_means2,'k')
# plt.plot(df_ramp_frequencies,plfp_height_means2,'b')
# plt.plot(df_ramp_frequencies,pp1.T,'.k')

# plt.fill_between(df_ramp_frequencies, lfp_height_means2-lfp_height_stds2, lfp_height_means2+lfp_height_stds2,alpha=0.2,color='grey')

# plt.plot(df_ramp_frequencies,ppp1.T,'.b')
# plt.plot(df_ramp_frequencies,plfp_height_means2,'b')
# plt.fill_between(df_ramp_frequencies, plfp_height_means2-plfp_height_stds2, plfp_height_means2+plfp_height_stds2,alpha=0.2,color='b')
# ax.set_xlim([5,1050])
# ax.set_ylim([0,np.max(pp1)])
# plt.legend(['mouse','phantom'],loc='lower right',framealpha=0.0,fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.tight_layout()
# plt.savefig('fft_pp.png', bbox_inches='tight')
# plt.show()


# #
# # FFT amplitude
# direct_10_df        = [6.09,11.23,5.87,0.9,0.09]
# # e115 t3, 7,8,9,10 
# modulated_10_df     = [171.15,176.38,177.69,178.59,179.35] #

# direct_mean     = np.mean(direct_10_df) 
# modulated_mean  = np.mean(modulated_10_df) 

# direct_std = np.std(direct_10_df) 
# modulated_std = np.std(modulated_10_df) 

# # T-test. 
# sample1 = direct_10_df
# sample2 = modulated_10_df
# t_stat, p_value = ttest_ind(sample1, sample2) 
# print('T-statistic value: ', t_stat) 
# print('P-Value: ', p_value)
# # 
# # Now do Bar plot. 
# # 

# # Create lists for the plot
# materials = ['Direct 10Hz', 'Modulated 10Hz']
# x_pos = np.arange(len(materials))
# CTEs = [direct_mean,modulated_mean]
# error = [direct_std,modulated_std]

# # Build the plot
# # fig, ax = plt.subplots()
# fig = plt.figure(figsize=(5,5))
# ax  = fig.add_subplot(111)

# ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,color='grey', ecolor='black', capsize=10)
# ax.set_xticks(x_pos)
# ax.set_xticklabels(materials)
# # ax.yaxis.grid(True)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # Save the figure and show
# plt.tight_layout()
# plt.savefig('bar_plot_with_error_bars.png')
# plt.show()


