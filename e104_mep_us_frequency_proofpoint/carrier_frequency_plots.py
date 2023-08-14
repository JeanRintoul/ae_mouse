'''

 in vivo analysis of difference and sum trends. 

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
import sys
from scipy.signal import iirfilter,sosfiltfilt
sys.path.append('D:\\ae_mouse')  #  so that we can import from the parent folder. 
import mouse_library as m
# 
# 
# 
filename1       = '250000.npz'
filename2       = '500000.npz'
filename3       = '750000.npz'
filename4       = '1000000.npz'

carrier_frequencies = [0.25,0.5,0.75,1]
# 
data1           = np.load(filename1)
d1        	  	= data1['outs_array']

data2           = np.load(filename2)
d2        	  	= data2['outs_array']

data3           = np.load(filename3)
d3        	  	= data3['outs_array']

data4           = np.load(filename4)
d4        	  	= data4['outs_array']
#

d1_means = np.mean(d1,0)
d1_stds  = np.std(d1,0)
d2_means = np.mean(d2,0)
d2_stds  = np.std(d2,0)
d3_means = np.mean(d3,0)
d3_stds  = np.std(d3,0)
d4_means = np.mean(d4,0)
d4_stds  = np.std(d4,0)
# #
# print (d1_means)
dfs = [d1_means[0],d2_means[0],d3_means[0],d4_means[0] ]
sfs = [d1_means[1],d2_means[1],d3_means[1],d4_means[1] ]
c1s = [d1_means[2],d2_means[2],d3_means[2],d4_means[2] ]
c2s = [d1_means[3],d2_means[3],d3_means[3],d4_means[3] ]
# group_array = np.vstack( (d1,d2,d3,d4))
# 
# dfs = group_array[:,0]
print ( 'dfs',dfs )

# 
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
plt.plot(carrier_frequencies,dfs,'.k-')
plt.plot(carrier_frequencies,sfs,'.r-')
ax.set_xlabel('Frequency(MHz)')
ax.set_ylabel('Volts ($\mu$V)')
plt.legend(['difference freqs','sum frequencies'])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = 'mixing_frequencies_with_carrier.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
plt.plot(carrier_frequencies,c1s,'.k-')
plt.plot(carrier_frequencies,c2s,'.r-')
ax.set_xlabel('Frequency(MHz)')
ax.set_ylabel('Volts ($\mu$V)')
plt.legend(['carrier 1','carrier 2'])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = 'carrier_amplitudes_with_frequency.png'
plt.savefig(plot_filename)
plt.show()

# cf250 = [11,12,13]
# cf500 = [10,14,15]
# cf750 = [16,17,18]
# cf100 = [19,20,21]

# batch 		= cf100
# carrier 	= 1000000

# savepath        = 'D:\\ae_mouse\\e104_mep_us_frequency_proofpoint\\t1\\'
# gain            = 500
# Fs              = 5e6
# duration        = 4.0	
# timestep        = 1.0/Fs
# N               = int(Fs*duration)
# marker_channel  = 7 
# i_channel       = 5
# v_channel       = 6   
# m_channel       = 0     
# rf_channel      = 4
# t 				= np.linspace(0, duration, N, endpoint=False)
# start_pause     = int(0.25 * N+1)
# end_pause       = int(0.75 * N-1)
# xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# frequencies     = xf[1:(end_pause-start_pause)//2]

# dfx 		= 1000 

# carrier2 	= carrier +dfx
# sfx = carrier + carrier2
# df_idx = m.find_nearest(frequencies,dfx)
# sf_idx = m.find_nearest(frequencies,sfx)
# carrier_idx = m.find_nearest(frequencies,carrier)
# carrier2_idx = m.find_nearest(frequencies,carrier2)

# outs_array = []
# # Loops through each batch and save the df, sum, and carrier amplitudes.  
# for i in range(len(batch)):
# 	file_number = batch[i]
# 	filename    = savepath + 't'+str(file_number)+'_stream.npy'
# 	data = np.load(filename)
# 	a,b = data.shape
# 	# 
# 	# convert it to microvolts by taking the gain into account. 
# 	fsignal 		= 1e6*data[m_channel]/gain
# 	# 
# 	fft_data        = fft(fsignal[start_pause:end_pause])
# 	fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]
# 	# 
# 	output = [fft_data[df_idx],fft_data[sf_idx],fft_data[carrier_idx],fft_data[carrier2_idx]]
# 	print ('output',output)
# 	outs_array.append(output)

# outs_array = np.array(outs_array)
# outfile                      = str(carrier)+'.npz'
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
# np.savez(outfile,outs_array)
# print ('saved out a data file!')
# 
# 
# It could clearly be mains noise. blarg. 
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(t,fsignal,'k')
# ax2 = fig.add_subplot(212)
# plt.plot(frequencies,fft_data,'k')
# ax2.set_xlim([0,1100])
# ax2.set_ylim([0,1])
# plt.show()

# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(t,1e6*data[m_channel]/gain,'k')
# ax.set_xlim([0,duration])
# # ax.set_xlim([1.755,1.756])
# ax.set_ylabel('Volts ($\mu$V)')
# ax.set_xlabel('Time(s)')
# plt.legend(['measurement channel at focus'],loc='upper right')
# ax2 = fig.add_subplot(212)
# plt.plot(t,10*data[rf_channel],'r')
# ax2.set_xlim([0,duration])
# ax2.set_ylabel('Volts (V)')
# plt.legend(['rf monitor channel'],loc='upper right')
# # ax2.set_xlim([1.755,1.756])
# ax.set_xlabel('time (s)')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = 'draw_data.png'
# plt.savefig(plot_filename)
# plt.show()
# 

# 
# low  = 1000-5
# high = 1000 +5

# sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# f1signal  = sosfiltfilt(sos_low, fsignal)


# low  = 3000-5
# high = 3000 +5

# sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# f2signal  = sosfiltfilt(sos_low, fsignal)


# low  = 1001000 -5
# high = 1001000 +5

# sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# f3signal  = sosfiltfilt(sos_low, fsignal)


# low  = 500000 -5000
# high = 500000 +5000
# high = 10
# sos_low = iirfilter(17, [high], rs=60, btype='lowpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# f4signal  = sosfiltfilt(sos_low, fsignal)

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(t,f4signal,'k')
# # ax.set_xlim([1.755,1.756])
# plot_filename = 'dc_components.png'
# plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(t,f1signal,'k')
# plt.legend(['1kHz filtered'],loc='upper right')
# ax.set_xlim([1.755,1.756])
# ax2 = fig.add_subplot(212)
# plt.plot(t,fsignal,'m')
# ax2.set_xlim([1.755,1.756])
# plt.legend(['raw single pulse'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = 'dcomponents.png'
# plt.savefig(plot_filename)
# plt.show()
# # 
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_data,'k')
# # ax.set_xlim([0,1000000])
# ax.set_xlim([0,5000])
# ax.set_ylim([0,2.5])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = 'dFFT.png'
# plt.savefig(plot_filename)
# plt.show()
# # 
# 

