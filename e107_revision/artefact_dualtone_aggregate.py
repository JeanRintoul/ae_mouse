'''

Title: dual sine artefact comparison test. 
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

saved_outputs 	= []
f1 				= 70 
f2 				= 1020 
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx
# 
# file_list 	= [5,6,7,8]  		# the two frequencies are 70Hz and 1020Hz. 
file_list  = [9,10,11,12]  	# the two frequencies are 70Hz and 1020PRF at 500kHz. 

# file_number     = 5
gain            = 1000
# in the mouse yesterday. 
savepath        = 'D:\\ae_mouse\\e107_revision\\t3_phantom\\'
Fs              = 5e6
duration        = 6.0	
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 4
# 
for i in range(len(file_list)):
	file_number = file_list[i]
	# 
	filename    = savepath + 't'+str(file_number)+'_stream.npy'
	data = np.load(filename)
	a,b = data.shape
	t = np.linspace(0, duration, N, endpoint=False)
	# convert it to microvolts by taking the gain into account. 
	fsignal = 1e6*data[m_channel]/gain
	vsignal = data[v_channel]
	rfsignal = 10*data[rf_channel] 
	# 
	# find the fft of the data. 
	start_pause     = int(0.25 * N+1)
	end_pause       = int(0.75 * N-1)
	window          = np.hanning(end_pause-start_pause)
	fft_data        = fft(fsignal[start_pause:end_pause])
	fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]
	xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
	frequencies     = xf[1:(end_pause-start_pause)//2]
	# amplitude at 70Hz
	f1_idx = find_nearest(frequencies,f1)
	f2_idx = find_nearest(frequencies,f2)
	print ('amplitude at applied fields(microvolts) p-p', 2*fft_data[f1_idx],2*fft_data[f2_idx])
	sum_idx = find_nearest(frequencies,f1+f2)
	diff_idx = find_nearest(frequencies,f2-f1)
	print ('amplitude at sum + diff(microvolts) p-p', 2*fft_data[sum_idx],2*fft_data[diff_idx])
	# Noise floor calcultation. 
	decibel_fft = 20*np.log10(abs(fft_data) )
	noise 		= np.median(decibel_fft[40:sum_idx-2])
	# print ('noise',noise)
	# print ('sum and diffs',decibel_fft[diff_idx],decibel_fft[sum_idx])
	# this is the data required to be saved for each loop. 
	print (decibel_fft[f1_idx]-noise,decibel_fft[f2_idx]-noise,decibel_fft[diff_idx]-noise,decibel_fft[sum_idx]-noise)
	# 
	saved_outputs.append([decibel_fft[f1_idx]-noise,decibel_fft[f2_idx]-noise,decibel_fft[diff_idx]-noise,decibel_fft[sum_idx]-noise])

	fig = plt.figure(figsize=(10,6))
	ax = fig.add_subplot(211)
	plt.plot(t,fsignal,'k')
	ax2 = fig.add_subplot(212)
	plt.plot(frequencies,fft_data,'k')
	ax2.set_xlim([0,1300])
	plt.show()
#
# save out an .npz 
# 
outfile                      = 'dual_sine_artefact_test_PRF1020.npz'
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,saved_outputs=saved_outputs)
print ('saved out a data file!')


# 

# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(t,fsignal,'k')
# ax2 = fig.add_subplot(312)
# plt.plot(t,rfsignal,'k')
# ax2 = fig.add_subplot(313)
# plt.plot(t,data[1],'k')

# plot_filename = 'draw_data.png'
# plt.savefig(plot_filename)
# plt.show()
# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(t,1+rfsignal/np.max(rfsignal),'r')
# plt.plot(t,vsignal/np.max(vsignal),'m')
# plt.plot(t,-1-fsignal/np.max(fsignal),'k')
# ax2 = fig.add_subplot(312)
# plt.plot(t,demodulated_signal,'k')
# ax3 = fig.add_subplot(313)
# plt.plot(frequencies,fft_data,'k')
# plt.plot(frequencies,fft_rfdata,'r')
# # ax2.set_xlim([0,1100])
# # ax2.set_xlim([0,5])
# # ax2.set_ylim([0,20])
# # plot_filename = 'draw_data.png'
# # plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(frequencies,fft_data,'k')
# # plt.plot(frequencies,fft_rfdata,'r')
# ax.set_xlim([1000000-200,1000000+200])
# ax2 = fig.add_subplot(212)
# plt.plot(frequencies,fft_data,'k')
# # plt.plot(frequencies,fft_rfdata,'r')
# ax2.set_xlim([500000-200,500000+200])
# # ax2.set_xlim([0,5])
# # ax2.set_ylim([0,20])
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

