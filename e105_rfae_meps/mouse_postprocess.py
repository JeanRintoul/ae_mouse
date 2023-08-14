'''

Copy Xizi's post-processing pipeline as exactly and unimaginatively as possible. 
Author: Jean Rintoul
Date: 31/07/2023

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
# 
# files 1-10. 

file_list_start = 3 
file_list_end  	= 3
file_list = np.linspace(file_list_start,file_list_end,(file_list_end-file_list_start+1),dtype=np.int16)
print ('file list:', file_list)

gain           = 1000
savepath       = 'D:\\ae_mouse\\e105_rfae_meps\\t3_mouse\\'
# 
Fs              = 1e4
duration        = 12.0	
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 4
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx

# Downsample the signal to 600 Hz. 
low 					      = 0.5
new_Fs                        = 500 
downsampling_factor           = int(Fs/new_Fs)
high = new_Fs 
low_filter = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

# print ('new N ',len(dsignal))

outputs = []
for i in range(len(file_list)):
	file_number = file_list[i]
	filename    = savepath + 't'+str(file_number)+'_stream.npy'
	data = np.load(filename)
	a,b = data.shape
	t = np.linspace(0, duration, N, endpoint=False)
	# convert it to microvolts by taking the gain into account. 
	fsignal = 1e6*data[m_channel]/gain
	# rfsignal = 10*data[rf_channel] 
	fsignal = fsignal-np.mean(fsignal)

	# start_pause     = int(0.0 * N)
	# end_pause       = int( N)
	# fft_data        = fft(fsignal[start_pause:end_pause])
	# fft_data        = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]
	# xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
	# frequencies     = xf[1:(end_pause-start_pause)//2]
	# 
	# fig = plt.figure(figsize=(10,6))
	# ax = fig.add_subplot(111)
	# plt.plot(frequencies,fft_data)
	# plt.show()

	# 
	# start_idx = 75
	# end_idx   = 59492
	# s = fsignal[start_idx:end_idx]
	s = fsignal
	signal     = sosfiltfilt(low_filter, s)
	dsignal = signal[::downsampling_factor]
	t = t[::downsampling_factor]
	# concatenate 
	# outputs=np.concatenate((outputs,fsignal[start_idx:end_idx]))
	# 
	if i > 0:
		outputs=outputs+dsignal
	else: 
		outputs = dsignal
	print (outputs.shape)
	# 
	low  = 2.0
	high = 4
	signal_filter = iirfilter(17, [low,high], rs=60, btype='bandpass',
	                       analog=False, ftype='cheby2', fs=new_Fs,
	                       output='sos')
	filtered_signal     = sosfiltfilt(signal_filter, dsignal)


	fig = plt.figure(figsize=(10,6))
	ax = fig.add_subplot(211)
	plt.plot(t,filtered_signal)
	# plt.plot(t[start_idx],filtered_signal[start_idx],'.r')
	# plt.plot(t[end_idx],filtered_signal[end_idx],'.r')	
	ax2 = fig.add_subplot(212)
	plt.plot(dsignal)
	# plt.plot(start_idx,fsignal[start_idx],'.r')
	# plt.plot(end_idx,fsignal[end_idx],'.r')
	plt.show()
# 
# There are a ton of discontinuities in the data. 
# 
outfile                      = 'mouse_demodulation.npz'
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,outputs =outputs,new_Fs=new_Fs)
print ('saved out a data file!')
# 
