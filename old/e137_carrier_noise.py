#!/usr/bin/python
'''
 Author: Jean Rintoul Date: 18/01/2023

'''
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib
from scipy import interpolate
import serial, time
from subprocess import check_output
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.fft import fft, fftfreq
import mouse_library as m
import pandas as pd
from datetime import datetime
from scipy.signal import iirfilter,sosfiltfilt
from scipy.signal import hilbert
# 
test_no 					= 7
current_signal_frequency 	= 8    # This is set up for a 2Hz VEP. 
gain                		= 500
Fs                  		= 5e6
carrier             		= 500000
measurement_channel 		= 0
duration 					= 12
bandwidth_of_interest    	= 30 
calculate_nth 				= 5
beta 						= 14 # for the kaiser window. 
prf 						= 1.0
#
print ('current_signal_frequency: ', current_signal_frequency)
start_time = np.round(0.8/duration ,2)
end_time   = np.round((duration - 0.2)/duration,2)
print ('start and end',start_time,end_time)
# 
def demodulate(in_signal,carrier_f,t): 
    return np.abs(in_signal*np.exp(2*np.pi*1j*carrier_f*t))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   

sos_lowpass  = iirfilter(17, [10], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
aeti_variables = {
'type':'demodulation',        # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': Fs,                     # 
'USMEP': 1,                   # when usmep == 1, the current and pressure Fs can be different to the recorded Fs. In this case the US is at 5Mhz, but the recording frequency is 100kHz. This decreases the amount of data that needs to be dumped to disk. 
'duration': duration, 
'position': test_no,
'pressure_amplitude': 0.1,      	# how much is lost through skull??? 400kPa, 0.08 is about 200kPz. 0.15 is about 400kPa. 
'pressure_frequency': carrier,
# 'pressure_prf':prf,             	# pulse repetition frequency for the sine wave. Hz.
# 'pressure_ISI':0.0,             	# inter trial interval in seconds. 
# 'pressure_burst_length': 1.0,  		# burst length in seconds.       
'current_amplitude':0.0,    
'current_frequency':carrier,        # 
# 'current_ISI':0.0,  				# time in seconds between repeats. default length of time for pulse is 0.05s. see c code. 
# 'current_burst_length':1.0,    		# 50ms 
# 'current_prf':prf,
# 'ti_frequency': carrier + dfx,  # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 0,                # the channel of the measurement probe. 
'rf_monitor_channel': 4,        # this output of the rf amplifier.  
'e_channel': 6,                 # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5,   # this is the current measurement channel of the transformer. 
'marker_channel':7,
'end_null': 0.0,                # start of end null. 
'end_pause': end_time,          # start of end ramp
'start_null': 0.1,              # percent of file set to zero at the beginning. 
'start_pause': start_time,      # percent of file in ramp mode or null at start.
'no_ramp':0.0,                  # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                    # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':1,             # the current and voltage monitor both have attenuators on them 
'command_c':'code\\mouse_stream',
'save_folder_path':'D:\\ae_mouse\\e137_ae_neural_decoding',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
# 
# 
timestep = 1.0/Fs
N = int(Fs*duration)
print("expected no. samples:",N)
t               = np.linspace(0, duration, N, endpoint=False)
start_pause     = int(aeti_variables['start_pause'] * N+1)
end_pause       = int(aeti_variables['end_pause'] * N-1)
rf_channel      = aeti_variables['rf_monitor_channel']
marker_channel  = aeti_variables['marker_channel']
m_channel       = aeti_variables['ae_channel'] 
v_channel       = aeti_variables['e_channel'] 
i_channel       = aeti_variables['current_monitor_channel'] 
current_signal_frequency = aeti_variables['current_frequency'] 
acoustic_frequency = aeti_variables['pressure_frequency']
Fs          = aeti_variables['Fs'] 
duration    = aeti_variables['duration'] 
savepath    = aeti_variables['save_folder_path']
window      = np.kaiser( (end_pause-start_pause), beta )
print('window len:',len(window))

# 
# Do a recording and copy it into the experiment folder. 
result, data_out      = m.aeti_recording(**aeti_variables)
data                  = m.time_align_copy_to_folder_and_return_data(**aeti_variables)
fsignal 		      = 1e6*data[m_channel]/gain
# #
# 
start_pause  = int(2*Fs) 
end_pause    = int(10*Fs)
# 
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]

beta        = 20
window      = np.kaiser( (end_pause-start_pause), beta )
print('window len:',len(window))
# 
fft_m  				= fft(fsignal[start_pause:end_pause]*window)
fft_m  				= np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
# 
fft_rf              = fft(data[rf_channel][start_pause:end_pause]*window)
fft_rf              = np.abs(2.0/(end_pause-start_pause) * (fft_rf))[1:(end_pause-start_pause)//2]
# 
# 
# It would be nice to have an SNR metric here going out to 5Hz from the carrier, using a kaiser window. 
# Since there is no definition of signal in this case, it might just be better to do a mean of the noise. 
# This is the start and end point to calculate the SNR from. 
start_idx   = find_nearest(frequencies,carrier + int(1) )   # 1-5Hz near the carrier 
end_idx     = find_nearest(frequencies,carrier + int(5) )
carrier_idx = find_nearest(frequencies,carrier)
noise_mean  = np.mean(fft_m[start_idx:end_idx])
noise_std   = np.std(fft_m[start_idx:end_idx])
print ('noise mean/std, carrier amplitude: ',np.round(noise_mean,2),np.round(noise_std,2),fft_m[carrier_idx])
# 
# 
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(311)
# plt.plot(t,10*data[rf_channel],'r')
plt.plot(t,fsignal,'k')
ax2 = fig.add_subplot(312)
plt.plot(frequencies,fft_m,'k')
ax2.set_xlim([carrier - 10,carrier + 10])
# ax3.set_xlim([0,10])
ax2.set_ylim([0,2])
ax3 = fig.add_subplot(313)
plt.plot(frequencies,fft_rf,'r')
ax3.set_xlim([carrier - 50,carrier + 50])
# ax3.set_xlim([0,10])
# ax3.set_ylim([0,200])
plt.show()
# 
# # Try a template that is an 8 Hz sinusoid.  
# template     = [0]*int(1.0*Fs)     # 1 second 
# for j in range(len(template)):
#     template[j] = np.cos( 2*np.pi*(current_signal_frequency)*j*timestep)  
# # 
# frequencies_of_interest = [carrier+current_signal_frequency,carrier-current_signal_frequency]
# summed_total 			= [0]*t
# average_signal 			= [0]*t
# summed_corrd 			= [0]*t
# marker_summation    	= [0]*t
# rf_summation        	= [0]*t
# # Do averaging. 
# x 						= range(multifile_start,multifile_end)
# print ('x',x )
# iteration_number = 0 
# for i in x:
# 	print (i)
# 	# Divide by this number to get the average.  
# 	iteration_number = iteration_number + 1
# 	aeti_variables['position'] = i
# 	# Do a recording and copy it into the experiment folder. 
# 	result, data_out      = m.aeti_recording(**aeti_variables)
# 	data                  = m.time_align_copy_to_folder_and_return_data(**aeti_variables)
# 	fsignal 		      = 1e6*data[m_channel]/gain
# 	# 
# 	rfsignal            = 10*data[rf_channel]  
# 	marker              = data[marker_channel]
# 	marker_summation    = marker_summation + marker
# 	rf_summation        = rf_summation + rfsignal
# 	# This adds each signal together(they should be time synced), to decrease the SNR. 
# 	if i == 0:
# 		summed_total = fsignal 
# 	else: 
# 		summed_total = summed_total + fsignal
# 		# print (summed_total.shape)
# 	# calculate the average signal. 
# 	average_signal = summed_total/(iteration_number)
# 	# this is the matched filter. 
# 	filtered_signal   = sosfiltfilt(sos_demodulate_bandpass, fsignal) 
# 	demodulate_signal = demodulate(filtered_signal,carrier,t)
# 	# These are the two signals to enter the matched filter.  
# 	demodulate_signal = sosfiltfilt(sos_lowpass,demodulate_signal)
# 	x = demodulate_signal
# 	x = x - x.mean()
# 	# Try a template that is an 8 Hz sinusoid.     
# 	corrd = signal.correlate(x,template, mode='same')
# 	if i == 0:
# 		summed_corrd = corrd 
# 	else: 
# 		summed_corrd = summed_corrd + corrd 
# 	average_corrd = summed_corrd/(iteration_number)
#     # 
# 	# Every 4th one, just to check what I am getting... is it any good? And also calc for the last one. 
# 	if (iteration_number%calculate_nth == 0 or i == (multifile_end-1) ):
# 		# calculate the SNR. It should increase each time due to averaging. 
# 		fft_f  = fft(fsignal[start_pause:end_pause]*window)
# 		fft_f  = np.abs(2.0/(end_pause-start_pause) * (fft_f))[1:(end_pause-start_pause)//2]
# 		fft_mk = fft(average_signal[start_pause:end_pause]*window)
# 		fft_mk = np.abs(2.0/(end_pause-start_pause) * (fft_mk))[1:(end_pause-start_pause)//2]
# 		fft_m  = fft(average_signal[start_pause:end_pause])
# 		fft_m  = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
# 		# 
# 		fft_corrd  = fft(average_corrd[start_pause:end_pause])
# 		fft_corrd  = np.abs(2.0/(end_pause-start_pause) * (fft_corrd))[1:(end_pause-start_pause)//2]
# 		# 
# 		# Plot the calculated FFT. 
# 		fig = plt.figure(figsize=(5,5))
# 		ax = fig.add_subplot(211)
# 		plt.plot(frequencies,fft_mk,'k')
# 		ax.set_xlim([carrier-bandwidth_of_interest,carrier+bandwidth_of_interest])
# 		plt.legend(['fft around carrier signal (Kaiser)'],loc='upper right')
# 		ax2 = fig.add_subplot(212)
# 		plt.plot(frequencies,fft_m,'r')
# 		plt.plot(frequencies,fft_corrd/np.max(fft_corrd),'b')
# 		ax2.set_xlim([0,bandwidth_of_interest])
# 		plt.legend(['fft signal (Rec)','mf demod'],loc='upper right')
# 		plt.show()
# 		# 
# 		# If it looks bad for some reason? For now, just note down the file number.  	
# 		signal_totals                   = []
# 		kaiser_signal_totals            = []
# 		individual_signal_totals        = []
# 		interest_frequencies         	= []
# 		for n in range(len(frequencies_of_interest)): # sum all frequencies per unit time.
# 		  df_idx = find_nearest(frequencies,frequencies_of_interest[n])
# 		  # also eliminate on either side of bins of interest as I don't have great bin resolution.    
# 		  interest_frequencies.append(df_idx)    
# 		  kaiser_signal_totals.append(fft_mk[df_idx])
# 		  signal_totals.append(fft_m[df_idx])
# 		  individual_signal_totals.append(fft_f[df_idx])
# 		#  
# 		# This is the start and end point to calculate the SNR from. 
# 		start_idx = find_nearest(frequencies,carrier + int(current_signal_frequency/2) )   # after the first 5Hz. 
# 		end_idx = find_nearest(frequencies,carrier + bandwidth_of_interest)
# 		# # print ('end idx',end_idx)
# 		dis_kaiser_signal_totals      = []
# 		dis_signal_totals             = []  # 
# 		dis_individual_signal_totals  = []  # 
# 		for n in range(start_idx, end_idx): # sum all frequencies per unit time.
# 			if n not in interest_frequencies: 
# 				dis_kaiser_signal_totals.append(fft_mk[n])
# 				dis_signal_totals.append(fft_m[n])
# 				dis_individual_signal_totals.append(fft_f[n])
# 		# 			
# 		signal_snr         		= 20*np.log(np.mean(signal_totals)/np.mean(dis_signal_totals))
# 		kaiser_signal_snr  		= 20*np.log(np.mean(kaiser_signal_totals)/np.mean(dis_kaiser_signal_totals))
# 		individual_signal_snr  	= 20*np.log(np.mean(individual_signal_totals)/np.mean(dis_individual_signal_totals))
# 		# Print all the calculated SNRs. 
# 		print ('signal snr/kaiser snr/individual_signal snr: ',np.round(signal_snr,2),np.round(kaiser_signal_snr,2),np.round(individual_signal_snr,2 ))
# # 
# # 
# outfile = 'e136_tphantom.npz'
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
# np.savez(outfile,rf_summation=rf_summation,marker_summation=marker_summation,average_signal=average_signal,average_corrd=average_corrd,n=iteration_number)
# print ('saved out a data file!')
# # print ('start and end:',start_pause,end_pause)
# # 
# rfsignal = 10*data[rf_channel]
# fig 	 = plt.figure(figsize=(5,5))
# ax 		 = fig.add_subplot(111)
# plt.plot(t,rfsignal,'k')
# plt.show()
# # 
# fft_mk = fft(average_signal[start_pause:end_pause]*window)
# fft_mk = np.abs(2.0/(end_pause-start_pause) * (fft_mk))[1:(end_pause-start_pause)//2]

# fft_m = fft(average_signal[start_pause:end_pause])
# fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
# # 
# # 
# df = int(abs(carrier - current_signal_frequency))
# print ('df is',df)
# sf = int(abs(carrier + current_signal_frequency))
# carrier_idx = m.find_nearest(frequencies,acoustic_frequency)
# df_idx = m.find_nearest(frequencies,df)
# sf_idx = m.find_nearest(frequencies,sf)
# # 
# # prf_idx = m.find_nearest(frequencies,prf)
# # sig_idx = m.find_nearest(frequencies,current_signal_frequency)
# # prf_df_idx = m.find_nearest(frequencies,(current_signal_frequency+prf) )
# # 
# print ('df/sf/carrier:',frequencies[df_idx],frequencies[sf_idx],frequencies[carrier_idx])
# print ('df and sf:',2*fft_m[df_idx],2*fft_m[sf_idx],2*fft_m[carrier_idx])
# # 
# # print ('df and sf:',2*fft_m[prf_idx],2*fft_m[sig_idx],2*(fft_m[prf_df_idx] ))
# # print ('ratio orig/df:',fft_m[sig_idx]/(fft_m[prf_df_idx] ))

# filtered_data 		  = sosfiltfilt(sos_demodulate_bandpass, average_signal) 
# analytical_signal     = hilbert(filtered_data) # Hilbert demodulate.  
# h_signal              = -np.abs(analytical_signal)
# demodulated_signal    = sosfiltfilt(sos_lowpass, h_signal) 
# # 
# fft_d = fft(demodulated_signal[start_pause:end_pause])
# fft_d = np.abs(2.0/(end_pause-start_pause) * (fft_d))[1:(end_pause-start_pause)//2]


# current_idx 	= find_nearest(frequencies,current_signal_frequency)

# print ('current size: ',fft_m[current_idx]*2)

# plt.rc('font', family='serif')
# plt.rc('font', serif='Arial')
# plt.rcParams['axes.linewidth'] = 2
# saveprefix = '\\images\\'

# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_mk,'k')
# plt.plot(frequencies,fft_m,'g')
# # plt.xlim([0,bandwidth_of_interest])
# plt.ylim([0,200])
# plt.xlim([carrier - bandwidth_of_interest - 5,carrier + bandwidth_of_interest + 5])
# # plt.legend(['kaiser','rectangular'],loc='upper right')
# plt.tight_layout()
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# # plt.legend(['Demodulated \n FFT'],loc='upper right',frameon=False,framealpha=1.0, fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = savepath+saveprefix+'averaged_fft_at_carrier.png'
# plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(311)
# plt.plot(t,data[m_channel],'k')
# # plt.plot(frequencies,fft_mk,'b')
# # plt.plot(frequencies,fft_m,'k')
# # plt.xlim([0,bandwidth_of_interest])
# # plt.ylim([0,500])
# # plt.legend(['kaiser','rectangular'],loc='upper right')
# # plt.plot(t,demodulated_signal,'k')
# # plt.plot(t,rfsignal,'r')
# ax2 = fig.add_subplot(312)
# plt.plot(frequencies,fft_mk,'b')
# plt.plot(frequencies,fft_m,'k')
# # plt.plot(frequencies,fft_d,'r')
# # plt.plot(frequencies,fft_m,'k')
# ax2.set_xlim([0,bandwidth_of_interest])
# # ax2.set_ylim([0,50])
# ax3 = fig.add_subplot(313)
# plt.axvline(x=carrier,color='b')
# plt.axvline(x=carrier-current_signal_frequency,color='r')
# plt.axvline(x=carrier+current_signal_frequency,color='r')
# # plt.plot(frequencies,fft_m,'b')
# plt.plot(frequencies,fft_mk,'k')
# ax3.set_ylim([0,4])
# ax3.set_xlim([carrier-bandwidth_of_interest,carrier+bandwidth_of_interest])

# plt.tight_layout()
# plot_filename = savepath+saveprefix+'FFT_continuous.png'
# plt.savefig(plot_filename)
# plt.show()


