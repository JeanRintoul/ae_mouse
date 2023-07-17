"""
Aggregate summary files. 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import iirfilter,sosfiltfilt
from scipy.fft import fft,fftshift
import sys
sys.path.append('D:\\mouse_aeti')  #  so that we can import from the parent folder. 
import mouse_library as m

# t3 saline phantom test. 10 gain. 
#  7hz acoustically connected files. 
# acoustic_connection 	= [5,6,7,8,9,10]
# #  7hz not acoustically connected files. 
# no_acoustic_connection 	= [13,14,15,16,17,18]


# t4 saline phantom test. 10 gain. 
#  7hz acoustically connected files. 
# acoustic_connection     = [29,30,31,32,33,35,35]
# #  7hz not acoustically connected files. 
# no_acoustic_connection  = [15,16,17,18,19,20]

# t5 mouse phantom test. v electrodes only. 
#  7hz acoustically connected files. 
acoustic_connection     = [12,13,4]
acoustic_connection     = [14]

#  iso tests: 
# acoustic_connection     = [4,5,6,7,8,9]
# iso level = [2.5,2.0,1.5,1.0,0.6,2.5]
# max microvolt level = [3740,4800,5136,4300,5760,3900 ]
# there may be an upward trend... as iso is decreased for the peak value. 
# but I don't have enough data to say. Really no real trend. 
# 

#  7hz not acoustically connected files. 
no_acoustic_connection  = [23,24,25,26]


# t2 mouse test. 1000 gain. 
# acoustic_connection = np.linspace(9,41,(41-9+1))
# no_acoustic_connection= np.linspace(42,72,(72-42+1))
# print ('no_acoustic_connection',no_acoustic_connection)

files 					= acoustic_connection
# data_filepath           = 'D:\\mouse_aeti\\e90_demod\\t2\\'
# data_filepath           = 'D:\\mouse_aeti\\e90_demod\\t4_phantom_artefact_noise_reduction\\'
data_filepath           = 'D:\\mouse_aeti\\e94_ae_amplitudes\\t1\\'

f1                      = 'acoustic_connection_summary_data.npz'



def demodulate(measured_signal,carrier_f,t):
    # That's interesting, if I add a DC offset here, I obtain the result correctly, otherwise the result is rectified into obscurity. 
    offset = 1000
    offset_adjustment = offset*np.sin(2 * np.pi * carrier_f * t)
    IQ = (measured_signal+offset_adjustment)*np.exp(1j*(2*np.pi*carrier_f*t )) 
    # idown = measured_signal*np.cos(2*np.pi*carrier_f*t)
    # qdown = -measured_signal*np.sin(2*np.pi*carrier_f*t)    
    idown = np.real(IQ)
    qdown = np.imag(IQ)
    v = idown + 1j*qdown
    mag = np.abs(v)
    mag = mag - np.mean(mag)
    return mag

# define the carrier frequency. 
fc          = 500000
Fs          = 5e6
timestep    = 1.0/Fs
duration    = 4.0
N           = int(Fs*duration)

m_channel   = 0
rf_channel  = 2
v_channel   = 6             # this is the voltage measured between the stimulator probes. 
current_monitor_channel = 5

gain        = 10
# print("expected no. samples:",N)
t               = np.linspace(0, duration, N, endpoint=False)
start_pause     = int(0.25 * N+1)
end_pause       = int(0.875 * N-1)

# print ('start and end:',start_pause,end_pause)
low                     = 2.5
high                    = 300
sos_band                = iirfilter(17, [high], rs=60, btype='lowpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')
# 
xf 						= np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies 			= xf[1:(end_pause-start_pause)//2]
frequency_idx           = m.find_nearest(frequencies,high)
# 
l = fc - 2*high
h = fc + 2*high
sos_carrier_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')


l = 50
h = 150
# l = 500000 + 499905 - 50 
# h = 500000 + 499905 + 50 
sos_df_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')


newN = end_pause - start_pause
window = np.hanning(newN)
# 
total_rawdata  = []
total_demod    = []
total_rf       = []
total_rfft     = []
total_dfft     = []
total_rffft    = []
total_fft_vs   = []
total_fft_ms   = []

for i in range(len(files)):
    file_number     = int(files[i])
    print ('current file:',str(file_number))
    filename        = data_filepath + 't'+str(file_number)+'_stream.npy'
    data            = np.load(filename)
    a,b             = data.shape
    # print ('shape',a,b)
    rawdata         = 1e6*data[m_channel]/gain
    rfdata          = 10*data[rf_channel]
    vdata           = 1e6*data[v_channel]
    # to avoid spectral smearing, remove the sharp on/off transients and retain only the periodic data.
    # reduce spectral leakage from filters. 
    # t2       = t[start_pause:end_pause]    
    # rawdata2 = rawdata[start_pause:end_pause]
    # rfdata2  = rfdata[start_pause:end_pause]
    # 
    #     
    vcraw           = sosfiltfilt(sos_df_band, vdata) 
    fraw            = sosfiltfilt(sos_df_band, rawdata) 
    demod_data      = demodulate(fraw,fc,t)
    rawdata         = rawdata - np.mean(rawdata)
    # Now filter both data the same way. 
    filtered_rawdata    = sosfiltfilt(sos_band, rawdata)
    filtered_demoddata  = sosfiltfilt(sos_band, demod_data)
    # 
    fft_m = fft(rawdata[start_pause:end_pause]*window)
    fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
    
    fft_v = fft(vdata[start_pause:end_pause]*window)
    fft_v = np.abs(2.0/(end_pause-start_pause) * (fft_v))[1:(end_pause-start_pause)//2]
    
    # 
    fft_rf= fft(rfdata[start_pause:end_pause]*window)
    fft_rf = np.abs(2.0/(end_pause-start_pause) * (fft_rf))[1:(end_pause-start_pause)//2]
    # 
    fft_demod = fft(filtered_demoddata[start_pause:end_pause]*window)
    fft_demod = np.abs(2.0/(end_pause-start_pause) * (fft_demod))[1:(end_pause-start_pause)//2]
	#    
    # fft_demod = fft(filtered_demoddata)
    # fft_demod = np.abs(2.0/(end_pause-start_pause) * (fft_demod))[1:(end_pause-start_pause)//2]
    # 
	# Decimate filtered rawdata, filtered demoddata and save to file. 
	#    
    # Decimate arrays before plotting, so it is less memory intensive. 
    desired_plot_sample_rate = 1e5
    decimation_factor 		 = int(Fs/desired_plot_sample_rate)  
    filtered_rawdata 		 = filtered_rawdata[::decimation_factor]      
    filtered_demoddata 		 = filtered_demoddata[::decimation_factor]    
    td 						 = t[::decimation_factor]  	

	#  
	#  
    total_rfft.append(fft_m[0:frequency_idx])
    total_dfft.append(fft_demod[0:frequency_idx])
    # Decimate the data, and save into big arrays for later plotting. 
    total_rawdata.append(filtered_rawdata)
    total_demod.append(filtered_demoddata)
    total_rffft.append(fft_rf[0:frequency_idx])
    total_fft_ms.append(fft_m)
    total_fft_vs.append(fft_v)
    # 
    # 
    fig = plt.figure(figsize=(6,4))
    ax  = fig.add_subplot(111)
    plt.plot(t,vcraw,'k')
    ax.set_xlim([0,duration])
    # ax.set_ylim([-350,350])
    ax.set_ylabel('Volts ($\mu$V)')
    ax.set_xlabel('Time (s)')
    # Save the figure and show
    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('carrier_filtered.png')
    plt.show()
    # 
    #     
    # plot the carrier filtered time series data. 
    # fig = plt.figure(figsize=(6,4))
    # ax  = fig.add_subplot(111)
    # plt.plot(t,fraw,'k')
    # ax.set_xlim([0,8])
    # # ax.set_ylim([-350,350])
    # ax.set_ylabel('Volts ($\mu$V)')
    # ax.set_xlabel('Time (s)')
    # # Save the figure and show
    # plt.tight_layout()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.savefig('carrier_filtered_saline.png')
    # plt.show()

total_td = td
#  
# What plots do I actually want? 
# a bar graph with mean and std of with and without an acoustic connection. 
# recreate a plot with all demod and raw data summed? 
# 
# Another simple test is to see if the mixing frequency occurs on the RF channel? i.e. is there a 7hz? 
# 
outfile=f1   # save out the data. 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile, td= td,total_fft_vs=total_fft_vs,total_rawdata=total_rawdata,total_fft_ms=total_fft_ms,total_rffft=total_rffft ,total_demod=total_demod,frequencies=frequencies[0:frequency_idx],f = frequencies,total_rfft=total_rfft,total_dfft=total_dfft)
print ('saved out a data file!')

# #  this data is somewhat filtered. 
# filename          	= 'demod_pressure_summary_data.npz'
# data              	= np.load(filename)
# d_total_rd       	= np.array(data['total_rd'])
# d_total_dd          = np.array(data['total_dd'])
# d_frequencies       = data['frequencies']
# d_total_rfft        = np.array(data['total_rfft'])
# d_total_dfft        = np.array(data['total_dfft'])
# marker       		= data['total_mm']

# a,b = d_total_rd.shape
# print ('with pressure shape',d_total_rd.shape)

# nfilename          	= 'demod_nopressure_summary_data.npz'
# # nfilename          	= 'demod_nopressure_summary_data300f3.npz'
# ndata              	= np.load(nfilename)
# n_total_rd       	= np.array(ndata['total_rd'])
# n_total_dd          = np.array(ndata['total_dd'])
# n_frequencies       = ndata['frequencies']
# n_total_rfft        = np.array(ndata['total_rfft'])
# n_total_dfft        = np.array(ndata['total_dfft'])
# nmarker       		= ndata['total_mm']

# c,d = n_total_rd.shape
# print ('without pressure shape',n_total_rd.shape)
# # 
# n_events,N 	= d_total_rd.shape
# Fs 			= 5e6 
# timestep 	= 1/Fs 
# duration 	= timestep*N
# t        	= np.linspace(0, duration, N, endpoint=False)
# # 
# # pressure applied with acoustic connection
# average_VEP 		= np.mean(d_total_rd,axis=0)
# std_VEP     		= np.std(d_total_rd,axis=0)
# average_demod_VEP 	= np.mean(d_total_dd,axis=0)
# std_demod_VEP     	= np.std(d_total_dd,axis=0)
# average_VEP_fft 	= np.mean(d_total_rfft,axis=0)
# average_demod_fft 	= np.mean(d_total_dfft,axis=0)
# std_VEP_fft 		= np.std(d_total_rfft,axis=0)
# std_demod_fft 		= np.std(d_total_dfft,axis=0)

# # no acoustic connection 
# n_average_VEP 			= np.mean(n_total_rd,axis=0)
# n_std_VEP     			= np.std(n_total_rd,axis=0)
# n_average_demod_VEP 	= np.mean(n_total_dd,axis=0)
# n_std_demod_VEP     	= np.std(n_total_dd,axis=0)
# n_average_VEP_fft 		= np.mean(n_total_rfft,axis=0)
# n_average_demod_fft 	= np.mean(n_total_dfft,axis=0)
# n_std_VEP_fft 			= np.std(n_total_rfft,axis=0)
# n_std_demod_fft 		= np.std(n_total_dfft,axis=0)
# # 
# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(111)
# ax.plot(t,np.max(average_demod_VEP)*marker/np.max(marker),'g')
# ax.plot(t,-average_demod_VEP/np.max(-average_demod_VEP),'k')
# ax.plot(t,n_average_demod_VEP/np.max(n_average_demod_VEP),'r')
# ax.plot(t,average_VEP/np.max(average_VEP),'b')

# # plt.fill_between(t, n_average_demod_VEP - n_std_demod_VEP, n_average_demod_VEP + n_std_demod_VEP,
# #                   color='r', alpha=0.5)
# # plt.fill_between(t, average_demod_VEP - std_demod_VEP, average_demod_VEP + std_demod_VEP,
# #                   color='k', alpha=0.5)


# # plt.legend(['led on/off','acoustic connection','no acoustic_connection','real VEP'])
# plt.legend(['led on/off','acoustic connection','no acoustic_connection','vep'])
# ax.set_xlim([0,np.max(t)])
# ax.set_ylabel('norm Volts ($\mu$V)')
# ax.set_xlabel('Time(s)')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = 'demod_vep_individual_trials.png'
# plt.savefig(plot_filename)
# plt.show()


# # # FFT 
# fig = plt.figure(figsize=(10,6))
# ax2  = fig.add_subplot(111)
# plt.plot(d_frequencies,average_demod_fft/a,'k')
# plt.plot(n_frequencies,n_average_demod_fft/c,'r')
# plt.legend(['acoustic connection','no acoustic_connection'])
# ax2.set_xlim([0,np.max(d_frequencies)])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = 'vep_individual_trials_fft.png'
# plt.savefig(plot_filename)
# plt.show()
