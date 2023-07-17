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

# t2 saline phantom test. 1000 gain. 
#  5hz acoustically connected files. 
acoustic_connection 	= [2,3,4]
# #  5hz not acoustically connected files. 
no_acoustic_connection 	= [26,27,28]


# 
file_list_start = 44
file_list_end   = 84
acoustic_connection = np.linspace(file_list_start,file_list_end,(file_list_end-file_list_start+1),dtype=np.int16)


#  5hz not acoustically connected files. 
# no_acoustic_connection  = [26,27,28]

file_list_start = 44
file_list_end   = 46
no_acoustic_connection = np.linspace(file_list_start,file_list_end,(file_list_end-file_list_start+1),dtype=np.int16)


files 					= no_acoustic_connection
# data_filepath           = 'D:\\mouse_aeti\\e96_demod\\t2\\'
data_filepath           = 'D:\\mouse_aeti\\e82_ae_demod\\t9\\'

f1                      = 'no_acoustic_connection_summary_data.npz'

# def demodulate(measured_signal,carrier_f,t):
#     offset              = np.min(measured_signal)
#     offset_adjustment   = offset*np.cos(2 * np.pi * carrier_f * t)
#     IQ                  = (measured_signal+offset_adjustment)*np.exp(1j*(2*np.pi*carrier_f*t )) 
#     idown               = np.real(IQ)
#     qdown               = np.imag(IQ)
#     idown               = sosfiltfilt(lp_filter, idown)
#     qdown               = sosfiltfilt(lp_filter, qdown)  
#     rsignal             = -(idown + qdown)
#     rsignal             = rsignal - np.mean(rsignal) 
#     return rsignal

# def demodulate(measured_signal,carrier_f,t):
#     # That's interesting, if I add a DC offset here, I obtain the result correctly, otherwise the result is rectified into obscurity. 
#     offset = 1000
#     offset_adjustment = offset*np.sin(2 * np.pi * carrier_f * t)
#     IQ = (measured_signal+offset_adjustment)*np.exp(1j*(2*np.pi*carrier_f*t )) 
#     # idown = measured_signal*np.cos(2*np.pi*carrier_f*t)
#     # qdown = -measured_signal*np.sin(2*np.pi*carrier_f*t)    
#     idown = np.real(IQ)
#     qdown = np.imag(IQ)
#     v = idown + 1j*qdown
#     mag = np.abs(v)
#     mag = mag - np.mean(mag)
#     return mag

# define the carrier frequency. 
fc          = 500000
Fs          = 5e6
timestep    = 1.0/Fs
duration    = 8.0
N           = int(Fs*duration)
gain        = 1000
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0  
rf_channel      = 2  

# print("expected no. samples:",N)
t               = np.linspace(0, duration, N, endpoint=False)
start_pause     = int(0.25 * N+1)
end_pause       = int(0.875 * N-1)

# print ('start and end:',start_pause,end_pause)
# low                     = 2.5
# high                    = 300
# sos_band                = iirfilter(17, [high], rs=60, btype='lowpass',
#                             analog=False, ftype='cheby2', fs=Fs,
#                             output='sos')
# # 
newN = end_pause - start_pause
# window = np.hanning(newN)
xf 						= np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies 			= xf[1:(end_pause-start_pause)//2]
# frequency_idx           = m.find_nearest(frequencies,high)
# # 
# l = fc - 2*high
# h = fc + 2*high
# sos_carrier_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')


# filter_cutoff           = 200
# lp_filter               = iirfilter(17, [filter_cutoff], rs=60, btype='lowpass',
#                             analog=False, ftype='cheby2', fs=Fs,
#                             output='sos')



# 
# total_rawdata  = []
# total_demod    = []
# total_rf       = []
# total_rfft     = []
# total_dfft     = []
# total_rffft    = []
total_fft_vs   = []
# total_fft_ms   = []

# xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# frequencies     = xf[1:(end_pause-start_pause)//2]


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

    fft_m = fft(rawdata[start_pause:end_pause])
    fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
    

    total_fft_vs.append(fft_m)
    # 
    # 
    # fig = plt.figure(figsize=(6,4))
    # ax  = fig.add_subplot(111)
    # plt.plot(frequencies,fft_m,'k')
    # # ax.set_xlim([0,duration])
    # # ax.set_ylim([-350,350])
    # # ax.set_ylabel('Volts ($\mu$V)')
    # # ax.set_xlabel('Time (s)')
    # # Save the figure and show
    # plt.tight_layout()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.savefig('fft_data.png')
    # plt.show()
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

# total_td = td
#  
outfile=f1   # save out the data. 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,total_fft_vs=total_fft_vs,f = frequencies)
print ('saved out a data file!')


