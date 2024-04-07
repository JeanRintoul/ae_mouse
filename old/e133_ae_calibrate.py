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
# 

gain                = 500
# 
Fs                  = 5e6
carrier             = 500000
measurement_channel = 0
# this is the variable to change. 
# dfx     = 2
# prf     = 1020
# compute the start and end points to be accurate. 
duration = 4
test_no  = 1
# 
# current_signal_frequency = carrier - 8000
# 
current_signal_frequency = 498020
print ('current_signal_frequency: ', current_signal_frequency)
start_time = np.round(0.8/duration ,2)
end_time   = np.round((duration - 1.0)/duration,2)
print ('start and end',start_time,end_time)
# 
# 
aeti_variables = {
'type':'demodulation',        # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': Fs,                     # 
'USMEP': 1,                   # when usmep == 1, the current and pressure Fs can be different to the recorded Fs. In this case the US is at 5Mhz, but the recording frequency is 100kHz. This decreases the amount of data that needs to be dumped to disk. 
'duration': duration, 
'position': test_no,
'pressure_amplitude': 0.08,        # how much is lost through skull??? 400kPa, 0.08 is about 200kPz. 0.15 is about 400kPa. 
'pressure_frequency': carrier,
# 'pressure_prf':prf,             # pulse repetition frequency for the sine wave. Hz.
# 'pressure_ISI':0.0,             # inter trial interval in seconds. 
# 'pressure_burst_length': 0.1,   # burst length in seconds. 
'current_amplitude':0.05,       
'current_frequency':current_signal_frequency,   # 
# 'current_ISI':0.0,  				# time in seconds between repeats. default length of time for pulse is 0.05s. see c code. 
# 'current_burst_length':0.05,    	# 50ms 
# 'current_prf':0,
# 'ti_frequency': carrier + dfx,  # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 0,                # the channel of the measurement probe. 
'rf_monitor_channel': 4,        # this output of the rf amplifier.  
'e_channel': 6,                 # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5,   # this is the current measurement channel of the transformer. 
'marker_channel':7,
'end_null': 0.0,                # start of end null. 
'end_pause': end_time,          # start of end ramp
'start_null': 0.05,             # percent of file set to zero at the beginning. 
'start_pause': start_time,      # percent of file in ramp mode or null at start.
'no_ramp':0.0,                  # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                    # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':1,             # the current and voltage monitor both have attenuators on them 
'command_c':'code\\mouse_stream',
'save_folder_path':'D:\\ae_mouse\\e133_phantom_ae_prf_amplitudes',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
# 
# x = range(1,11)
# print ('x',x )
# for i in x:
#     print (i)
#     aeti_variables['position'] = i
#  Do a recording and copy it into the experiment folder. 
result, data_out            = m.aeti_recording(**aeti_variables)
data                        = m.copy_to_folder_and_return_data(**aeti_variables)
# 
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# plt.plot(data[1],'k')
# # ax.set_xlim([start,stop])
# plt.show()
#
# # #  
rf_channel = aeti_variables['rf_monitor_channel']
marker_channel = aeti_variables['marker_channel']
m_channel = aeti_variables['ae_channel'] 
v_channel = aeti_variables['e_channel'] 
i_channel = aeti_variables['current_monitor_channel'] 
current_signal_frequency = aeti_variables['current_frequency'] 
acoustic_frequency = aeti_variables['pressure_frequency']
Fs       = aeti_variables['Fs'] 
duration = aeti_variables['duration'] 
savepath = aeti_variables['save_folder_path']
#  
timestep = 1.0/Fs
N = int(Fs*duration)
print("expected no. samples:",N)
t               = np.linspace(0, duration, N, endpoint=False)
start_pause     = int(aeti_variables['start_pause'] * N+1)
end_pause       = int(aeti_variables['end_pause'] *N-1)
# print ('start and end:',start_pause,end_pause)

# this is if I use the preamp. 
fsignal 		      = 1e6*data[m_channel]/gain

fft_m = fft(fsignal[start_pause:end_pause])
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
# 
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 

df = int(abs(carrier - current_signal_frequency))
print ('df is',df)
sf = int(abs(carrier + current_signal_frequency))

carrier_idx = m.find_nearest(frequencies,acoustic_frequency)
df_idx = m.find_nearest(frequencies,df)
sf_idx = m.find_nearest(frequencies,sf)
# 
# prf_idx = m.find_nearest(frequencies,prf)
# sig_idx = m.find_nearest(frequencies,current_signal_frequency)
# prf_df_idx = m.find_nearest(frequencies,(current_signal_frequency+prf) )
# # 
print ('df/sf/carrier:',frequencies[df_idx],frequencies[sf_idx],frequencies[carrier_idx])
print ('df and sf:',2*fft_m[df_idx],2*fft_m[sf_idx],2*fft_m[carrier_idx])
# 
# print ('df and sf:',2*fft_m[prf_idx],2*fft_m[sig_idx],2*(fft_m[prf_df_idx] ))
# print ('ratio orig/df:',fft_m[sig_idx]/(fft_m[prf_df_idx] ))
# 
# 
# vdata = data[v_channel]
# idata = 0.1*data[i_channel] # 100mA per V i.e. 100*data[i_channel] gives i in mA. 
# impedance = np.max(vdata)/np.max(idata)
# print ('v,i(ma),z:',np.max(vdata),np.max(1000*idata),impedance)

plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2

# 
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(311)
plt.plot(t,data[v_channel],'k')
plt.plot(t,data[m_channel],'r')
ax2 = fig.add_subplot(312)
plt.axvline(x=frequencies[df_idx],color='r')
plt.axvline(x=frequencies[sf_idx],color='r')
plt.plot(frequencies,fft_m,'k')
# ax2.set_xlim([carrier - current_signal_frequency - 1000,carrier + current_signal_frequency + 1000])
ax2.set_xlim([carrier - current_signal_frequency - 100,carrier - current_signal_frequency + 100])
# ax2.set_xlim([0,1040])
ax2.set_ylim([0,1])
# ax2.set_xlim([0,1e6])
ax3 = fig.add_subplot(313)
plt.plot(t,10*data[rf_channel],'k')
plt.show()


# rf_channel = 2
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# plt.plot(t,10*data[rf_channel])
# plt.show()

# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(311)
# plt.plot(t,data[0],'k')
# ax.set_xlim([start,stop])
# ax2 = fig.add_subplot(312)
# plt.plot(t,lfp_data,'k')
# ax3 = fig.add_subplot(313)
# plt.plot(t,data[rf_channel],'r')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# # plt.yticks(fontsize=16)
# # plt.xticks(fontsize=16)
# # plt.yticks([])
# # plt.xticks([])
# plt.tight_layout()
# plot_filename = savepath + '\\t'+str(test_no)+'_output_wave.png'
# plt.savefig(plot_filename)
# plt.show()

# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# # plt.plot(t,data[measurement_channel],'k')
# # plt.plot(t,data[rf_channel],'k')
# # # plt.plot(t,data[v_channel],'k')
# # ax2 = fig.add_subplot(212)
# plt.plot(frequencies,fft_m,'k')
# ax.set_xlim([0,110])
# # ax3 = fig.add_subplot(313)
# # plt.plot(t,data[i_channel],'r')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # ax2.spines['right'].set_visible(False)
# # ax2.spines['top'].set_visible(False)
# # ax3.spines['right'].set_visible(False)
# # ax3.spines['top'].set_visible(False)
# plot_filename = savepath + '\\t'+str(test_no)+'_fft_wave.png'
# plt.savefig(plot_filename)
# plt.show()
# # # # 



# = = 
# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(t,data[measurement_channel],'k')
# ax2 = fig.add_subplot(312)
# plt.plot(t,lfp_data,'k')
# plt.plot(t,10*data[measurement_channel]/np.max(data[measurement_channel]),'r')
# plt.legend(['lfp'],loc='upper right')
# ax3 = fig.add_subplot(313)
# plt.plot(t,spike_data,'k')
# plt.plot(t,10*data[measurement_channel]/np.max(data[measurement_channel]),'r')
# plt.legend(['spike'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plot_filename = savepath + '\\t'+str(test_no)+'_estim.png'
# plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(411)
# plt.plot(t,data[m_channel],'k')
# ax2 = fig.add_subplot(412)
# plt.plot(t,data[v_channel],'k')
# ax3 = fig.add_subplot(413)
# plt.plot(t,data[i_channel],'k')
# ax4 = fig.add_subplot(414)
# # plt.plot(t,data[rf_channel],'k')
# # plt.plot(frequencies,fft_us,'k')
# plt.show()
# # 
# print ('carrier:',2*fft_m[carrier_idx])
# print ('rf antenna f:',2*fft_m[v_idx])
# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(411)
# plt.plot(frequencies,fft_v,'r')
# ax.set_ylabel('Volts(V)')
# plt.legend(['voltage output monitor'],loc='upper right')

# ax2 = fig.add_subplot(412)
# plt.plot(frequencies,fft_m,'k')
# ax2.set_ylabel('Volts ($\mu$V)')
# ax2.set_xlim([carrier-20000,carrier+20000])
# ax2.set_ylim([0,200])
# # ax2.set_xlim([400000,600000])
# ax2.set_xlabel('Frequencies(Hz)')
# plt.legend(['measurement in electrolyte'],loc='upper right')

# ax3 = fig.add_subplot(413)
# plt.plot(frequencies,fft_m,'k')
# ax3.set_xlim([0,dfx*1.5])
# # ax3.set_ylim([0,200])
# plt.legend(['dfx close up'],loc='upper right')

# ax4 = fig.add_subplot(414)
# plt.plot(frequencies,fft_m,'k')
# ax4.set_xlim([2*carrier,2*carrier+dfx*2])
# # ax4.set_ylim([0,200])
# plt.legend(['sfx close up'],loc='upper right')

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# ax4.spines['right'].set_visible(False)
# ax4.spines['top'].set_visible(False)

# plt.suptitle('RF transmitter/receiver test')
# # plot_filename = savepath + '\\rf_receiver_test.png'
# plot_filename = savepath + '\\t'+str(test_no)+'_rf.png'
# plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(t,data[v_channel],'r')

# ax2 = fig.add_subplot(212)
# plt.plot(t,lfp_data,'k')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = savepath + '\\lfp_filtered.png'
# plt.savefig(plot_filename)
# plt.show()

# First plot is the raw signal, 
# Second plot is the 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# # plt.plot(t,10*data[rf_channel],color='b')
# # plt.plot(t,10*data[v_channel],color='r')
# plt.plot(frequencies,fft_us,'b')
# plt.plot(frequencies,fft_v,'r')
# # plt.plot(frequencies,fft_m,'k')
# plt.legend(['fft us','fft v'],loc='upper right')
# ax.set_xlim([0,int(np.max(frequencies))/2])

# ax2 = fig.add_subplot(212)
# plt.axvline(df,color='k')
# plt.axvline(sf,color='k')
# # plt.plot(frequencies,fft_us,'b')

# plt.plot(frequencies,1e6*fft_v,'r')
# plt.plot(frequencies,fft_m,'c')
# # plt.plot(frequencies,fft_i,'orange')
# # ax2.set_ylim([0,500])
# ax2.set_xlim([acoustic_frequency - 2*current_signal_frequency,acoustic_frequency + 2*current_signal_frequency])
# # ax2.set_xlim([0,40])
# ax2.set_ylim([0,2000])
# plt.legend(['fft us','fft v'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plt.suptitle('AE location calibration')
# plot_filename = savepath + '\\t'+str(test_no)+'_datacheck.png'
# plt.savefig(plot_filename)
# plt.show()
# 
# 
# At low frequencies, I don't have enough repeats, and I have too much Johnson noise
# to see the difference frequency. I definitely cannot see it in the 
# df = abs(current_signal_frequency - acoustic_frequency)
# print ('df:', df)

# # low  = df - 4
# if low < 0:
#     low = 0.1
# high = df+4
# sos_lowdf = iirfilter(17, [low,high], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# fsignal  = sosfiltfilt(sos_lowdf, 1e6*data[m_channel]/gain )
# fvsignal  = sosfiltfilt(sos_lowdf, data[v_channel] )
# # 
# subsection = fsignal[start_pause:end_pause]
# df_pp_height = np.max(subsection) -np.min(subsection)
# print ('df pp height:', df_pp_height)
# 
# plot of what comes through the preamp
# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(311)
# # plt.plot(t,10*data[rf_channel],'r')
# # plt.plot(t,data[v_channel],'k')
# plt.plot(10*data[rf_channel],'r')
# plt.plot(data[v_channel],'k')
# plt.legend(['rf chan','v chan'],loc='upper right')
# ax2  = fig.add_subplot(312)
# # plt.plot(t,fsignal,'k')
# plt.plot(fsignal,'k')
# plt.legend(['tight filtered df'],loc='upper right')
# ax3  = fig.add_subplot(313)
# plt.plot(frequencies,fft_m,'k')
# ax3.set_xlim([0,100])
# ax3.set_ylim([0,1000])
# plt.legend(['measurement channel fft'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plot_filename = savepath + '\\t'+str(test_no)+'low_frequency_df.png'
# plt.savefig(plot_filename)
# plt.show()


# 
# plot of the output currents and voltages. 
# fig = plt.figure(figsize=(10,6))
# ax  = fig.add_subplot(311)
# plt.plot(t,data[v_channel],'r')
# plt.legend(['v monitor raw'],loc='upper right')
# ax2  = fig.add_subplot(312)
# plt.plot(t,i_data,'k')
# plt.legend(['i monitor'],loc='upper right')
# ax3  = fig.add_subplot(313)
# plt.plot(frequencies,fft_v,'r')
# plt.plot(frequencies,fft_i,'k')
# ax3.set_xlim([0,int(np.max(frequencies))/2])
# plt.legend(['voltage out fft(V)','current fft'],loc='upper right')
# # plot_filename = savepath + '\\t'+str(test_no)+'_v_i_vhannel.png'
# # plt.savefig(plot_filename)
# plt.show()




