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
# Increment this for each test. 
# ch5 has the differential measurement of the function generator. 
# 
first = 125
last  = 140

test_no  = 3
gain     = 500
duration = 6 
# 
prf     = 1000     # 
Fs      = 5e6   # this is the max frequency I can have on the recording. The function gneerators can go higher though. 
carrier = 2000

dfx     = 10 # This means we should get one dfx cycle in the pulse. 

measurement_channel         = 0
time_to_start_in_seconds    = 1.0
time_to_end_in_seconds      = 0.8
start_time_null_in_seconds  = 0.5
end_time_null_in_seconds    = 0.0
start_null_time = np.round(start_time_null_in_seconds/duration ,2)
end_null_time   = np.round(end_time_null_in_seconds/duration ,2)
start_time      = np.round(time_to_start_in_seconds/duration ,2)
end_time        = np.round((duration - time_to_end_in_seconds)/duration,2)
print ('start and end',start_time,end_time,start_null_time,end_null_time)

aeti_variables = {
'type':'demodulation',        # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': Fs,                     # 
# 'Fs': Fs,                   # 
# 'Fs': 1e6,                  # this will change just the sample rate, and not the f generated rate if those two lines are left uncommented. 
'USMEP': 1,                   # when usmep == 1, the current and pressure Fs can be different to the recorded Fs. In this case the US is at 5Mhz, but the recording frequency is 100kHz. This decreases the amount of data that needs to be dumped to disk. 
'duration': duration,         # 
'position': test_no,          # 
'pressure_amplitude': 12,    # how much is lost through skull??? 400kPa, 0.08 is about 200kPz. 0.15 is about 400kPa. 
'pressure_frequency': carrier,
# 'pi_frequency': carrier + dfx, 
# 'pressure_fswitching2': carrier+dfx, # both signals are output on the antenna. 
# 'pressure_ISI':0,
# 'pressure_prf':prf,             # pulse repetition frequency for the sine wave. Hz. 
# 'pressure_burst_length':0.01,   # in seconds(maxes out at 50% duty cycle). pressure burst length is calculated as: prf_counter/gen_pressure_sample_frequency
# 'jitter_range':0.0,
'current_amplitude': 6,         # its actually a voltage .. Volts. 
'current_frequency': carrier+dfx,   # 
# 'ti_frequency': carrier + dfx,  # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
# 'current_fswitching': carrier + dfx,
# 'current_ISI':0,
# 'current_prf':prf,      # twice per second. 
# 'current_burst_length':0.0001, # Duration should be? 
'ae_channel': 0,                  # the channel of the measurement probe. 
'rf_monitor_channel': 4,          # this output of the rf amplifier.  
# 'e_channel': 6,                 # this is the voltage measured between the stimulator probes. 
# 'current_monitor_channel': 5,   # this is the current measurement channel of the transformer. 
'e_channel': 6,                   # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5,     # this is the current measurement channel of the transformer. 
'marker_channel':7,
# 'end_null': 0.1,                # start of end null. 
# 'end_pause': 0.8,               # start of end ramp
# 'start_null': 0.2,              # percent of file set to zero at the beginning. 
# 'start_pause': 0.3,             # percent of file in ramp mode or null at start.
'end_null': end_null_time,        # start of end null. 
'end_pause': end_time,            # start of end ramp
'start_null': start_null_time,    # percent of file set to zero at the beginning. 
'start_pause': start_time,        # percent of file in ramp mode or null at start.
'no_ramp':0.0,                    # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                      # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':1,               # the current and voltage monitor both have attenuators on them 
'command_c':'code\\mouse_stream',
'save_folder_path':'D:\\ae_mouse\\e128_USbringup_phantom',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
# 
carriers =[40,100, 500, 1000, 1500, 2000, 4000, 8000, 10000, 30000, 50000,100000,250000,500000,750000,1000000,2000000]

thingstosave = []
for i in range(len(carriers)):
	print ('iteration/carrier:',i,carriers[i])
	aeti_variables['current_frequency'] = carriers[i]+dfx
	aeti_variables['pressure_frequency'] = carriers[i]
	aeti_variables['position'] = np.round(carriers[i],0)
	#  Do a recording and copy it into the experiment folder. 
	result, data_out            = m.aeti_recording(**aeti_variables)
	data                        = m.copy_to_folder_and_return_data(**aeti_variables)
	# #  
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
	fsignal             = 1e6*data[m_channel]/gain
	# this is a stand in for my testing signal. 
	fsignal         = 1e6*data[1]
	vsignal         = data[v_channel]

	rfdata              = 10*data[rf_channel]
	fft_m = fft(fsignal[start_pause:end_pause])
	fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
	fft_v = fft(vsignal[start_pause:end_pause])
	fft_v = np.abs(2.0/(end_pause-start_pause) * (fft_v))[1:(end_pause-start_pause)//2]

	xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
	frequencies = xf[1:(end_pause-start_pause)//2]
	dfx_idx = m.find_nearest(frequencies,dfx)
	carrier_idx = m.find_nearest(frequencies,carriers[i])
	prf_idx = m.find_nearest(frequencies,prf)
	print ('amplitudes of interest dfx,carrier,vout:',2*fft_m[dfx_idx],2*fft_m[carrier_idx],2*fft_v[carrier_idx])
	# calculate the current passing through. 
	resistor_current_mon    = 49.9  # 49.9 Ohms for current monitor, 
	# Now calculate the resistance and the reactance.   
	i_data = np.max(data[i_channel])/resistor_current_mon
	v_data = 1*np.max(data[v_channel])
	z = v_data/i_data 
	print ('i(ma),v,z',i_data*2*1000, v_data*2,z)
	thingstosave.append([i_data,v_data,z,2*fft_m[dfx_idx],2*fft_m[carrier_idx],2*fft_v[carrier_idx]])


# 
outfile = aeti_variables['save_folder_path']+'\\transmission_info.npz'
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,thingstosave=thingstosave)
print ('saved out a data file!')

things = np.array(thingstosave)
print ('things shape: ',things.shape)
dfs = things[:,3]
carriers_amplitudes = things[:,4]
carriers_amplitudes_mon = things[:,5]
zs = things[:,2]
print ('dfs:',dfs )
print ('carrier amplitudes:',carriers_amplitudes )


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(411)
plt.plot(carriers,dfs,'k')
ax2 = fig.add_subplot(412)
plt.plot(carriers,carriers_amplitudes,'r')
ax3 = fig.add_subplot(413)
plt.plot(carriers,zs,'b')
ax4 = fig.add_subplot(414)
plt.plot(carriers,carriers_amplitudes_mon,'g')
plot_filename = aeti_variables['save_folder_path']+'\\carrier_vs_dfs.png'
plt.savefig(plot_filename)
plt.show()
# 
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(411)
# plt.plot(t,data[v_channel],'k')
# ax2 = fig.add_subplot(412)
# plt.plot(t,data[i_channel],'k')
# ax3 = fig.add_subplot(413)
# plt.plot(frequencies,fft_m,'k')
# ax3.set_xlim([0,100])
# ax3.set_ylim([0,10000])
# ax4 = fig.add_subplot(414)
# plt.plot(t,data[1],'k')
# plt.show()


