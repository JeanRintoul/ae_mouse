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
# 
test_no = 4
gain    = 500
# 
# This control of the sample rate is to avoid aliasing in the recorded electric signal. 
# The sample rate should be an integer sub-multiple of the 
# current_frequency = 499990
# 
# Fs = current_frequency*10 
# Fs = 5e6
dfx = 1000 
# 
aeti_variables = {
'type':'demodulation',        # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 5e6,                    # 
# 'Fs': Fs,                   # 
# 'Fs': 1e6,                  # this will change just the sample rate, and not the f generated rate if those two lines are left uncommented. 
# 'USMEP': 1,                 # when usmep == 1, the current and pressure Fs can be different to the recorded Fs. In this case the US is at 5Mhz, but the recording frequency is 100kHz. This decreases the amount of data that needs to be dumped to disk. 
'duration': 4.0, 
'position': test_no,
'pressure_amplitude': 0.05,      # how much is lost through skull??? 400kPa, 0.08 is about 200kPz. 0.15 is about 400kPa. 
'pressure_frequency': 500000.0,
'pi_frequency': 500000 +dfx,
'current_amplitude': 0.0,      # its actually a voltage .. Volts. 
'current_frequency': 8000,     # 
# 'current_frequency': current_frequency, # 
# 'current_frequency': 499996,  # 
#'current_frequency': 500000,   # 
#'current_frequency': 500200,   # 
# 'current_frequency': 4,       # 
# 'current_amplitude': 6.0,     # 
# 'ti_frequency': 0,            # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 0,                # the channel of the measurement probe. 
'rf_monitor_channel': 4,        # this output of the rf amplifier.  
# 'e_channel': 6,                 # this is the voltage measured between the stimulator probes. 
# 'current_monitor_channel': 5,   # this is the current measurement channel of the transformer. 
'e_channel': 6,                 # this is the voltage measured between the stimulator probes. 
'current_monitor_channel': 5,   # this is the current measurement channel of the transformer. 
'marker_channel':7,
'end_null': 0.0,               # start of end null. 
'end_pause': 1.0,             # start of end ramp
# 'start_null': 0.36,           # percent of file set to zero at the beginning. 
# 'start_pause': 0.4,           # percent of file in ramp mode or null at start.
# 'end_null': 0.0,               # start of end null. 
# 'end_pause': 1.0,             # start of end ramp
# 'start_null': 0.0,           # percent of file set to zero at the beginning. 
# 'start_pause': 0.0,           # percent of file in ramp mode or null at start.
'start_null': 0.05,            # percent of file set to zero at the beginning. 
'start_pause': 0.1,            # percent of file in ramp mode or null at start.
'no_ramp':0.0,                  # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':gain,                    # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':1,             # the current and voltage monitor both have attenuators on them 
'command_c':'code\\mouse_stream',
'save_folder_path':'D:\\ae_mouse\\e103_phantom_tests',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}
# 
# 
# 
frequencies_list = list(np.linspace(30,int(1e6),num=100))
# frequencies_list = [100000,500000,1000000] # frequencies for dummy test. 
print ('frequencies', frequencies_list)
# 
Fs       = aeti_variables['Fs'] 
duration = aeti_variables['duration'] 
timestep = 1.0/Fs
N = int(Fs*duration)
t               = np.linspace(0, duration, N, endpoint=False)
start_pause     = int(aeti_variables['start_pause'] * N+1)
end_pause       = int(aeti_variables['end_pause'] *N-1)
savepath = aeti_variables['save_folder_path']
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 
# 
data_response       = []
for i in range(len(frequencies_list)): 
  # f =  round(frequencies[i],2)
  f = round(frequencies_list[i],2)
  print ('f',f)
  aeti_variables['pressure_frequency'] = f
  aeti_variables['pi_frequency'] = f + 1000 
  print('pressure_frequency(Hz): ',f)  
  aeti_variables['position']  = f   # this is to label the file differently. 
  result, data_out            = m.aeti_recording(**aeti_variables)
  data                        = m.copy_to_folder_and_return_data(**aeti_variables)
  # get the rf channel of the data and save off the fft amplitude at the pressure frequency. 
  rf_channel = aeti_variables['rf_monitor_channel']
  m_channel = aeti_variables['ae_channel']
  fft_us = fft(data[rf_channel][start_pause:end_pause])
  fft_us = np.abs(2.0/(end_pause-start_pause) * (fft_us))[1:(end_pause-start_pause)//2]
  carrier_idx = m.find_nearest(frequencies,f)
  amplitude_rf = fft_us[carrier_idx]
  #
  dual_carrier_idx = m.find_nearest(frequencies,(f+1000) )  
  amplitude_rf2 = fft_us[dual_carrier_idx]  
  #
  fft_m = 1e6*fft(data[m_channel][start_pause:end_pause])/gain
  fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]
  #
  carrier_amplitude_measured = fft_m[carrier_idx]
  dual_carrier_amplitude_measured = fft_m[dual_carrier_idx]  
  # 
  diff_frequency_idx = m.find_nearest(frequencies,1000)  
  sum_frequency_idx = m.find_nearest(frequencies,2*f+1000)   
  #
  df_amplitude_measured   = fft_m[diff_frequency_idx]
  sum_amplitude_measured  = fft_m[sum_frequency_idx]  
  # 
  df_amplitude_rf   = fft_us[diff_frequency_idx]
  sum_amplitude_rf  = fft_us[sum_frequency_idx]  
  # 
  # 
  data_response.append([f,amplitude_rf,amplitude_rf2,df_amplitude_rf,sum_amplitude_rf,carrier_amplitude_measured,dual_carrier_amplitude_measured,df_amplitude_measured,sum_amplitude_measured])


#  #  #  #  #  #  #  #  #  #  #  #  #  #  #
d = np.array(data_response)
print ('d shape',d.shape,d)
d_f          = d[:,0]
d_rf1        = d[:,1]
d_rf2        = d[:,2]
d_rf_df      = d[:,3]
d_rf_sf      = d[:,4]
d_mf_c1      = d[:,5]
d_mf_c2      = d[:,6]
d_mf_df      = d[:,7]
d_mf_sf      = d[:,8]

outfile                      = savepath + '\\dual_wave.npz'
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,d=d)
print ('saved out a data file!')

#  #  Now save all these out to file. #  #  #  #  #  #  #  #  #  #  #
# print ('d_f',d_f)

# print ('d_rf',d_rf)

# print ('d_mf',d_mf)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(311)
plt.plot(d_f,d_rf1,'-k.')
plt.plot(d_f,d_rf2,'-r.')
ax.set_ylabel('Volts (V)')
plt.legend(['rf monitor c1','rf monitor c2'],loc='upper right')
ax.set_xlim([0,1000010])
ax2 = fig.add_subplot(312)
plt.plot(d_f,d_mf_c1,'-k.')
plt.plot(d_f,d_mf_c2,'-r.')
plt.legend(['measured carrier 1 amplitudes@electrode', 'measured carrier 1 amplitudes@electrode'],loc='upper right')
ax2.set_ylabel('Volts ($\mu$V)')
ax2.set_xlabel('Frequency(Hz)')
ax2.set_xlim([0,1000010])

ax3 = fig.add_subplot(313)
plt.plot(d_f,d_mf_df,'-k.')
plt.plot(d_f,d_mf_sf,'-r.')
plt.legend(['difference amplitudes@electrode', 'sum amplitudes@electrode'],loc='upper right')
ax3.set_ylabel('Volts ($\mu$V)')
ax3.set_xlabel('Frequency(Hz)')
ax3.set_xlim([0,1000010])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
plt.suptitle('RF versus measured carrier response amplitude')
plot_filename = savepath + '\\t'+str(test_no)+'_rf_response.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(d_f,d_rf1/np.max(d_rf1),'-k.')
plt.plot(d_f,d_rf2/np.max(d_rf2),'-b.')
plt.plot(d_f,d_mf_c1/np.max(d_mf_c1),'-r.')
plt.plot(d_f,d_mf_c2/np.max(d_mf_c2),'-m.')
ax.set_xlim([0,1000010])
ax.set_ylabel('Normalized Volts (V)')
plt.legend(['rf monitor c1','rf monitor c2','measured carrier 1 amplitude@electrode','measured carrier 2 amplitude@electrode'],loc='upper right')
ax2.set_ylabel('normalized')
ax2.set_xlabel('Frequency(Hz)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.suptitle('RF versus measured carrier response shape(normalized)')
plot_filename = savepath + '\\t'+str(test_no)+'_rf_response_normalized.png'
plt.savefig(plot_filename)
plt.show()



fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(d_f,d_mf_df/np.max(d_mf_df),'-k.')
plt.plot(d_f,d_mf_sf/np.max(d_mf_sf),'-r.')

plt.plot(d_f,d_rf_df/np.max(d_rf_df),'-b.')
plt.plot(d_f,d_rf_sf/np.max(d_rf_sf),'-m.')
ax.set_xlim([0,1000010])
ax.set_ylabel('Normalized Volts (V)')
plt.legend(['difference measured','sum measured','rf difference','rf sum'],loc='upper right')
ax.set_xlabel('Frequency(Hz)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.suptitle('Difference and sum frequencies')
plot_filename = savepath + 'diffsum_electrode_response_normalized.png'
plt.savefig(plot_filename)
plt.show()
# # 
