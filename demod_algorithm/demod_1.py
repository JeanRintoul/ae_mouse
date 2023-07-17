'''

Title: IQ demodulation realdata
Function: Perform IQ demodulation. 
This function has a rolling window pearson cross-correlation. 

Author: Jean Rintoul
Date: 18.03.2022

Updated: 22.05.2022


'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import kaiserord
from scipy import signal
from scipy.signal import iirfilter,sosfilt
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
#
# I still have the rectification problem in the demodulation result. 
#
def demodulate(measured_signal):
    # That's interesting, if I add a DC offset here, I obtain the result correctly, otherwise the result is rectified into obscurity. 
    # adding an offset changes the signals recovered size.. 
    offset = 0
    offset_adjustment = offset*np.sin(2 * np.pi * carrier_f * t)
    IQ = (measured_signal+offset_adjustment)*np.exp(1j*(2*np.pi*carrier_f*t )) 
    idown = np.real(IQ)
    qdown = np.imag(IQ)
    idown = measured_signal*np.cos(2*np.pi*carrier_f*t)
    qdown = -measured_signal*np.sin(2*np.pi*carrier_f*t)    
    v = idown + 1j*qdown
    # calculate the phase in order to figure out the sign of the recovered wave. 

    mag   = np.abs(v)               # magnitude
    phase = np.unwrap(np.angle(v))  # instantaneous phase
    inst_freq = np.diff(phase)/(2*np.pi)*Fs #instantaneous frequency
    # Regenerate the carrier from the instantaneous phase
    regenerated_carrier = np.cos(phase)
    # 
    rsignal = mag

    rsignal      = sosfiltfilt(lp_filter, rsignal)
    return mag,phase,regenerated_carrier,inst_freq,rsignal

def demodulate2(measured_signal):
    IQ = measured_signal*np.exp(1j*(2*np.pi*carrier_f*t )) 
    idown = np.real(IQ)
    qdown = np.imag(IQ)
    # 
    idown = sosfiltfilt(lp_filter,idown)
    qdown = sosfiltfilt(lp_filter,qdown)
    # 
    v = idown + 1j*qdown
    # calculate the phase in order to figure out the sign of the recovered envelope. 
    mag     = np.abs(v)             # magnitude
    phase = np.unwrap(np.angle(v))  # instantaneous phase
    inst_freq = np.diff(phase)/(2*np.pi)*Fs #instantaneous frequency
    # Regenerate the carrier from the instantaneous phase
    regenerated_carrier = np.cos(phase)
    #  
    regenerated_carrier[regenerated_carrier >= 0] = 1
    regenerated_carrier[regenerated_carrier < 0] = 0

    rsignal = mag
    for i in range(len(regenerated_carrier)):
        if regenerated_carrier[i]>0:
            rsignal[i] = -mag[i]
        else:
            rsignal[i] = mag[i]

    return mag,phase,regenerated_carrier,inst_freq,rsignal

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx    

# mouse saline phantom. 
# filename = 'mouse_saline_phantom.npy'
# ae_channel   = 6
# rf_channel   = 2
# vmeasure_channel    = 5
# duration = 0.1   # uren phantom duration
# duration = 8.0  #  mouse saline phantom duration
# start_slice  = 3.0
# end_slice    = 3.3
# frequency_of_interest = 70

# uren phantom
filename = 'uren_demod_stream.npy'
rf_channel          = 0    #  this is the rf amplifier output monitor. 
v_channel           = 4    #  this is output voltage monitor after David Bono's voltage transformer i.e. the voltage being applied across the stimulation electrodes. This measurement uses a 10x attenuator so I don't draw any of the power away. 
ae_channel          = 6    #  this is measuring the acoustoelectric voltage i.e. it goes through a hardware high pass filter before being amplified. 
vmeasure_channel    = 5    #  this is measured voltage before the hardware filter. It is measured with a 10x attenuator so as not to impact the ampliture of the signal going through the hardware filter. 
duration            = 0.1   # uren phantom duration
start_slice         = 0.03
end_slice           = 0.08
frequency_of_interest = 15000 

# mouse
# filename            = 't7_stream.npy'
# ae_channel          = 6
# rf_channel          = 2
# vmeasure_channel    = 6
# duration     = 4.0    # mouse saline phantom duration
# start_slice  = 1.0
# end_slice    = 1.4
# frequency_of_interest = 70 


Fs          = 5e6     # sample rate
timestep    = 1.0/Fs
N           = int(Fs*duration)
print("expected no. samples:",N)
t           = np.linspace(0, duration, N, endpoint=False)
xf          = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies = xf[1:N//2]
data        = np.load(filename)

carrier_f               = 500000
filter_cutoff           = 20000
lp_filter               = iirfilter(17, [filter_cutoff], rs=60, btype='lowpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')
# 
lcut                    = carrier_f - 2*frequency_of_interest 
hcut                    = carrier_f + 2*frequency_of_interest 
input_band_filter       = iirfilter(17, [lcut,hcut], rs=60, btype='bandpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')
# 
input_cut_filter = iirfilter(17, [carrier_f-5,carrier_f+5], rs=60, btype='bandstop',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')
# 

# the demodulated signal will have a phase lag due to the time it takes for acoustic signal to travel.. or capacitance? 
# 

input_signal            = sosfiltfilt(input_band_filter, data[ae_channel])
input_signal            = sosfiltfilt(input_cut_filter, input_signal)



demodulated_signal,phase,regenerated_carrier,inst_freq,rsignal  = demodulate2(input_signal )
# demodulated_signal,phase,regenerated_carrier,inst_freq,rsignal = demodulate(input_signal )

raw_v_at_electrode      = sosfiltfilt(lp_filter, data[ae_channel])

# remove the starting ramp and ending ramp. 
val,idx_start   = find_nearest(t,start_slice)
val,idx_end     = find_nearest(t,end_slice)

# clip the start and end areas away where the current signal is ramping. 
raw_v_at_electrode = raw_v_at_electrode[idx_start:idx_end]
demodulated_signal = demodulated_signal[idx_start:idx_end]
t_cut              = t[idx_start:idx_end]
phase              = phase[idx_start:idx_end]
re_carrier         = regenerated_carrier[idx_start:idx_end]
inst_freq          = inst_freq[idx_start:idx_end]
rsignal             = rsignal[idx_start:idx_end]

# Correctly conpute the lag between the two signals. 
x = raw_v_at_electrode
y = rsignal
correlation = signal.correlate(x-np.mean(x), y - np.mean(y), mode="full")
lags = signal.correlation_lags(len(x), len(y), mode="full")
idx_lag = -lags[np.argmax(correlation)]
print ('lag is',idx_lag)
idx_lag = 0 
#
if idx_lag > 0:
    raw_v_at_electrode = raw_v_at_electrode[:(len(demodulated_signal)-idx_lag)]
    t_cut              = t_cut[idx_lag:]
    demodulated_signal = demodulated_signal[idx_lag:]
    re_carrier = re_carrier[idx_lag:]
    inst_freq = inst_freq[idx_lag:]
    phase = phase[idx_lag:]
    rsignal = rsignal[idx_lag:]
else: 
    raw_v_at_electrode = raw_v_at_electrode[-idx_lag:]
    t_cut              = t_cut[-idx_lag:]
    demodulated_signal = demodulated_signal[:(len(demodulated_signal)+idx_lag)]
    re_carrier = re_carrier[-idx_lag:]
    inst_freq = inst_freq[-idx_lag:]
    phase = phase[-idx_lag:]
    rsignal = rsignal[-idx_lag:]

# remove the mean.  
raw_v_at_electrode = raw_v_at_electrode-np.mean(raw_v_at_electrode)

df = pd.DataFrame({'x': raw_v_at_electrode, 'y': rsignal })
window          = 12000 #this is the number of samples to be used in the rolling cross-correlation. 
# window          = len(demodulated_signal)
rolling_corr    = df['x'].rolling(window).corr(df['y'])

print ('rolling corr max:',np.max(rolling_corr))  # 
result = np.nanmedian(rolling_corr)
print ('median corr: ',result)
max_index = np.argmax(rolling_corr) 


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(3,1,1)
# plt.plot(t_cut,demodulated_signal,'black')
plt.plot(t_cut,np.max(rsignal)*raw_v_at_electrode/np.max(raw_v_at_electrode),'r')
plt.plot(t_cut,rsignal,'k')
# plt.legend(['raw v at electrode','demodulated signal'],loc ='lower right')
ax2 = fig.add_subplot(3,1,2)

plt.plot(t_cut,re_carrier/np.max(re_carrier),'black')
plt.plot(t_cut,demodulated_signal/np.max(demodulated_signal),'red')
# plt.plot(t_cut,inst_freq,'purple')
# plt.plot(t_cut,phase,'purple')

ax3 = fig.add_subplot(3,1,3)
plt.plot(t_cut,abs(rolling_corr),'.b')
plt.legend(['abs(correlation) window:'+str(window)],loc ='lower right')
plt.show()





