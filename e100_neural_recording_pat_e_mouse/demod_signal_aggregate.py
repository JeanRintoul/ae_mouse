'''

Title: vep signal inspection
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
# 
# Convincing ones. 
# the first part of file 2 is AWESOME in terms of demodulation accuracy. 
# file 5. 
# file 7 has a little bit of good, then it goes bad at acoustic cut point. 
# file 10 has a little good. little
# file 11 little good. 
# 
# file 15 has a clear phase inversion. 
# 
# file 16 is awesome. 
# file 18 has good bits. 
# 19 is pretty awesome. 
# file 21 has awesome parts. 

# 3 appears to have phase inversion. 
# Each time there is a break in the RF data in amplitude, something weird happens to the demodulated signal. 
# file 5 is pretty good. 
# 6 is bad. 
# 
# We need to use a PRF to avoid reflections dominating the modulated data? 
# Can we have a larger signal at 1kHz? 
# 
file_list_start  = 22  # acoustically not connected.
file_list_end    = 42
# 
# file_list_start  = 2   # acoustically connected.
# file_list_end    = 21
# 
file_list = np.linspace(file_list_start,file_list_end,(file_list_end-file_list_start+1),dtype=np.int16)
print ('file list:', file_list)
# 
# 
file_number     = 26
gain            = 1000
savepath        = 'D:\\mouse_aeti\\e100_neural_recording_pat_e_mouse\\t3_delta_wave_demodulation\\'
Fs              = 5e6
duration        = 8.0
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     
rf_channel      = 4 

def demodulate(measured_signal):
    offset              = np.min(measured_signal)
    offset_adjustment   = offset*np.cos(2 * np.pi * carrier_f * t)
    IQ                  = (measured_signal+offset_adjustment)*np.exp(1j*(2*np.pi*carrier_f*t )) 
    idown               = np.real(IQ)
    qdown               = np.imag(IQ)
    idown               = sosfiltfilt(lp_filter, idown)
    qdown               = sosfiltfilt(lp_filter, qdown)  
    rsignal             = -(idown + qdown)
    rsignal             = rsignal - np.mean(rsignal) 
    return rsignal

carrier_f               = 500000
# 
# the demodulation accuracy is proportional to the size of the harmonic sum, as a proxy for frequency mixing. 
# carrier_f = carrier_f*2
# 
filter_cutoff           = 200


low  = 1
# high = 1000 
high = 16
high = 40
sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')


sos_center_cut = iirfilter(17, [carrier_f-2,carrier_f+2], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

# define the carrier frequency. 
fc = carrier_f
l = fc - high
h = fc + high
sos_carrier_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

lp_filter               = iirfilter(17, [filter_cutoff], rs=60, btype='lowpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')


carrier_filter   = iirfilter(17, [carrier_f-2,carrier_f+2], rs=60, btype='bandpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')

double_carrier_filter   = iirfilter(17, [carrier_f*2-2,carrier_f*2+2], rs=60, btype='bandpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')

s_data = []
corr_data =[]
for i in range(len(file_list)):
    file_number = file_list[i]

    filename    = savepath + 't'+str(file_number)+'_stream.npy'
    data = np.load(filename)
    a,b = data.shape
    t = np.linspace(0, duration, N, endpoint=False)
    # convert it to microvolts by taking the gain into account. 
    fsignal = 1e6*data[m_channel]/gain
    unfiltered_data = fsignal
    rf_data = 10*data[rf_channel]
    # do the demodulation
    fraw           = sosfiltfilt(sos_carrier_band, fsignal) 
    fraw           = sosfiltfilt(sos_center_cut, fraw) 
    # 
    demod_data     = demodulate(fraw)
    # 
    demodulated     = sosfiltfilt(sos_low, demod_data)
    fsignal         = sosfiltfilt(sos_low, fsignal)
    # 
    df              = pd.DataFrame({'x': np.real(fsignal), 'y': np.real(demodulated) })
    window          = 5000000 # correlate over a whole second. 
    rolling_corr    = df['x'].rolling(window).corr(df['y'])
    # print ('len rolling corr:',len(rolling_corr),len(t))
    rolling_corr = rolling_corr*rolling_corr
    # 
    carrier_signal         = sosfiltfilt(carrier_filter, unfiltered_data)
    double_carrier_signal  = sosfiltfilt(double_carrier_filter, unfiltered_data)
    # 
    start_time  = int(0.25*N)
    end_time    = int(0.8*N)
    # simpler if I just take the median  
    corr_portion = np.median(rolling_corr[start_time:end_time])

    corr_data.append(corr_portion)

    selected_corrs = []
    for j in range(len(rolling_corr)):
        if np.max(double_carrier_signal[j:j+100]) > 400 :     # if the sum frequency is bigger than 500 microvolts in amplitude. 
            # print ('big un')
            selected_corrs.append(rolling_corr[j])
    if len(selected_corrs) <= 1:
        selected_corrs.append(0)
    selected = np.median(selected_corrs)
    s_data.append(selected)
    print ('median corr and selected:', corr_portion,selected)
    print ('corr_data shape:',len(corr_data))

outfile                      = 'no_acoustic_connection_corrs.npz'
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,corr_data=corr_data,s_data = s_data)
print ('saved out a data file!')
