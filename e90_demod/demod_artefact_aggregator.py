"""
There appeaer to be sometime discontinuities at the location of the demod VEP.
Even when gain is 1000. What is happening here? Maybe my sample rate is too low for 672khz? 

Maybe there is some strange aliasing effect?
I could: 
1. increase sample rate. 
2. change to lower frequency transducer. i.e. 500kHz. 
3. add a filter that starts at Fs/2 to minimize this aliasing effect. Its possible that higher frequency harmonics are causing this step deviation. 

This also needs to cycle through in an automated fashion. 

"""
import sys
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib
import matplotlib.cm as cm
from scipy import interpolate
from numpy.linalg import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy.signal import hilbert, chirp
from tkinter import *
import matplotlib.colors as colors
from matplotlib.widgets import TextBox
from scipy import signal
from scipy.signal import iirfilter,sosfiltfilt
from scipy.fft import fft,fftshift
import pandas as pd
# 
# f1 = 'demod_pressure_summary_data.npz'
# start_files             = 21  #  pressure connected. 14hz
# end_files               = 31  #  pressure connected. 14hz
# 
f1 = 'demod_nopressure_summary_data.npz'
start_files             = 43  #  no pressure connected. 14 hz
end_files               = 53  #  no pressure connected. 14hz
# 
data_filepath           = 'D:\\mouse_aeti\\e86_demod\\t10_mouse\\'
factor                  = 1  #  this is how many repeats we want in the data. 
vep_factor              = 1 

mains_filter            = False # runs faster without the mains filter - dont really need for high N. 
# Instead of events, if this is called it will just continuously loop until it finishes. Great if I want to go away and do something else. 
automated               = False
files                   = np.linspace(start_files,end_files,(end_files-start_files+1),dtype=np.int16)
gain                    = 1 

fc                      = 500000
led_frequency           = 14 # in Hz. 
duration                = 8.0 # time in S. 
# print ('duration',duration)
led_duration            = 1/(2*led_frequency)
Fs                      = 5e6
timestep                = 1.0/Fs
N                       = int(Fs*duration)
marker_channel          = 7 
m_channel               = 4 
rf_channel              = 2
t                       = np.linspace(0, duration, N, endpoint=False)
# 
low  = 4.5 
high = 100 
# high = 50 
sos_band                = iirfilter(17, [low,high], rs=60, btype='bandpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')
# 
# remove the original VEP signal from the data such that it doesn't interfere with the demodulation. 
l = fc - 2*high 
h = fc + 2*high
sos_carrier_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

# decimation parameters. 
new_Fs                        = 50000 
downsampling_factor           = int(Fs/new_Fs)

# These are the variables which we wish to obtain at the end. 
factor_duration                 = int(led_duration*Fs*factor)
time_segment                    = np.linspace(0, timestep*factor_duration*2, factor_duration*2, endpoint=False) 
time_segment                    = time_segment - led_duration
# print ('factor length:', factor_duration)
idx                             = 0 

def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx

newN = int(factor_duration*2)
window = np.hanning(newN)
xf = np.fft.fftfreq(newN, d=timestep)[:newN//2]
frequencies = xf[1:newN//2] 
print ('bin size:',frequencies[1]-frequencies[0])
Nyquist = int(Fs/2)
frequency_Nyquist_idx   = find_nearest(frequencies,Nyquist)
frequency_idx           = find_nearest(frequencies,high)
print ('f nyquist index, high cut index: ',frequency_Nyquist_idx,frequency_idx)

# Read in the baseline flash. 
filename          = 'chosen_vep_f'+str(vep_factor)+'.npz'
data              = np.load(filename)
chosen_VEP        = data['chosen_VEP']
chosen_marker     = data['chosen_marker']
chosen_VEP = chosen_VEP - np.mean(chosen_VEP)

# these are the lists and array which will be saved out to file for later plotting and analysis. 
total_mm    = np.zeros((factor_duration*2))[::downsampling_factor]
total_rd    = []
total_dd    = []
total_rf    = []
total_rfft  = []
total_dfft  = []

# IQ demodulate function. 
# I use constant to get rid of the rectification around zero issue. 
def demodulate(measured_signal,carrier_f,t):
    # That's interesting, if I add a DC offset here, I obtain the result correctly, otherwise the result is rectified into obscurity. 
    offset = 100
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
# 
#     
for x in range(len(files)):

    file_number     = files[x]
    print ('current file:',str(file_number))
    filename        = data_filepath + 't'+str(file_number)+'_stream.npy'
    data            = np.load(filename)
    a,b             = data.shape
    # print ('shape',a,b)
    rawdata         = 1e6*data[m_channel]/gain
    rawdata         = rawdata - np.mean(rawdata)
    rfdata          = 10*data[rf_channel]
    # this comes out with similar frequencies, and looking like the inverse 180 degree phase shift of the data. 
    fraw                    = sosfiltfilt(sos_carrier_band, rawdata) 
    # print ('fraw len',len(fraw),len(rawdata),len(t) )
    demodulated_signal      = demodulate(fraw,fc,t)
    # filter the raw data the same way as the demod signal is filtered. 
    rawdata                 = sosfiltfilt(sos_band, rawdata)
    demodulated_signal      = sosfiltfilt(sos_band, demodulated_signal)
    markerdata              = np.array(data[marker_channel])
    marker                  = markerdata/np.max(markerdata)
    diffs   = np.diff( markerdata )
    zarray  = t*[0]
    indexes = np.argwhere(diffs > 0.2)[:,0] # leading edge
    marker_up_length = int(led_duration*Fs)
    for i in range(len(indexes)):
        if i > 0:
            zarray[indexes[i]:(indexes[i]+marker_up_length) ] = 1
    markerdata = zarray

    rd = np.zeros((factor_duration*2))
    dd = np.zeros((factor_duration*2))
    mm = np.zeros((factor_duration*2))
    rf = np.zeros((factor_duration*2))
    good_events = 0 
    for i in range(len(indexes)):  #  

        if i > 0 and i%factor == 0 : # ensure there is no overlap between epochs. 
            # control the start and stop points of the epoch. 
            startid = indexes[i] - int(factor_duration/(2*factor) ) 
            stopid  = indexes[i] + int(2*factor_duration - int(factor_duration/(2*factor)  ))
            # print ('start and stop markers',startid,stopid)
            # this is baseline subtraction when its a single VEP. 
            prestimulus_period          = rawdata[startid:indexes[i]]
            demod_prestimulus_period    = demodulated_signal[startid:indexes[i]]
            baseline                    = np.mean(prestimulus_period)
            demod_baseline              = np.mean(demod_prestimulus_period)
            # It's here we need to make a decision about the rf data. 
            rd  = rawdata[startid:stopid]
            dd  = demodulated_signal[startid:stopid]
            mm  = markerdata[startid:stopid]

            mdiffs = np.diff(mm)
            marker_cnt = np.argwhere(mdiffs > 0.2)[:,0] # leading edge
            # print ('marker count len',len(marker_cnt))
            rf  = rfdata[startid:stopid]
            # subtract the baseline. 
            rd  = rawdata[startid:stopid] - baseline
            dd  = demodulated_signal[startid:stopid] - demod_baseline
            # 
            # Now match the shape of the VEP with the chosen VEP shape. 
            df              = pd.DataFrame({'x': rd, 'y': chosen_VEP })
            window          = len(rd)
            window          = int(len(rd)/(10*factor*2) ) # this is the number of samples to be used in the rolling cross-correlation. 
            rolling_corr    = df['x'].rolling(window).corr(df['y'])
            # 
            median_corrs         = []
            vep_locations        = []
            area_of_interest_len = 300000
            for p in range(factor):
                # vep_location = round ((2+p*4) *(2*factor_duration) / (factor*4 ))
                vep_location = round ((2+p*4) *(2*factor_duration) / (factor*4 ))                
                vep_locations.append(vep_location)
                correlation = np.nanmedian(rolling_corr[vep_location:(vep_location+area_of_interest_len)])
                median_corrs.append(correlation)
            # print ('median corrs',median_corrs)
            median_corr = np.median(median_corrs)
            print ('mean corr',median_corr)
            # 
            # Really I just want a high correlation at the part of the file where there is the VEP.     
            # fig = plt.figure(figsize=(10,6))
            # ax  = fig.add_subplot(211)
            # plt.plot(rd)
            # plt.plot(mm)
            # for n in range(factor):
            #     plt.axvline(x=vep_locations[n],color='r')
            #     plt.axvline(x=vep_locations[n]+area_of_interest_len,color='r')    
            # # plt.axvline(x=vep_locations[1],color='r')
            # # plt.axvline(x=vep_locations[1]+area_of_interest_len,color='r')                      
            # ax2 = fig.add_subplot(212)
            # plt.plot(rolling_corr)
            # plt.plot(mm)
            # plt.suptitle(str(median_corr))
            # plt.show()
            # 
            # Append only the good veps, and where the full array has markers until the end. 
            if median_corr > 0.4 and len(marker_cnt) >= factor: 
                print ('good one!')
                good_events += 1
                # Do FFT calcs here before downsampling occurs to retain maximum bin size. 
                # Now calculate the FFTs to save out to file. 
                fft_rawdata = fft(rd*window)
                fft_rawdata = np.abs(2.0/newN * (fft_rawdata))[1:newN//2]
                fft_demoddata = fft(dd*window)
                fft_demoddata = np.abs(2.0/newN * (fft_demoddata))[1:newN//2]
                # Put the fft into an array, such that it can later be processed. 
                total_rfft.append(fft_rawdata[0:frequency_idx])
                total_dfft.append(fft_demoddata[0:frequency_idx])
                # Decimate the data, and save into big arrays for later plotting. 
                total_rd.append(rd[::downsampling_factor])
                total_dd.append(dd[::downsampling_factor])
                total_mm = np.add(total_mm,mm[::downsampling_factor])           
                # total_rf.append(rf[::downsampling_factor])
                # 

    # if good_events > 0:
    #     mm = mm/n_events
    #     dd = dd/n_events
    #     rd = rd/n_events
    #     rf = rf/n_events
            # Decide later if I want to do epoch removal.  
            # Now we want to eliminate flashes where the pressure changed suddenly.                

outfile=f1   # save out the data. 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile, total_rd=total_rd, total_dd=total_dd,frequencies=frequencies[0:frequency_idx],total_rfft=total_rfft,total_dfft=total_dfft,total_mm=total_mm)
print ('saved out a data file!')




